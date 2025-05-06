# File name  :    tensor_expert.py
# Author     :    xiaocuicui
# Time       :    2024/07/09 10:28:25
# Version    :    V1.0
# Abstract   :        

import os
import sys
file_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(file_path)
sys.path.append(os.path.join(file_path, '../../../communication/collective_communication/'))
sys.path.append(os.path.join(file_path, '../../../components/'))

from typing import List, Tuple

from reducescatter import reducescatter, reducelocal
from allgather import allgather
from allreduce import allreduce
from alltoall import ordertoorder, alltoall

from NodeNetwork import NodeNetwork


class tensor_expert(object):

    def __init__(self, topology: str, algorithm: dict):

        self.allreduce = allreduce(topology, algorithm["allreduce"])
        self.allgather = allgather(topology, algorithm["allgather"])
        self.alltoall = alltoall(topology, algorithm["alltoall"])
        self.reducescatter = reducescatter(topology, algorithm["reducescatter"])
        self.ordertoorder = ordertoorder(topology, algorithm["ordertoorder"])
        self.reducelocal = reducelocal(topology, algorithm["reducelocal"])


    def base(
        self,
        whole_nodes: NodeNetwork, current_event_tag: int, current_dependency_list: List[List[List[int]]], 
        source_nodes_coordinates_list: List[Tuple[int]],
        source_x_number: int, source_y_number: int, source_z_number: int,
        topology_x_limitation: int, topology_y_limitation: int, topology_z_limitation: int,
        data_parallelism_degree: List[int], top_k: int, 
        message_flits: int,     
        reduction: float,
        medium_nodes_coordinates_list=None,
        medium_x_number=None, medium_y_number=None, medium_z_number=None,
        target_nodes_coordinates_list=None,
        target_x_number=None, target_y_number=None, target_z_number=None,
        latency=None, bandwidth=None            
    ):
        
        group_x_number = source_x_number // data_parallelism_degree[0]
        group_y_number = source_y_number // data_parallelism_degree[1]
        group_z_number = source_z_number // data_parallelism_degree[2]

        groups_nodes_coordinates_list = []
        for group_z_idx in range(data_parallelism_degree[2]):
            for group_y_idx in range(data_parallelism_degree[1]):
                for group_x_idx in range(data_parallelism_degree[0]):
                    group = []
                    for z_idx in range(group_z_number):
                        for y_idx in range(group_y_number):
                            for x_idx in range(group_x_number):
                                group_node_z = z_idx + group_z_idx * group_z_number
                                group_node_y = y_idx + group_y_idx * group_y_number
                                group_node_x = x_idx + group_x_idx * group_x_number
                                group_node_idx = group_node_z * source_y_number * source_x_number + group_node_y * source_x_number + group_node_x
                                group.append(source_nodes_coordinates_list[group_node_idx])
                    groups_nodes_coordinates_list.append(group)

        # first allreduce
        groups_current_dependency_list = []
        for group_z_idx in range(data_parallelism_degree[2]):
            for group_y_idx in range(data_parallelism_degree[1]):
                for group_x_idx in range(data_parallelism_degree[0]):
                    group = []
                    for z_idx in range(group_z_number):
                        group_z_list = []
                        for y_idx in range(group_y_number):
                            group_row_list = []
                            for x_idx in range(group_x_number):
                                group_node_z = z_idx + group_z_idx * group_z_number
                                group_node_y = y_idx + group_y_idx * group_y_number
                                group_node_x = x_idx + group_x_idx * group_x_number
                                group_node_idx = group_node_z * source_y_number * source_x_number + group_node_y * source_x_number + group_node_x
                                group_row_list.append(current_dependency_list[group_node_z][group_node_y][group_node_x])
                            group_z_list.append(group_row_list)
                        group.append(group_z_list)
                    groups_current_dependency_list.append(group)

        group_allreduce_event_tag = current_event_tag
        allreduce_dependency_list = [[[[] for _ in range(source_x_number)] for _ in range(source_y_number)] for _ in range(source_z_number)]

        for group_z_idx in range(data_parallelism_degree[2]):
            for group_y_idx in range(data_parallelism_degree[1]):
                for group_x_idx in range(data_parallelism_degree[0]):
                    group_idx = group_z_idx * data_parallelism_degree[1] * data_parallelism_degree[0] + group_y_idx * data_parallelism_degree[0] + group_x_idx
                    group_coordinates = groups_nodes_coordinates_list[group_idx]
                    group_dependency = groups_current_dependency_list[group_idx]
                    group_allreduce_event_tag, group_allreduce_dependency_list = self.allreduce.cal_time(
                        whole_nodes=whole_nodes, current_event_tag=group_allreduce_event_tag, current_dependency_list=group_dependency,
                        source_nodes_coordinates_list=group_coordinates,
                        source_x_number=group_x_number, source_y_number=group_y_number, source_z_number=group_z_number,
                        topology_x_limitation=topology_x_limitation, topology_y_limitation=topology_y_limitation, topology_z_limitation=topology_z_limitation,
                        message_flits=message_flits, 
                        latency=latency, bandwidth=bandwidth, reduction=reduction
                    )

                    for z_idx in range(group_z_number):
                        for y_idx in range(group_y_number):
                            for x_idx in range(group_x_number):
                                group_node_z = z_idx + group_z_idx * group_z_number
                                group_node_y = y_idx + group_y_idx * group_y_number
                                group_node_x = x_idx + group_x_idx * group_x_number
                                group_node_idx = group_node_z * source_y_number * source_x_number + group_node_y * source_x_number + group_node_x
                                allreduce_dependency_list[group_node_z][group_node_y][group_node_x] = group_allreduce_dependency_list[z_idx][y_idx][x_idx]

        # first alltoall
        alltoall_event_tag, alltoall_dependency_list = self.alltoall.cal_time(
            whole_nodes=whole_nodes, current_event_tag=group_allreduce_event_tag, current_dependency_list=allreduce_dependency_list,
            source_nodes_coordinates_list=source_nodes_coordinates_list,
            source_x_number=source_x_number, source_y_number=source_y_number, source_z_number=source_z_number,
            topology_x_limitation=topology_x_limitation, topology_y_limitation=topology_y_limitation, topology_z_limitation=topology_z_limitation,
            message_flits=message_flits*top_k*data_parallelism_degree[0]*data_parallelism_degree[1]*data_parallelism_degree[2]//source_x_number//source_y_number//source_z_number, 
            latency=latency, bandwidth=bandwidth, reduction=reduction
        )

        return alltoall_event_tag, alltoall_dependency_list


    def fusion(
        self,
        whole_nodes: NodeNetwork, current_event_tag: int, current_dependency_list: List[List[List[int]]], 
        source_nodes_coordinates_list: List[Tuple[int]],
        source_x_number: int, source_y_number: int, source_z_number: int,
        topology_x_limitation: int, topology_y_limitation: int, topology_z_limitation: int,
        data_parallelism_degree: List[int], top_k: int, 
        message_flits: int,     
        reduction: float,
        medium_nodes_coordinates_list=None,
        medium_x_number=None, medium_y_number=None, medium_z_number=None,
        target_nodes_coordinates_list=None,
        target_x_number=None, target_y_number=None, target_z_number=None,
        latency=None, bandwidth=None            
    ):
        
        group_x_number = source_x_number // data_parallelism_degree[0]
        group_y_number = source_y_number // data_parallelism_degree[1]
        group_z_number = source_z_number // data_parallelism_degree[2]

        groups_nodes_coordinates_list = []
        for group_z_idx in range(data_parallelism_degree[2]):
            for group_y_idx in range(data_parallelism_degree[1]):
                for group_x_idx in range(data_parallelism_degree[0]):
                    group = []
                    for z_idx in range(group_z_number):
                        for y_idx in range(group_y_number):
                            for x_idx in range(group_x_number):
                                group_node_z = z_idx + group_z_idx * group_z_number
                                group_node_y = y_idx + group_y_idx * group_y_number
                                group_node_x = x_idx + group_x_idx * group_x_number
                                group_node_idx = group_node_z * source_y_number * source_x_number + group_node_y * source_x_number + group_node_x
                                group.append(source_nodes_coordinates_list[group_node_idx])
                    groups_nodes_coordinates_list.append(group)

        # first reducescatter
        groups_current_dependency_list = []
        for group_z_idx in range(data_parallelism_degree[2]):
            for group_y_idx in range(data_parallelism_degree[1]):
                for group_x_idx in range(data_parallelism_degree[0]):
                    group = []
                    for z_idx in range(group_z_number):
                        group_z_list = []
                        for y_idx in range(group_y_number):
                            group_row_list = []
                            for x_idx in range(group_x_number):
                                group_node_z = z_idx + group_z_idx * group_z_number
                                group_node_y = y_idx + group_y_idx * group_y_number
                                group_node_x = x_idx + group_x_idx * group_x_number
                                group_node_idx = group_node_z * source_y_number * source_x_number + group_node_y * source_x_number + group_node_x
                                group_row_list.append(current_dependency_list[group_node_z][group_node_y][group_node_x])
                            group_z_list.append(group_row_list)
                        group.append(group_z_list)
                    groups_current_dependency_list.append(group)

        group_reducescatter_event_tag = current_event_tag
        reducescatter_dependency_list = [[[[] for _ in range(source_x_number)] for _ in range(source_y_number)] for _ in range(source_z_number)]

        for group_z_idx in range(data_parallelism_degree[2]):
            for group_y_idx in range(data_parallelism_degree[1]):
                for group_x_idx in range(data_parallelism_degree[0]):
                    group_idx = group_z_idx * data_parallelism_degree[1] * data_parallelism_degree[0] + group_y_idx * data_parallelism_degree[0] + group_x_idx
                    group_coordinates = groups_nodes_coordinates_list[group_idx]
                    group_dependency = groups_current_dependency_list[group_idx]
                    group_reducescatter_event_tag, group_reducescatter_dependency_list = self.reducescatter.cal_time(
                        whole_nodes=whole_nodes, current_event_tag=group_reducescatter_event_tag, current_dependency_list=group_dependency,
                        source_nodes_coordinates_list=group_coordinates,
                        source_x_number=group_x_number, source_y_number=group_y_number, source_z_number=group_z_number,
                        topology_x_limitation=topology_x_limitation, topology_y_limitation=topology_y_limitation, topology_z_limitation=topology_z_limitation,
                        message_flits=message_flits, 
                        latency=latency, bandwidth=bandwidth, reduction=reduction
                    )

                    for z_idx in range(group_z_number):
                        for y_idx in range(group_y_number):
                            for x_idx in range(group_x_number):
                                group_node_z = z_idx + group_z_idx * group_z_number
                                group_node_y = y_idx + group_y_idx * group_y_number
                                group_node_x = x_idx + group_x_idx * group_x_number
                                group_node_idx = group_node_z * source_y_number * source_x_number + group_node_y * source_x_number + group_node_x
                                reducescatter_dependency_list[group_node_z][group_node_y][group_node_x] = group_reducescatter_dependency_list[z_idx][y_idx][x_idx]

        # first alltoall
        alltoall_event_tag, alltoall_dependency_list = self.alltoall.cal_time(
            whole_nodes=whole_nodes, current_event_tag=group_reducescatter_event_tag, current_dependency_list=reducescatter_dependency_list,
            source_nodes_coordinates_list=source_nodes_coordinates_list,
            source_x_number=source_x_number, source_y_number=source_y_number, source_z_number=source_z_number,
            topology_x_limitation=topology_x_limitation, topology_y_limitation=topology_y_limitation, topology_z_limitation=topology_z_limitation,
            message_flits=message_flits*top_k*data_parallelism_degree[0]*data_parallelism_degree[1]*data_parallelism_degree[2]//source_x_number//source_y_number//source_z_number, 
            latency=latency, bandwidth=bandwidth, reduction=reduction
        )

        return alltoall_event_tag, alltoall_dependency_list


class tensor_expert_half(object):

    def __init__(self, topology: str, algorithm: dict):

        self.allreduce = allreduce(topology, algorithm["allreduce"])
        self.alltoall = alltoall(topology, algorithm["alltoall"])
        self.reducescatter = reducescatter(topology, algorithm["reducescatter"])
        self.ordertoorder = ordertoorder(topology, algorithm["ordertoorder"])
        self.reducelocal = reducelocal(topology, algorithm["reducelocal"])


    def base(
        self,
        whole_nodes: NodeNetwork, current_event_tag: int, current_dependency_list: List[List[List[int]]], 
        source_nodes_coordinates_list: List[Tuple[int]],
        source_x_number: int, source_y_number: int, source_z_number: int,
        topology_x_limitation: int, topology_y_limitation: int, topology_z_limitation: int,
        data_parallelism_degree: List[int], top_k: int, 
        message_flits: int,     
        reduction: float,
        medium_nodes_coordinates_list=None,
        medium_x_number=None, medium_y_number=None, medium_z_number=None,
        target_nodes_coordinates_list=None,
        target_x_number=None, target_y_number=None, target_z_number=None,
        latency=None, bandwidth=None            
    ):
        
        group_x_number = source_x_number // data_parallelism_degree[0]
        group_y_number = source_y_number // data_parallelism_degree[1]
        group_z_number = source_z_number // data_parallelism_degree[2]

        groups_nodes_coordinates_list = []
        for group_z_idx in range(data_parallelism_degree[2]):
            for group_y_idx in range(data_parallelism_degree[1]):
                for group_x_idx in range(data_parallelism_degree[0]):
                    group = []
                    for z_idx in range(group_z_number):
                        for y_idx in range(group_y_number):
                            for x_idx in range(group_x_number):
                                group_node_z = z_idx + group_z_idx * group_z_number
                                group_node_y = y_idx + group_y_idx * group_y_number
                                group_node_x = x_idx + group_x_idx * group_x_number
                                group_node_idx = group_node_z * source_y_number * source_x_number + group_node_y * source_x_number + group_node_x
                                group.append(source_nodes_coordinates_list[group_node_idx])
                    groups_nodes_coordinates_list.append(group)

        # first allreduce
        groups_current_dependency_list = []
        for group_z_idx in range(data_parallelism_degree[2]):
            for group_y_idx in range(data_parallelism_degree[1]):
                for group_x_idx in range(data_parallelism_degree[0]):
                    group = []
                    for z_idx in range(group_z_number):
                        group_z_list = []
                        for y_idx in range(group_y_number):
                            group_row_list = []
                            for x_idx in range(group_x_number):
                                group_node_z = z_idx + group_z_idx * group_z_number
                                group_node_y = y_idx + group_y_idx * group_y_number
                                group_node_x = x_idx + group_x_idx * group_x_number
                                group_node_idx = group_node_z * source_y_number * source_x_number + group_node_y * source_x_number + group_node_x
                                group_row_list.append(current_dependency_list[group_node_z][group_node_y][group_node_x])
                            group_z_list.append(group_row_list)
                        group.append(group_z_list)
                    groups_current_dependency_list.append(group)

        group_allreduce_event_tag = current_event_tag
        allreduce_dependency_list = [[[[] for _ in range(source_x_number)] for _ in range(source_y_number)] for _ in range(source_z_number)]

        for group_z_idx in range(data_parallelism_degree[2]):
            for group_y_idx in range(data_parallelism_degree[1]):
                for group_x_idx in range(data_parallelism_degree[0]):
                    group_idx = group_z_idx * data_parallelism_degree[1] * data_parallelism_degree[0] + group_y_idx * data_parallelism_degree[0] + group_x_idx
                    group_coordinates = groups_nodes_coordinates_list[group_idx]
                    group_dependency = groups_current_dependency_list[group_idx]
                    group_allreduce_event_tag, group_allreduce_dependency_list = self.allreduce.cal_time(
                        whole_nodes=whole_nodes, current_event_tag=group_allreduce_event_tag, current_dependency_list=group_dependency,
                        source_nodes_coordinates_list=group_coordinates,
                        source_x_number=group_x_number, source_y_number=group_y_number, source_z_number=group_z_number,
                        topology_x_limitation=topology_x_limitation, topology_y_limitation=topology_y_limitation, topology_z_limitation=topology_z_limitation,
                        message_flits=message_flits, 
                        latency=latency, bandwidth=bandwidth, reduction=reduction
                    )

                    for z_idx in range(group_z_number):
                        for y_idx in range(group_y_number):
                            for x_idx in range(group_x_number):
                                group_node_z = z_idx + group_z_idx * group_z_number
                                group_node_y = y_idx + group_y_idx * group_y_number
                                group_node_x = x_idx + group_x_idx * group_x_number
                                group_node_idx = group_node_z * source_y_number * source_x_number + group_node_y * source_x_number + group_node_x
                                allreduce_dependency_list[group_node_z][group_node_y][group_node_x] = group_allreduce_dependency_list[z_idx][y_idx][x_idx]

        # first ordertoorder
        ordertoorder_event_tag, ordertoorder_dependency_list = self.ordertoorder.cal_time(
            whole_nodes=whole_nodes, current_event_tag=group_allreduce_event_tag, current_dependency_list=allreduce_dependency_list,
            source_nodes_coordinates_list=source_nodes_coordinates_list,
            source_x_number=source_x_number, source_y_number=source_y_number, source_z_number=source_z_number,
            topology_x_limitation=topology_x_limitation, topology_y_limitation=topology_y_limitation, topology_z_limitation=topology_z_limitation,
            data_parallelism_degree=data_parallelism_degree, 
            message_flits=message_flits*top_k*data_parallelism_degree[0]*data_parallelism_degree[1]*data_parallelism_degree[2], 
            latency=latency, bandwidth=bandwidth, reduction=reduction
        )

        return ordertoorder_event_tag, ordertoorder_dependency_list


    def fusion(
        self,
        whole_nodes: NodeNetwork, current_event_tag: int, current_dependency_list: List[List[List[int]]], 
        source_nodes_coordinates_list: List[Tuple[int]],
        source_x_number: int, source_y_number: int, source_z_number: int,
        topology_x_limitation: int, topology_y_limitation: int, topology_z_limitation: int,
        data_parallelism_degree: List[int], top_k: int, 
        message_flits: int,     
        reduction: float,
        medium_nodes_coordinates_list=None,
        medium_x_number=None, medium_y_number=None, medium_z_number=None,
        target_nodes_coordinates_list=None,
        target_x_number=None, target_y_number=None, target_z_number=None,
        latency=None, bandwidth=None            
    ):
        
        group_x_number = source_x_number // data_parallelism_degree[0]
        group_y_number = source_y_number // data_parallelism_degree[1]
        group_z_number = source_z_number // data_parallelism_degree[2]

        groups_nodes_coordinates_list = []
        for group_z_idx in range(data_parallelism_degree[2]):
            for group_y_idx in range(data_parallelism_degree[1]):
                for group_x_idx in range(data_parallelism_degree[0]):
                    group = []
                    for z_idx in range(group_z_number):
                        for y_idx in range(group_y_number):
                            for x_idx in range(group_x_number):
                                group_node_z = z_idx + group_z_idx * group_z_number
                                group_node_y = y_idx + group_y_idx * group_y_number
                                group_node_x = x_idx + group_x_idx * group_x_number
                                group_node_idx = group_node_z * source_y_number * source_x_number + group_node_y * source_x_number + group_node_x
                                group.append(source_nodes_coordinates_list[group_node_idx])
                    groups_nodes_coordinates_list.append(group)

        # first reducescatter
        groups_current_dependency_list = []
        for group_z_idx in range(data_parallelism_degree[2]):
            for group_y_idx in range(data_parallelism_degree[1]):
                for group_x_idx in range(data_parallelism_degree[0]):
                    group = []
                    for z_idx in range(group_z_number):
                        group_z_list = []
                        for y_idx in range(group_y_number):
                            group_row_list = []
                            for x_idx in range(group_x_number):
                                group_node_z = z_idx + group_z_idx * group_z_number
                                group_node_y = y_idx + group_y_idx * group_y_number
                                group_node_x = x_idx + group_x_idx * group_x_number
                                group_node_idx = group_node_z * source_y_number * source_x_number + group_node_y * source_x_number + group_node_x
                                group_row_list.append(current_dependency_list[group_node_z][group_node_y][group_node_x])
                            group_z_list.append(group_row_list)
                        group.append(group_z_list)
                    groups_current_dependency_list.append(group)

        group_reducescatter_event_tag = current_event_tag
        reducescatter_dependency_list = [[[[] for _ in range(source_x_number)] for _ in range(source_y_number)] for _ in range(source_z_number)]

        for group_z_idx in range(data_parallelism_degree[2]):
            for group_y_idx in range(data_parallelism_degree[1]):
                for group_x_idx in range(data_parallelism_degree[0]):
                    group_idx = group_z_idx * data_parallelism_degree[1] * data_parallelism_degree[0] + group_y_idx * data_parallelism_degree[0] + group_x_idx
                    group_coordinates = groups_nodes_coordinates_list[group_idx]
                    group_dependency = groups_current_dependency_list[group_idx]
                    group_reducescatter_event_tag, group_reducescatter_dependency_list = self.reducescatter.cal_time(
                        whole_nodes=whole_nodes, current_event_tag=group_reducescatter_event_tag, current_dependency_list=group_dependency,
                        source_nodes_coordinates_list=group_coordinates,
                        source_x_number=group_x_number, source_y_number=group_y_number, source_z_number=group_z_number,
                        topology_x_limitation=topology_x_limitation, topology_y_limitation=topology_y_limitation, topology_z_limitation=topology_z_limitation,
                        message_flits=message_flits, 
                        latency=latency, bandwidth=bandwidth, reduction=reduction
                    )

                    for z_idx in range(group_z_number):
                        for y_idx in range(group_y_number):
                            for x_idx in range(group_x_number):
                                group_node_z = z_idx + group_z_idx * group_z_number
                                group_node_y = y_idx + group_y_idx * group_y_number
                                group_node_x = x_idx + group_x_idx * group_x_number
                                group_node_idx = group_node_z * source_y_number * source_x_number + group_node_y * source_x_number + group_node_x
                                reducescatter_dependency_list[group_node_z][group_node_y][group_node_x] = group_reducescatter_dependency_list[z_idx][y_idx][x_idx]

        # first alltoall
        alltoall_event_tag, alltoall_dependency_list = self.alltoall.cal_time(
            whole_nodes=whole_nodes, current_event_tag=group_reducescatter_event_tag, current_dependency_list=reducescatter_dependency_list,
            source_nodes_coordinates_list=source_nodes_coordinates_list,
            source_x_number=source_x_number, source_y_number=source_y_number, source_z_number=source_z_number,
            topology_x_limitation=topology_x_limitation, topology_y_limitation=topology_y_limitation, topology_z_limitation=topology_z_limitation,
            message_flits=message_flits*top_k*data_parallelism_degree[0]*data_parallelism_degree[1]*data_parallelism_degree[2], 
            latency=latency, bandwidth=bandwidth, reduction=reduction
        )

        return alltoall_event_tag, alltoall_dependency_list


    def fusion_firststep(
        self,
        whole_nodes: NodeNetwork, current_event_tag: int, current_dependency_list: List[List[List[int]]], 
        source_nodes_coordinates_list: List[Tuple[int]],
        source_x_number: int, source_y_number: int, source_z_number: int,
        topology_x_limitation: int, topology_y_limitation: int, topology_z_limitation: int,
        data_parallelism_degree: List[int], top_k: int, 
        message_flits: int,     
        reduction: float,
        medium_nodes_coordinates_list=None,
        medium_x_number=None, medium_y_number=None, medium_z_number=None,
        target_nodes_coordinates_list=None,
        target_x_number=None, target_y_number=None, target_z_number=None,
        latency=None, bandwidth=None            
    ):
        
        group_x_number = source_x_number // data_parallelism_degree[0]
        group_y_number = source_y_number // data_parallelism_degree[1]
        group_z_number = source_z_number // data_parallelism_degree[2]

        groups_nodes_coordinates_list = []
        for group_z_idx in range(data_parallelism_degree[2]):
            for group_y_idx in range(data_parallelism_degree[1]):
                for group_x_idx in range(data_parallelism_degree[0]):
                    group = []
                    for z_idx in range(group_z_number):
                        for y_idx in range(group_y_number):
                            for x_idx in range(group_x_number):
                                group_node_z = z_idx + group_z_idx * group_z_number
                                group_node_y = y_idx + group_y_idx * group_y_number
                                group_node_x = x_idx + group_x_idx * group_x_number
                                group_node_idx = group_node_z * source_y_number * source_x_number + group_node_y * source_x_number + group_node_x
                                group.append(source_nodes_coordinates_list[group_node_idx])
                    groups_nodes_coordinates_list.append(group)

        # first reducescatter
        groups_current_dependency_list = []
        for group_z_idx in range(data_parallelism_degree[2]):
            for group_y_idx in range(data_parallelism_degree[1]):
                for group_x_idx in range(data_parallelism_degree[0]):
                    group = []
                    for z_idx in range(group_z_number):
                        group_z_list = []
                        for y_idx in range(group_y_number):
                            group_row_list = []
                            for x_idx in range(group_x_number):
                                group_node_z = z_idx + group_z_idx * group_z_number
                                group_node_y = y_idx + group_y_idx * group_y_number
                                group_node_x = x_idx + group_x_idx * group_x_number
                                group_node_idx = group_node_z * source_y_number * source_x_number + group_node_y * source_x_number + group_node_x
                                group_row_list.append(current_dependency_list[group_node_z][group_node_y][group_node_x])
                            group_z_list.append(group_row_list)
                        group.append(group_z_list)
                    groups_current_dependency_list.append(group)

        group_reducescatter_event_tag = current_event_tag
        reducescatter_dependency_list = [[[] for _ in range(source_x_number)] for _ in range(source_y_number)]

        for group_z_idx in range(data_parallelism_degree[2]):
            for group_y_idx in range(data_parallelism_degree[1]):
                for group_x_idx in range(data_parallelism_degree[0]):
                    group_idx = group_z_idx * data_parallelism_degree[0] * data_parallelism_degree[1] + group_y_idx * data_parallelism_degree[0] + group_x_idx
                    group_coordinates = groups_nodes_coordinates_list[group_idx]
                    group_dependency = groups_current_dependency_list[group_idx]
                    group_reducescatter_event_tag, group_reducescatter_dependency_list = self.reducescatter.cal_time(
                        whole_nodes=whole_nodes, current_event_tag=group_reducescatter_event_tag, current_dependency_list=group_dependency,
                        source_nodes_coordinates_list=group_coordinates,
                        source_x_number=group_x_number, source_y_number=group_y_number, source_z_number=group_z_number,
                        topology_x_limitation=topology_x_limitation, topology_y_limitation=topology_y_limitation, topology_z_limitation=topology_z_limitation,
                        message_flits=message_flits, 
                        latency=latency, bandwidth=bandwidth, reduction=reduction
                    )

                    for z_idx in range(group_z_number):
                        for y_idx in range(group_y_number):
                            for x_idx in range(group_x_number):
                                group_node_z = z_idx + group_z_idx * group_z_number
                                group_node_y = y_idx + group_y_idx * group_y_number
                                group_node_x = x_idx + group_x_idx * group_x_number
                                group_node_idx = group_node_y * source_x_number + group_node_x
                                reducescatter_dependency_list[group_node_z][group_node_y][group_node_x] = group_reducescatter_dependency_list[z_idx][y_idx][x_idx]

        return group_reducescatter_event_tag, reducescatter_dependency_list




if __name__ == "__main__":

    sys.path.append(os.path.join(file_path, '../../../components/'))
    from Launcher import build_3D, launch


    '''
    parameters
    '''
    node_k = 2
    node_n = 3
    ni_k = node_k
    ni_n = node_n
    cfg_topology = "torus"
    algorithm_dict = {
        "allreduce": "hierarchicalring", 
        "alltoall": "hierarchicalring", 
        "reducescatter": "hierarchicalring", 
        "reducelocal": "base", 
        "ordertoorder": "hierarchicalring"
    }

    source_nodes = [(0, 0, 0), (0, 0, 1), (0, 1, 0), (0, 1, 1)]
    source_shape = [1, 2, 2]
    data_parallelism_degree = [2, 1, 1]
    top_k = 2
    target_nodes = [(1, 0, 0), (1, 0, 1), (1, 1, 0), (1, 1, 1)]
    target_shape = [1, 2, 2]
    whole_flits = 294912 
    reduction_cores = 1 / 1024


    '''
    build
    '''
    te_base_node_network, te_base_noc_network = build_3D(
        node_k=node_k, node_n=node_n, ni_k=ni_k, ni_n=ni_n, 
        cfg_topology=cfg_topology
    )

    # tp_fusion_node_network, tp_fusion_noc_network = build_3D(
    #     node_k=node_k, node_n=node_n, ni_k=ni_k, ni_n=ni_n,
    #     cfg_topology=cfg_topology
    # )

    '''
    User logic
    '''

    te = tensor_expert(        
        topology=(cfg_topology + str(node_n) + "d"), 
        algorithm=algorithm_dict
    )

    initial_te_base_event_tag = 0
    initial_te_base_dependency_list = [[[[] for _ in range(source_shape[2])] for _ in range(source_shape[1])] for _ in range(source_shape[0])]

    te_base_event_tag, te_base_dependency_list = te.base(
        whole_nodes=te_base_node_network, current_event_tag=initial_te_base_event_tag, current_dependency_list=initial_te_base_dependency_list,
        source_nodes_coordinates_list=source_nodes,
        source_x_number=source_shape[2], source_y_number=source_shape[1], source_z_number=source_shape[0],
        topology_x_limitation=node_k, topology_y_limitation=node_k, topology_z_limitation=node_k,
        data_parallelism_degree=data_parallelism_degree, top_k=top_k, 
        message_flits=whole_flits, 
        latency=None, bandwidth=None, reduction=reduction_cores
    )    


    '''
    launch
    '''
    te_base_run_cost = launch(
        whole_nodes=te_base_node_network, booksim_net=te_base_noc_network
    )


