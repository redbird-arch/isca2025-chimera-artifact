# File name  :    sequence_expert.py
# Author     :    xiaocuicui
# Time       :    2024/07/14 22:34:12
# Version    :    V1.0
# Abstract   :        

import os
import sys
file_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(file_path)
sys.path.append(os.path.join(file_path, '../../../communication/collective_communication/'))
sys.path.append(os.path.join(file_path, '../../../components/'))

from typing import List, Tuple

from reducescatter import reducelocal
from allgather import allgather
from alltoall import alltoall

from NodeNetwork import NodeNetwork


class sequence_expert(object):

    def __init__(self, topology: str, algorithm: dict):

        self.allgather = allgather(topology, algorithm["allgather"])
        self.reducelocal = reducelocal(topology, algorithm["reducelocal"])
        self.alltoall = alltoall(topology, algorithm["alltoall"])


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
                                group_node_idx = group_node_z * source_x_number * source_y_number + group_node_y * source_x_number + group_node_x
                                group.append(source_nodes_coordinates_list[group_node_idx])
                    groups_nodes_coordinates_list.append(group)

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
                                group_node_idx = group_node_z * source_x_number * source_y_number + group_node_y * source_x_number + group_node_x
                                group_row_list.append(current_dependency_list[group_node_z][group_node_y][group_node_x])
                            group_z_list.append(group_row_list)
                        group.append(group_z_list)
                    groups_current_dependency_list.append(group)

        group_allgather_event_tag = current_event_tag
        allgather_dependency_list = [[[[] for _ in range(source_x_number)] for _ in range(source_y_number)] for _ in range(source_z_number)]

        for group_z_idx in range(data_parallelism_degree[2]):
            for group_y_idx in range(data_parallelism_degree[1]):
                for group_x_idx in range(data_parallelism_degree[0]):
                    group_idx = group_z_idx * data_parallelism_degree[1] * data_parallelism_degree[0] + group_y_idx * data_parallelism_degree[0] + group_x_idx
                    group_coordinates = groups_nodes_coordinates_list[group_idx]
                    group_dependency = groups_current_dependency_list[group_idx]
                    group_allgather_event_tag, group_allgather_dependency_list = self.allgather.cal_time(
                        whole_nodes=whole_nodes, current_event_tag=group_allgather_event_tag, current_dependency_list=group_dependency,
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
                                group_node_idx = group_node_z * source_x_number * source_y_number + group_node_y * source_x_number + group_node_x
                                allgather_dependency_list[group_node_z][group_node_y][group_node_x] = group_allgather_dependency_list[z_idx][y_idx][x_idx]

        alltoall_event_tag, alltoall_dependency_list = self.alltoall.cal_time(
            whole_nodes=whole_nodes, current_event_tag=group_allgather_event_tag, current_dependency_list=allgather_dependency_list,
            source_nodes_coordinates_list=source_nodes_coordinates_list,
            source_x_number=source_x_number, source_y_number=source_y_number, source_z_number=source_z_number,
            topology_x_limitation=topology_x_limitation, topology_y_limitation=topology_y_limitation, topology_z_limitation=topology_z_limitation,
            message_flits=message_flits*top_k*data_parallelism_degree[0]*data_parallelism_degree[1]*data_parallelism_degree[2], 
            latency=latency, bandwidth=bandwidth, reduction=reduction
        )

        return alltoall_event_tag, alltoall_dependency_list


    def trans(
        self,
        whole_nodes: NodeNetwork, current_event_tag: int, current_dependency_list: List[List[List[int]]], 
        source_nodes_coordinates_list: List[Tuple[int]],
        source_x_number: int, source_y_number: int, source_z_number: int,
        topology_x_limitation: int, topology_y_limitation: int, topology_z_limitation: int,
        data_parallelism_degree: List[int], top_k: int, 
        message_flits: int,     
        reduction: float,
        medium_nodes_coordinates_list=None,
        medium_x_number=None, medium_y_number=None, meeidum_z_number=None,
        target_nodes_coordinates_list=None,
        target_x_number=None, target_y_number=None, target_z_number=None,
        latency=None, bandwidth=None            
    ):
        
        alltoall_event_tag, alltoall_dependency_list = self.alltoall.cal_time(
            whole_nodes=whole_nodes, current_event_tag=current_event_tag, current_dependency_list=current_dependency_list,
            source_nodes_coordinates_list=source_nodes_coordinates_list,
            source_x_number=source_x_number, source_y_number=source_y_number, source_z_number=source_z_number,
            topology_x_limitation=topology_x_limitation, topology_y_limitation=topology_y_limitation, topology_z_limitation=topology_z_limitation,
            message_flits=message_flits*top_k*data_parallelism_degree[0]*data_parallelism_degree[1]*data_parallelism_degree[2], 
            latency=latency, bandwidth=bandwidth, reduction=reduction
        )

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
                                group_node_idx = group_node_z * source_x_number * source_y_number + group_node_y * source_x_number + group_node_x
                                group.append(source_nodes_coordinates_list[group_node_idx])
                    groups_nodes_coordinates_list.append(group)

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
                                group_node_idx = group_node_z * source_x_number * source_y_number + group_node_y * source_x_number + group_node_x
                                group_row_list.append(alltoall_dependency_list[group_node_z][group_node_y][group_node_x])
                            group_z_list.append(group_row_list)
                        group.append(group_z_list)
                    groups_current_dependency_list.append(group)

        group_alltoall_event_tag = alltoall_event_tag
        after_alltoall_dependency_list = [[[[] for _ in range(source_x_number)] for _ in range(source_y_number)] for _ in range(source_z_number)]

        for group_z_idx in range(data_parallelism_degree[2]):
            for group_y_idx in range(data_parallelism_degree[1]):
                for group_x_idx in range(data_parallelism_degree[0]):
                    group_idx = group_z_idx * data_parallelism_degree[1] * data_parallelism_degree[0] + group_y_idx * data_parallelism_degree[0] + group_x_idx
                    group_coordinates = groups_nodes_coordinates_list[group_idx]
                    group_dependency = groups_current_dependency_list[group_idx]
                    group_alltoall_event_tag, group_alltoall_dependency_list = self.alltoall.cal_time(
                        whole_nodes=whole_nodes, current_event_tag=group_alltoall_event_tag, current_dependency_list=group_dependency,
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
                                group_node_idx = group_node_z * source_x_number * source_y_number + group_node_y * source_x_number + group_node_x
                                after_alltoall_dependency_list[group_node_z][group_node_y][group_node_x] = group_alltoall_dependency_list[z_idx][y_idx][x_idx]

        return group_alltoall_event_tag, after_alltoall_dependency_list


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
        
        alltoall_event_tag, alltoall_dependency_list = self.alltoall.cal_time(
            whole_nodes=whole_nodes, current_event_tag=current_event_tag, current_dependency_list=current_dependency_list,
            source_nodes_coordinates_list=source_nodes_coordinates_list,
            source_x_number=source_x_number, source_y_number=source_y_number, source_z_number=source_z_number,
            topology_x_limitation=topology_x_limitation, topology_y_limitation=topology_y_limitation, topology_z_limitation=topology_z_limitation,
            message_flits=message_flits*top_k*data_parallelism_degree[0]*data_parallelism_degree[1]*data_parallelism_degree[2], 
            latency=latency, bandwidth=bandwidth, reduction=reduction
        )   

        return alltoall_event_tag, alltoall_dependency_list


class sequence_expert_backward(object):

    def __init__(self, topology: str, algorithm: dict):

        self.allgather = allgather(topology, algorithm["allgather"])
        self.reducelocal = reducelocal(topology, algorithm["reducelocal"])
        self.alltoall = alltoall(topology, algorithm["alltoall"])


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
                                group_node_idx = group_node_z * source_x_number * source_y_number + group_node_y * source_x_number + group_node_x
                                group.append(source_nodes_coordinates_list[group_node_idx])
                    groups_nodes_coordinates_list.append(group)

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
                                group_node_idx = group_node_z * source_x_number * source_y_number + group_node_y * source_x_number + group_node_x
                                group_row_list.append(current_dependency_list[group_node_z][group_node_y][group_node_x])
                            group_z_list.append(group_row_list)
                        group.append(group_z_list)
                    groups_current_dependency_list.append(group)

        group_allgather_event_tag = current_event_tag
        allgather_dependency_list = [[[[] for _ in range(source_x_number)] for _ in range(source_y_number)] for _ in range(source_z_number)]

        for group_z_idx in range(data_parallelism_degree[2]):
            for group_y_idx in range(data_parallelism_degree[1]):
                for group_x_idx in range(data_parallelism_degree[0]):
                    group_idx = group_z_idx * data_parallelism_degree[1] * data_parallelism_degree[0] + group_y_idx * data_parallelism_degree[0] + group_x_idx
                    group_coordinates = groups_nodes_coordinates_list[group_idx]
                    group_dependency = groups_current_dependency_list[group_idx]
                    group_allgather_event_tag, group_allgather_dependency_list = self.allgather.cal_time(
                        whole_nodes=whole_nodes, current_event_tag=group_allgather_event_tag, current_dependency_list=group_dependency,
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
                                group_node_idx = group_node_z * source_x_number * source_y_number + group_node_y * source_x_number + group_node_x
                                allgather_dependency_list[group_node_z][group_node_y][group_node_x] = group_allgather_dependency_list[z_idx][y_idx][x_idx]

        alltoall_event_tag, alltoall_dependency_list = self.alltoall.cal_time(
            whole_nodes=whole_nodes, current_event_tag=group_allgather_event_tag, current_dependency_list=allgather_dependency_list,
            source_nodes_coordinates_list=source_nodes_coordinates_list,
            source_x_number=source_x_number, source_y_number=source_y_number, source_z_number=source_z_number,
            topology_x_limitation=topology_x_limitation, topology_y_limitation=topology_y_limitation, topology_z_limitation=topology_z_limitation,
            message_flits=message_flits*top_k*data_parallelism_degree[0]*data_parallelism_degree[1]*data_parallelism_degree[2], 
            latency=latency, bandwidth=bandwidth, reduction=reduction
        )

        return alltoall_event_tag, alltoall_dependency_list


    def trans(
        self,
        whole_nodes: NodeNetwork, current_event_tag: int, current_dependency_list: List[List[List[int]]], 
        source_nodes_coordinates_list: List[Tuple[int]],
        source_x_number: int, source_y_number: int, source_z_number: int,
        topology_x_limitation: int, topology_y_limitation: int, topology_z_limitation: int,
        data_parallelism_degree: List[int], top_k: int, 
        message_flits: int,     
        reduction: float,
        medium_nodes_coordinates_list=None,
        medium_x_number=None, medium_y_number=None, meeidum_z_number=None,
        target_nodes_coordinates_list=None,
        target_x_number=None, target_y_number=None, target_z_number=None,
        latency=None, bandwidth=None            
    ):
        
        alltoall_event_tag, alltoall_dependency_list = self.alltoall.cal_time(
            whole_nodes=whole_nodes, current_event_tag=current_event_tag, current_dependency_list=current_dependency_list,
            source_nodes_coordinates_list=source_nodes_coordinates_list,
            source_x_number=source_x_number, source_y_number=source_y_number, source_z_number=source_z_number,
            topology_x_limitation=topology_x_limitation, topology_y_limitation=topology_y_limitation, topology_z_limitation=topology_z_limitation,
            message_flits=message_flits*top_k*data_parallelism_degree[0]*data_parallelism_degree[1]*data_parallelism_degree[2], 
            latency=latency, bandwidth=bandwidth, reduction=reduction
        )

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
                                group_node_idx = group_node_z * source_x_number * source_y_number + group_node_y * source_x_number + group_node_x
                                group.append(source_nodes_coordinates_list[group_node_idx])
                    groups_nodes_coordinates_list.append(group)

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
                                group_node_idx = group_node_z * source_x_number * source_y_number + group_node_y * source_x_number + group_node_x
                                group_row_list.append(alltoall_dependency_list[group_node_z][group_node_y][group_node_x])
                            group_z_list.append(group_row_list)
                        group.append(group_z_list)
                    groups_current_dependency_list.append(group)

        group_alltoall_event_tag = alltoall_event_tag
        after_alltoall_dependency_list = [[[[] for _ in range(source_x_number)] for _ in range(source_y_number)] for _ in range(source_z_number)]

        for group_z_idx in range(data_parallelism_degree[2]):
            for group_y_idx in range(data_parallelism_degree[1]):
                for group_x_idx in range(data_parallelism_degree[0]):
                    group_idx = group_z_idx * data_parallelism_degree[1] * data_parallelism_degree[0] + group_y_idx * data_parallelism_degree[0] + group_x_idx
                    group_coordinates = groups_nodes_coordinates_list[group_idx]
                    group_dependency = groups_current_dependency_list[group_idx]
                    group_alltoall_event_tag, group_alltoall_dependency_list = self.alltoall.cal_time(
                        whole_nodes=whole_nodes, current_event_tag=group_alltoall_event_tag, current_dependency_list=group_dependency,
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
                                group_node_idx = group_node_z * source_x_number * source_y_number + group_node_y * source_x_number + group_node_x
                                after_alltoall_dependency_list[group_node_z][group_node_y][group_node_x] = group_alltoall_dependency_list[z_idx][y_idx][x_idx]

        return group_alltoall_event_tag, after_alltoall_dependency_list


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
        
        alltoall_event_tag, alltoall_dependency_list = self.alltoall.cal_time(
            whole_nodes=whole_nodes, current_event_tag=current_event_tag, current_dependency_list=current_dependency_list,
            source_nodes_coordinates_list=source_nodes_coordinates_list,
            source_x_number=source_x_number, source_y_number=source_y_number, source_z_number=source_z_number,
            topology_x_limitation=topology_x_limitation, topology_y_limitation=topology_y_limitation, topology_z_limitation=topology_z_limitation,
            message_flits=message_flits*top_k*data_parallelism_degree[0]*data_parallelism_degree[1]*data_parallelism_degree[2], 
            latency=latency, bandwidth=bandwidth, reduction=reduction
        )   

        return alltoall_event_tag, alltoall_dependency_list


class sequence_expert_moe(object):

    def __init__(self, topology: str, algorithm: dict):

        self.allgather = allgather(topology, algorithm["allgather"])
        self.reducelocal = reducelocal(topology, algorithm["reducelocal"])
        self.alltoall = alltoall(topology, algorithm["alltoall"])


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
                                group_node_idx = group_node_z * source_x_number * source_y_number + group_node_y * source_x_number + group_node_x
                                group.append(source_nodes_coordinates_list[group_node_idx])
                    groups_nodes_coordinates_list.append(group)

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
                                group_node_idx = group_node_z * source_x_number * source_y_number + group_node_y * source_x_number + group_node_x
                                group_row_list.append(current_dependency_list[group_node_z][group_node_y][group_node_x])
                            group_z_list.append(group_row_list)
                        group.append(group_z_list)
                    groups_current_dependency_list.append(group)

        group_allgather_event_tag = current_event_tag
        allgather_dependency_list = [[[[] for _ in range(source_x_number)] for _ in range(source_y_number)] for _ in range(source_z_number)]

        for group_z_idx in range(data_parallelism_degree[2]):
            for group_y_idx in range(data_parallelism_degree[1]):
                for group_x_idx in range(data_parallelism_degree[0]):
                    group_idx = group_z_idx * data_parallelism_degree[1] * data_parallelism_degree[0] + group_y_idx * data_parallelism_degree[0] + group_x_idx
                    group_coordinates = groups_nodes_coordinates_list[group_idx]
                    group_dependency = groups_current_dependency_list[group_idx]
                    group_allgather_event_tag, group_allgather_dependency_list = self.allgather.cal_time(
                        whole_nodes=whole_nodes, current_event_tag=group_allgather_event_tag, current_dependency_list=group_dependency,
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
                                group_node_idx = group_node_z * source_x_number * source_y_number + group_node_y * source_x_number + group_node_x
                                allgather_dependency_list[group_node_z][group_node_y][group_node_x] = group_allgather_dependency_list[z_idx][y_idx][x_idx]

        alltoall_event_tag, alltoall_dependency_list = self.alltoall.cal_time(
            whole_nodes=whole_nodes, current_event_tag=group_allgather_event_tag, current_dependency_list=allgather_dependency_list,
            source_nodes_coordinates_list=source_nodes_coordinates_list,
            source_x_number=source_x_number, source_y_number=source_y_number, source_z_number=source_z_number,
            topology_x_limitation=topology_x_limitation, topology_y_limitation=topology_y_limitation, topology_z_limitation=topology_z_limitation,
            message_flits=message_flits*top_k, 
            latency=latency, bandwidth=bandwidth, reduction=reduction
        )

        alltoall_event_tag, alltoall_dependency_list = self.alltoall.cal_time(
            whole_nodes=whole_nodes, current_event_tag=alltoall_event_tag, current_dependency_list=alltoall_dependency_list,
            source_nodes_coordinates_list=source_nodes_coordinates_list,
            source_x_number=source_x_number, source_y_number=source_y_number, source_z_number=source_z_number,
            topology_x_limitation=topology_x_limitation, topology_y_limitation=topology_y_limitation, topology_z_limitation=topology_z_limitation,
            message_flits=message_flits*top_k, 
            latency=latency, bandwidth=bandwidth, reduction=reduction
        )

        if (top_k > 1):
            reducelocal_event_tag, reducelocal_dependency_list = self.reducelocal.cal_time(
                whole_nodes=whole_nodes, current_event_tag=alltoall_event_tag, current_dependency_list=alltoall_dependency_list,
                source_nodes_coordinates_list=source_nodes_coordinates_list,
                source_x_number=source_x_number, source_y_number=source_y_number, source_z_number=source_z_number,
                topology_x_limitation=topology_x_limitation, topology_y_limitation=topology_y_limitation, topology_z_limitation=topology_z_limitation,
                message_flits=message_flits*data_parallelism_degree[0]*data_parallelism_degree[1], 
                latency=latency, bandwidth=bandwidth, reduction=reduction
            )
        else:
            reducelocal_event_tag = alltoall_event_tag
            reducelocal_dependency_list = alltoall_dependency_list        

        group_alltoall_event_tag = reducelocal_event_tag
        next_alltoall_dependency_list = [[[[] for _ in range(source_x_number)] for _ in range(source_y_number)] for _ in range(source_z_number)]

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
                                group_node_idx = group_node_z * source_x_number * source_y_number + group_node_y * source_x_number + group_node_x
                                group_row_list.append(reducelocal_dependency_list[group_node_z][group_node_y][group_node_x])
                            group_z_list.append(group_row_list)
                        group.append(group_z_list)
                    groups_current_dependency_list.append(group)

        for group_z_idx in range(data_parallelism_degree[2]):
            for group_y_idx in range(data_parallelism_degree[1]):
                for group_x_idx in range(data_parallelism_degree[0]):
                    group_idx = group_z_idx * data_parallelism_degree[1] * data_parallelism_degree[0] + group_y_idx * data_parallelism_degree[0] + group_x_idx
                    group_coordinates = groups_nodes_coordinates_list[group_idx]
                    group_dependency = groups_current_dependency_list[group_idx]

                    group_alltoall_event_tag, group_alltoall_dependency_list = self.alltoall.cal_time(
                        whole_nodes=whole_nodes, current_event_tag=group_alltoall_event_tag, current_dependency_list=group_dependency,
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
                                group_node_idx = group_node_z * source_x_number * source_y_number + group_node_y * source_x_number + group_node_x
                                next_alltoall_dependency_list[group_node_z][group_node_y][group_node_x] = group_alltoall_dependency_list[z_idx][y_idx][x_idx]

        return group_alltoall_event_tag, next_alltoall_dependency_list


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
        
        alltoall_event_tag, alltoall_dependency_list = self.alltoall.cal_time(
            whole_nodes=whole_nodes, current_event_tag=current_event_tag, current_dependency_list=current_dependency_list,
            source_nodes_coordinates_list=source_nodes_coordinates_list,
            source_x_number=source_x_number, source_y_number=source_y_number, source_z_number=source_z_number,
            topology_x_limitation=topology_x_limitation, topology_y_limitation=topology_y_limitation, topology_z_limitation=topology_z_limitation,
            message_flits=message_flits*top_k, 
            latency=latency, bandwidth=bandwidth, reduction=reduction
        )   

        alltoall_event_tag, alltoall_dependency_list = self.alltoall.cal_time(
            whole_nodes=whole_nodes, current_event_tag=alltoall_event_tag, current_dependency_list=alltoall_dependency_list,
            source_nodes_coordinates_list=source_nodes_coordinates_list,
            source_x_number=source_x_number, source_y_number=source_y_number, soruce_z_number=source_z_number,
            topology_x_limitation=topology_x_limitation, topology_y_limitation=topology_y_limitation, topology_z_limitation=topology_z_limitation,
            message_flits=message_flits*top_k, 
            latency=latency, bandwidth=bandwidth, reduction=reduction
        )

        if (top_k > 1):
            reducelocal_event_tag, reducelocal_dependency_list = self.reducelocal.cal_time(
                whole_nodes=whole_nodes, current_event_tag=alltoall_event_tag, current_dependency_list=alltoall_dependency_list,
                source_nodes_coordinates_list=source_nodes_coordinates_list,
                source_x_number=source_x_number, source_y_number=source_y_number, source_z_number=source_z_number,
                topology_x_limitation=topology_x_limitation, topology_y_limitation=topology_y_limitation, topology_z_limitation=topology_z_limitation,
                message_flits=message_flits*data_parallelism_degree[0]*data_parallelism_degree[1]*data_parallelism_degree[2], 
                latency=latency, bandwidth=bandwidth, reduction=reduction
            )
        else:
            reducelocal_event_tag = alltoall_event_tag
            reducelocal_dependency_list = alltoall_dependency_list

        return reducelocal_event_tag, reducelocal_dependency_list






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
        "allgather": "hierarchicalring", 
        "reducelocal": "base", 
        "alltoall": "hierarchicalring"
    }

    source_nodes = [(0, 0, 0), (0, 0, 1)]
    source_shape = [1, 1, 2]
    data_parallelism_degree = [2, 1, 1]
    top_k = 2
    target_nodes = [(0, 1, 0), (0, 1, 1)]
    target_shape = [1, 1, 2]
    whole_flits = 294912 
    reduction_cores = 1 / 1024


    '''
    build
    '''
    se_base_node_network, se_base_noc_network = build_3D(
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

    se = sequence_expert(        
        topology=(cfg_topology + str(node_n) + "d"), 
        algorithm=algorithm_dict
    )

    initial_se_base_event_tag = 0
    initial_se_base_dependency_list = [[[[] for _ in range(source_shape[2])] for _ in range(source_shape[1])] for _ in range(source_shape[0])]

    se_base_event_tag, se_base_dependency_list = se.fusion(
        whole_nodes=se_base_node_network, current_event_tag=initial_se_base_event_tag, current_dependency_list=initial_se_base_dependency_list,
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
    se_base_run_cost = launch(
        whole_nodes=se_base_node_network, booksim_net=se_base_noc_network
    )


