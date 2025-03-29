# File name  :    tensor_pipeline.py
# Author     :    xiaocuicui
# Time       :    2024/07/07 19:24:10
# Version    :    V1.0
# Abstract   :        

import os
import sys
file_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(file_path)
sys.path.append(os.path.join(file_path, '../../../communication/collective_communication/'))
sys.path.append(os.path.join(file_path, '../../../components/'))

from typing import List, Tuple

from allgather import allgather
from reducescatter import reducescatter
from allreduce import allreduce
from multicast import pointtopoint, manytomanymulticast

from NodeNetwork import NodeNetwork


class tensor_pipeline(object):

    def __init__(self, topology: str, algorithm: dict):

        self.allgather = allgather(topology, algorithm["allgather"])
        self.allreduce = allreduce(topology, algorithm["allreduce"])
        self.reducescatter = reducescatter(topology, algorithm["reducescatter"])
        self.pointtopoint = pointtopoint(topology, algorithm["pointtopoint"])
        self.manytomanymulticast = manytomanymulticast(topology, algorithm["manytomanymulticast"])

    def megatron(
        self,
        whole_nodes: NodeNetwork, current_event_tag: int, current_dependency_list: List[List[List[int]]], 
        source_nodes_coordinates_list: List[Tuple[int]],
        target_nodes_coordinates_list: List[Tuple[int]],
        source_x_number: int, source_y_number: int, source_z_number: int,
        target_x_number: int, target_y_number: int, target_z_number: int,
        topology_x_limitation: int, topology_y_limitation: int, topology_z_limitation: int,
        message_flits: int,     
        reduction: float,
        medium_nodes_coordinates_list=None,
        medium_x_number=None, medium_y_number=None, medium_z_number=None,
        data_parallelism_degree=None, top_k=None, 
        latency=None, bandwidth=None
    ):

        allreduce_event_tag, allreduce_dependency_list = self.allreduce.cal_time(
            whole_nodes=whole_nodes, current_event_tag=current_event_tag, current_dependency_list=current_dependency_list,
            source_nodes_coordinates_list=source_nodes_coordinates_list,
            source_x_number=source_x_number, source_y_number=source_y_number, source_z_number=source_z_number,
            topology_x_limitation=topology_x_limitation, topology_y_limitation=topology_y_limitation, topology_z_limitation=topology_z_limitation,
            message_flits=message_flits, 
            latency=latency, bandwidth=bandwidth, reduction=reduction
        )

        pointtopoint_event_tag, pointtopoint_dependency_list = self.pointtopoint.cal_time(
            whole_nodes=whole_nodes, current_event_tag=allreduce_event_tag, current_dependency_list=allreduce_dependency_list,
            source_nodes_coordinates_list=source_nodes_coordinates_list,
            target_nodes_coordinates_list=target_nodes_coordinates_list,
            source_x_number=source_x_number, source_y_number=source_y_number, source_z_number=source_z_number,
            target_x_number=target_x_number, target_y_number=target_y_number, target_z_number=target_z_number,
            topology_x_limitation=topology_x_limitation, topology_y_limitation=topology_y_limitation, topology_z_limitation=topology_z_limitation,
            message_flits=message_flits, 
            latency=latency, bandwidth=bandwidth, reduction=reduction
        )

        return pointtopoint_event_tag, pointtopoint_dependency_list


    def alpa(
        self,
        whole_nodes: NodeNetwork, current_event_tag: int, current_dependency_list: List[List[List[int]]], 
        source_nodes_coordinates_list: List[Tuple[int]],
        target_nodes_coordinates_list: List[Tuple[int]],
        source_x_number: int, source_y_number: int, source_z_number: int,
        target_x_number: int, target_y_number: int, target_z_number: int,
        topology_x_limitation: int, topology_y_limitation: int, topology_z_limitation: int,
        message_flits: int,     
        reduction: float,
        medium_nodes_coordinates_list=None,
        medium_x_number=None, medium_y_number=None, medium_z_number=None,
        data_parallelism_degree=None, top_k=None,        
        latency=None, bandwidth=None            
    ):

        allreduce_event_tag, allreduce_dependency_list = self.allreduce.cal_time(
            whole_nodes=whole_nodes, current_event_tag=current_event_tag, current_dependency_list=current_dependency_list,
            source_nodes_coordinates_list=source_nodes_coordinates_list,
            source_x_number=source_x_number, source_y_number=source_y_number, source_z_number=source_z_number,
            topology_x_limitation=topology_x_limitation, topology_y_limitation=topology_y_limitation, topology_z_limitation=topology_z_limitation,
            message_flits=message_flits, 
            latency=latency, bandwidth=bandwidth, reduction=reduction
        )

        manytomanymulticast_event_tag, manytomanymulticast_dependency_list = self.manytomanymulticast.cal_time(
            whole_nodes=whole_nodes, current_event_tag=allreduce_event_tag, current_dependency_list=allreduce_dependency_list,
            source_nodes_coordinates_list=source_nodes_coordinates_list,
            target_nodes_coordinates_list=target_nodes_coordinates_list,
            source_x_number=source_x_number, source_y_number=source_y_number, source_z_number=source_z_number,
            target_x_number=target_x_number, target_y_number=target_y_number, target_z_number=target_z_number,
            topology_x_limitation=topology_x_limitation, topology_y_limitation=topology_y_limitation, topology_z_limitation=topology_z_limitation,
            message_flits=message_flits, 
            latency=latency, bandwidth=bandwidth, reduction=reduction
        )

        allgather_event_tag, allgather_dependency_list = self.allgather.cal_time(
            whole_nodes=whole_nodes, current_event_tag=manytomanymulticast_event_tag, current_dependency_list=manytomanymulticast_dependency_list,
            source_nodes_coordinates_list=target_nodes_coordinates_list,
            source_x_number=target_x_number, source_y_number=target_y_number, source_z_number=target_z_number,
            topology_x_limitation=topology_x_limitation, topology_y_limitation=topology_y_limitation, topology_z_limitation=topology_z_limitation,
            message_flits=message_flits, 
            latency=latency, bandwidth=bandwidth, reduction=reduction
        )

        return allgather_event_tag, allgather_dependency_list


    def fusion(
        self,
        whole_nodes: NodeNetwork, current_event_tag: int, current_dependency_list: List[List[List[int]]], 
        source_nodes_coordinates_list: List[Tuple[int]],
        target_nodes_coordinates_list: List[Tuple[int]],
        source_x_number: int, source_y_number: int, source_z_number: int,
        target_x_number: int, target_y_number: int, target_z_number: int,
        topology_x_limitation: int, topology_y_limitation: int, topology_z_limitation: int,
        message_flits: int,     
        reduction: float,
        medium_nodes_coordinates_list=None,
        medium_x_number=None, medium_y_number=None, medium_z_number=None,
        data_parallelism_degree=None, top_k=None,        
        latency=None, bandwidth=None               
    ):
        
        reducescatter_event_tag, reducescatter_dependency_list = self.reducescatter.cal_time(
            whole_nodes=whole_nodes, current_event_tag=current_event_tag, current_dependency_list=current_dependency_list,
            source_nodes_coordinates_list=source_nodes_coordinates_list,
            source_x_number=source_x_number, source_y_number=source_y_number, source_z_number=source_z_number,
            topology_x_limitation=topology_x_limitation, topology_y_limitation=topology_y_limitation, topology_z_limitation=topology_z_limitation,
            message_flits=message_flits, 
            latency=latency, bandwidth=bandwidth, reduction=reduction
        )

        manytomanymulticast_event_tag, manytomanymulticast_dependency_list = self.manytomanymulticast.cal_time(
            whole_nodes=whole_nodes, current_event_tag=reducescatter_event_tag, current_dependency_list=reducescatter_dependency_list,
            source_nodes_coordinates_list=source_nodes_coordinates_list,
            target_nodes_coordinates_list=target_nodes_coordinates_list,
            source_x_number=source_x_number, source_y_number=source_y_number, source_z_number=source_z_number,
            target_x_number=target_x_number, target_y_number=target_y_number, target_z_number=target_z_number,
            topology_x_limitation=topology_x_limitation, topology_y_limitation=topology_y_limitation, topology_z_limitation=topology_z_limitation,
            message_flits=message_flits, 
            latency=latency, bandwidth=bandwidth, reduction=reduction
        )

        allgather_event_tag, allgather_dependency_list = self.allgather.cal_time(
            whole_nodes=whole_nodes, current_event_tag=manytomanymulticast_event_tag, current_dependency_list=manytomanymulticast_dependency_list,
            source_nodes_coordinates_list=target_nodes_coordinates_list,
            source_x_number=target_x_number, source_y_number=target_y_number, source_z_number=target_z_number,
            topology_x_limitation=topology_x_limitation, topology_y_limitation=topology_y_limitation, topology_z_limitation=topology_z_limitation,
            message_flits=message_flits, 
            latency=latency, bandwidth=bandwidth, reduction=reduction
        )

        return allgather_event_tag, allgather_dependency_list




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
        "allreduce": "hierarchicalring", 
        "reducescatter": "hierarchicalring", 
        "pointtopoint": "base", 
        "manytomanymulticast": "alpa"
    }

    source_nodes = [(0, 0, 0), (0, 0, 1)]
    source_shape = [1, 1, 2]
    target_nodes = [(0, 1, 0), (0, 1, 1)]
    target_shape = [1, 1, 2]
    whole_flits = 294912 
    reduction_cores = 1 / 1024


    '''
    build
    '''
    tp_alpa_node_network, tp_alpa_noc_network = build_3D(
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

    tp = tensor_pipeline(
        topology=(cfg_topology + str(node_n) + "d"), 
        algorithm=algorithm_dict
    )

    initial_tp_alpa_event_tag = 0
    initial_tp_alpa_dependency_list = [[[[] for _ in range(source_shape[2])] for _ in range(source_shape[1])] for _ in range(source_shape[0])]

    tp_alpa_event_tag, tp_alpa_dependency_list = tp.alpa(
        whole_nodes=tp_alpa_node_network, current_event_tag=initial_tp_alpa_event_tag, current_dependency_list=initial_tp_alpa_dependency_list,
        source_nodes_coordinates_list=source_nodes,
        target_nodes_coordinates_list=target_nodes,
        source_x_number=source_shape[2], source_y_number=source_shape[1], source_z_number=source_shape[0],
        target_x_number=target_shape[2], target_y_number=target_shape[1], target_z_number=target_shape[0],
        topology_x_limitation=node_k, topology_y_limitation=node_k, topology_z_limitation=node_k,
        message_flits=whole_flits, 
        latency=None, bandwidth=None, reduction=reduction_cores
    )    

    # initial_tp_fusion_event_tag = 0
    # initial_tp_fusion_dependency_list = [[[] for _ in range(source_shape[1])] for _ in range(source_shape[0])]

    # tp_fusion_event_tag, tp_fusion_dependency_list = tp.fusion(
    #     whole_nodes=tp_fusion_node_network, current_event_tag=initial_tp_fusion_event_tag, current_dependency_list=initial_tp_fusion_dependency_list,
    #     source_nodes_coordinates_list=source_nodes,
    #     target_nodes_coordinates_list=target_nodes,
    #     source_x_number=source_shape[1], source_y_number=source_shape[0],
    #     target_x_number=target_shape[1], target_y_number=target_shape[0],
    #     topology_x_limitation=node_k, topology_y_limitation=node_k,
    #     message_flits=whole_flits, 
    #     latency=None, bandwidth=None, reduction=reduction_cores
    # )

    '''
    launch
    '''
    tp_alpa_run_cost = launch(
        whole_nodes=tp_alpa_node_network, booksim_net=tp_alpa_noc_network
    )

    # tp_fusion_run_cost = launch(
        # whole_nodes=tp_fusion_node_network, booksim_net=tp_fusion_noc_network
    # )

    # print("fusion / alpa speed-up is ", tp_alap_run_cost / tp_fusion_run_cost)


