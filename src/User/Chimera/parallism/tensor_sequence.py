# File name  :    tensor_sequence.py
# Author     :    xiaocuicui
# Time       :    2024/07/13 00:26:59
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
from allreduce import allreduce
from reducescatter import reducescatter

from NodeNetwork import NodeNetwork


class tensor_sequence(object):

    def __init__(self, topology: str, algorithm: dict):

        self.allgather = allgather(topology, algorithm["allgather"])
        self.allreduce = allreduce(topology, algorithm["allreduce"])
        self.reducescatter = reducescatter(topology, algorithm["reducescatter"])


    def base(
        self,
        whole_nodes: NodeNetwork, current_event_tag: int, current_dependency_list: List[List[List[int]]], 
        source_nodes_coordinates_list: List[Tuple[int]],
        target_nodes_coordinates_list: List[Tuple[int]],
        source_x_number: int, source_y_number: int,
        target_x_number: int, target_y_number: int,
        topology_x_limitation: int, topology_y_limitation: int, 
        message_flits: int,     
        reduction: float,
        medium_nodes_coordinates_list=None,
        medium_x_number=None, medium_y_number=None,
        data_parallelism_degree=None, top_k=None,             
        latency=None, bandwidth=None
    ): 

        allreduce_event_tag, allreduce_dependency_list = self.allreduce.cal_time(
            whole_nodes=whole_nodes, current_event_tag=current_event_tag, current_dependency_list=current_dependency_list,
            source_nodes_coordinates_list=source_nodes_coordinates_list,
            source_x_number=source_x_number, source_y_number=source_y_number,
            topology_x_limitation=topology_x_limitation, topology_y_limitation=topology_y_limitation,
            message_flits=message_flits, 
            latency=latency, bandwidth=bandwidth, reduction=reduction
        )

        return allreduce_event_tag, allreduce_dependency_list


    def megatron(
        self,
        whole_nodes: NodeNetwork, current_event_tag: int, current_dependency_list: List[List[List[int]]], 
        source_nodes_coordinates_list: List[Tuple[int]],
        target_nodes_coordinates_list: List[Tuple[int]],
        source_x_number: int, source_y_number: int,
        target_x_number: int, target_y_number: int,
        topology_x_limitation: int, topology_y_limitation: int, 
        message_flits: int,     
        reduction: float,
        medium_nodes_coordinates_list=None,
        medium_x_number=None, medium_y_number=None,
        data_parallelism_degree=None, top_k=None,        
        latency=None, bandwidth=None            
    ): 
        
        reducescatter_event_tag, reducescatter_dependency_list = self.reducescatter.cal_time(
            whole_nodes=whole_nodes, current_event_tag=current_event_tag, current_dependency_list=current_dependency_list,
            source_nodes_coordinates_list=source_nodes_coordinates_list,
            source_x_number=source_x_number, source_y_number=source_y_number,
            topology_x_limitation=topology_x_limitation, topology_y_limitation=topology_y_limitation,
            message_flits=message_flits, 
            latency=latency, bandwidth=bandwidth, reduction=reduction
        )

        return reducescatter_event_tag, reducescatter_dependency_list




if __name__ == "__main__":

    sys.path.append(os.path.join(file_path, '../../../components/'))
    from Launcher import build, launch


    '''
    parameters
    '''
    node_k = 8
    node_n = 2
    ni_k = node_k
    ni_n = node_n
    cfg_topology = "mesh"
    algorithm_dict = {
        "allgather": "hierarchicalring", 
        "allreduce": "hierarchicalring"
    }

    source_nodes = [(0, 0), (0, 1), (0, 2), (0, 3), (1, 0), (1, 1), (1, 2), (1, 3)]
    source_shape = [2, 4]
    target_nodes = [(0, 4), (0, 5), (0, 6), (0, 7), (1, 4), (1, 5), (1, 6), (1, 7)]
    target_shape = [2, 4]
    whole_flits = 294912 
    reduction_cores = 1 / 1024


    '''
    build
    '''
    ts_base_node_network, ts_base_noc_network = build(
        node_k=node_k, node_n=node_n, ni_k=ni_k, ni_n=ni_n, 
        cfg_topology=cfg_topology
    )

    '''
    User logic
    '''

    ts = tensor_sequence(
        topology=(cfg_topology + str(node_n) + "d"), 
        algorithm=algorithm_dict
    )

    initial_ts_base_event_tag = 0
    initial_ts_base_dependency_list = [[[] for _ in range(source_shape[1])] for _ in range(source_shape[0])]

    ts_base_event_tag, ts_base_dependency_list = ts.base(
        whole_nodes=ts_base_node_network, current_event_tag=initial_ts_base_event_tag, current_dependency_list=initial_ts_base_dependency_list,
        source_nodes_coordinates_list=source_nodes,
        target_nodes_coordinates_list=target_nodes,
        source_x_number=source_shape[1], source_y_number=source_shape[0],
        target_x_number=target_shape[1], target_y_number=target_shape[0],
        topology_x_limitation=node_k, topology_y_limitation=node_k,
        message_flits=whole_flits, 
        latency=None, bandwidth=None, reduction=reduction_cores
    )        


    '''
    launch
    '''
    ts_base_run_cost = launch(
        whole_nodes=ts_base_node_network, booksim_net=ts_base_noc_network
    )

