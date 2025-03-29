# File name  :    pipeline_expert.py
# Author     :    xiaocuicui
# Time       :    2024/07/14 13:34:01
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
from alltoall import alltoall
from multicast import pointtopoint, manytomanymulticast

from NodeNetwork import NodeNetwork


class pipeline_expert(object):

    def __init__(self, topology: str, algorithm: dict):

        self.reducelocal = reducelocal(topology, algorithm["reducelocal"])
        self.alltoall = alltoall(topology, algorithm["alltoall"])
        self.pointtopoint = pointtopoint(topology, algorithm["pointtopoint"])
        self.manytomanymulticast = manytomanymulticast(topology, algorithm["manytomanymulticast"])


    def base(
        self,
        whole_nodes: NodeNetwork, current_event_tag: int, current_dependency_list: List[List[List[int]]], 
        source_nodes_coordinates_list: List[Tuple[int]],
        medium_nodes_coordinates_list: List[Tuple[int]],
        target_nodes_coordinates_list: List[Tuple[int]],
        source_x_number: int, source_y_number: int,
        medium_x_number: int, medium_y_number: int,
        target_x_number: int, target_y_number: int,
        topology_x_limitation: int, topology_y_limitation: int, 
        data_parallelism_degree: List[int], top_k: int, 
        message_flits: int,     
        reduction: float,
        latency=None, bandwidth=None            
    ): 
        
        pointtopoint_event_tag, pointtopoint_dependency_list = self.pointtopoint.cal_time(
            whole_nodes=whole_nodes, current_event_tag=current_event_tag, current_dependency_list=current_dependency_list,
            source_nodes_coordinates_list=source_nodes_coordinates_list,
            target_nodes_coordinates_list=medium_nodes_coordinates_list,
            source_x_number=source_x_number, source_y_number=source_y_number,
            target_x_number=medium_x_number, target_y_number=medium_y_number,
            topology_x_limitation=topology_x_limitation, topology_y_limitation=topology_y_limitation,
            message_flits=message_flits, 
            latency=latency, bandwidth=bandwidth, reduction=reduction
        )

        alltoall_event_tag, alltoall_dependency_list = self.alltoall.cal_time(
            whole_nodes=whole_nodes, current_event_tag=pointtopoint_event_tag, current_dependency_list=pointtopoint_dependency_list,
            source_nodes_coordinates_list=medium_nodes_coordinates_list,
            source_x_number=medium_x_number, source_y_number=medium_y_number, 
            topology_x_limitation=topology_x_limitation, topology_y_limitation=topology_y_limitation,
            message_flits=message_flits*top_k*data_parallelism_degree[0]*data_parallelism_degree[1], 
            latency=latency, bandwidth=bandwidth, reduction=reduction
        )

        return alltoall_event_tag, alltoall_dependency_list

        # alltoall_event_tag, alltoall_dependency_list = self.alltoall.cal_time(
        #     whole_nodes=whole_nodes, current_event_tag=alltoall_event_tag, current_dependency_list=alltoall_dependency_list,
        #     source_nodes_coordinates_list=medium_nodes_coordinates_list,
        #     source_x_number=medium_x_number, source_y_number=medium_y_number, 
        #     topology_x_limitation=topology_x_limitation, topology_y_limitation=topology_y_limitation, 
        #     message_flits=message_flits*top_k, 
        #     latency=latency, bandwidth=bandwidth, reduction=reduction
        # )

        # if (top_k > 1):
        #     reducelocal_event_tag, reducelocal_dependency_list = self.reducelocal.cal_time(
        #         whole_nodes=whole_nodes, current_event_tag=alltoall_event_tag, current_dependency_list=alltoall_dependency_list,
        #         source_nodes_coordinates_list=source_nodes_coordinates_list,
        #         source_x_number=source_x_number, source_y_number=source_y_number,
        #         topology_x_limitation=topology_x_limitation, topology_y_limitation=topology_y_limitation,
        #         message_flits=message_flits*data_parallelism_degree[0]*data_parallelism_degree[1], 
        #         latency=latency, bandwidth=bandwidth, reduction=reduction
        #     )
        # else:
        #     reducelocal_event_tag = alltoall_event_tag
        #     reducelocal_dependency_list = alltoall_dependency_list

        # pointtopoint_event_tag, pointtopoint_dependency_list = self.pointtopoint.cal_time(
        #     whole_nodes=whole_nodes, current_event_tag=reducelocal_event_tag, current_dependency_list=reducelocal_dependency_list,
        #     source_nodes_coordinates_list=medium_nodes_coordinates_list,
        #     target_nodes_coordinates_list=target_nodes_coordinates_list,
        #     source_x_number=medium_x_number, source_y_number=medium_y_number,
        #     target_x_number=target_x_number, target_y_number=target_y_number,
        #     topology_x_limitation=topology_x_limitation, topology_y_limitation=topology_y_limitation,
        #     message_flits=message_flits, 
        #     latency=latency, bandwidth=bandwidth, reduction=reduction
        # )

        # return pointtopoint_event_tag, pointtopoint_dependency_list


    def fusion(
        self,
        whole_nodes: NodeNetwork, current_event_tag: int, current_dependency_list: List[List[List[int]]], 
        source_nodes_coordinates_list: List[Tuple[int]],
        medium_nodes_coordinates_list: List[Tuple[int]],
        target_nodes_coordinates_list: List[Tuple[int]],
        source_x_number: int, source_y_number: int,
        medium_x_number: int, medium_y_number: int,
        target_x_number: int, target_y_number: int,
        topology_x_limitation: int, topology_y_limitation: int, 
        data_parallelism_degree: List[int], top_k: int, 
        message_flits: int,     
        reduction: float,
        latency=None, bandwidth=None                  
    ): 
        
        manytomanymulticast_event_tag, manytomanymulticast_dependency_list = self.manytomanymulticast.cal_time(
            whole_nodes=whole_nodes, current_event_tag=current_event_tag, current_dependency_list=current_dependency_list,
            source_nodes_coordinates_list=source_nodes_coordinates_list,
            target_nodes_coordinates_list=medium_nodes_coordinates_list,
            source_x_number=source_x_number, source_y_number=source_y_number,
            target_x_number=medium_x_number, target_y_number=medium_y_number,
            topology_x_limitation=topology_x_limitation, topology_y_limitation=topology_y_limitation,
            message_flits=message_flits*top_k*data_parallelism_degree[0]*data_parallelism_degree[1], 
            latency=latency, bandwidth=bandwidth, reduction=reduction
        )

        return manytomanymulticast_event_tag, manytomanymulticast_dependency_list

        # if (top_k > 1):
        #     reducelocal_event_tag, reducelocal_dependency_list = self.reducelocal.cal_time(
        #         whole_nodes=whole_nodes, current_event_tag=manytomanymulticast_event_tag, current_dependency_list=manytomanymulticast_dependency_list,
        #         source_nodes_coordinates_list=medium_nodes_coordinates_list,
        #         source_x_number=medium_x_number, source_y_number=medium_y_number,
        #         topology_x_limitation=topology_x_limitation, topology_y_limitation=topology_y_limitation,
        #         message_flits=message_flits*data_parallelism_degree[0]*data_parallelism_degree[1], 
        #         latency=latency, bandwidth=bandwidth, reduction=reduction
        #     )
        # else:
        #     reducelocal_event_tag = manytomanymulticast_event_tag
        #     reducelocal_dependency_list = manytomanymulticast_dependency_list

        # print(reducelocal_dependency_list)

        # manytomanymulticast_event_tag, manytomanymulticast_dependency_list = self.manytomanymulticast.cal_time(
        #     whole_nodes=whole_nodes, current_event_tag=reducelocal_event_tag, current_dependency_list=reducelocal_dependency_list,
        #     source_nodes_coordinates_list=medium_nodes_coordinates_list,
        #     target_nodes_coordinates_list=target_nodes_coordinates_list,
        #     source_x_number=medium_x_number, source_y_number=medium_y_number,
        #     target_x_number=target_x_number, target_y_number=target_y_number,
        #     topology_x_limitation=topology_x_limitation, topology_y_limitation=topology_y_limitation,
        #     message_flits=message_flits, 
        #     latency=latency, bandwidth=bandwidth, reduction=reduction
        # )

        # return manytomanymulticast_event_tag, manytomanymulticast_dependency_list




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
        "reducelocal": "base", 
        "alltoall": "hierarchicalring", 
        "pointtopoint": "base", 
        "manytomanymulticast": "alpa"
    }

    source_nodes = [(0, 0), (0, 1), (0, 2), (0, 3), (1, 0), (1, 1), (1, 2), (1, 3)]
    source_shape = [2, 4]
    data_parallelism_degree = [2, 2]
    top_k = 2
    medium_nodes = [(0, 4), (0, 5), (0, 6), (0, 7), (1, 4), (1, 5), (1, 6), (1, 7)]
    medium_shape = [2, 4]
    target_nodes = [(2, 4), (2, 5), (2, 6), (2, 7), (3, 4), (3, 5), (3, 6), (3, 7)]
    target_shape = [2, 4]
    whole_flits = 294912 
    reduction_cores = 1 / 1024


    '''
    build
    '''
    pe_base_node_network, pe_base_noc_network = build(
        node_k=node_k, node_n=node_n, ni_k=ni_k, ni_n=ni_n, 
        cfg_topology=cfg_topology
    )


    '''
    User logic
    '''

    pe = pipeline_expert(
        topology=(cfg_topology + str(node_n) + "d"), 
        algorithm=algorithm_dict
    )

    initial_pe_base_event_tag = 0
    initial_pe_base_dependency_list = [[[] for _ in range(source_shape[1])] for _ in range(source_shape[0])]

    pe_base_event_tag, pe_base_dependency_list = pe.base(
        whole_nodes=pe_base_node_network, current_event_tag=initial_pe_base_event_tag, current_dependency_list=initial_pe_base_dependency_list,
        source_nodes_coordinates_list=source_nodes,
        medium_nodes_coordinates_list=medium_nodes,
        target_nodes_coordinates_list=target_nodes,
        source_x_number=source_shape[1], source_y_number=source_shape[0],
        medium_x_number=medium_shape[1], medium_y_number=medium_shape[0],
        target_x_number=target_shape[1], target_y_number=target_shape[0],
        topology_x_limitation=node_k, topology_y_limitation=node_k,
        data_parallelism_degree=data_parallelism_degree, top_k=top_k,
        message_flits=whole_flits, 
        latency=None, bandwidth=None, reduction=reduction_cores
    )    


    '''
    launch
    '''
    pe_base_run_cost = launch(
        whole_nodes=pe_base_node_network, booksim_net=pe_base_noc_network
    )




