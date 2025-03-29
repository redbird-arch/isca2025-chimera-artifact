# File name  :    allreduce.py
# Author     :    xiaocuicui
# Time       :    2024/06/30 18:22:35
# Version    :    V1.0
# Abstract   :        

import os
import sys
file_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(file_path)
sys.path.append(os.path.join(file_path, '../../components/'))
sys.path.append(os.path.join(file_path, '../backend/'))
sys.path.append(os.path.join(file_path, '../../utils/'))


from typing import List, Tuple

from NodeNetwork import NodeNetwork
from NodeNetwork_2D import NodeNetwork_2D
from Event import Event, CommunicationEvent, ComputationEvent

from config_para import modify_topology_cfg_file
from Booksim_Api import BookSim_Interface
from Runner import run

from reducescatter import reducescatter_mesh2d_hierarchicalring, reducescatter_torus2d_hierarchicalring, reducescatter_torus3d_hierarchicalring, reducescatter_dgx2_havlingdoubling
from allgather import allgather_mesh2d_hierarchicalring, allgather_torus2d_hierarchicalring, allgather_torus3d_hierarchicalring, allgather_dgx2_havlingdoubling


'''
hierarchicalring is from <Exhaustive Study of Hierarchical AllReduce Patterns for Large Messages Between GPUs>.
It means that do the communication per dimension.
'''

def allreduce_mesh2d_hierarchicalring(
    whole_nodes: NodeNetwork, current_event_tag: int, current_dependency_list: List[List[List[int]]], 
    source_nodes_coordinates_list: List[Tuple[int]],
    source_x_number: int, source_y_number: int,
    topology_x_limitation: int, topology_y_limitation: int, 
    message_flits: int, 
    reduction: float,
    latency=None, bandwidth=None
) -> int:

    reducescatter_event_tag, reducescatter_dependency_list = reducescatter_mesh2d_hierarchicalring(
        whole_nodes=whole_nodes, current_event_tag=current_event_tag, current_dependency_list=current_dependency_list,
        source_nodes_coordinates_list=source_nodes_coordinates_list,
        source_x_number=source_x_number, source_y_number=source_y_number,
        topology_x_limitation=topology_x_limitation, topology_y_limitation=topology_y_limitation,
        message_flits=message_flits, 
        latency=latency, bandwidth=bandwidth, reduction=reduction
    )

    # print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
    # print("reducescatter_event_tag: ", reducescatter_event_tag)
    # print("reducescatter_dependency_list: ", reducescatter_dependency_list)
    # print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!")

    allgather_event_tag, allgather_dependency_list = allgather_mesh2d_hierarchicalring(
        whole_nodes=whole_nodes, current_event_tag=reducescatter_event_tag, current_dependency_list=reducescatter_dependency_list,
        source_nodes_coordinates_list=source_nodes_coordinates_list,
        source_x_number=source_x_number, source_y_number=source_y_number,
        topology_x_limitation=topology_x_limitation, topology_y_limitation=topology_y_limitation,
        message_flits=message_flits, 
        latency=latency, bandwidth=bandwidth, reduction=None
    )        

    return allgather_event_tag, allgather_dependency_list


def allreduce_torus2d_hierarchicalring(
    whole_nodes: NodeNetwork, current_event_tag: int, current_dependency_list: List[List[List[int]]], 
    source_nodes_coordinates_list: List[Tuple[int]],
    source_x_number: int, source_y_number: int,
    topology_x_limitation: int, topology_y_limitation: int, 
    message_flits: int, 
    reduction: float,
    latency=None, bandwidth=None
) -> int:

    reducescatter_event_tag, reducescatter_dependency_list = reducescatter_torus2d_hierarchicalring(
        whole_nodes=whole_nodes, current_event_tag=current_event_tag, current_dependency_list=current_dependency_list,
        source_nodes_coordinates_list=source_nodes_coordinates_list,
        source_x_number=source_x_number, source_y_number=source_y_number,
        topology_x_limitation=topology_x_limitation, topology_y_limitation=topology_y_limitation,
        message_flits=message_flits, 
        latency=latency, bandwidth=bandwidth, reduction=reduction
    )

    # print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
    # print("reducescatter_event_tag: ", reducescatter_event_tag)
    # print("reducescatter_dependency_list: ", reducescatter_dependency_list)
    # print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!")

    allgather_event_tag, allgather_dependency_list = allgather_torus2d_hierarchicalring(
        whole_nodes=whole_nodes, current_event_tag=reducescatter_event_tag, current_dependency_list=reducescatter_dependency_list,
        source_nodes_coordinates_list=source_nodes_coordinates_list,
        source_x_number=source_x_number, source_y_number=source_y_number,
        topology_x_limitation=topology_x_limitation, topology_y_limitation=topology_y_limitation,
        message_flits=message_flits, 
        latency=latency, bandwidth=bandwidth, reduction=None
    )        

    return allgather_event_tag, allgather_dependency_list


def allreduce_torus3d_hierarchicalring(
    whole_nodes: NodeNetwork, current_event_tag: int, current_dependency_list: List[List[List[int]]], 
    source_nodes_coordinates_list: List[Tuple[int]],
    source_x_number: int, source_y_number: int, source_z_number: int,
    topology_x_limitation: int, topology_y_limitation: int, topology_z_limitation: int,
    message_flits: int, 
    reduction: float,
    latency=None, bandwidth=None
) -> int:

    reducescatter_event_tag, reducescatter_dependency_list = reducescatter_torus3d_hierarchicalring(
        whole_nodes=whole_nodes, current_event_tag=current_event_tag, current_dependency_list=current_dependency_list,
        source_nodes_coordinates_list=source_nodes_coordinates_list,
        source_x_number=source_x_number, source_y_number=source_y_number, source_z_number=source_z_number,
        topology_x_limitation=topology_x_limitation, topology_y_limitation=topology_y_limitation, topology_z_limitation=topology_z_limitation,
        message_flits=message_flits, 
        latency=latency, bandwidth=bandwidth, reduction=reduction
    )

    # print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
    # print("reducescatter_event_tag: ", reducescatter_event_tag)
    # print("reducescatter_dependency_list: ", reducescatter_dependency_list)
    # print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!")

    allgather_event_tag, allgather_dependency_list = allgather_torus3d_hierarchicalring(
        whole_nodes=whole_nodes, current_event_tag=reducescatter_event_tag, current_dependency_list=reducescatter_dependency_list,
        source_nodes_coordinates_list=source_nodes_coordinates_list,
        source_x_number=source_x_number, source_y_number=source_y_number, source_z_number=source_z_number,
        topology_x_limitation=topology_x_limitation, topology_y_limitation=topology_y_limitation, topology_z_limitation=topology_z_limitation,
        message_flits=message_flits, 
        latency=latency, bandwidth=bandwidth, reduction=None
    )        

    return allgather_event_tag, allgather_dependency_list


def allreduce_dgx2_havlingdoubling(
    whole_nodes: NodeNetwork_2D, current_event_tag: int, current_dependency_list: List[List[List[int]]], 
    source_nodes_coordinates_list: List[Tuple[int]],
    source_x_number: int, source_y_number: int,
    topology_x_limitation: int, topology_y_limitation: int, 
    message_flits: int, 
    reduction: float,
    latency=None, bandwidth=None
) -> int:

    reducescatter_event_tag, reducescatter_dependency_list = reducescatter_dgx2_havlingdoubling(
        whole_nodes=whole_nodes, current_event_tag=current_event_tag, current_dependency_list=current_dependency_list,
        source_nodes_coordinates_list=source_nodes_coordinates_list,
        source_x_number=source_x_number, source_y_number=source_y_number,
        topology_x_limitation=topology_x_limitation, topology_y_limitation=topology_y_limitation,
        message_flits=message_flits, 
        latency=latency, bandwidth=bandwidth, reduction=reduction
    )

    # print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
    # print("reducescatter_event_tag: ", reducescatter_event_tag)
    # print("reducescatter_dependency_list: ", reducescatter_dependency_list)
    # print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!")

    allgather_event_tag, allgather_dependency_list = allgather_dgx2_havlingdoubling(
        whole_nodes=whole_nodes, current_event_tag=reducescatter_event_tag, current_dependency_list=reducescatter_dependency_list,
        source_nodes_coordinates_list=source_nodes_coordinates_list,
        source_x_number=source_x_number, source_y_number=source_y_number,
        topology_x_limitation=topology_x_limitation, topology_y_limitation=topology_y_limitation,
        message_flits=message_flits, 
        latency=latency, bandwidth=bandwidth, reduction=None
    )        

    return allgather_event_tag, allgather_dependency_list


class allreduce(object):

    def __init__(self, topology: str, algorithm: str):
        communication_name = self.__class__.__name__
        collective_selected = communication_name + "_" + topology + "_" + algorithm
        self.cal_time = globals()[collective_selected]



if __name__ == "__main__":

    node_k = 2
    node_n = 3
    ni_k = node_k
    ni_n = node_n
    cfg_topology = "torus"
    cfg_filepath = os.path.join(file_path, '../backend/booksim2/runfiles/mesh_o_torus_py.cfg')
    modify_topology_cfg_file(cfg_filepath, cfg_topology, node_k, node_n)

    # test_topology = cfg_topology + str(node_n) + "d"
    # test_source_nodes = [(0, 0), (0, 1), (1, 0), (1, 1)]
    # test_source_shape = [2, 2]
    # test_whole_flits = 1024

    test_topology = cfg_topology + str(node_n) + "d"
    test_source_nodes = [(0, 0, 0), (0, 0, 1), (0, 1, 0), (0, 1, 1), (1, 0, 0), (1, 0, 1), (1, 1, 0), (1, 1, 1)]
    test_source_shape = [2, 2, 2]
    test_whole_flits = 4096
    test_reduction_cores = 1 / 1024
    # test_reduction_cores = None

    current_event_tag = 0
    initial_dependency_list = [[[[] for _ in range(test_source_shape[2])] for _ in range(test_source_shape[1])] for _ in range(test_source_shape[0])]

    test_communication_algorithm = "hierarchicalring"

    node_network = NodeNetwork(node_k, node_n, ni_k, ni_n)

    communication_scheduler = allreduce(test_topology, test_communication_algorithm)

    now_event_tag, _ = communication_scheduler.cal_time(
        whole_nodes=node_network, current_event_tag=current_event_tag, current_dependency_list=initial_dependency_list,
        source_nodes_coordinates_list=test_source_nodes,
        source_x_number=test_source_shape[2], source_y_number=test_source_shape[1],
        source_z_number=test_source_shape[0],
        topology_x_limitation=node_k, topology_y_limitation=node_k, topology_z_limitation=node_k,
        message_flits=test_whole_flits, 
        latency=None, bandwidth=None, reduction=test_reduction_cores
    )
    print(f"current_event_tag: {now_event_tag}")
    node_network.show_nodes_events()
    print("====================================")
    

    print("====================================")
    print("Begin to build nis...")
    node_network.build_nis_events()
    node_network.show_nis_events()    


    noc_network = BookSim_Interface(cfg_filepath)
    print("====================================")
    print("Begin to run...")
    end_cycle = run(whole_nodes=node_network, current_cycle=0, network=noc_network)
    print("************************************")    
    print(f"end_cycle: {end_cycle}")
    print("************************************")


    # cfg_filepath = os.path.join(file_path, '../backend/booksim2/runfiles/dgx2.cfg')
    # test_cfg_topology = "dgx2"
    # test_source_nodes = [(0, 0), (0, 1), (0, 2), (0, 3), (1, 0), (1, 1), (1, 2), (1, 3)]
    # test_source_shape = [2, 4]
    # test_whole_flits = 294912

    # current_event_tag = 0
    # initial_dependency_list = [[[] for _ in range(test_source_shape[1])] for _ in range(test_source_shape[0])]

    # test_communication_algorithm = "havlingdoubling"

    # node_network = NodeNetwork_2D(8, 2, 8, 2)
    # node_network.create_nodes()

    # communication_scheduler = allreduce(test_cfg_topology, test_communication_algorithm)

    # now_event_tag, _ = communication_scheduler.cal_time(
    #     whole_nodes=node_network, current_event_tag=current_event_tag, current_dependency_list=initial_dependency_list,
    #     source_nodes_coordinates_list=test_source_nodes,
    #     source_x_number=test_source_shape[1], source_y_number=test_source_shape[0],
    #     topology_x_limitation=8, topology_y_limitation=2,
    #     message_flits=test_whole_flits, 
    #     latency=None, bandwidth=None, reduction=None
    # )
    # print(f"current_event_tag: {now_event_tag}")
    # node_network.show_nodes_events()
    # print("====================================")

    # print("====================================")
    # print("Begin to build nis...")
    # node_network.build_nis_events()
    # node_network.show_nis_events()


    # noc_network = BookSim_Interface(cfg_filepath)
    # print("====================================")
    # print("Begin to run...")
    # end_cycle = run(whole_nodes=node_network, current_cycle=0, network=noc_network)
    # print("************************************")
    # print(f"end_cycle: {end_cycle}")
    # print("************************************")






