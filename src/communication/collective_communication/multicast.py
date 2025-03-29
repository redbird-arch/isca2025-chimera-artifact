# File name  :    multicast.py
# Author     :    xiaocuicui
# Time       :    2024/07/01 21:26:53
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


'''
The collective communication method in this file almost are from LLM inference or traning in the senerio of multiple clusters.
'''

'''
pointtopotin: the two cluster should have the same number of nodes.
The nodes in two clusters which have the same index will have communication.
eg: cluster1: [node1, node2, node3], cluster2: [nodeA, nodeB, nodeC]
    node1 -> nodeA, node2 -> nodeB, node3 -> nodeC
'''

def pointtopoint_mesh2d_base(
    whole_nodes: NodeNetwork, current_event_tag: int, current_dependency_list: List[List[List[int]]], 
    source_nodes_coordinates_list: List[Tuple[int]],
    target_nodes_coordinates_list: List[Tuple[int]],
    target_x_number: int, target_y_number: int,
    topology_x_limitation: int, topology_y_limitation: int, 
    message_flits: int,     
    source_x_number=None, source_y_number=None,
    latency=None, bandwidth=None, reduction=None        
):

    nodes_number = len(source_nodes_coordinates_list)
    topleft_node = source_nodes_coordinates_list[0]
    topleft_x_coord = topleft_node[1]
    topleft_y_coord = topleft_node[0]

    pointtopoint_done_dependency_list = []

    last_dependency_list = current_dependency_list

    for task_idx in range(nodes_number):

        source_node_coordinate = source_nodes_coordinates_list[task_idx]
        send_message_source_idx = source_node_coordinate[0] * topology_x_limitation + source_node_coordinate[1]
        target_node_coordinate = target_nodes_coordinates_list[task_idx]
        send_message_target_idx = target_node_coordinate[0] * topology_x_limitation + target_node_coordinate[1]

        communication_event = CommunicationEvent(current_event_tag, send_message_source_idx, send_message_target_idx, message_flits)
        communication_event.build_dependency(last_dependency_list[source_node_coordinate[0]-topleft_y_coord][source_node_coordinate[1]-topleft_x_coord])
        pointtopoint_done_dependency_list.append(communication_event.event_tag)
        current_event_tag += 1
        whole_nodes.nodes[send_message_source_idx].event_queue.append(communication_event)

    pass_dependency_list = [[pointtopoint_done_dependency_list for _ in range(target_x_number)] for _ in range(target_y_number)]

    return current_event_tag, pass_dependency_list


def pointtopoint_torus2d_base(
    whole_nodes: NodeNetwork, current_event_tag: int, current_dependency_list: List[List[List[int]]], 
    source_nodes_coordinates_list: List[Tuple[int]],
    target_nodes_coordinates_list: List[Tuple[int]],
    target_x_number: int, target_y_number: int,
    topology_x_limitation: int, topology_y_limitation: int, 
    message_flits: int,     
    source_x_number=None, source_y_number=None,
    latency=None, bandwidth=None, reduction=None        
):

    nodes_number = len(source_nodes_coordinates_list)
    topleft_node = source_nodes_coordinates_list[0]
    topleft_x_coord = topleft_node[1]
    topleft_y_coord = topleft_node[0]

    pointtopoint_done_dependency_list = []

    last_dependency_list = current_dependency_list

    for task_idx in range(nodes_number):

        source_node_coordinate = source_nodes_coordinates_list[task_idx]
        send_message_source_idx = source_node_coordinate[0] * topology_x_limitation + source_node_coordinate[1]
        target_node_coordinate = target_nodes_coordinates_list[task_idx]
        send_message_target_idx = target_node_coordinate[0] * topology_x_limitation + target_node_coordinate[1]

        communication_event = CommunicationEvent(current_event_tag, send_message_source_idx, send_message_target_idx, message_flits)
        communication_event.build_dependency(last_dependency_list[source_node_coordinate[0]-topleft_y_coord][source_node_coordinate[1]-topleft_x_coord])
        pointtopoint_done_dependency_list.append(communication_event.event_tag)
        current_event_tag += 1
        whole_nodes.nodes[send_message_source_idx].event_queue.append(communication_event)

    pass_dependency_list = [[pointtopoint_done_dependency_list for _ in range(target_x_number)] for _ in range(target_y_number)]

    return current_event_tag, pass_dependency_list


def pointtopoint_torus3d_base(
    whole_nodes: NodeNetwork, current_event_tag: int, current_dependency_list: List[List[List[int]]],
    source_nodes_coordinates_list: List[Tuple[int]],
    target_nodes_coordinates_list: List[Tuple[int]],
    target_x_number: int, target_y_number: int, target_z_number: int,
    topology_x_limitation: int, topology_y_limitation: int, topology_z_limitation: int,
    message_flits: int,
    source_x_number=None, source_y_number=None, source_z_number=None,
    latency=None, bandwidth=None, reduction=None
):
    
    nodes_number = len(source_nodes_coordinates_list)
    topleft_node = source_nodes_coordinates_list[0]
    topleft_z_coord = topleft_node[2]
    topleft_x_coord = topleft_node[1]
    topleft_y_coord = topleft_node[0]

    pointtopoint_done_dependency_list = []

    last_dependency_list = current_dependency_list

    for task_idx in range(nodes_number):

        source_node_coordinate = source_nodes_coordinates_list[task_idx]
        send_message_source_idx = source_node_coordinate[0] * topology_x_limitation * topology_z_limitation + source_node_coordinate[1] * topology_z_limitation + source_node_coordinate[2]
        target_node_coordinate = target_nodes_coordinates_list[task_idx]
        send_message_target_idx = target_node_coordinate[0] * topology_x_limitation * topology_z_limitation + target_node_coordinate[1] * topology_z_limitation + target_node_coordinate[2]

        communication_event = CommunicationEvent(current_event_tag, send_message_source_idx, send_message_target_idx, message_flits)
        communication_event.build_dependency(last_dependency_list[source_node_coordinate[0]-topleft_y_coord][source_node_coordinate[1]-topleft_x_coord][source_node_coordinate[2]-topleft_z_coord])
        pointtopoint_done_dependency_list.append(communication_event.event_tag)
        current_event_tag += 1
        whole_nodes.nodes[send_message_source_idx].event_queue.append(communication_event)

    pass_dependency_list = [[[pointtopoint_done_dependency_list for _ in range(target_x_number)] for _ in range(target_y_number)] for _ in range(target_z_number)]

    return current_event_tag, pass_dependency_list
    

def pointtopoint_dgx2_base(
    whole_nodes: NodeNetwork_2D, current_event_tag: int, current_dependency_list: List[List[List[int]]], 
    source_nodes_coordinates_list: List[Tuple[int]],
    target_nodes_coordinates_list: List[Tuple[int]],
    target_x_number: int, target_y_number: int,
    topology_x_limitation: int, topology_y_limitation: int, 
    message_flits: int,     
    source_x_number=None, source_y_number=None,
    latency=None, bandwidth=None, reduction=None        
):

    nodes_number = len(source_nodes_coordinates_list)
    topleft_node = source_nodes_coordinates_list[0]
    topleft_x_coord = topleft_node[1]
    topleft_y_coord = topleft_node[0]

    pointtopoint_done_dependency_list = []

    last_dependency_list = current_dependency_list

    for task_idx in range(nodes_number):

        source_node_coordinate = source_nodes_coordinates_list[task_idx]
        send_message_source_idx = source_node_coordinate[0] * topology_x_limitation + source_node_coordinate[1]
        target_node_coordinate = target_nodes_coordinates_list[task_idx]
        send_message_target_idx = target_node_coordinate[0] * topology_x_limitation + target_node_coordinate[1]

        communication_event = CommunicationEvent(current_event_tag, send_message_source_idx, send_message_target_idx, message_flits)
        communication_event.build_dependency(last_dependency_list[source_node_coordinate[0]-topleft_y_coord][source_node_coordinate[1]-topleft_x_coord])
        pointtopoint_done_dependency_list.append(communication_event.event_tag)
        current_event_tag += 1
        whole_nodes.nodes[send_message_source_idx].event_queue.append(communication_event)

    pass_dependency_list = [[pointtopoint_done_dependency_list for _ in range(target_x_number)] for _ in range(target_y_number)]

    return current_event_tag, pass_dependency_list



def manytomanymulticast_mesh2d_alpa(
    whole_nodes: NodeNetwork, current_event_tag: int, current_dependency_list: List[List[List[int]]], 
    source_nodes_coordinates_list: List[Tuple[int]],
    target_nodes_coordinates_list: List[Tuple[int]],
    target_x_number: int, target_y_number: int,
    topology_x_limitation: int, topology_y_limitation: int, 
    message_flits: int,     
    source_x_number=None, source_y_number=None,
    latency=None, bandwidth=None, reduction=None 
):

    topleft_node = source_nodes_coordinates_list[0]
    topleft_x_coord = topleft_node[1]
    topleft_y_coord = topleft_node[0]

    source_nodes_number = len(source_nodes_coordinates_list)
    target_nodes_number = len(target_nodes_coordinates_list)

    manytomanymulticast_done_dependency_list = []

    last_dependency_list = current_dependency_list

    for source_node_idx in range(source_nodes_number):

        source_node_coordinate = source_nodes_coordinates_list[source_node_idx]
        send_message_source_idx = source_node_coordinate[0] * topology_x_limitation + source_node_coordinate[1]

        for target_node_idx in range(target_nodes_number):

            target_node_idx_ordered = (source_node_idx + target_node_idx) % source_nodes_number

            target_node_coordinate = target_nodes_coordinates_list[target_node_idx_ordered]
            send_message_target_idx = target_node_coordinate[0] * topology_x_limitation + target_node_coordinate[1]

            communication_event = CommunicationEvent(current_event_tag, send_message_source_idx, send_message_target_idx, int(message_flits / target_nodes_number / source_nodes_number))
            communication_event.build_dependency(last_dependency_list[source_node_coordinate[0]-topleft_y_coord][source_node_coordinate[1]-topleft_x_coord])
            manytomanymulticast_done_dependency_list.append(communication_event.event_tag)
            current_event_tag += 1
            whole_nodes.nodes[send_message_source_idx].event_queue.append(communication_event)

    pass_dependency_list = [[manytomanymulticast_done_dependency_list for _ in range(target_x_number)] for _ in range(target_y_number)]

    return current_event_tag, pass_dependency_list


def manytomanymulticast_torus2d_alpa(
    whole_nodes: NodeNetwork, current_event_tag: int, current_dependency_list: List[List[List[int]]], 
    source_nodes_coordinates_list: List[Tuple[int]],
    target_nodes_coordinates_list: List[Tuple[int]],
    target_x_number: int, target_y_number: int,
    topology_x_limitation: int, topology_y_limitation: int, 
    message_flits: int,     
    source_x_number=None, source_y_number=None,
    latency=None, bandwidth=None, reduction=None 
):

    topleft_node = source_nodes_coordinates_list[0]
    topleft_x_coord = topleft_node[1]
    topleft_y_coord = topleft_node[0]

    source_nodes_number = len(source_nodes_coordinates_list)
    target_nodes_number = len(target_nodes_coordinates_list)

    manytomanymulticast_done_dependency_list = []

    last_dependency_list = current_dependency_list

    for source_node_idx in range(source_nodes_number):

        source_node_coordinate = source_nodes_coordinates_list[source_node_idx]
        send_message_source_idx = source_node_coordinate[0] * topology_x_limitation + source_node_coordinate[1]

        for target_node_idx in range(target_nodes_number):

            target_node_idx_ordered = (source_node_idx + target_node_idx) % source_nodes_number

            target_node_coordinate = target_nodes_coordinates_list[target_node_idx_ordered]
            send_message_target_idx = target_node_coordinate[0] * topology_x_limitation + target_node_coordinate[1]

            communication_event = CommunicationEvent(current_event_tag, send_message_source_idx, send_message_target_idx, int(message_flits / target_nodes_number / source_nodes_number))
            communication_event.build_dependency(last_dependency_list[source_node_coordinate[0]-topleft_y_coord][source_node_coordinate[1]-topleft_x_coord])
            manytomanymulticast_done_dependency_list.append(communication_event.event_tag)
            current_event_tag += 1
            whole_nodes.nodes[send_message_source_idx].event_queue.append(communication_event)

    pass_dependency_list = [[manytomanymulticast_done_dependency_list for _ in range(target_x_number)] for _ in range(target_y_number)]

    return current_event_tag, pass_dependency_list


def manytomanymulticast_torus3d_alpa(
    whole_nodes: NodeNetwork, current_event_tag: int, current_dependency_list: List[List[List[int]]], 
    source_nodes_coordinates_list: List[Tuple[int]],
    target_nodes_coordinates_list: List[Tuple[int]],
    target_x_number: int, target_y_number: int, target_z_number: int,
    topology_x_limitation: int, topology_y_limitation: int, topology_z_limitation: int, 
    message_flits: int,     
    source_x_number=None, source_y_number=None, source_z_number=None,
    latency=None, bandwidth=None, reduction=None 
):

    topleft_node = source_nodes_coordinates_list[0]
    topleft_z_coord = topleft_node[2]
    topleft_x_coord = topleft_node[1]
    topleft_y_coord = topleft_node[0]

    source_nodes_number = len(source_nodes_coordinates_list)
    target_nodes_number = len(target_nodes_coordinates_list)

    manytomanymulticast_done_dependency_list = []

    last_dependency_list = current_dependency_list

    for source_node_idx in range(source_nodes_number):

        source_node_coordinate = source_nodes_coordinates_list[source_node_idx]
        send_message_source_idx = source_node_coordinate[0] * topology_x_limitation * topology_y_limitation + source_node_coordinate[1] * topology_x_limitation + source_node_coordinate[2]

        for target_node_idx in range(target_nodes_number):

            target_node_idx_ordered = (source_node_idx + target_node_idx) % source_nodes_number

            target_node_coordinate = target_nodes_coordinates_list[target_node_idx_ordered]
            send_message_target_idx = target_node_coordinate[0] * topology_x_limitation * topology_y_limitation + target_node_coordinate[1] * topology_x_limitation + target_node_coordinate[2]

            communication_event = CommunicationEvent(current_event_tag, send_message_source_idx, send_message_target_idx, int(message_flits / target_nodes_number / source_nodes_number))
            communication_event.build_dependency(last_dependency_list[source_node_coordinate[0]-topleft_y_coord][source_node_coordinate[1]-topleft_x_coord][source_node_coordinate[2]-topleft_z_coord])
            manytomanymulticast_done_dependency_list.append(communication_event.event_tag)
            current_event_tag += 1
            whole_nodes.nodes[send_message_source_idx].event_queue.append(communication_event)

    pass_dependency_list = [[[manytomanymulticast_done_dependency_list for _ in range(target_x_number)] for _ in range(target_y_number)] for _ in range(target_z_number)]

    return current_event_tag, pass_dependency_list


def manytomanymulticast_dgx2_alpa(
    whole_nodes: NodeNetwork_2D, current_event_tag: int, current_dependency_list: List[List[List[int]]], 
    source_nodes_coordinates_list: List[Tuple[int]],
    target_nodes_coordinates_list: List[Tuple[int]],
    target_x_number: int, target_y_number: int,
    topology_x_limitation: int, topology_y_limitation: int, 
    message_flits: int,     
    source_x_number=None, source_y_number=None,
    latency=None, bandwidth=None, reduction=None 
):

    topleft_node = source_nodes_coordinates_list[0]
    topleft_x_coord = topleft_node[1]
    topleft_y_coord = topleft_node[0]

    source_nodes_number = len(source_nodes_coordinates_list)
    target_nodes_number = len(target_nodes_coordinates_list)

    manytomanymulticast_done_dependency_list = []

    last_dependency_list = current_dependency_list

    for source_node_idx in range(source_nodes_number):

        source_node_coordinate = source_nodes_coordinates_list[source_node_idx]
        send_message_source_idx = source_node_coordinate[0] * topology_x_limitation + source_node_coordinate[1]

        for target_node_idx in range(target_nodes_number):

            target_node_idx_ordered = (source_node_idx + target_node_idx) % source_nodes_number

            target_node_coordinate = target_nodes_coordinates_list[target_node_idx_ordered]
            send_message_target_idx = target_node_coordinate[0] * topology_x_limitation + target_node_coordinate[1]

            communication_event = CommunicationEvent(current_event_tag, send_message_source_idx, send_message_target_idx, int(message_flits / target_nodes_number / source_nodes_number))
            communication_event.build_dependency(last_dependency_list[source_node_coordinate[0]-topleft_y_coord][source_node_coordinate[1]-topleft_x_coord])
            manytomanymulticast_done_dependency_list.append(communication_event.event_tag)
            current_event_tag += 1
            whole_nodes.nodes[send_message_source_idx].event_queue.append(communication_event)

    pass_dependency_list = [[manytomanymulticast_done_dependency_list for _ in range(target_x_number)] for _ in range(target_y_number)]

    return current_event_tag, pass_dependency_list


class pointtopoint(object):

    def __init__(self, topology: str, algorithm: str):
        communication_name = self.__class__.__name__
        collective_selected = communication_name + "_" + topology + "_" + algorithm
        self.cal_time = globals()[collective_selected]


class manytomanymulticast(object):

    def __init__(self, topology: str, algorithm: str):
        communication_name = self.__class__.__name__
        collective_selected = communication_name + "_" + topology + "_" + algorithm
        self.cal_time = globals()[collective_selected]




if __name__ == "__main__":

    # node_k = 8
    # node_n = 2
    # ni_k = node_k
    # ni_n = node_n
    # cfg_topology = "mesh"
    # cfg_filepath = os.path.join(file_path, '../backend/booksim2/runfiles/mesh_o_torus_py.cfg')
    # modify_topology_cfg_file(cfg_filepath, cfg_topology, node_k, node_n)

    # test_topology = cfg_topology + str(node_n) + "d"
    # test_source_nodes = [(0, 0), (0, 1), (0, 2), (0, 3), (1, 0), (1, 1), (1, 2), (1, 3)]
    # test_source_shape = [2, 4]
    # test_target_nodes = [(0, 4), (0, 5), (0, 6), (0, 7), (1, 4), (1, 5), (1, 6), (1, 7)]
    # test_target_shape = [2, 4]
    # test_whole_flits = 294912 

    # current_event_tag = 0
    # initial_dependency_list = [[[] for _ in range(test_source_shape[1])] for _ in range(test_source_shape[0])]

    # test_communication_algorithm = "base"

    # node_network = NodeNetwork(node_k, node_n, ni_k, ni_n)
    # node_network.create_nodes()

    # communication_scheduler = pointtopoint(test_topology, test_communication_algorithm)

    # now_event_tag, _ = communication_scheduler.cal_time(
    #     whole_nodes=node_network, current_event_tag=current_event_tag, current_dependency_list=initial_dependency_list,
    #     source_nodes_coordinates_list=test_source_nodes,
    #     target_nodes_coordinates_list=test_target_nodes,
    #     source_x_number=test_source_shape[0], source_y_number=test_source_shape[1],
    #     target_x_number=test_target_shape[0], target_y_number=test_target_shape[1],
    #     topology_x_limitation=node_k, topology_y_limitation=node_k,
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



    '''
    |||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||
    '''
    node_k = 2
    node_n = 3
    ni_k = node_k
    ni_n = node_n
    cfg_topology = "torus"
    cfg_filepath = os.path.join(file_path, '../backend/booksim2/runfiles/mesh_o_torus_py.cfg')
    modify_topology_cfg_file(cfg_filepath, cfg_topology, node_k, node_n)

    test_topology = cfg_topology + str(node_n) + "d"   
    test_source_nodes = [(0, 0, 0), (0, 0, 1), (0, 1, 0), (0, 1, 1)]
    test_source_shape = [1, 2, 2]
    test_target_nodes = [(1, 0, 0), (1, 0, 1), (1, 1, 0), (1, 1, 1)]
    test_target_shape = [1, 2, 2]
    test_whole_flits = 4 * 1024 * 1024

    current_event_tag = 0
    initial_dependency_list = [[[[] for _ in range(test_source_shape[2])] for _ in range(test_source_shape[1])] for _ in range(test_source_shape[0])]

    test_communication_algorithm = "alpa"

    node_network = NodeNetwork(node_k, node_n, ni_k, ni_n)
    node_network.create_nodes()

    communication_scheduler = manytomanymulticast(test_topology, test_communication_algorithm)

    now_event_tag, _ = communication_scheduler.cal_time(
        whole_nodes=node_network, current_event_tag=current_event_tag, current_dependency_list=initial_dependency_list,
        source_nodes_coordinates_list=test_source_nodes,
        target_nodes_coordinates_list=test_target_nodes,
        source_x_number=test_source_shape[2], source_y_number=test_source_shape[1], source_z_number=test_source_shape[0],
        target_x_number=test_target_shape[2], target_y_number=test_target_shape[1], target_z_number=test_target_shape[0],
        topology_x_limitation=node_k, topology_y_limitation=node_k, topology_z_limitation=node_k,
        message_flits=test_whole_flits, 
        latency=None, bandwidth=None, reduction=None
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



    # '''
    # |||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||
    # '''
    # node_k = 8
    # node_n = 2
    # ni_k = node_k
    # ni_n = node_n
    # cfg_topology = "dgx2"
    # cfg_filepath = os.path.join(file_path, '../backend/booksim2/runfiles/dgx2.cfg')
    # # modify_topology_cfg_file(cfg_filepath, cfg_topology, node_k, node_n)

    # # test_topology = cfg_topology + str(node_n) + "d"
    # test_topology = cfg_topology
    # test_source_nodes = [(0, 0), (0, 1), (0, 2), (0, 3)]
    # test_source_shape = [1, 4]
    # test_target_nodes = [(1, 0), (1, 1), (1, 2), (1, 3)]
    # test_target_shape = [1, 4]
    # test_whole_flits = 4 * 1024 * 1024 

    # current_event_tag = 0
    # initial_dependency_list = [[[] for _ in range(test_source_shape[1])] for _ in range(test_source_shape[0])]

    # test_communication_algorithm = "alpa"

    # node_network = NodeNetwork_2D(node_k, node_n, ni_k, ni_n)
    # node_network.create_nodes()

    # communication_scheduler = manytomanymulticast(test_topology, test_communication_algorithm)

    # now_event_tag, _ = communication_scheduler.cal_time(
    #     whole_nodes=node_network, current_event_tag=current_event_tag, current_dependency_list=initial_dependency_list,
    #     source_nodes_coordinates_list=test_source_nodes,
    #     target_nodes_coordinates_list=test_target_nodes,
    #     source_x_number=test_source_shape[1], source_y_number=test_source_shape[0],
    #     target_x_number=test_target_shape[1], target_y_number=test_target_shape[0],
    #     topology_x_limitation=node_k, topology_y_limitation=node_k,
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


