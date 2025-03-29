# File name  :    alltoall.py
# Author     :    xiaocuicui
# Time       :    2024/07/01 21:15:37
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
import numpy as np

from NodeNetwork import NodeNetwork
from NodeNetwork_2D import NodeNetwork_2D
from Event import Event, CommunicationEvent, ComputationEvent

from config_para import modify_topology_cfg_file
from Booksim_Api import BookSim_Interface
from Runner import run


def alltoall_mesh2d_hierarchicalring(
    whole_nodes: NodeNetwork, current_event_tag: int, current_dependency_list: List[List[List[int]]],
    source_nodes_coordinates_list: List[Tuple[int]],
    source_x_number: int, source_y_number: int,
    topology_x_limitation: int, topology_y_limitation: int, 
    message_flits: int, 
    reduction: float,
    latency=None, bandwidth=None
) -> int:

    nodes_number = source_x_number * source_y_number
    initial_event_tag = current_event_tag

    alltoall_done_dependency_list = []

    '''
    all nodes lists should be ordered by idx
    
    '''
    topleft_node = source_nodes_coordinates_list[0]
    topleft_x_coord = topleft_node[1]
    topleft_y_coord = topleft_node[0]
    # x first
    for step_x in range(source_x_number - 1):
        if step_x == 0:
            last_dependency_list = current_dependency_list
            node_dependency_list = [[[] for _ in range(source_x_number)] for _ in range(source_y_number)]
        else:
            last_dependency_list = node_dependency_list
            node_dependency_list = [[[] for _ in range(source_x_number)] for _ in range(source_y_number)]
        for y_idx in range(source_y_number):
            send_message_source_y = topleft_y_coord + y_idx
            send_message_target_y = send_message_source_y
            send_message_target_y_idx = y_idx
            for x_idx in range(source_x_number):
                send_message_source_x = x_idx + topleft_x_coord
                send_message_source_idx = send_message_source_y * topology_x_limitation + send_message_source_x
                send_message_target_x = (x_idx + step_x + 1) % source_x_number + topleft_x_coord
                send_message_target_x_idx = (x_idx + step_x + 1) % source_x_number
                send_message_target_idx = send_message_target_y * topology_x_limitation + send_message_target_x
                # print(f"send_message_source_idx: {send_message_source_idx}, send_message_target_idx: {send_message_target_idx}")
                # print(whole_nodes.nodes[send_message_source_idx].coordinate)
                if (send_message_source_y, send_message_source_x) != whole_nodes.nodes[send_message_source_idx].coordinate:
                    raise ValueError(f"source node idx {send_message_source_idx} is not matched with source node coordinate {send_message_source_y, send_message_source_x}")
                if (send_message_target_y, send_message_target_x) != whole_nodes.nodes[send_message_target_idx].coordinate:
                    raise ValueError(f"target node idx {send_message_target_idx} is not matched with target node coordinate {send_message_target_y, send_message_target_x}")
                node_dependency_list[send_message_target_y_idx][send_message_target_x_idx].append(current_event_tag)
                communication_event = CommunicationEvent(current_event_tag, send_message_source_idx, send_message_target_idx, message_flits / source_x_number / nodes_number)
                communication_event.build_dependency(last_dependency_list[y_idx][x_idx])
                alltoall_done_dependency_list.append(current_event_tag)
                current_event_tag += 1
                whole_nodes.nodes[send_message_source_idx].event_queue.append(communication_event)

    # y second
    for step_y in range(source_y_number - 1):
        if source_x_number == 1:
            if step_y == 0:
                last_dependency_list = current_dependency_list
                node_dependency_list = [[[] for _ in range(source_x_number)] for _ in range(source_y_number)]
            else:
                last_dependency_list = node_dependency_list
                node_dependency_list = [[[] for _ in range(source_x_number)] for _ in range(source_y_number)]
        else:
            last_dependency_list = node_dependency_list
            node_dependency_list = [[[] for _ in range(source_x_number)] for _ in range(source_y_number)]
        for x_idx in range(source_x_number):
            send_message_source_x = topleft_x_coord + x_idx
            send_message_target_x = send_message_source_x
            send_message_target_x_idx = x_idx
            for y_idx in range(source_y_number):
                send_message_source_y = y_idx + topleft_y_coord
                send_message_source_idx = send_message_source_y * topology_x_limitation + send_message_source_x
                send_message_target_y = (y_idx + step_y + 1) % source_y_number + topleft_y_coord
                send_message_target_y_idx = (y_idx + step_y + 1) % source_y_number
                send_message_target_idx = send_message_target_y * topology_x_limitation + send_message_target_x
                if (send_message_source_y, send_message_source_x) != whole_nodes.nodes[send_message_source_idx].coordinate:
                    raise ValueError(f"source node idx {send_message_source_idx} is not matched with source node coordinate {send_message_source_y, send_message_source_x}")
                if (send_message_target_y, send_message_target_x) != whole_nodes.nodes[send_message_target_idx].coordinate:
                    raise ValueError(f"target node idx {send_message_target_idx} is not matched with target node coordinate {send_message_target_y, send_message_target_x}")
                node_dependency_list[send_message_target_y_idx][send_message_target_x_idx].append(current_event_tag)
                alltoall_done_dependency_list.append(current_event_tag)
                communication_event = CommunicationEvent(current_event_tag, send_message_source_idx, send_message_target_idx, message_flits / source_y_number / nodes_number)
                communication_event.build_dependency(last_dependency_list[y_idx][x_idx])
                current_event_tag += 1
                whole_nodes.nodes[send_message_source_idx].event_queue.append(communication_event)

    pass_dependency_list = [[alltoall_done_dependency_list for _ in range(source_x_number)] for _ in range(source_y_number)]

    return current_event_tag, pass_dependency_list


def alltoall_torus2d_hierarchicalring(
    whole_nodes: NodeNetwork, current_event_tag: int, current_dependency_list: List[List[List[int]]],
    source_nodes_coordinates_list: List[Tuple[int]],
    source_x_number: int, source_y_number: int,
    topology_x_limitation: int, topology_y_limitation: int, 
    message_flits: int, 
    reduction: float,
    latency=None, bandwidth=None
) -> int:

    nodes_number = source_x_number * source_y_number
    initial_event_tag = current_event_tag

    alltoall_done_dependency_list = []

    '''
    all nodes lists should be ordered by idx
    
    '''
    topleft_node = source_nodes_coordinates_list[0]
    topleft_x_coord = topleft_node[1]
    topleft_y_coord = topleft_node[0]
    # x first
    for step_x in range(source_x_number - 1):
        if step_x == 0:
            last_dependency_list = current_dependency_list
            node_dependency_list = [[[] for _ in range(source_x_number)] for _ in range(source_y_number)]
        else:
            last_dependency_list = node_dependency_list
            node_dependency_list = [[[] for _ in range(source_x_number)] for _ in range(source_y_number)]
        for y_idx in range(source_y_number):
            send_message_source_y = topleft_y_coord + y_idx
            send_message_target_y = send_message_source_y
            send_message_target_y_idx = y_idx
            for x_idx in range(source_x_number):
                send_message_source_x = x_idx + topleft_x_coord
                send_message_source_idx = send_message_source_y * topology_x_limitation + send_message_source_x
                send_message_target_x = (x_idx + step_x + 1) % source_x_number + topleft_x_coord
                send_message_target_x_idx = (x_idx + step_x + 1) % source_x_number
                send_message_target_idx = send_message_target_y * topology_x_limitation + send_message_target_x
                # print(f"send_message_source_idx: {send_message_source_idx}, send_message_target_idx: {send_message_target_idx}")
                # print(whole_nodes.nodes[send_message_source_idx].coordinate)
                if (send_message_source_y, send_message_source_x) != whole_nodes.nodes[send_message_source_idx].coordinate:
                    raise ValueError(f"source node idx {send_message_source_idx} is not matched with source node coordinate {send_message_source_y, send_message_source_x}")
                if (send_message_target_y, send_message_target_x) != whole_nodes.nodes[send_message_target_idx].coordinate:
                    raise ValueError(f"target node idx {send_message_target_idx} is not matched with target node coordinate {send_message_target_y, send_message_target_x}")
                node_dependency_list[send_message_target_y_idx][send_message_target_x_idx].append(current_event_tag)
                communication_event = CommunicationEvent(current_event_tag, send_message_source_idx, send_message_target_idx, message_flits / source_x_number / nodes_number)
                communication_event.build_dependency(last_dependency_list[y_idx][x_idx])
                alltoall_done_dependency_list.append(current_event_tag)
                current_event_tag += 1
                whole_nodes.nodes[send_message_source_idx].event_queue.append(communication_event)

    # y second
    for step_y in range(source_y_number - 1):
        if source_x_number == 1:
            if step_y == 0:
                last_dependency_list = current_dependency_list
                node_dependency_list = [[[] for _ in range(source_x_number)] for _ in range(source_y_number)]
            else:
                last_dependency_list = node_dependency_list
                node_dependency_list = [[[] for _ in range(source_x_number)] for _ in range(source_y_number)]
        else:
            last_dependency_list = node_dependency_list
            node_dependency_list = [[[] for _ in range(source_x_number)] for _ in range(source_y_number)]
        for x_idx in range(source_x_number):
            send_message_source_x = topleft_x_coord + x_idx
            send_message_target_x = send_message_source_x
            send_message_target_x_idx = x_idx
            for y_idx in range(source_y_number):
                send_message_source_y = y_idx + topleft_y_coord
                send_message_source_idx = send_message_source_y * topology_x_limitation + send_message_source_x
                send_message_target_y = (y_idx + step_y + 1) % source_y_number + topleft_y_coord
                send_message_target_y_idx = (y_idx + step_y + 1) % source_y_number
                send_message_target_idx = send_message_target_y * topology_x_limitation + send_message_target_x
                if (send_message_source_y, send_message_source_x) != whole_nodes.nodes[send_message_source_idx].coordinate:
                    raise ValueError(f"source node idx {send_message_source_idx} is not matched with source node coordinate {send_message_source_y, send_message_source_x}")
                if (send_message_target_y, send_message_target_x) != whole_nodes.nodes[send_message_target_idx].coordinate:
                    raise ValueError(f"target node idx {send_message_target_idx} is not matched with target node coordinate {send_message_target_y, send_message_target_x}")
                node_dependency_list[send_message_target_y_idx][send_message_target_x_idx].append(current_event_tag)
                alltoall_done_dependency_list.append(current_event_tag)
                communication_event = CommunicationEvent(current_event_tag, send_message_source_idx, send_message_target_idx, message_flits / source_y_number / nodes_number)
                communication_event.build_dependency(last_dependency_list[y_idx][x_idx])
                current_event_tag += 1
                whole_nodes.nodes[send_message_source_idx].event_queue.append(communication_event)

    pass_dependency_list = [[alltoall_done_dependency_list for _ in range(source_x_number)] for _ in range(source_y_number)]

    return current_event_tag, pass_dependency_list


def alltoall_torus3d_hierarchicalring(
    whole_nodes: NodeNetwork, current_event_tag: int, current_dependency_list: List[List[List[int]]],
    source_nodes_coordinates_list: List[Tuple[int]],
    source_x_number: int, source_y_number: int, source_z_number: int,
    topology_x_limitation: int, topology_y_limitation: int, topology_z_limitation: int,
    message_flits: int,
    reduction: float,
    latency=None, bandwidth=None
) -> int:
    
    nodes_number = source_x_number * source_y_number * source_z_number
    initial_event_tag = current_event_tag

    alltoall_done_dependency_list = []

    topleft_node = source_nodes_coordinates_list[0]
    topleft_z_coord = topleft_node[0]
    topleft_y_coord = topleft_node[1]
    topleft_x_coord = topleft_node[2]

    for step_x in range(source_x_number - 1):
        if step_x == 0:
            last_dependency_list = current_dependency_list
            node_dependency_list = [[[[] for _ in range(source_x_number)] for _ in range(source_y_number)] for _ in range(source_z_number)]
        else:
            last_dependency_list = node_dependency_list
            node_dependency_list = [[[[] for _ in range(source_x_number)] for _ in range(source_y_number)] for _ in range(source_z_number)]
        for z_idx in range(source_z_number):
            send_message_source_z = topleft_z_coord + z_idx
            for y_idx in range(source_y_number):
                send_message_source_y = topleft_y_coord + y_idx
                for x_idx in range(source_x_number):
                    send_message_source_x = topleft_x_coord + x_idx
                    send_message_target_x = (x_idx + step_x + 1) % source_x_number + topleft_x_coord
                    send_message_target_x_idx = (x_idx + step_x + 1) % source_x_number
                    send_message_target_y = send_message_source_y
                    send_message_target_z = send_message_source_z
                    send_message_source_idx = send_message_source_z * (topology_y_limitation * topology_x_limitation) + send_message_source_y * topology_x_limitation + send_message_source_x
                    send_message_target_idx = send_message_target_z * (topology_y_limitation * topology_x_limitation) + send_message_target_y * topology_x_limitation + send_message_target_x
                    if (send_message_source_z, send_message_source_y, send_message_source_x) != whole_nodes.nodes[send_message_source_idx].coordinate:
                        raise ValueError(f"source node idx {send_message_source_idx} is not matched with source node coordinate {send_message_source_z, send_message_source_y, send_message_source_x}")
                    if (send_message_target_z, send_message_target_y, send_message_target_x) != whole_nodes.nodes[send_message_target_idx].coordinate:
                        raise ValueError(f"target node idx {send_message_target_idx} is not matched with target node coordinate {send_message_target_z, send_message_target_y, send_message_target_x}")
                    node_dependency_list[z_idx][y_idx][send_message_target_x_idx].append(current_event_tag)
                    communication_event = CommunicationEvent(current_event_tag, send_message_source_idx, send_message_target_idx, message_flits / source_x_number / nodes_number)
                    communication_event.build_dependency(last_dependency_list[z_idx][y_idx][x_idx])
                    alltoall_done_dependency_list.append(current_event_tag)
                    current_event_tag += 1
                    whole_nodes.nodes[send_message_source_idx].event_queue.append(communication_event)

    for step_y in range(source_y_number - 1):
        if source_x_number == 1:
            if step_y == 0:
                last_dependency_list = current_dependency_list
                node_dependency_list = [[[[] for _ in range(source_x_number)] for _ in range(source_y_number)] for _ in range(source_z_number)]
            else:
                last_dependency_list = node_dependency_list
                node_dependency_list = [[[[] for _ in range(source_x_number)] for _ in range(source_y_number)] for _ in range(source_z_number)]
        else:
            last_dependency_list = node_dependency_list
            node_dependency_list = [[[[] for _ in range(source_x_number)] for _ in range(source_y_number)] for _ in range(source_z_number)]
        for z_idx in range(source_z_number):
            for x_idx in range(source_x_number):
                for y_idx in range(source_y_number):
                    send_message_source_z = topleft_z_coord + z_idx
                    send_message_source_y = topleft_y_coord + y_idx
                    send_message_source_x = topleft_x_coord + x_idx
                    send_message_target_y = (y_idx + step_y + 1) % source_y_number + topleft_y_coord
                    send_message_target_y_idx = (y_idx + step_y + 1) % source_y_number
                    send_message_target_z = send_message_source_z
                    send_message_target_x = send_message_source_x
                    send_message_source_idx = send_message_source_z * (topology_y_limitation * topology_x_limitation) + send_message_source_y * topology_x_limitation + send_message_source_x
                    send_message_target_idx = send_message_target_z * (topology_y_limitation * topology_x_limitation) + send_message_target_y * topology_x_limitation + send_message_target_x
                    if (send_message_source_z, send_message_source_y, send_message_source_x) != whole_nodes.nodes[send_message_source_idx].coordinate:
                        raise ValueError(f"source node idx {send_message_source_idx} is not matched with source node coordinate {send_message_source_z, send_message_source_y, send_message_source_x}")
                    if (send_message_target_z, send_message_target_y, send_message_target_x) != whole_nodes.nodes[send_message_target_idx].coordinate:
                        raise ValueError(f"target node idx {send_message_target_idx} is not matched with target node coordinate {send_message_target_z, send_message_target_y, send_message_target_x}")
                    node_dependency_list[z_idx][send_message_target_y_idx][x_idx].append(current_event_tag)
                    communication_event = CommunicationEvent(current_event_tag, send_message_source_idx, send_message_target_idx, message_flits / source_y_number / nodes_number)
                    communication_event.build_dependency(last_dependency_list[z_idx][y_idx][x_idx])
                    alltoall_done_dependency_list.append(current_event_tag)
                    current_event_tag += 1
                    whole_nodes.nodes[send_message_source_idx].event_queue.append(communication_event)

    for step_z in range(source_z_number - 1):
        if source_x_number == 1 and source_y_number == 1:
            if step_z == 0:
                last_dependency_list = current_dependency_list
                node_dependency_list = [[[[] for _ in range(source_x_number)] for _ in range(source_y_number)] for _ in range(source_z_number)]
            else:
                last_dependency_list = node_dependency_list
                node_dependency_list = [[[[] for _ in range(source_x_number)] for _ in range(source_y_number)] for _ in range(source_z_number)]
        else:
            last_dependency_list = node_dependency_list
            node_dependency_list = [[[[] for _ in range(source_x_number)] for _ in range(source_y_number)] for _ in range(source_z_number)]
        for y_idx in range(source_y_number):
            for x_idx in range(source_x_number):
                for z_idx in range(source_z_number):
                    send_message_source_z = topleft_z_coord + z_idx
                    send_message_source_y = topleft_y_coord + y_idx
                    send_message_source_x = topleft_x_coord + x_idx
                    send_message_target_z = (z_idx + step_z + 1) % source_z_number + topleft_z_coord
                    send_message_target_z_idx = (z_idx + step_z + 1) % source_z_number
                    send_message_target_y = send_message_source_y
                    send_message_target_x = send_message_source_x
                    send_message_source_idx = send_message_source_z * (topology_y_limitation * topology_x_limitation) + send_message_source_y * topology_x_limitation + send_message_source_x
                    send_message_target_idx = send_message_target_z * (topology_y_limitation * topology_x_limitation) + send_message_target_y * topology_x_limitation + send_message_target_x
                    if (send_message_source_z, send_message_source_y, send_message_source_x) != whole_nodes.nodes[send_message_source_idx].coordinate:
                        raise ValueError(f"source node idx {send_message_source_idx} is not matched with source node coordinate {send_message_source_z, send_message_source_y, send_message_source_x}")
                    if (send_message_target_z, send_message_target_y, send_message_target_x) != whole_nodes.nodes[send_message_target_idx].coordinate:
                        raise ValueError(f"target node idx {send_message_target_idx} is not matched with target node coordinate {send_message_target_z, send_message_target_y, send_message_target_x}")
                    node_dependency_list[send_message_target_z_idx][y_idx][x_idx].append(current_event_tag)
                    communication_event = CommunicationEvent(current_event_tag, send_message_source_idx, send_message_target_idx, message_flits / source_z_number / nodes_number)
                    communication_event.build_dependency(last_dependency_list[z_idx][y_idx][x_idx])
                    alltoall_done_dependency_list.append(current_event_tag)
                    current_event_tag += 1
                    whole_nodes.nodes[send_message_source_idx].event_queue.append(communication_event)

    pass_dependency_list = [[[alltoall_done_dependency_list for _ in range(source_x_number)] for _ in range(source_y_number)] for _ in range(source_z_number)]

    return current_event_tag, pass_dependency_list    


def alltoall_dgx2_havlingdoubling(
    whole_nodes: NodeNetwork_2D, current_event_tag: int, current_dependency_list: List[List[List[int]]], 
    source_nodes_coordinates_list: List[Tuple[int]],
    source_x_number: int, source_y_number: int,
    topology_x_limitation: int, topology_y_limitation: int, 
    message_flits: int, 
    latency=None, bandwidth=None, reduction=None    
):
    
    # x is intra-dimension
    # y is inter-dimension
    nodes_number = source_x_number * source_y_number
    initial_event_tag = current_event_tag

    allgather_done_dependency_list = []

    topleft_node = source_nodes_coordinates_list[0]
    topleft_x_coord = topleft_node[1]
    topleft_y_coord = topleft_node[0]

    judge_p = int(np.log2(nodes_number))
    p = int(np.floor(np.log2(nodes_number)))

    if judge_p == p:
        # lg(n) steps
        for step_idx in range(p):
            if step_idx == 0:
                last_dependency_list = current_dependency_list
                node_dependency_list = [[[] for _ in range(source_x_number)] for _ in range(source_y_number)]
            else:
                last_dependency_list = node_dependency_list
                node_dependency_list = [[[] for _ in range(source_x_number)] for _ in range(source_y_number)]
            # each process in n nodes communicate with 2^i distance node with communication (2^i)*(1/n)
            distance = 2 ** step_idx
            size_ratio = 0.5
            iter_list = list(range(nodes_number))
            for card_idx in range(nodes_number//2):
                source_node_idx = iter_list[0]
                source_node_coordinate = source_nodes_coordinates_list[source_node_idx]
                send_message_source_idx = source_node_coordinate[0] * topology_x_limitation + source_node_coordinate[1]
                target_node_idx = (source_node_idx + distance) % nodes_number
                target_node_coordinate = source_nodes_coordinates_list[target_node_idx]
                send_message_target_idx = target_node_coordinate[0] * topology_x_limitation + target_node_coordinate[1]
                if source_node_coordinate != whole_nodes.nodes[send_message_source_idx].coordinate:
                    raise ValueError(f"source node idx {send_message_source_idx} is not matched with source node coordinate {source_node_coordinate}")
                if target_node_coordinate != whole_nodes.nodes[send_message_target_idx].coordinate:
                    raise ValueError(f"target node idx {send_message_target_idx} is not matched with target node coordinate {target_node_coordinate}")

                node_dependency_list[target_node_coordinate[0]-topleft_y_coord][target_node_coordinate[1]-topleft_x_coord].append(current_event_tag)
                communication_event = CommunicationEvent(current_event_tag, send_message_source_idx, send_message_target_idx, message_flits * size_ratio / nodes_number)
                communication_event.build_dependency(last_dependency_list[source_node_coordinate[0]-topleft_y_coord][source_node_coordinate[1]-topleft_x_coord])
                current_event_tag += 1
                whole_nodes.nodes[send_message_source_idx].event_queue.append(communication_event)

                node_dependency_list[source_node_coordinate[0]-topleft_y_coord][source_node_coordinate[1]-topleft_x_coord].append(current_event_tag)
                communication_event = CommunicationEvent(current_event_tag, send_message_target_idx, send_message_source_idx, message_flits * size_ratio / nodes_number)
                communication_event.build_dependency(last_dependency_list[target_node_coordinate[0]-topleft_y_coord][target_node_coordinate[1]-topleft_x_coord])
                current_event_tag += 1
                whole_nodes.nodes[send_message_target_idx].event_queue.append(communication_event)

                iter_list.remove(source_node_idx)
                iter_list.remove(target_node_idx)

    else:
        # TODO: implement the non-power of 2 nodes number
        raise ValueError(f"nodes_number {nodes_number} is not a power of 2")

    return current_event_tag, node_dependency_list


def ordertoorder_mesh2d_hierarchicalring(
    whole_nodes: NodeNetwork, current_event_tag: int, current_dependency_list: List[List[List[int]]],
    source_nodes_coordinates_list: List[Tuple[int]],
    source_x_number: int, source_y_number: int,
    topology_x_limitation: int, topology_y_limitation: int, 
    data_parallelism_degree: List[int], 
    message_flits: int, 
    reduction: float,
    latency=None, bandwidth=None
) -> int:

    nodes_number = source_x_number * source_y_number
    initial_event_tag = current_event_tag

    ordertoorder_done_dependency_list = []

    '''
    all nodes lists should be ordered by idx  
    '''

    group_x_number = source_x_number // data_parallelism_degree[0]
    group_y_number = source_y_number // data_parallelism_degree[1]
    group_nodes_number = group_x_number * group_y_number
    ordertoorder_groups_list = []
    for group_y_idx in range(data_parallelism_degree[1]):
        group_y_nodes_list = []
        for group_x_idx in range(data_parallelism_degree[0]):
            group_x_nodes_list = []
            for node_y_idx in range(group_y_number):
                for node_x_idx in range(group_x_number):
                    group_node_x_idx = group_x_idx * group_x_number + node_x_idx
                    group_node_y_idx = group_y_idx * group_y_number + node_y_idx
                    group_node_idx = group_node_y_idx * source_x_number + group_node_x_idx
                    group_x_nodes_list.append(source_nodes_coordinates_list[group_node_idx])
            group_y_nodes_list.append(group_x_nodes_list)
        ordertoorder_groups_list.append(group_y_nodes_list)

    # separate group
    for group_y_idx in range(data_parallelism_degree[1]):
        for group_x_idx in range(data_parallelism_degree[0]):

            # group logic
            topleft_node = ordertoorder_groups_list[group_y_idx][group_x_idx][0]
            topleft_x_coord = topleft_node[1]
            topleft_y_coord = topleft_node[0]
            # x first
            for step_x in range(group_x_number - 1):
                if step_x == 0:
                    last_dependency_list = current_dependency_list
                    node_dependency_list = [[[] for _ in range(source_x_number)] for _ in range(source_y_number)]
                else:
                    last_dependency_list = node_dependency_list
                    node_dependency_list = [[[] for _ in range(source_x_number)] for _ in range(source_y_number)]
                for y_idx in range(group_y_number):
                    send_message_source_y = topleft_y_coord + y_idx
                    send_message_target_y = send_message_source_y
                    send_message_target_y_idx = y_idx
                    for x_idx in range(group_x_number):
                        send_message_source_x = x_idx + topleft_x_coord
                        send_message_source_idx = send_message_source_y * topology_x_limitation + send_message_source_x
                        send_message_target_x = (x_idx + step_x + 1) % group_x_number + topleft_x_coord
                        send_message_target_x_idx = (x_idx + step_x + 1) % group_x_number
                        send_message_target_idx = send_message_target_y * topology_x_limitation + send_message_target_x
                        # print(f"send_message_source_idx: {send_message_source_idx}, send_message_target_idx: {send_message_target_idx}")
                        # print(whole_nodes.nodes[send_message_source_idx].coordinate)
                        if (send_message_source_y, send_message_source_x) != whole_nodes.nodes[send_message_source_idx].coordinate:
                            raise ValueError(f"source node idx {send_message_source_idx} is not matched with source node coordinate {send_message_source_y, send_message_source_x}")
                        if (send_message_target_y, send_message_target_x) != whole_nodes.nodes[send_message_target_idx].coordinate:
                            raise ValueError(f"target node idx {send_message_target_idx} is not matched with target node coordinate {send_message_target_y, send_message_target_x}")
                        node_dependency_list[send_message_target_y_idx][send_message_target_x_idx].append(current_event_tag)
                        communication_event = CommunicationEvent(current_event_tag, send_message_source_idx, send_message_target_idx, message_flits / group_x_number / group_nodes_number)
                        communication_event.build_dependency(last_dependency_list[send_message_source_y-topleft_y_coord][send_message_source_x-topleft_x_coord])
                        ordertoorder_done_dependency_list.append(current_event_tag)
                        current_event_tag += 1
                        whole_nodes.nodes[send_message_source_idx].event_queue.append(communication_event)

            # y second
            for step_y in range(group_y_number - 1):
                if group_x_number == 1:
                    if step_y == 0:
                        last_dependency_list = current_dependency_list
                        node_dependency_list = [[[] for _ in range(source_x_number)] for _ in range(source_y_number)]
                    else:
                        last_dependency_list = node_dependency_list
                        node_dependency_list = [[[] for _ in range(source_x_number)] for _ in range(source_y_number)]
                else:
                    last_dependency_list = node_dependency_list
                    node_dependency_list = [[[] for _ in range(source_x_number)] for _ in range(source_y_number)]
                for x_idx in range(group_x_number):
                    send_message_source_x = topleft_x_coord + x_idx
                    send_message_target_x = send_message_source_x
                    send_message_target_x_idx = x_idx
                    for y_idx in range(group_y_number):
                        send_message_source_y = y_idx + topleft_y_coord
                        send_message_source_idx = send_message_source_y * topology_x_limitation + send_message_source_x
                        send_message_target_y = (y_idx + step_y + 1) % group_y_number + topleft_y_coord
                        send_message_target_y_idx = (y_idx + step_y + 1) % group_y_number
                        send_message_target_idx = send_message_target_y * topology_x_limitation + send_message_target_x
                        if (send_message_source_y, send_message_source_x) != whole_nodes.nodes[send_message_source_idx].coordinate:
                            raise ValueError(f"source node idx {send_message_source_idx} is not matched with source node coordinate {send_message_source_y, send_message_source_x}")
                        if (send_message_target_y, send_message_target_x) != whole_nodes.nodes[send_message_target_idx].coordinate:
                            raise ValueError(f"target node idx {send_message_target_idx} is not matched with target node coordinate {send_message_target_y, send_message_target_x}")
                        node_dependency_list[send_message_target_y_idx][send_message_target_x_idx].append(current_event_tag)
                        communication_event = CommunicationEvent(current_event_tag, send_message_source_idx, send_message_target_idx, message_flits / group_y_number / group_nodes_number)
                        communication_event.build_dependency(last_dependency_list[send_message_source_y-topleft_y_coord][send_message_source_x-topleft_x_coord])
                        ordertoorder_done_dependency_list.append(current_event_tag)
                        current_event_tag += 1
                        whole_nodes.nodes[send_message_source_idx].event_queue.append(communication_event)

    pass_dependency_list = [[ordertoorder_done_dependency_list for _ in range(source_x_number)] for _ in range(source_y_number)]

    return current_event_tag, pass_dependency_list


def ordertoorder_torus2d_hierarchicalring(
    whole_nodes: NodeNetwork, current_event_tag: int, current_dependency_list: List[List[List[int]]],
    source_nodes_coordinates_list: List[Tuple[int]],
    source_x_number: int, source_y_number: int,
    topology_x_limitation: int, topology_y_limitation: int, 
    data_parallelism_degree: List[int], 
    message_flits: int, 
    reduction: float,
    latency=None, bandwidth=None
) -> int:

    nodes_number = source_x_number * source_y_number
    initial_event_tag = current_event_tag

    ordertoorder_done_dependency_list = []

    '''
    all nodes lists should be ordered by idx  
    '''

    group_x_number = source_x_number // data_parallelism_degree[0]
    group_y_number = source_y_number // data_parallelism_degree[1]
    group_nodes_number = group_x_number * group_y_number
    ordertoorder_groups_list = []
    for group_y_idx in range(data_parallelism_degree[1]):
        group_y_nodes_list = []
        for group_x_idx in range(data_parallelism_degree[0]):
            group_x_nodes_list = []
            for node_y_idx in range(group_y_number):
                for node_x_idx in range(group_x_number):
                    group_node_x_idx = group_x_idx * group_x_number + node_x_idx
                    group_node_y_idx = group_y_idx * group_y_number + node_y_idx
                    group_node_idx = group_node_y_idx * source_x_number + group_node_x_idx
                    group_x_nodes_list.append(source_nodes_coordinates_list[group_node_idx])
            group_y_nodes_list.append(group_x_nodes_list)
        ordertoorder_groups_list.append(group_y_nodes_list)

    # separate group
    for group_y_idx in range(data_parallelism_degree[1]):
        for group_x_idx in range(data_parallelism_degree[0]):

            # group logic
            topleft_node = ordertoorder_groups_list[group_y_idx][group_x_idx][0]
            topleft_x_coord = topleft_node[1]
            topleft_y_coord = topleft_node[0]
            # x first
            for step_x in range(group_x_number - 1):
                if step_x == 0:
                    last_dependency_list = current_dependency_list
                    node_dependency_list = [[[] for _ in range(source_x_number)] for _ in range(source_y_number)]
                else:
                    last_dependency_list = node_dependency_list
                    node_dependency_list = [[[] for _ in range(source_x_number)] for _ in range(source_y_number)]
                for y_idx in range(group_y_number):
                    send_message_source_y = topleft_y_coord + y_idx
                    send_message_target_y = send_message_source_y
                    send_message_target_y_idx = y_idx
                    for x_idx in range(group_x_number):
                        send_message_source_x = x_idx + topleft_x_coord
                        send_message_source_idx = send_message_source_y * topology_x_limitation + send_message_source_x
                        send_message_target_x = (x_idx + step_x + 1) % group_x_number + topleft_x_coord
                        send_message_target_x_idx = (x_idx + step_x + 1) % group_x_number
                        send_message_target_idx = send_message_target_y * topology_x_limitation + send_message_target_x
                        # print(f"send_message_source_idx: {send_message_source_idx}, send_message_target_idx: {send_message_target_idx}")
                        # print(whole_nodes.nodes[send_message_source_idx].coordinate)
                        if (send_message_source_y, send_message_source_x) != whole_nodes.nodes[send_message_source_idx].coordinate:
                            raise ValueError(f"source node idx {send_message_source_idx} is not matched with source node coordinate {send_message_source_y, send_message_source_x}")
                        if (send_message_target_y, send_message_target_x) != whole_nodes.nodes[send_message_target_idx].coordinate:
                            raise ValueError(f"target node idx {send_message_target_idx} is not matched with target node coordinate {send_message_target_y, send_message_target_x}")
                        node_dependency_list[send_message_target_y_idx][send_message_target_x_idx].append(current_event_tag)
                        communication_event = CommunicationEvent(current_event_tag, send_message_source_idx, send_message_target_idx, message_flits / group_x_number / group_nodes_number)
                        communication_event.build_dependency(last_dependency_list[send_message_source_y-topleft_y_coord][send_message_source_x-topleft_x_coord])
                        ordertoorder_done_dependency_list.append(current_event_tag)
                        current_event_tag += 1
                        whole_nodes.nodes[send_message_source_idx].event_queue.append(communication_event)

            # y second
            for step_y in range(group_y_number - 1):
                if group_x_number == 1:
                    if step_y == 0:
                        last_dependency_list = current_dependency_list
                        node_dependency_list = [[[] for _ in range(source_x_number)] for _ in range(source_y_number)]
                    else:
                        last_dependency_list = node_dependency_list
                        node_dependency_list = [[[] for _ in range(source_x_number)] for _ in range(source_y_number)]
                else:
                    last_dependency_list = node_dependency_list
                    node_dependency_list = [[[] for _ in range(source_x_number)] for _ in range(source_y_number)]
                for x_idx in range(group_x_number):
                    send_message_source_x = topleft_x_coord + x_idx
                    send_message_target_x = send_message_source_x
                    send_message_target_x_idx = x_idx
                    for y_idx in range(group_y_number):
                        send_message_source_y = y_idx + topleft_y_coord
                        send_message_source_idx = send_message_source_y * topology_x_limitation + send_message_source_x
                        send_message_target_y = (y_idx + step_y + 1) % group_y_number + topleft_y_coord
                        send_message_target_y_idx = (y_idx + step_y + 1) % group_y_number
                        send_message_target_idx = send_message_target_y * topology_x_limitation + send_message_target_x
                        if (send_message_source_y, send_message_source_x) != whole_nodes.nodes[send_message_source_idx].coordinate:
                            raise ValueError(f"source node idx {send_message_source_idx} is not matched with source node coordinate {send_message_source_y, send_message_source_x}")
                        if (send_message_target_y, send_message_target_x) != whole_nodes.nodes[send_message_target_idx].coordinate:
                            raise ValueError(f"target node idx {send_message_target_idx} is not matched with target node coordinate {send_message_target_y, send_message_target_x}")
                        node_dependency_list[send_message_target_y_idx][send_message_target_x_idx].append(current_event_tag)
                        communication_event = CommunicationEvent(current_event_tag, send_message_source_idx, send_message_target_idx, message_flits / group_y_number / group_nodes_number)
                        communication_event.build_dependency(last_dependency_list[send_message_source_y-topleft_y_coord][send_message_source_x-topleft_x_coord])
                        ordertoorder_done_dependency_list.append(current_event_tag)
                        current_event_tag += 1
                        whole_nodes.nodes[send_message_source_idx].event_queue.append(communication_event)

    pass_dependency_list = [[ordertoorder_done_dependency_list for _ in range(source_x_number)] for _ in range(source_y_number)]

    return current_event_tag, pass_dependency_list


def ordertoorder_torus3d_hierarchicalring(
    whole_nodes: NodeNetwork, current_event_tag: int, current_dependency_list: List[List[List[int]]],
    source_nodes_coordinates_list: List[Tuple[int]],
    source_x_number: int, source_y_number: int, source_z_number: int,
    topology_x_limitation: int, topology_y_limitation: int, topology_z_limitation: int,
    data_parallelism_degree: List[int], 
    message_flits: int, 
    reduction: float,
    latency=None, bandwidth=None
) -> int:
    
    nodes_number = source_x_number * source_y_number * source_z_number
    initial_event_tag = current_event_tag

    ordertoorder_done_dependency_list = []

    group_x_number = source_x_number // data_parallelism_degree[0]
    group_y_number = source_y_number // data_parallelism_degree[1]
    group_z_number = source_z_number // data_parallelism_degree[2]
    group_nodes_number = group_x_number * group_y_number * group_z_number

    ordertoorder_groups_list = []
    for group_z_idx in range(data_parallelism_degree[2]):
        group_z_nodes_list = []
        for group_y_idx in range(data_parallelism_degree[1]):
            group_y_nodes_list = []
            for group_x_idx in range(data_parallelism_degree[0]):
                group_x_nodes_list = []
                for node_z_idx in range(group_z_number):
                    for node_y_idx in range(group_y_number):
                        for node_x_idx in range(group_x_number):
                            group_node_x_idx = group_x_idx * group_x_number + node_x_idx
                            group_node_y_idx = group_y_idx * group_y_number + node_y_idx
                            group_node_z_idx = group_z_idx * group_z_number + node_z_idx
                            group_node_idx = group_node_z_idx * (source_x_number * source_y_number) \
                                             + group_node_y_idx * source_x_number \
                                             + group_node_x_idx
                            group_x_nodes_list.append(source_nodes_coordinates_list[group_node_idx])
                group_y_nodes_list.append(group_x_nodes_list)
            group_z_nodes_list.append(group_y_nodes_list)
        ordertoorder_groups_list.append(group_z_nodes_list)

    for group_z_idx in range(data_parallelism_degree[2]):
        for group_y_idx in range(data_parallelism_degree[1]):
            for group_x_idx in range(data_parallelism_degree[0]):
                group_nodes_list = ordertoorder_groups_list[group_z_idx][group_y_idx][group_x_idx]
                topleft_z_coord, topleft_y_coord, topleft_x_coord = group_nodes_list[0]

                for step_x in range(group_x_number - 1):
                    if step_x == 0:
                        last_dependency_list = current_dependency_list
                        node_dependency_list = [
                            [
                                [[] for _ in range(source_x_number)]
                                for _ in range(source_y_number)
                            ]
                            for _ in range(source_z_number)
                        ]
                    else:
                        last_dependency_list = node_dependency_list
                        node_dependency_list = [
                            [
                                [[] for _ in range(source_x_number)]
                                for _ in range(source_y_number)
                            ]
                            for _ in range(source_z_number)
                        ]
                    for z_idx in range(group_z_number):
                        send_message_source_z = topleft_z_coord + z_idx
                        for y_idx in range(group_y_number):
                            send_message_source_y = topleft_y_coord + y_idx
                            for x_idx in range(group_x_number):
                                send_message_source_x = topleft_x_coord + x_idx
                                send_message_target_x = ((x_idx + step_x + 1) % group_x_number) + topleft_x_coord
                                send_message_target_z = send_message_source_z
                                send_message_target_y = send_message_source_y
                                send_message_source_idx = (
                                    send_message_source_z * (topology_y_limitation * topology_x_limitation)
                                    + send_message_source_y * topology_x_limitation
                                    + send_message_source_x
                                )
                                send_message_target_idx = (
                                    send_message_target_z * (topology_y_limitation * topology_x_limitation)
                                    + send_message_target_y * topology_x_limitation
                                    + send_message_target_x
                                )
                                if (send_message_source_z, send_message_source_y, send_message_source_x) != whole_nodes.nodes[send_message_source_idx].coordinate:
                                    raise ValueError(f"source node idx {send_message_source_idx} is not matched with source node coordinate {send_message_source_z, send_message_source_y, send_message_source_x}")
                                if (send_message_target_z, send_message_target_y, send_message_target_x) != whole_nodes.nodes[send_message_target_idx].coordinate:
                                    raise ValueError(f"target node idx {send_message_target_idx} is not matched with target node coordinate {send_message_target_z, send_message_target_y, send_message_target_x}")
                                node_dependency_list[send_message_source_z - topleft_z_coord][send_message_source_y - topleft_y_coord][send_message_target_x - topleft_x_coord].append(current_event_tag)
                                communication_event = CommunicationEvent(
                                    current_event_tag,
                                    send_message_source_idx,
                                    send_message_target_idx,
                                    message_flits / group_x_number / group_nodes_number # x-ring usage
                                )
                                communication_event.build_dependency(
                                    last_dependency_list[send_message_source_z - topleft_z_coord][send_message_source_y - topleft_y_coord][send_message_source_x - topleft_x_coord]
                                )
                                ordertoorder_done_dependency_list.append(current_event_tag)
                                current_event_tag += 1
                                whole_nodes.nodes[send_message_source_idx].event_queue.append(communication_event)

                for step_y in range(group_y_number - 1):
                    if group_x_number == 1 and group_z_number == 1:
                        if step_y == 0:
                            last_dependency_list = current_dependency_list
                            node_dependency_list = [
                                [
                                    [[] for _ in range(source_x_number)]
                                    for _ in range(source_y_number)
                                ]
                                for _ in range(source_z_number)
                            ]
                        else:
                            last_dependency_list = node_dependency_list
                            node_dependency_list = [
                                [
                                    [[] for _ in range(source_x_number)]
                                    for _ in range(source_y_number)
                                ]
                                for _ in range(source_z_number)
                            ]
                    else:
                        last_dependency_list = node_dependency_list
                        node_dependency_list = [
                            [
                                [[] for _ in range(source_x_number)]
                                for _ in range(source_y_number)
                            ]
                            for _ in range(source_z_number)
                        ]
                    for z_idx in range(group_z_number):
                        for x_idx in range(group_x_number):
                            for y_idx in range(group_y_number):
                                send_message_source_z = topleft_z_coord + z_idx
                                send_message_source_y = topleft_y_coord + y_idx
                                send_message_source_x = topleft_x_coord + x_idx
                                send_message_target_y = ((y_idx + step_y + 1) % group_y_number) + topleft_y_coord
                                send_message_target_z = send_message_source_z
                                send_message_target_x = send_message_source_x
                                send_message_source_idx = (
                                    send_message_source_z * (topology_y_limitation * topology_x_limitation)
                                    + send_message_source_y * topology_x_limitation
                                    + send_message_source_x
                                )
                                send_message_target_idx = (
                                    send_message_target_z * (topology_y_limitation * topology_x_limitation)
                                    + send_message_target_y * topology_x_limitation
                                    + send_message_target_x
                                )
                                if (send_message_source_z, send_message_source_y, send_message_source_x) != whole_nodes.nodes[send_message_source_idx].coordinate:
                                    raise ValueError(f"source node idx {send_message_source_idx} is not matched with source node coordinate {send_message_source_z, send_message_source_y, send_message_source_x}")
                                if (send_message_target_z, send_message_target_y, send_message_target_x) != whole_nodes.nodes[send_message_target_idx].coordinate:
                                    raise ValueError(f"target node idx {send_message_target_idx} is not matched with target node coordinate {send_message_target_z, send_message_target_y, send_message_target_x}")
                                node_dependency_list[send_message_source_z - topleft_z_coord][send_message_target_y - topleft_y_coord][send_message_source_x - topleft_x_coord].append(current_event_tag)
                                communication_event = CommunicationEvent(
                                    current_event_tag,
                                    send_message_source_idx,
                                    send_message_target_idx,
                                    message_flits / group_y_number / group_nodes_number # y-ring usage
                                )
                                communication_event.build_dependency(
                                    last_dependency_list[send_message_source_z - topleft_z_coord][send_message_source_y - topleft_y_coord][send_message_source_x - topleft_x_coord]
                                )
                                ordertoorder_done_dependency_list.append(current_event_tag)
                                current_event_tag += 1
                                whole_nodes.nodes[send_message_source_idx].event_queue.append(communication_event)

                for step_z in range(group_z_number - 1):
                    if group_x_number == 1 and group_y_number == 1:
                        if step_z == 0:
                            last_dependency_list = current_dependency_list
                            node_dependency_list = [
                                [
                                    [[] for _ in range(source_x_number)]
                                    for _ in range(source_y_number)
                                ]
                                for _ in range(source_z_number)
                            ]
                        else:
                            last_dependency_list = node_dependency_list
                            node_dependency_list = [
                                [
                                    [[] for _ in range(source_x_number)]
                                    for _ in range(source_y_number)
                                ]
                                for _ in range(source_z_number)
                            ]
                    else:
                        last_dependency_list = node_dependency_list
                        node_dependency_list = [
                            [
                                [[] for _ in range(source_x_number)]
                                for _ in range(source_y_number)
                            ]
                            for _ in range(source_z_number)
                        ]
                    for y_idx in range(group_y_number):
                        for x_idx in range(group_x_number):
                            for z_idx in range(group_z_number):
                                send_message_source_z = topleft_z_coord + z_idx
                                send_message_source_y = topleft_y_coord + y_idx
                                send_message_source_x = topleft_x_coord + x_idx
                                send_message_target_z = ((z_idx + step_z + 1) % group_z_number) + topleft_z_coord
                                send_message_target_y = send_message_source_y
                                send_message_target_x = send_message_source_x
                                send_message_source_idx = (
                                    send_message_source_z * (topology_y_limitation * topology_x_limitation)
                                    + send_message_source_y * topology_x_limitation
                                    + send_message_source_x
                                )
                                send_message_target_idx = (
                                    send_message_target_z * (topology_y_limitation * topology_x_limitation)
                                    + send_message_target_y * topology_x_limitation
                                    + send_message_target_x
                                )
                                if (send_message_source_z, send_message_source_y, send_message_source_x) != whole_nodes.nodes[send_message_source_idx].coordinate:
                                    raise ValueError(f"source node idx {send_message_source_idx} is not matched with source node coordinate {send_message_source_z, send_message_source_y, send_message_source_x}")
                                if (send_message_target_z, send_message_target_y, send_message_target_x) != whole_nodes.nodes[send_message_target_idx].coordinate:
                                    raise ValueError(f"target node idx {send_message_target_idx} is not matched with target node coordinate {send_message_target_z, send_message_target_y, send_message_target_x}")
                                node_dependency_list[send_message_target_z - topleft_z_coord][send_message_source_y - topleft_y_coord][send_message_source_x - topleft_x_coord].append(current_event_tag)
                                communication_event = CommunicationEvent(
                                    current_event_tag,
                                    send_message_source_idx,
                                    send_message_target_idx,
                                    message_flits / group_z_number / group_nodes_number # z-ring usage
                                )
                                communication_event.build_dependency(
                                    last_dependency_list[send_message_source_z - topleft_z_coord][send_message_source_y - topleft_y_coord][send_message_source_x - topleft_x_coord]
                                )
                                ordertoorder_done_dependency_list.append(current_event_tag)
                                current_event_tag += 1
                                whole_nodes.nodes[send_message_source_idx].event_queue.append(communication_event)

    pass_dependency_list = [
        [
            [ordertoorder_done_dependency_list for _ in range(source_x_number)]
            for _ in range(source_y_number)
        ]
        for _ in range(source_z_number)
    ]

    return current_event_tag, pass_dependency_list


def ordertoorder_dgx2_havlingdoubling(
    whole_nodes: NodeNetwork_2D, current_event_tag: int, current_dependency_list: List[List[List[int]]],
    source_nodes_coordinates_list: List[Tuple[int]],
    source_x_number: int, source_y_number: int,
    topology_x_limitation: int, topology_y_limitation: int, 
    data_parallelism_degree: List[int], 
    message_flits: int, 
    reduction: float,
    latency=None, bandwidth=None
) -> int:
    
    nodes_number = source_x_number * source_y_number
    initial_event_tag = current_event_tag

    ordertoorder_done_dependency_list = []

    '''
    all nodes lists should be ordered by idx  
    '''

    group_x_number = source_x_number // data_parallelism_degree[0]
    group_y_number = source_y_number // data_parallelism_degree[1]
    group_nodes_number = group_x_number * group_y_number
    ordertoorder_groups_list = []
    for group_y_idx in range(data_parallelism_degree[1]):
        group_y_nodes_list = []
        for group_x_idx in range(data_parallelism_degree[0]):
            group_x_nodes_list = []
            for node_y_idx in range(group_y_number):
                for node_x_idx in range(group_x_number):
                    group_node_x_idx = group_x_idx * group_x_number + node_x_idx
                    group_node_y_idx = group_y_idx * group_y_number + node_y_idx
                    group_node_idx = group_node_y_idx * source_x_number + group_node_x_idx
                    group_x_nodes_list.append(source_nodes_coordinates_list[group_node_idx])
            group_y_nodes_list.append(group_x_nodes_list)
        ordertoorder_groups_list.append(group_y_nodes_list)    

    for group_y_idx in range(data_parallelism_degree[1]):
        for group_x_idx in range(data_parallelism_degree[0]):

            topleft_node = ordertoorder_groups_list[group_y_idx][group_x_idx][0]
            topleft_x_coord = topleft_node[1]
            topleft_y_coord = topleft_node[0]

            judge_p = int(np.log2(group_x_number*group_y_number))
            p = int(np.floor(np.log2(group_x_number*group_y_number)))

            if judge_p == p:
                for step_idx in range(p):
                    if step_idx == 0:
                        last_dependency_list = current_dependency_list
                        node_dependency_list = [[[] for _ in range(source_x_number)] for _ in range(source_y_number)]
                    else:
                        last_dependency_list = node_dependency_list
                        node_dependency_list = [[[] for _ in range(source_x_number)] for _ in range(source_y_number)]
                    distance = 2 ** step_idx
                    size_ratio = 0.5
                    iter_list = list(range(group_x_number*group_y_number))
                    for card_idx in range(group_x_number*group_y_number//2):
                        source_node_idx = iter_list[0]
                        source_node_coordinate = ordertoorder_groups_list[group_y_idx][group_x_idx][source_node_idx]
                        send_message_source_idx = source_node_coordinate[0] * topology_x_limitation + source_node_coordinate[1]
                        target_node_idx = (source_node_idx + distance) % (group_x_number*group_y_number)
                        target_node_coordinate = ordertoorder_groups_list[group_y_idx][group_x_idx][target_node_idx]
                        send_message_target_idx = target_node_coordinate[0] * topology_x_limitation + target_node_coordinate[1]
                        if source_node_coordinate != whole_nodes.nodes[send_message_source_idx].coordinate:
                            raise ValueError(f"source node idx {send_message_source_idx} is not matched with source node coordinate {source_node_coordinate}")
                        if target_node_coordinate != whole_nodes.nodes[send_message_target_idx].coordinate:
                            raise ValueError(f"target node idx {send_message_target_idx} is not matched with target node coordinate {target_node_coordinate}")

                        node_dependency_list[target_node_coordinate[0]-topleft_y_coord][target_node_coordinate[1]-topleft_x_coord].append(current_event_tag)
                        communication_event = CommunicationEvent(current_event_tag, send_message_source_idx, send_message_target_idx, message_flits * size_ratio / group_nodes_number)
                        communication_event.build_dependency(last_dependency_list[source_node_coordinate[0]-topleft_y_coord][source_node_coordinate[1]-topleft_x_coord])
                        ordertoorder_done_dependency_list.append(current_event_tag)
                        current_event_tag += 1
                        whole_nodes.nodes[send_message_source_idx].event_queue.append(communication_event)

                        node_dependency_list[source_node_coordinate[0]-topleft_y_coord][source_node_coordinate[1]-topleft_x_coord].append(current_event_tag)
                        communication_event = CommunicationEvent(current_event_tag, send_message_target_idx, send_message_source_idx, message_flits * size_ratio / group_nodes_number)
                        communication_event.build_dependency(last_dependency_list[target_node_coordinate[0]-topleft_y_coord][target_node_coordinate[1]-topleft_x_coord])
                        ordertoorder_done_dependency_list.append(current_event_tag)
                        current_event_tag += 1
                        whole_nodes.nodes[send_message_target_idx].event_queue.append(communication_event)

                        iter_list.remove(source_node_idx)
                        iter_list.remove(target_node_idx)

    pass_dependency_list = [[ordertoorder_done_dependency_list for _ in range(source_x_number)] for _ in range(source_y_number)]

    return current_event_tag, pass_dependency_list




class alltoall(object):

    def __init__(self, topology: str, algorithm: str):
        communication_name = self.__class__.__name__
        collective_selected = communication_name + "_" + topology + "_" + algorithm
        self.cal_time = globals()[collective_selected]


class ordertoorder(object):

    def __init__(self, topology: str, algorithm: str):
        communication_name = self.__class__.__name__
        collective_selected = communication_name + "_" + topology + "_" + algorithm
        self.cal_time = globals()[collective_selected]



if __name__ == "__main__":

    node_k = 4
    node_n = 3
    ni_k = node_k
    ni_n = node_n
    cfg_topology = "torus"
    cfg_filepath = os.path.join(file_path, '../backend/booksim2/runfiles/mesh_o_torus_py.cfg')

    # TODO: this parameter can be modified 
    flit_size = 128
    flit_capacity = 2 * flit_size
    modify_topology_cfg_file(cfg_filepath, cfg_topology, node_k, node_n)

    # test_topology = cfg_topology + str(node_n) + "d"
    # test_source_nodes = [(0, 0), (0, 1), (1, 0), (1, 1)]
    # test_source_shape = [2, 2]
    # test_whole_flits = 1024

    test_topology = cfg_topology + str(node_n) + "d"
    test_source_nodes = [(0, 0, 0), (0, 0, 1), (0, 0, 2), (0, 0, 3), (0, 1, 0), (0, 1, 1), (0, 1, 2), (0, 1, 3), (0, 2, 0), (0, 2, 1), (0, 2, 2), (0, 2, 3), (0, 3, 0), (0, 3, 1), (0, 3, 2), (0, 3, 3), (1, 0, 0), (1, 0, 1), (1, 0, 2), (1, 0, 3), (1, 1, 0), (1, 1, 1), (1, 1, 2), (1, 1, 3), (1, 2, 0), (1, 2, 1), (1, 2, 2), (1, 2, 3), (1, 3, 0), (1, 3, 1), (1, 3, 2), (1, 3, 3), (2, 0, 0), (2, 0, 1), (2, 0, 2), (2, 0, 3), (2, 1, 0), (2, 1, 1), (2, 1, 2), (2, 1, 3), (2, 2, 0), (2, 2, 1), (2, 2, 2), (2, 2, 3), (2, 3, 0), (2, 3, 1), (2, 3, 2), (2, 3, 3), (3, 0, 0), (3, 0, 1), (3, 0, 2), (3, 0, 3), (3, 1, 0), (3, 1, 1), (3, 1, 2), (3, 1, 3), (3, 2, 0), (3, 2, 1), (3, 2, 2), (3, 2, 3), (3, 3, 0), (3, 3, 1), (3, 3, 2), (3, 3, 3)]
    test_source_shape = [4, 4, 4]
    # test_data_parallelism_degree = [1, 1]
    test_data_parallelism_degree = [2, 2, 2]
    # test_whole_flits = 4096
    test_whole_flits = 294912

    current_event_tag = 0
    initial_dependency_list = [[[[] for _ in range(test_source_shape[2])] for _ in range(test_source_shape[1])] for _ in range(test_source_shape[0])]

    test_communication_algorithm = "hierarchicalring"

    node_network = NodeNetwork(node_k, node_n, ni_k, ni_n)
    node_network.create_nodes()

    communication_scheduler = ordertoorder(test_topology, test_communication_algorithm)

    now_event_tag, _ = communication_scheduler.cal_time(
        whole_nodes=node_network, current_event_tag=current_event_tag, current_dependency_list=initial_dependency_list,
        source_nodes_coordinates_list=test_source_nodes,
        source_x_number=test_source_shape[0], source_y_number=test_source_shape[1], source_z_number=test_source_shape[2],
        topology_x_limitation=node_k, topology_y_limitation=node_k, topology_z_limitation=node_k,
        data_parallelism_degree=test_data_parallelism_degree,
        message_flits=test_whole_flits, 
        latency=None, bandwidth=None, reduction=None
    )
    print(f"current_event_tag: {now_event_tag}")
    # node_network.show_nodes_events()
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

