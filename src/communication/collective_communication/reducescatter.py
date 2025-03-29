# File name  :    reducescatter.py
# Author     :    xiaocuicui
# Time       :    2024/06/30 14:48:07
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


def reducelocal_mesh2d_base(
    whole_nodes: NodeNetwork, current_event_tag: int, current_dependency_list: List[List[List[int]]], 
    source_nodes_coordinates_list: List[Tuple[int]],
    source_x_number: int, source_y_number: int,
    topology_x_limitation: int, topology_y_limitation: int,     
    message_flits: int, 
    reduction: float,
    latency=None, bandwidth=None   
): 
    
    reducelocal_done_dependency_list = [[[] for _ in range(source_x_number)] for _ in range(source_y_number)]

    topleft_node = source_nodes_coordinates_list[0]
    topleft_x_coord = topleft_node[1]
    topleft_y_coord = topleft_node[0]
    for y_idx in range(source_y_number):
        source_y = topleft_y_coord + y_idx
        for x_idx in range(source_x_number):
            source_x = x_idx + topleft_x_coord
            soure_idx = source_y * topology_x_limitation + source_x
            computation_event = ComputationEvent(current_event_tag, soure_idx)
            computation_event.build_dependency(current_dependency_list[y_idx][x_idx])
            computation_event.reduce_cal(message_flits, reduction)
            reducelocal_done_dependency_list[y_idx][x_idx].append(current_event_tag)
            current_event_tag += 1
            whole_nodes.nodes[soure_idx].event_queue.append(computation_event)

    return current_event_tag, reducelocal_done_dependency_list            


def reducelocal_torus2d_base(
    whole_nodes: NodeNetwork, current_event_tag: int, current_dependency_list: List[List[List[int]]], 
    source_nodes_coordinates_list: List[Tuple[int]],
    source_x_number: int, source_y_number: int,
    topology_x_limitation: int, topology_y_limitation: int,     
    message_flits: int, 
    reduction: float,
    latency=None, bandwidth=None   
): 
    
    reducelocal_done_dependency_list = [[[] for _ in range(source_x_number)] for _ in range(source_y_number)]

    topleft_node = source_nodes_coordinates_list[0]
    topleft_x_coord = topleft_node[1]
    topleft_y_coord = topleft_node[0]
    for y_idx in range(source_y_number):
        source_y = topleft_y_coord + y_idx
        for x_idx in range(source_x_number):
            source_x = x_idx + topleft_x_coord
            soure_idx = source_y * topology_x_limitation + source_x
            computation_event = ComputationEvent(current_event_tag, soure_idx)
            computation_event.build_dependency(current_dependency_list[y_idx][x_idx])
            computation_event.reduce_cal(message_flits, reduction)
            reducelocal_done_dependency_list[y_idx][x_idx].append(current_event_tag)
            current_event_tag += 1
            whole_nodes.nodes[soure_idx].event_queue.append(computation_event)

    return current_event_tag, reducelocal_done_dependency_list     


def reducelocal_torus3d_base(
    whole_nodes: NodeNetwork, current_event_tag: int, current_dependency_list: List[List[List[int]]],
    source_nodes_coordinates_list: List[Tuple[int]],
    source_x_number: int, source_y_number: int, source_z_number: int,
    topology_x_limitation: int, topology_y_limitation: int, topology_z_limitation: int,
    message_flits: int,
    reduction: float,
    latency=None, bandwidth=None
) -> int:
    
    reducelocal_done_dependency_list = [[[] for _ in range(source_x_number)] for _ in range(source_y_number)]

    topleft_node = source_nodes_coordinates_list[0]
    topleft_x_coord = topleft_node[2]
    topleft_y_coord = topleft_node[1]
    topleft_z_coord = topleft_node[0]
    for z_idx in range(source_z_number):
        source_z = topleft_z_coord + z_idx
        for y_idx in range(source_y_number):
            source_y = topleft_y_coord + y_idx
            for x_idx in range(source_x_number):
                source_x = x_idx + topleft_x_coord
                soure_idx = source_z * topology_y_limitation * topology_x_limitation + source_y * topology_x_limitation + source_x
                computation_event = ComputationEvent(current_event_tag, soure_idx)
                computation_event.build_dependency(current_dependency_list[z_idx][y_idx][x_idx])
                computation_event.reduce_cal(message_flits, reduction)
                reducelocal_done_dependency_list[z_idx][y_idx][x_idx].append(current_event_tag)
                current_event_tag += 1
                whole_nodes.nodes[soure_idx].event_queue.append(computation_event)

    return current_event_tag, reducelocal_done_dependency_list


def reducelocal_dgx2_base(
    whole_nodes: NodeNetwork_2D, current_event_tag: int, current_dependency_list: List[List[List[int]]], 
    source_nodes_coordinates_list: List[Tuple[int]],
    source_x_number: int, source_y_number: int,
    topology_x_limitation: int, topology_y_limitation: int,     
    message_flits: int, 
    reduction: float,
    latency=None, bandwidth=None   
): 
    
    reducelocal_done_dependency_list = [[[] for _ in range(source_x_number)] for _ in range(source_y_number)]

    topleft_node = source_nodes_coordinates_list[0]
    topleft_x_coord = topleft_node[1]
    topleft_y_coord = topleft_node[0]
    for y_idx in range(source_y_number):
        source_y = topleft_y_coord + y_idx
        for x_idx in range(source_x_number):
            source_x = x_idx + topleft_x_coord
            soure_idx = source_y * topology_x_limitation + source_x
            computation_event = ComputationEvent(current_event_tag, soure_idx)
            computation_event.build_dependency(current_dependency_list[y_idx][x_idx])
            computation_event.reduce_cal(message_flits, reduction)
            reducelocal_done_dependency_list[y_idx][x_idx].append(current_event_tag)
            current_event_tag += 1
            whole_nodes.nodes[soure_idx].event_queue.append(computation_event)

    return current_event_tag, reducelocal_done_dependency_list   


'''
hierarchicalring is from <Exhaustive Study of Hierarchical AllReduce Patterns for Large Messages Between GPUs>.
It means that do the communication per dimension.
'''

def reducescatter_mesh2d_hierarchicalring(
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

    reducescatter_done_dependency_list = []

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
                if x_idx == source_x_number - 1:
                    send_message_target_x = topleft_x_coord
                    send_message_target_x_idx = 0
                else:
                    send_message_target_x = x_idx + 1
                    send_message_target_x_idx = x_idx + 1
                send_message_target_idx = send_message_target_y * topology_x_limitation + send_message_target_x
                # print(f"send_message_source_idx: {send_message_source_idx}, send_message_target_idx: {send_message_target_idx}")
                # print(whole_nodes.nodes[send_message_source_idx].coordinate)
                if (send_message_source_y, send_message_source_x) != whole_nodes.nodes[send_message_source_idx].coordinate:
                    raise ValueError(f"source node idx {send_message_source_idx} is not matched with source node coordinate {send_message_source_y, send_message_source_x}")
                if (send_message_target_y, send_message_target_x) != whole_nodes.nodes[send_message_target_idx].coordinate:
                    raise ValueError(f"target node idx {send_message_target_idx} is not matched with target node coordinate {send_message_target_y, send_message_target_x}")
                
                if reduction == None:
                    node_dependency_list[send_message_target_y_idx][send_message_target_x_idx].append(current_event_tag)
                    communication_event = CommunicationEvent(current_event_tag, send_message_source_idx, send_message_target_idx, message_flits / source_x_number)
                    communication_event.build_dependency(last_dependency_list[y_idx][x_idx])
                    current_event_tag += 1
                    whole_nodes.nodes[send_message_source_idx].event_queue.append(communication_event)                
                else:
                    communication_event = CommunicationEvent(current_event_tag, send_message_source_idx, send_message_target_idx, message_flits / source_x_number)
                    communication_event.build_dependency(last_dependency_list[y_idx][x_idx])
                    current_event_tag += 1
                    whole_nodes.nodes[send_message_source_idx].event_queue.append(communication_event)

                    compuatation_event = ComputationEvent(current_event_tag, send_message_source_idx)
                    compuatation_event.reduce_cal(message_flits / source_x_number, reduction)
                    compuatation_event.build_dependency([communication_event.event_tag])
                    node_dependency_list[send_message_target_y_idx][send_message_target_x_idx].append(current_event_tag)
                    reducescatter_done_dependency_list.append(current_event_tag)
                    current_event_tag += 1
                    whole_nodes.nodes[send_message_target_idx].event_queue.append(compuatation_event)

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
                if y_idx == source_y_number - 1:
                    send_message_target_y = topleft_y_coord
                    send_message_target_y_idx = 0
                else:
                    send_message_target_y = y_idx + 1
                    send_message_target_y_idx = y_idx + 1
                send_message_target_idx = send_message_target_y * topology_x_limitation + send_message_target_x
                if (send_message_source_y, send_message_source_x) != whole_nodes.nodes[send_message_source_idx].coordinate:
                    raise ValueError(f"source node idx {send_message_source_idx} is not matched with source node coordinate {send_message_source_y, send_message_source_x}")
                if (send_message_target_y, send_message_target_x) != whole_nodes.nodes[send_message_target_idx].coordinate:
                    raise ValueError(f"target node idx {send_message_target_idx} is not matched with target node coordinate {send_message_target_y, send_message_target_x}")
                
                if reduction == None:
                    node_dependency_list[send_message_target_y_idx][send_message_target_x_idx].append(current_event_tag)
                    communication_event = CommunicationEvent(current_event_tag, send_message_source_idx, send_message_target_idx, message_flits / nodes_number)
                    communication_event.build_dependency(last_dependency_list[y_idx][x_idx])
                    current_event_tag += 1
                    whole_nodes.nodes[send_message_source_idx].event_queue.append(communication_event)
                else:
                    communication_event = CommunicationEvent(current_event_tag, send_message_source_idx, send_message_target_idx, message_flits / nodes_number)
                    communication_event.build_dependency(last_dependency_list[y_idx][x_idx])
                    current_event_tag += 1
                    whole_nodes.nodes[send_message_source_idx].event_queue.append(communication_event)

                    compuatation_event = ComputationEvent(current_event_tag, send_message_source_idx)
                    compuatation_event.reduce_cal(message_flits / nodes_number, reduction)
                    compuatation_event.build_dependency([communication_event.event_tag])
                    node_dependency_list[send_message_target_y_idx][send_message_target_x_idx].append(current_event_tag)
                    reducescatter_done_dependency_list.append(current_event_tag)
                    current_event_tag += 1
                    whole_nodes.nodes[send_message_target_idx].event_queue.append(compuatation_event)

    pass_dependency_list = [[reducescatter_done_dependency_list for _ in range(source_x_number)] for _ in range(source_y_number)]

    return current_event_tag, pass_dependency_list


def reducescatter_torus2d_hierarchicalring(
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

    reducescatter_done_dependency_list = []

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
                if x_idx == source_x_number - 1:
                    send_message_target_x = topleft_x_coord
                    send_message_target_x_idx = 0
                else:
                    send_message_target_x = x_idx + 1
                    send_message_target_x_idx = x_idx + 1
                send_message_target_idx = send_message_target_y * topology_x_limitation + send_message_target_x
                # print(f"send_message_source_idx: {send_message_source_idx}, send_message_target_idx: {send_message_target_idx}")
                # print(whole_nodes.nodes[send_message_source_idx].coordinate)
                if (send_message_source_y, send_message_source_x) != whole_nodes.nodes[send_message_source_idx].coordinate:
                    raise ValueError(f"source node idx {send_message_source_idx} is not matched with source node coordinate {send_message_source_y, send_message_source_x}")
                if (send_message_target_y, send_message_target_x) != whole_nodes.nodes[send_message_target_idx].coordinate:
                    raise ValueError(f"target node idx {send_message_target_idx} is not matched with target node coordinate {send_message_target_y, send_message_target_x}")
                
                if reduction == None:
                    node_dependency_list[send_message_target_y_idx][send_message_target_x_idx].append(current_event_tag)
                    communication_event = CommunicationEvent(current_event_tag, send_message_source_idx, send_message_target_idx, message_flits / source_x_number)
                    communication_event.build_dependency(last_dependency_list[y_idx][x_idx])
                    current_event_tag += 1
                    whole_nodes.nodes[send_message_source_idx].event_queue.append(communication_event)                
                else:
                    communication_event = CommunicationEvent(current_event_tag, send_message_source_idx, send_message_target_idx, message_flits / source_x_number)
                    communication_event.build_dependency(last_dependency_list[y_idx][x_idx])
                    current_event_tag += 1
                    whole_nodes.nodes[send_message_source_idx].event_queue.append(communication_event)

                    compuatation_event = ComputationEvent(current_event_tag, send_message_source_idx)
                    compuatation_event.reduce_cal(message_flits / source_x_number, reduction)
                    compuatation_event.build_dependency([communication_event.event_tag])
                    node_dependency_list[send_message_target_y_idx][send_message_target_x_idx].append(current_event_tag)
                    reducescatter_done_dependency_list.append(current_event_tag)
                    current_event_tag += 1
                    whole_nodes.nodes[send_message_target_idx].event_queue.append(compuatation_event)

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
                if y_idx == source_y_number - 1:
                    send_message_target_y = topleft_y_coord
                    send_message_target_y_idx = 0
                else:
                    send_message_target_y = y_idx + 1
                    send_message_target_y_idx = y_idx + 1
                send_message_target_idx = send_message_target_y * topology_x_limitation + send_message_target_x
                if (send_message_source_y, send_message_source_x) != whole_nodes.nodes[send_message_source_idx].coordinate:
                    raise ValueError(f"source node idx {send_message_source_idx} is not matched with source node coordinate {send_message_source_y, send_message_source_x}")
                if (send_message_target_y, send_message_target_x) != whole_nodes.nodes[send_message_target_idx].coordinate:
                    raise ValueError(f"target node idx {send_message_target_idx} is not matched with target node coordinate {send_message_target_y, send_message_target_x}")
                
                if reduction == None:
                    node_dependency_list[send_message_target_y_idx][send_message_target_x_idx].append(current_event_tag)
                    communication_event = CommunicationEvent(current_event_tag, send_message_source_idx, send_message_target_idx, message_flits / nodes_number)
                    communication_event.build_dependency(last_dependency_list[y_idx][x_idx])
                    current_event_tag += 1
                    whole_nodes.nodes[send_message_source_idx].event_queue.append(communication_event)
                else:
                    communication_event = CommunicationEvent(current_event_tag, send_message_source_idx, send_message_target_idx, message_flits / nodes_number)
                    communication_event.build_dependency(last_dependency_list[y_idx][x_idx])
                    current_event_tag += 1
                    whole_nodes.nodes[send_message_source_idx].event_queue.append(communication_event)

                    compuatation_event = ComputationEvent(current_event_tag, send_message_source_idx)
                    compuatation_event.reduce_cal(message_flits / nodes_number, reduction)
                    compuatation_event.build_dependency([communication_event.event_tag])
                    node_dependency_list[send_message_target_y_idx][send_message_target_x_idx].append(current_event_tag)
                    reducescatter_done_dependency_list.append(current_event_tag)
                    current_event_tag += 1
                    whole_nodes.nodes[send_message_target_idx].event_queue.append(compuatation_event)

    pass_dependency_list = [[reducescatter_done_dependency_list for _ in range(source_x_number)] for _ in range(source_y_number)]

    return current_event_tag, pass_dependency_list


def reducescatter_torus3d_hierarchicalring(
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

    reducescatter_done_dependency_list = []

    '''
    all nodes lists should be ordered by idx
    '''

    fronttopleft_node = source_nodes_coordinates_list[0]
    topleft_z_coord = fronttopleft_node[0]
    topleft_y_coord = fronttopleft_node[1]
    topleft_x_coord = fronttopleft_node[2]

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
                    if x_idx == source_x_number - 1:
                        send_message_target_x = topleft_x_coord
                        send_message_target_x_idx = 0
                    else:
                        send_message_target_x = topleft_x_coord + x_idx + 1
                        send_message_target_x_idx = x_idx + 1
                    send_message_target_y = send_message_source_y
                    send_message_target_z = send_message_source_z
                    send_message_source_idx = send_message_source_z * (topology_y_limitation * topology_x_limitation) + send_message_source_y * topology_x_limitation + send_message_source_x
                    send_message_target_idx = send_message_target_z * (topology_y_limitation * topology_x_limitation) + send_message_target_y * topology_x_limitation + send_message_target_x
                    if (send_message_source_z, send_message_source_y, send_message_source_x) != whole_nodes.nodes[send_message_source_idx].coordinate:
                        raise ValueError(f"source node idx {send_message_source_idx} is not matched with source node coordinate {send_message_source_z, send_message_source_y, send_message_source_x}")
                    if (send_message_target_z, send_message_target_y, send_message_target_x) != whole_nodes.nodes[send_message_target_idx].coordinate:
                        raise ValueError(f"target node idx {send_message_target_idx} is not matched with target node coordinate {send_message_target_z, send_message_target_y, send_message_target_x}")
                    if reduction is None:
                        node_dependency_list[z_idx][y_idx][send_message_target_x_idx].append(current_event_tag)
                        communication_event = CommunicationEvent(current_event_tag, send_message_source_idx, send_message_target_idx, message_flits / source_x_number)
                        communication_event.build_dependency(last_dependency_list[z_idx][y_idx][x_idx])
                        current_event_tag += 1
                        whole_nodes.nodes[send_message_source_idx].event_queue.append(communication_event)
                    else:
                        communication_event = CommunicationEvent(current_event_tag, send_message_source_idx, send_message_target_idx, message_flits / source_x_number)
                        communication_event.build_dependency(last_dependency_list[z_idx][y_idx][x_idx])
                        current_event_tag += 1
                        whole_nodes.nodes[send_message_source_idx].event_queue.append(communication_event)
                        compuatation_event = ComputationEvent(current_event_tag, send_message_source_idx)
                        compuatation_event.reduce_cal(message_flits / source_x_number, reduction)
                        compuatation_event.build_dependency([communication_event.event_tag])
                        node_dependency_list[z_idx][y_idx][send_message_target_x_idx].append(current_event_tag)
                        reducescatter_done_dependency_list.append(current_event_tag)
                        current_event_tag += 1
                        whole_nodes.nodes[send_message_target_idx].event_queue.append(compuatation_event)

    for step_y in range(source_y_number - 1):
        if step_y == 0:
            last_dependency_list = current_dependency_list
            node_dependency_list = [[[[] for _ in range(source_x_number)] for _ in range(source_y_number)] for _ in range(source_z_number)]
        else:
            last_dependency_list = node_dependency_list
            node_dependency_list = [[[[] for _ in range(source_x_number)] for _ in range(source_y_number)] for _ in range(source_z_number)]
        for z_idx in range(source_z_number):
            for x_idx in range(source_x_number):
                for y_idx in range(source_y_number):
                    send_message_source_y = topleft_y_coord + y_idx
                    send_message_source_x = topleft_x_coord + x_idx
                    send_message_source_z = topleft_z_coord + z_idx
                    if y_idx == source_y_number - 1:
                        send_message_target_y = topleft_y_coord
                        send_message_target_y_idx = 0
                    else:
                        send_message_target_y = topleft_y_coord + y_idx + 1
                        send_message_target_y_idx = y_idx + 1
                    send_message_target_x = send_message_source_x
                    send_message_target_z = send_message_source_z
                    send_message_source_idx = send_message_source_z * (topology_y_limitation * topology_x_limitation) + send_message_source_y * topology_x_limitation + send_message_source_x
                    send_message_target_idx = send_message_target_z * (topology_y_limitation * topology_x_limitation) + send_message_target_y * topology_x_limitation + send_message_target_x
                    if (send_message_source_z, send_message_source_y, send_message_source_x) != whole_nodes.nodes[send_message_source_idx].coordinate:
                        raise ValueError(f"source node idx {send_message_source_idx} is not matched with source node coordinate {send_message_source_z, send_message_source_y, send_message_source_x}")
                    if (send_message_target_z, send_message_target_y, send_message_target_x) != whole_nodes.nodes[send_message_target_idx].coordinate:
                        raise ValueError(f"target node idx {send_message_target_idx} is not matched with target node coordinate {send_message_target_z, send_message_target_y, send_message_target_x}")
                    if reduction is None:
                        node_dependency_list[z_idx][send_message_target_y_idx][x_idx].append(current_event_tag)
                        communication_event = CommunicationEvent(current_event_tag, send_message_source_idx, send_message_target_idx, message_flits / (source_x_number * source_y_number))
                        communication_event.build_dependency(last_dependency_list[z_idx][y_idx][x_idx])
                        current_event_tag += 1
                        whole_nodes.nodes[send_message_source_idx].event_queue.append(communication_event)
                    else:
                        communication_event = CommunicationEvent(current_event_tag, send_message_source_idx, send_message_target_idx, message_flits / (source_x_number * source_y_number))
                        communication_event.build_dependency(last_dependency_list[z_idx][y_idx][x_idx])
                        current_event_tag += 1
                        whole_nodes.nodes[send_message_source_idx].event_queue.append(communication_event)
                        compuatation_event = ComputationEvent(current_event_tag, send_message_source_idx)
                        compuatation_event.reduce_cal(message_flits / (source_x_number * source_y_number), reduction)
                        compuatation_event.build_dependency([communication_event.event_tag])
                        node_dependency_list[z_idx][send_message_target_y_idx][x_idx].append(current_event_tag)
                        reducescatter_done_dependency_list.append(current_event_tag)
                        current_event_tag += 1
                        whole_nodes.nodes[send_message_target_idx].event_queue.append(compuatation_event)

    for step_z in range(source_z_number - 1):
        last_dependency_list = node_dependency_list
        node_dependency_list = [[[[] for _ in range(source_x_number)] for _ in range(source_y_number)] for _ in range(source_z_number)]
        for y_idx in range(source_y_number):
            for x_idx in range(source_x_number):
                for z_idx in range(source_z_number):
                    send_message_source_y = topleft_y_coord + y_idx
                    send_message_source_x = topleft_x_coord + x_idx
                    send_message_source_z = topleft_z_coord + z_idx
                    if z_idx == source_z_number - 1:
                        send_message_target_z = topleft_z_coord
                        send_message_target_z_idx = 0
                    else:
                        send_message_target_z = topleft_z_coord + z_idx + 1
                        send_message_target_z_idx = z_idx + 1
                    send_message_target_y = send_message_source_y
                    send_message_target_x = send_message_source_x
                    send_message_source_idx = send_message_source_z * (topology_y_limitation * topology_x_limitation) + send_message_source_y * topology_x_limitation + send_message_source_x
                    send_message_target_idx = send_message_target_z * (topology_y_limitation * topology_x_limitation) + send_message_target_y * topology_x_limitation + send_message_target_x
                    if (send_message_source_z, send_message_source_y, send_message_source_x) != whole_nodes.nodes[send_message_source_idx].coordinate:
                        raise ValueError(f"source node idx {send_message_source_idx} is not matched with source node coordinate {send_message_source_z, send_message_source_y, send_message_source_x}")
                    if (send_message_target_z, send_message_target_y, send_message_target_x) != whole_nodes.nodes[send_message_target_idx].coordinate:
                        raise ValueError(f"target node idx {send_message_target_idx} is not matched with target node coordinate {send_message_target_z, send_message_target_y, send_message_target_x}")
                    if reduction is None:
                        node_dependency_list[send_message_target_z_idx][y_idx][x_idx].append(current_event_tag)
                        communication_event = CommunicationEvent(current_event_tag, send_message_source_idx, send_message_target_idx, message_flits / nodes_number)
                        communication_event.build_dependency(last_dependency_list[z_idx][y_idx][x_idx])
                        current_event_tag += 1
                        whole_nodes.nodes[send_message_source_idx].event_queue.append(communication_event)
                    else:
                        communication_event = CommunicationEvent(current_event_tag, send_message_source_idx, send_message_target_idx, message_flits / nodes_number)
                        communication_event.build_dependency(last_dependency_list[z_idx][y_idx][x_idx])
                        current_event_tag += 1
                        whole_nodes.nodes[send_message_source_idx].event_queue.append(communication_event)
                        compuatation_event = ComputationEvent(current_event_tag, send_message_source_idx)
                        compuatation_event.reduce_cal(message_flits / nodes_number, reduction)
                        compuatation_event.build_dependency([communication_event.event_tag])
                        node_dependency_list[send_message_target_z_idx][y_idx][x_idx].append(current_event_tag)
                        reducescatter_done_dependency_list.append(current_event_tag)
                        current_event_tag += 1
                        whole_nodes.nodes[send_message_target_idx].event_queue.append(compuatation_event)

    pass_dependency_list = [[[reducescatter_done_dependency_list for _ in range(source_x_number)] for _ in range(source_y_number)] for _ in range(source_z_number)]

    return current_event_tag, pass_dependency_list


def reducescatter_dgx2_havlingdoubling(
    whole_nodes: NodeNetwork_2D, current_event_tag: int, current_dependency_list: List[List[List[int]]], 
    source_nodes_coordinates_list: List[Tuple[int]],
    source_x_number: int, source_y_number: int,
    topology_x_limitation: int, topology_y_limitation: int, 
    message_flits: int, 
    reduction: float,
    latency=None, bandwidth=None 
):
    
    # x is intra-dimension
    # y is inter-dimension
    nodes_number = source_x_number * source_y_number
    initial_event_tag = current_event_tag

    reducescatter_done_dependency_list = []

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
            size_ratio = distance / nodes_number
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

                if reduction == None:
                    node_dependency_list[target_node_coordinate[0]-topleft_y_coord][target_node_coordinate[1]-topleft_x_coord].append(current_event_tag)
                    communication_event = CommunicationEvent(current_event_tag, send_message_source_idx, send_message_target_idx, message_flits * size_ratio)
                    communication_event.build_dependency(last_dependency_list[source_node_coordinate[0]-topleft_y_coord][source_node_coordinate[1]-topleft_x_coord])
                    current_event_tag += 1
                    whole_nodes.nodes[send_message_source_idx].event_queue.append(communication_event)

                    node_dependency_list[source_node_coordinate[0]-topleft_y_coord][source_node_coordinate[1]-topleft_x_coord].append(current_event_tag)
                    communication_event = CommunicationEvent(current_event_tag, send_message_target_idx, send_message_source_idx, message_flits * size_ratio)
                    communication_event.build_dependency(last_dependency_list[target_node_coordinate[0]-topleft_y_coord][target_node_coordinate[1]-topleft_x_coord])
                    current_event_tag += 1
                    whole_nodes.nodes[send_message_target_idx].event_queue.append(communication_event)
                else:
                    communication_event = CommunicationEvent(current_event_tag, send_message_source_idx, send_message_target_idx, message_flits * size_ratio)
                    communication_event.build_dependency(last_dependency_list[source_node_coordinate[0]-topleft_y_coord][source_node_coordinate[1]-topleft_x_coord])
                    current_event_tag += 1
                    whole_nodes.nodes[send_message_source_idx].event_queue.append(communication_event)

                    compuatation_event = ComputationEvent(current_event_tag, send_message_source_idx)
                    compuatation_event.reduce_cal(message_flits * size_ratio, reduction)
                    compuatation_event.build_dependency([communication_event.event_tag])
                    node_dependency_list[target_node_coordinate[0]-topleft_y_coord][target_node_coordinate[1]-topleft_x_coord].append(current_event_tag)
                    current_event_tag += 1
                    whole_nodes.nodes[send_message_target_idx].event_queue.append(compuatation_event)

                    communication_event = CommunicationEvent(current_event_tag, send_message_target_idx, send_message_source_idx, message_flits * size_ratio)
                    communication_event.build_dependency([compuatation_event.event_tag])
                    current_event_tag += 1
                    whole_nodes.nodes[send_message_target_idx].event_queue.append(communication_event)

                    compuatation_event = ComputationEvent(current_event_tag, send_message_target_idx)
                    compuatation_event.reduce_cal(message_flits * size_ratio, reduction)
                    compuatation_event.build_dependency([communication_event.event_tag])
                    node_dependency_list[source_node_coordinate[0]-topleft_y_coord][source_node_coordinate[1]-topleft_x_coord].append(current_event_tag)
                    current_event_tag += 1
                    whole_nodes.nodes[send_message_source_idx].event_queue.append(compuatation_event)

                iter_list.remove(source_node_idx)
                iter_list.remove(target_node_idx)

    else:
        # TODO: implement the non-power of 2 nodes number
        raise ValueError(f"nodes_number {nodes_number} is not a power of 2")

    return current_event_tag, node_dependency_list


class reducelocal(object):

    def __init__(self, topology: str, algorithm: str):
        communication_name = self.__class__.__name__
        collective_selected = communication_name + "_" + topology + "_" + algorithm
        self.cal_time = globals()[collective_selected]


class reducescatter(object):

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

    communication_scheduler = reducescatter(test_topology, test_communication_algorithm)

    now_event_tag, _ = communication_scheduler.cal_time(
        whole_nodes=node_network, current_event_tag=current_event_tag, current_dependency_list=initial_dependency_list,
        source_nodes_coordinates_list=test_source_nodes,
        source_x_number=test_source_shape[1], source_y_number=test_source_shape[0], source_z_number=test_source_shape[2],
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

    # communication_scheduler = reducescatter(test_cfg_topology, test_communication_algorithm)

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



