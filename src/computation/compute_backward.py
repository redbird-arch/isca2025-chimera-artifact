# File name  :    compute.py
# Author     :    xiaocuicui
# Time       :    2024/07/28 16:57:48
# Version    :    V1.0
# Abstract   :        

import os
import sys
file_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(file_path, '../components/'))
sys.path.append(os.path.join(file_path, '../utils/'))


from typing import List, Tuple
import numpy as np

from NodeNetwork import NodeNetwork
from NodeNetwork_2D import NodeNetwork_2D
from Event import Event, CommunicationEvent, ComputationEvent


def compute_2d_base(
    whole_nodes: NodeNetwork_2D, current_event_tag: int, current_dependency_list: List[List[List[int]]], 
    source_nodes_coordinates_list: List[Tuple[int]],
    source_x_number: int, source_y_number: int,
    topology_x_limitation: int, topology_y_limitation: int,     
    message_flits: int, 
    reduction: float,
    latency=None, bandwidth=None   
): 

    compute_done_dependency_list = [[[] for _ in range(source_x_number)] for _ in range(source_y_number)]

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
            computation_event.compute_cal(message_flits, 1)
            compute_done_dependency_list[y_idx][x_idx].append(current_event_tag)
            current_event_tag += 1
            whole_nodes.nodes[soure_idx].event_queue.append(computation_event)

    return current_event_tag, compute_done_dependency_list      


def compute_3d_base(
    whole_nodes: NodeNetwork, current_event_tag: int, current_dependency_list: List[List[List[int]]], 
    source_nodes_coordinates_list: List[Tuple[int]],
    source_x_number: int, source_y_number: int, source_z_number: int,
    topology_x_limitation: int, topology_y_limitation: int, topology_z_limitation: int,     
    message_flits: int, 
    reduction: float,
    latency=None, bandwidth=None   
): 

    compute_done_dependency_list = [[[[] for _ in range(source_x_number)] for _ in range(source_y_number)] for _ in range(source_z_number)]

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
                soure_idx = source_z * topology_x_limitation * topology_y_limitation + source_y * topology_x_limitation + source_x
                computation_event = ComputationEvent(current_event_tag, soure_idx)
                computation_event.build_dependency(current_dependency_list[z_idx][y_idx][x_idx])
                computation_event.compute_cal(message_flits, 1)
                compute_done_dependency_list[z_idx][y_idx][x_idx].append(current_event_tag)
                current_event_tag += 1
                whole_nodes.nodes[soure_idx].event_queue.append(computation_event)

    return current_event_tag, compute_done_dependency_list



