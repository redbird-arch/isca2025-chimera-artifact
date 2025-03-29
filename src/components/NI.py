# File name  :    NI.py
# Author     :    xiaocuicui
# Time       :    2024/06/16 11:06:33
# Version    :    V1.0
# Abstract   :        

import os
import sys
file_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(file_path)

from typing import List
import copy

from Event import CommunicationEvent
from Node import Node


class NodeInterface(object):
    '''
    deal the relationship between nodes and NIs
    eg: a router may be connected to multiple nodes    
    '''
    def __init__(self, idx:int, coordinate=None, mapping_table={}):

        self.idx = idx
        # coordinate is a tuple
        self.coordinate = coordinate
        self.mapping_table = mapping_table
        self.event_queue = []


    def idx_convert(self, nodes_communication_event: CommunicationEvent) -> CommunicationEvent:
        source_node_idx = nodes_communication_event.source_node_idx
        target_node_idx = nodes_communication_event.target_node_idx
        source_ni_idx = self.mapping_table[source_node_idx]
        target_ni_idx = self.mapping_table[target_node_idx]

        ni_event = copy.deepcopy(nodes_communication_event)
        ni_event.source_node_idx = source_ni_idx
        ni_event.target_node_idx = target_ni_idx

        return ni_event


    def collect_communication_events(self, ni_nodes_list: List[Node]):

        self.event_queue = []
        for node in ni_nodes_list:
            for event in node.event_queue:
                if isinstance(event, CommunicationEvent):
                    ni_event = self.idx_convert(event)
                    self.event_queue.append(ni_event) 
        




    def __str__(self):
        return f"NodeInterface {self.idx} at {self.coordinate}"
    __repr__ = __str__




