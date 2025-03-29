# File name  :    Event.py
# Author     :    xiaocuicui
# Time       :    2024/06/17 16:45:44
# Version    :    V1.0
# Abstract   :        

import os
import sys
file_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(file_path)


from typing import List
import math


class Event(object):

    def __init__(self, event_tag: int, event_type: str):

        self.event_tag = event_tag
        '''
        type: computation, communication
        '''
        self.event_type = event_type 
        self.built_flag = False
        self.start_time = None
        self.run_time = None
        self.end_time = None
        self.dependency_list = []


    def build_dependency(self, dependency_list: List[int]):

        self.dependency_list = dependency_list
        self.built_flag = True


    def update_one_dependency(self, dealt_tag: int):

        if dealt_tag in self.dependency_list:
            self.dependency_list.remove(dealt_tag)


    def update_dependency(self, dealt_list: List[int]):

        for dealt_tag in dealt_list:
            if dealt_tag in self.dependency_list:
                self.dependency_list.remove(dealt_tag)


    def __str__(self):
        return f"{self.event_type} Event {self.event_tag}"
    __repr__ = __str__



class CommunicationEvent(Event):

    def __init__(self, event_tag: int, source_node_idx: int, target_node_idx: int, message_bits: int):

        super(CommunicationEvent, self).__init__(event_tag, "communication")

        self.source_node_idx = source_node_idx
        self.target_node_idx = target_node_idx
        self.message_bits = int(message_bits)


    def __str__(self):
        return f"{self.event_type} Event {self.event_tag} from {self.source_node_idx} to {self.target_node_idx} depends on {self.dependency_list}"
    __repr__ = __str__



class ComputationEvent(Event):

    def __init__(self, event_tag: int, node_idx: int):

        super(ComputationEvent, self).__init__(event_tag, "computation")

        self.node_idx = node_idx


    def reduce_cal(self, flits: int, reduction: float):

        self.run_time = math.ceil(flits * reduction)


    def compute_cal(self, flits: int, reduction: float):

        self.run_time = math.ceil(flits / reduction)


    def __str__(self):
        return f"{self.event_type} Event {self.event_tag} on {self.node_idx} depends on {self.dependency_list}"
    __repr__ = __str__




