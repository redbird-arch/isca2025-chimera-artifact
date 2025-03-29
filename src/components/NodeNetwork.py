# File name  :    NodeNetwork.py
# Author     :    xiaocuicui
# Time       :    2024/06/16 13:35:50
# Version    :    V1.0
# Abstract   :        

import os
import sys
file_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(file_path)

import itertools

from Node import Node
from NI import NodeInterface


class NodeNetwork(object):

    def __init__(self, node_k: int, node_n: int, ni_k: int, ni_n: int):

        # the number of nodes in each dimension
        self.node_k = node_k
        # the number of dimensions
        self.node_n = node_n
        self.nodes_number = node_k ** node_n
        self.nodes = []
        self.create_nodes()

        self.ni_k = ni_k
        self.ni_n = ni_n
        self.nis_number = ni_k ** ni_n
        self.nis = []
        self.create_nis()

        self.current_cycle = 0
        
        '''
        The event is the same as the event which has been sent
        '''
        self.global_communication_events_send_queue = []
        self.global_communication_events_receive_queue = []

        # self.global_computation_events_begin_queue = []
        self.global_computation_events_done_queue = []

        self.finished_events_list = []
        

    def create_nodes(self):

        points = list(itertools.product(range(self.node_k), repeat=self.node_n))
        for node_idx in range(len(points)):
            node = Node(node_idx, points[node_idx])
            self.nodes.append(node)
            

    def show_nodes_events(self):

        for node in self.nodes:
            print(node)
            for event in node.event_queue:
                print(event)


    def create_nis(self):

        points = list(itertools.product(range(self.ni_k), repeat=self.ni_n))
        '''
        user can modify mapping_table by his logic
        Here: {node_idx: ni_idx, ...}
        '''
        nodes_per_ni = self.nodes_number // self.nis_number

        node_ni_table = {}
        ni_node_table = {}
        for ni_idx in range(self.nis_number):
            ni_node_table[ni_idx] = []
            for nodes_inni_idx in range(nodes_per_ni):
                node_idx = ni_idx * nodes_per_ni + nodes_inni_idx
                node_ni_table[node_idx] = ni_idx
                ni_node_table[ni_idx].append(node_idx)

        for ni_idx in range(len(points)):
            ni = NodeInterface(ni_idx, points[ni_idx], node_ni_table)
            self.nis.append(ni)

        self.node_ni_table = node_ni_table
        self.ni_node_table = ni_node_table


    def build_nis_events(self):
        for ni in self.nis:
            ni.collect_communication_events([self.nodes[node_idx] for node_idx in self.ni_node_table[ni.idx]])
    

    def show_nis_events(self):

        for ni in self.nis:
            print(ni)
            for event in ni.event_queue:
                print(event)


    def initialize_first_events_cycle(self, set_cycle):

        self.current_cycle = set_cycle
        for node in self.nodes:
            for event in node.event_queue:
                if event.built_flag and event.dependency_list == []:
                    event.start_time = self.current_cycle

        for ni in self.nis:
            for event in ni.event_queue:
                if event.built_flag == True and event.dependency_list == []:
                    event.start_time = self.current_cycle


    def generate_peek_events_list(self) -> list:

        peek_events_list = []
        for event in self.global_communication_events_receive_queue:
            if event.target_node_idx in peek_events_list:
                continue
            else:
                peek_events_list.append(event)

        return peek_events_list


    def exsisting_events(self) -> bool:

        for node in self.nodes:
            if node.event_queue:
                return True

        return False



if __name__ == "__main__":

    k = 4
    n = 3
    node_network = NodeNetwork(k, n, k, n)
    print(node_network.nodes)



