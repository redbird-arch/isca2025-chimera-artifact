# File name  :    Node.py
# Author     :    xiaocuicui
# Time       :    2024/06/16 11:12:02
# Version    :    V1.0
# Abstract   :        

import os
import sys
file_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(file_path)



class Node(object):

    def __init__(self, idx:int, coordinate=None):

        self.idx = idx
        # coordinate is a tuple
        self.coordinate = coordinate
        self.event_queue = []


    def __str__(self):
        return f"Node {self.idx} at {self.coordinate}"
    __repr__ = __str__



