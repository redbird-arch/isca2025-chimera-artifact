# File name  :    finished_segments.py
# Author     :    xiaocuicui
# Time       :    2025/02/18 10:00:18
# Version    :    V1.0
# Abstract   :        

import os
import sys
file_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(file_path)
sys.path.append(os.path.join(file_path, '../'))
sys.path.append(os.path.join(file_path, '../../../components/'))


from Event import Event

from typing import List


def segment_time(events: List[Event], records: List[int], start_idx_list: List[int], end_idx_list: List[int]):

    segment_start = 0
    segment_end = 0
    for event in events:
        for start_idx in start_idx_list:
            if start_idx == -1:
                continue
            if event.event_tag in records[start_idx][0][0]:
                if event.end_time > segment_start:
                    segment_start = event.end_time
        for end_idx in end_idx_list:
            if event.event_tag in records[end_idx][0][0]:
                if event.end_time > segment_end:
                    segment_end = event.end_time

    return segment_end - segment_start
                

def segment_time_3D(events: List[Event], records: List[int], start_idx_list: List[int], end_idx_list: List[int]):

    segment_start = 0
    segment_end = 0
    for event in events:
        for start_idx in start_idx_list:
            if start_idx == -1:
                continue
            if event.event_tag in records[start_idx][0][0][0]:
                if event.end_time > segment_start:
                    segment_start = event.end_time
        for end_idx in end_idx_list:
            if event.event_tag in records[end_idx][0][0][0]:
                if event.end_time > segment_end:
                    segment_end = event.end_time

    return segment_end - segment_start

