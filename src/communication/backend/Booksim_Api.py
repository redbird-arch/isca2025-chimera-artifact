# File name  :    api.py
# Author     :    xiaocuicui
# Time       :    2024/06/14 16:28:32
# Version    :    V1.0
# Abstract   :        

import os
import sys
file_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(file_path, '../backend/booksim2/src'))
sys.path.append(os.path.join(file_path, '../../utils/'))

from func import division_list

os.environ['BOOKSIMSRC'] = os.path.join(file_path, '../backend/booksim2/src')
import pybooksim

import math


class BookSim_FlitsBoard():

    def __init__(self, communication_pair, flits_number):

        self.communication_pair = communication_pair
        self.flits_number = flits_number

        self.flits_board = list(range(flits_number))
        self.peek_flag = False


    def message_done(self):

        if self.flits_board == []:
            self.peek_flag = True
            return self.peek_flag
        else:
            return False


    def __str__(self):
        return f"FlitsBoard {self.communication_pair} with {self.flits_number} flits"




class BookSim_Interface():

    def __init__(self, booksim_config):
        super().__init__()
        self.name = 'BookSim'
        self.booksim = pybooksim.BookSim(booksim_config)

        self.booksim_tag = 0
        self.booksim_event_flits_list = []

        self.wait_sending_list = {}


    def backend_idle(self):
        return self.booksim.Idle()


    def wait_sending(self, src_idx: int, dest_idx: int, event_tag: int, 
                     message_bits: int, flit_bytes=256):
        flits_list = division_list(message_bits, flit_bytes)
        flits_number = len(flits_list)
        src_dest_pair = (src_idx, dest_idx)
        self.booksim_event_flits_list.append(BookSim_FlitsBoard(src_dest_pair, flits_number))
        
        for flit_idx, flits in enumerate(flits_list):
            if event_tag not in self.wait_sending_list.keys():
                self.wait_sending_list[event_tag] = []

            if flit_idx == flits_number - 1:
                self.wait_sending_list[event_tag].append([flit_idx, src_idx, dest_idx, flits, True])
            else:
                self.wait_sending_list[event_tag].append([flit_idx, src_idx, dest_idx, flits, False])


    def send_message(self):

        sent_list = []

        if self.wait_sending_list == {}:
            return sent_list
        else:
            for event_tag, sending_task in self.wait_sending_list.items():
                for sending_list in sending_task[:]:
                    flit_idx = sending_list[0]
                    src_idx = sending_list[1]
                    dest_idx = sending_list[2]
                    flits = sending_list[3]
                    end = sending_list[4] 
                    msg_id = self.booksim.IssueMessage(flit_idx, src_idx, dest_idx, self.booksim_tag, flits, pybooksim.Message.GatherData, pybooksim.Message.HeadTail, 0, end)
                    if msg_id == -1:
                        break
                    else:
                        self.booksim_tag += 1
                        sending_task.remove(sending_list)
                        if end:
                            # print('send message {} from HMC-{} to HMC-{} with end {}'.format(msg_id, src_idx, dest_idx, end))
                            sent_list.append([src_idx, dest_idx, event_tag])

            keys_to_remove = [key for key, value in self.wait_sending_list.items() if value == []]
            for key in keys_to_remove:
                del self.wait_sending_list[key]

            return sent_list

        # msg_id = self.booksim.IssueMessage(flow=0, src=0, dest=3, id=0, msg_size=100, type=pybooksim.Message.GatherData, subtype=pybooksim.Message.HeadTail, timestep=0, end=True) 


    def wakeup(self, cur_cycle):
        self.booksim.SetSimTime(cur_cycle)
        self.booksim.WakeUp()


    def receive_message(self, dest_idx):
        # TODO: how distinguish two different message from the same source to the same destination
        flow, src_node, msgtype, end = self.booksim.PeekMessage(dest_idx, 0)
        if src_node != -1:
            self.booksim.DequeueMessage(dest_idx, 0)
            src_dest_pair = (src_node, dest_idx)
            # print('peek message {} flow from HMC-{} to HMC-{} with end {}'.format(flow, src_node, dest_idx, end))

            flow_dequeue_falg = False
            for board in self.booksim_event_flits_list[:]:
                if board.communication_pair == src_dest_pair and board.peek_flag == False:
                    if flow in board.flits_board:
                        board.flits_board.remove(flow)
                        flow_dequeue_falg = True
                        if board.message_done():
                            self.booksim_event_flits_list.remove(board)
                            return src_node
                        break
                    else:
                        continue
            
            if flow_dequeue_falg == False:
                raise ValueError('flow {} of {} dequeue failed'.format(flow, src_dest_pair))
            else:
                return -1

    
    def dequeue_message(self, dest_idx):
        self.booksim.DequeueMessage(dest_idx, 0)



if __name__ == '__main__':
    
    booksim_config = os.path.join(file_path, '../src/booksim2/runfiles/mesh2x2express.cfg')
    network = BookSim_Interface(booksim_config)
    network.send_message()
    print("66666666666666666666666666666666666666666666666666666666666")
    network.receive_message()







