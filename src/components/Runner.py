# File name  :    Runner.py
# Author     :    xiaocuicui
# Time       :    2024/06/18 14:00:09
# Version    :    V1.0
# Abstract   :        

import os
import sys
file_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(file_path)
sys.path.append(os.path.join(file_path, '../communication/backend/'))


from NodeNetwork import NodeNetwork
from NodeNetwork_2D import NodeNetwork_2D
from Booksim_Api import BookSim_Interface


def run(
    whole_nodes: NodeNetwork, current_cycle: int, 
    network: BookSim_Interface, print_flag: bool = True, booksim2_flit_units: int = 256
) -> int:

    now_cycle = current_cycle
    whole_nodes.initialize_first_events_cycle(now_cycle)
    try_issue_flag = False

    event_done_cnt = 0

    while(whole_nodes.exsisting_events() == True or whole_nodes.global_communication_events_receive_queue != []):
        # print(whole_nodes.global_communication_events_receive_queue)
        ''''
        TODO Here should be look the status of booksim2 by 
        idle_flag = network.backend_idle()
        '''
        # idle_flag = network.backend_idle()
        idle_flag = (whole_nodes.global_communication_events_receive_queue == []) and network.backend_idle()
        # if idle_flag:
        #     print("idle netowrk is running cycle: ", now_cycle)
        # else:
        #     print("busy netowrk is running cycle: ", now_cycle)

        if idle_flag and try_issue_flag == False:
            # print("the noc network is idle")
            next_cycle = sys.maxsize
            for node in whole_nodes.nodes:
                for event in node.event_queue:
                    # print(node, event)
                    if event.built_flag == True and event.dependency_list == []:
                        if event.start_time < next_cycle:
                            next_cycle = event.start_time
                        else:
                            continue

            if next_cycle == sys.maxsize:
                if whole_nodes.exsisting_events() == True:
                    for node in whole_nodes.nodes:
                        if node.event_queue:
                            print(node.idx, node.event_queue)
                    raise ValueError("Flying off with un issued events")
                elif whole_nodes.global_communication_events_receive_queue != []:
                    raise ValueError("Flying off with un received communication events")
                else:
                    raise ValueError("next_cycle by calculation is not updated")
            
            now_cycle = next_cycle
            try_issue_flag = True
        else:
            # print("deal with communication")
            # communication
            # print("Begin to deal with sending")
            for ni in whole_nodes.nis:
                for event in ni.event_queue:
                    # print(ni, event)
                    if event.built_flag == True and event.dependency_list == [] and event.start_time == now_cycle:
                        network.wait_sending(event.source_node_idx, event.target_node_idx, event.event_tag, event.message_bits, booksim2_flit_units)
                        event.start_time = now_cycle
                        whole_nodes.global_communication_events_send_queue.append(event)
                        whole_nodes.global_communication_events_receive_queue.append(event)
                        whole_nodes.finished_events_list.append(event)

            sent_list = network.send_message()
            for sent_message_info in sent_list:
                event_tag = sent_message_info[2]
                src_idx = sent_message_info[0]
                dest_idx = sent_message_info[1]
                if print_flag:
                    print("At ", now_cycle, ",", event_tag, "event has sent flits from ", src_idx, " to ", dest_idx)
                for ni in whole_nodes.nis:
                    for event in ni.event_queue:
                        if event.event_tag == event_tag:
                            event.end_time = now_cycle
                            event.run_time = event.end_time - event.start_time
                            break

            for event in whole_nodes.global_communication_events_send_queue[:]:
                event_tag = event.event_tag        

                event_found = False
                # update nodes
                for node in whole_nodes.nodes:
                    for node_event in node.event_queue[:]:
                        if node_event.event_tag == event_tag:
                            node.event_queue.remove(node_event)
                            event_found = True
                            break
                        else:
                            continue
                    if event_found:
                        break

                whole_nodes.global_communication_events_send_queue.remove(event)

            # print("{} cycle will be set", now_cycle)
            network.wakeup(now_cycle)

            # print("Begin to deal with receiving")
            for unpeek_event in whole_nodes.generate_peek_events_list():

                dest_idx = unpeek_event.target_node_idx
                # print("kankan zhege dest", dest_idx)
                src_idx = network.receive_message(dest_idx)
                if src_idx == -1 or src_idx == None:
                    continue
                else:
                    for event in whole_nodes.global_communication_events_receive_queue[:]:
                        if event.source_node_idx == src_idx and event.target_node_idx == dest_idx:
                            event.end_time = now_cycle
                            event.run_time = event.end_time - event.start_time
                            dependency_tag = event.event_tag
                            event_done_cnt += 1
                            if print_flag:
                                print("At ", now_cycle, ",", dependency_tag, "event receive ", event.message_bits, "flits from ", src_idx, " to ", dest_idx, " current total done events: ", event_done_cnt)
                            for node in whole_nodes.nodes:
                                for node_event in node.event_queue:
                                    # print(node, node_event, dependency_tag)
                                    node_event.update_one_dependency(dependency_tag)
                                    if node_event.start_time == None:
                                        node_event.start_time = now_cycle + 1
                                    elif node_event.start_time < now_cycle:
                                        node_event.start_time = now_cycle + 1
                                    else:
                                        continue
                            whole_nodes.global_communication_events_receive_queue.remove(event)

                            break

                        else:
                            continue

            # computation
            for node in whole_nodes.nodes:
                for event in node.event_queue[:]:
                    if event.built_flag == True and event.dependency_list == [] and event.start_time == now_cycle and event.event_type == "computation":
                        '''
                        TODO: dynamic use computation api
                        eg: event.end_time = event.start_time + Maestro.sim()
                        '''
                        event_done_cnt += 1
                        if print_flag:
                            print("At ", now_cycle, ",", "node ", node.idx, " is computing ", event.event_tag, " current total done events: ", event_done_cnt)
                        
                        event.start_time = now_cycle
                        event.end_time = event.start_time + event.run_time
                        whole_nodes.global_computation_events_done_queue.append(event)
                        whole_nodes.finished_events_list.append(event)
                        node.event_queue.remove(event)


            for computation_event in whole_nodes.global_computation_events_done_queue[:]:
                dependency_tag = computation_event.event_tag
                for node in whole_nodes.nodes:
                    for node_event in node.event_queue[:]:
                        node_event.update_one_dependency(dependency_tag)
                        if node_event.start_time == None:
                            node_event.start_time = computation_event.end_time + 1
                        elif node_event.start_time < computation_event.end_time + 1:
                            node_event.start_time = computation_event.end_time + 1
                        else:
                            continue
                whole_nodes.global_computation_events_done_queue.remove(computation_event)

            
            # update nis
            whole_nodes.build_nis_events()

            now_cycle += 1
            try_issue_flag = False

    # TODO: which cycle is the end time
    task_finish_cycle = now_cycle

    while (network.backend_idle() == False):

        network.wakeup(now_cycle)
        now_cycle += 1


    return task_finish_cycle



