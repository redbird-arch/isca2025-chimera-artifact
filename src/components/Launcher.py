# File name  :    Launcher.py
# Author     :    xiaocuicui
# Time       :    2024/07/07 20:52:56
# Version    :    V1.0
# Abstract   :        

import os
import sys
file_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(file_path)
sys.path.append(os.path.join(file_path, '../utils/'))
sys.path.append(os.path.join(file_path, '../communication/backend/'))


from typing import List, Tuple
import multiprocessing

from NodeNetwork import NodeNetwork
from NodeNetwork_2D import NodeNetwork_2D
from config_para import modify_topology_cfg_file
from Booksim_Api import BookSim_Interface
from Runner import run


def build(
    node_k: int, node_n: int, ni_k: int, ni_n: int,
    cfg_topology: str, 
    cfg_filepath = os.path.join(file_path, '../communication/backend/booksim2/runfiles/mesh_o_torus_py.cfg') 
):

    # modify_topology_cfg_file(cfg_filepath, cfg_topology, node_k, node_n)

    node_network = NodeNetwork_2D(node_k, node_n, ni_k, ni_n)

    noc_network = BookSim_Interface(cfg_filepath)

    return node_network, noc_network


def build_3D(
    node_k: int, node_n: int, ni_k: int, ni_n: int,
    cfg_topology: str, 
    cfg_filepath = os.path.join(file_path, '../communication/backend/booksim2/runfiles/mesh_o_torus_py.cfg') 
):

    # modify_topology_cfg_file(cfg_filepath, cfg_topology, node_k, node_n)

    node_network = NodeNetwork(node_k, node_n, ni_k, ni_n)

    noc_network = BookSim_Interface(cfg_filepath)

    return node_network, noc_network


def build_multiprocessing(
    copy_num: int,
    node_k: int, node_n: int, ni_k: int, ni_n: int,
    cfg_topology: str, 
    cfg_filepath = os.path.join(file_path, '../communication/backend/booksim2/runfiles/mesh_o_torus_py.cfg'), 
    print_flag=True
):

    modify_topology_cfg_file(cfg_filepath, cfg_topology, node_k, node_n)

    num_cores = multiprocessing.cpu_count()
    if print_flag:
        print("====================================")
        print("available cores: ", num_cores)
        print("taks number: ", copy_num)

    multi_node_k = [node_k] * copy_num
    multi_node_n = [node_n] * copy_num
    multi_ni_k = [ni_k] * copy_num
    multi_ni_n = [ni_n] * copy_num
    multi_cfg_topology = [cfg_topology] * copy_num
    multi_cfg_filepath = [cfg_filepath] * copy_num

    with multiprocessing.Pool() as pool:
        multi_node_network = pool.starmap(build, zip(multi_node_k, multi_node_n, multi_ni_k, multi_ni_n, multi_cfg_topology))

    for single_node_network in multi_node_network:
        single_node_network.create_nodes()

    with multiprocessing.Pool() as pool:
        multi_noc_network = pool.map(BookSim_Interface, multi_cfg_filepath)

    return multi_node_network, multi_noc_network


def schedule(
    whole_nodes: NodeNetwork
):
    '''
    user logic
    update events here
    only dependency is the most important 
    '''
    
    return 0


def launch(
    whole_nodes: NodeNetwork, booksim_net: BookSim_Interface, 
    print_flag=True, booksim2_flit_units=256
) -> int:

    if print_flag:
        print("====================================")
        whole_nodes.show_nodes_events()

    print("====================================")
    print("Begin to build NIs...")
    whole_nodes.build_nis_events()

    if print_flag:
        print("====================================")
        whole_nodes.show_nis_events()

    print("====================================")
    print("Begin to run...")
    end_cycle = run(whole_nodes=whole_nodes, current_cycle=0, network=booksim_net, print_flag=print_flag, booksim2_flit_units=booksim2_flit_units)
    print("************************************")    
    print(f"end_cycle: {end_cycle}")
    print("************************************")

    return end_cycle


def launch_multiprocessing(
    multi_whole_nodes: List[NodeNetwork], multi_booksim_net: List[BookSim_Interface],
    print_flag=True
):
    
    num_cores = multiprocessing.cpu_count()
    if print_flag:
        print("====================================")
        print("available cores: ", num_cores)
        print("taks number: ", len(multi_whole_nodes))

    multi_print_flag = [print_flag] * len(multi_whole_nodes)

    with multiprocessing.Pool(processes=num_cores) as pool:
        end_cycles = pool.starmap(launch, zip(multi_whole_nodes, multi_booksim_net, multi_print_flag))

    return end_cycles



