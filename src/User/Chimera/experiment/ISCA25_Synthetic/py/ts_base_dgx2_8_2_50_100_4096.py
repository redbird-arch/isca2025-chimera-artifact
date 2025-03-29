
import os
import sys
file_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(file_path)
sys.path.append(os.path.join(file_path, '../../../parallism/'))
sys.path.append(os.path.join(file_path, '../../../../../components/'))


from tensor_pipeline import tensor_pipeline
from tensor_expert import tensor_expert
from tensor_sequence import tensor_sequence
from pipeline_expert import pipeline_expert
from pipeline_sequence import pipeline_sequence
from sequence_expert import sequence_expert

from Launcher import build, launch


node_k = 8
node_n = 2
ni_k = node_k
ni_n = node_n
cfg_topology = "dgx2"
algorithm_dict = {'allgather': 'havlingdoubling', 'allreduce': 'havlingdoubling', 'reducescatter': 'havlingdoubling', 'reducelocal': 'base', 'alltoall': 'havlingdoubling', 'ordertoorder': 'havlingdoubling', 'pointtopoint': 'base', 'manytomanymulticast': 'alpa'}

source_nodes = [(0, 0), (0, 1), (0, 2), (0, 3)]
source_shape = [1, 4]
medium_nodes = None
medium_shape = [0, 0]
target_nodes = [(0, 4), (0, 5), (0, 6), (0, 7)]
target_shape = [1, 4]
data_parallelism_degree = None
top_k = None

whole_flits = 4096
reduction_cores = 9.556025235334156e-14

node_network, noc_network = build(
    node_k=node_k, node_n=node_k, ni_k=ni_k, ni_n=ni_k, 
    cfg_topology=cfg_topology, 
    cfg_filepath=os.path.join(file_path, "../cfg/dgx2_8_2_50_100.cfg")
)

ts = tensor_sequence(
    topology=("dgx2"), 
    algorithm=algorithm_dict
)

initial_event_tag = 0
initial_dependency_list = [[[] for _ in range(source_shape[1])] for _ in range(source_shape[0])]

ts_base_event_tag, ts_base_dependency_list = ts.base(
    whole_nodes=node_network, current_event_tag=initial_event_tag, current_dependency_list=initial_dependency_list,
    source_nodes_coordinates_list=source_nodes,
    medium_nodes_coordinates_list=medium_nodes,
    target_nodes_coordinates_list=target_nodes,
    source_x_number=source_shape[1], source_y_number=source_shape[0],
    medium_x_number=medium_shape[1], medium_y_number=medium_shape[0],
    target_x_number=target_shape[1], target_y_number=target_shape[0],
    topology_x_limitation=node_k, topology_y_limitation=node_n,
    data_parallelism_degree=data_parallelism_degree, top_k=top_k, 
    message_flits=whole_flits, 
    latency=None, bandwidth=None, reduction=reduction_cores
)

ts_base_run_cost = launch(
    whole_nodes=node_network, booksim_net=noc_network
)

with open(os.path.join(file_path,"../txt/"+os.path.splitext(os.path.basename(__file__))[0]+'.txt'), 'w') as f:
    # pickle.dump(ts_base_run_cost, f)
    f.write(str(ts_base_run_cost))


    