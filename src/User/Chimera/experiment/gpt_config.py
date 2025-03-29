# File name  :    gpt_config.py
# Author     :    xiaocuicui
# Time       :    2025/02/17 13:38:54
# Version    :    V1.0
# Abstract   :        

import os
import sys
file_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(file_path, '../'))

a100_mapping = '''clusters_number = 8

cluster0_nodes_list = [(0, 0), (0, 1)]
cluster1_nodes_list = [(0, 2), (0, 3)]
cluster2_nodes_list = [(0, 4), (0, 5)]
cluster3_nodes_list = [(0, 6), (0, 7)]
cluster4_nodes_list = [(1, 6), (1, 7)]
cluster5_nodes_list = [(1, 4), (1, 5)]
cluster6_nodes_list = [(1, 2), (1, 3)]
cluster7_nodes_list = [(1, 0), (1, 1)]

cluster0_shape = [1, 2]
cluster1_shape = [1, 2]
cluster2_shape = [1, 2]
cluster3_shape = [1, 2]
cluster4_shape = [1, 2]
cluster5_shape = [1, 2]
cluster6_shape = [1, 2]
cluster7_shape = [1, 2]

cluster_nodes_lists = [cluster0_nodes_list, cluster1_nodes_list, cluster2_nodes_list, cluster3_nodes_list, cluster4_nodes_list, cluster5_nodes_list, cluster6_nodes_list, cluster7_nodes_list]
cluster_shapes = [cluster0_shape, cluster1_shape, cluster2_shape, cluster3_shape, cluster4_shape, cluster5_shape, cluster6_shape, cluster7_shape]



node_k = 8
node_n = 2
ni_k = node_k
ni_n = node_n
cfg_topology = "dgx2"
algorithm_dict = {'allgather': 'havlingdoubling', 'allreduce': 'havlingdoubling', 'reducescatter': 'havlingdoubling', 'reducelocal': 'base', 'alltoall': 'havlingdoubling', 'ordertoorder': 'havlingdoubling', 'pointtopoint': 'base', 'manytomanymulticast': 'alpa'}

node_network, noc_network = build(
    node_k=node_k, node_n=node_n, ni_k=ni_k, ni_n=ni_n,
    cfg_topology=cfg_topology,
    cfg_filepath=os.path.join(file_path, "../cfg/a100.cfg")
)

tp = tensor_pipeline(
    topology=("dgx2"),
    algorithm=algorithm_dict
)

allreduce_api = allreduce(
    topology="dgx2",
    algorithm=algorithm_dict["allreduce"]
)

reducescatter_api = reducescatter(
    topology="dgx2",
    algorithm=algorithm_dict["reducescatter"]
)

allgather_api = allgather(
    topology="dgx2",
    algorithm=algorithm_dict["allgather"]
)

pointtopoint_api = pointtopoint(
    topology="dgx2",
    algorithm=algorithm_dict["pointtopoint"]
)

manytomany_api = manytomanymulticast(
    topology="dgx2",
    algorithm=algorithm_dict["manytomanymulticast"]
)


initial_event_tag = 0
initial_cluster0_dependency_list = [[[] for _ in range(cluster0_shape[1])] for _ in range(cluster0_shape[0])]
initial_cluster1_dependency_list = [[[] for _ in range(cluster1_shape[1])] for _ in range(cluster1_shape[0])]
initial_cluster2_dependency_list = [[[] for _ in range(cluster2_shape[1])] for _ in range(cluster2_shape[0])]
initial_cluster3_dependency_list = [[[] for _ in range(cluster3_shape[1])] for _ in range(cluster3_shape[0])]
initial_cluster4_dependency_list = [[[] for _ in range(cluster4_shape[1])] for _ in range(cluster4_shape[0])]
initial_cluster5_dependency_list = [[[] for _ in range(cluster5_shape[1])] for _ in range(cluster5_shape[0])]
initial_cluster6_dependency_list = [[[] for _ in range(cluster6_shape[1])] for _ in range(cluster6_shape[0])]
initial_cluster7_dependency_list = [[[] for _ in range(cluster7_shape[1])] for _ in range(cluster7_shape[0])]

initial_cluster_dependency_lists = [initial_cluster0_dependency_list, initial_cluster1_dependency_list, initial_cluster2_dependency_list, initial_cluster3_dependency_list, initial_cluster4_dependency_list, initial_cluster5_dependency_list, initial_cluster6_dependency_list, initial_cluster7_dependency_list]
'''

a100_part = '''for cluster_idx, cluster_step in enumerate([2, 3, 4]):
    for layer_step in range(2):'''

a100_cal = '''intra_0_time = segment_time(finished_events_list, record_event_tags, [-1], [0])
intra_1_time = segment_time(finished_events_list, record_event_tags, [0], [1])
intra_2_time = segment_time(finished_events_list, record_event_tags, [2], [3])
intra_3_time = segment_time(finished_events_list, record_event_tags, [5], [6])
inter_0_time = segment_time(finished_events_list, record_event_tags, [1], [2])
inter_1_time = segment_time(finished_events_list, record_event_tags, [4], [5])

whole_time = intra_0_time + intra_2_time * (clusters_number - 2) + intra_3_time + intra_1_time * (model_config['n_layer'] - clusters_number) + inter_0_time * (clusters_number - 2) + inter_1_time
'''




dojo1_mapping = '''clusters_number = 12

cluster0_nodes_list = [(0, 0), (0, 1)]
cluster1_nodes_list = [(1, 0), (1, 1)]
cluster2_nodes_list = [(2, 0), (2, 1)]
cluster3_nodes_list = [(3, 0), (3, 1)]
cluster4_nodes_list = [(4, 0), (4, 1)]
cluster5_nodes_list = [(4, 2), (4, 3)]
cluster6_nodes_list = [(3, 2), (3, 3)]
cluster7_nodes_list = [(2, 2), (2, 3)]
cluster8_nodes_list = [(1, 2), (1, 3)]
cluster9_nodes_list = [(0, 2), (0, 3)]
cluster10_nodes_list = [(0, 4), (1, 4)]
cluster11_nodes_list = [(2, 4), (3, 4)]

cluster0_shape = [1, 2]
cluster1_shape = [1, 2]
cluster2_shape = [1, 2]
cluster3_shape = [1, 2]
cluster4_shape = [1, 2]
cluster5_shape = [1, 2]
cluster6_shape = [1, 2]
cluster7_shape = [1, 2]
cluster8_shape = [1, 2]
cluster9_shape = [1, 2]
cluster10_shape = [2, 1]
cluster11_shape = [2, 1]

cluster_nodes_lists = [cluster0_nodes_list, cluster1_nodes_list, cluster2_nodes_list, cluster3_nodes_list, cluster4_nodes_list, cluster5_nodes_list, cluster6_nodes_list, cluster7_nodes_list, cluster8_nodes_list, cluster9_nodes_list, cluster10_nodes_list, cluster11_nodes_list]
cluster_shapes = [cluster0_shape, cluster1_shape, cluster2_shape, cluster3_shape, cluster4_shape, cluster5_shape, cluster6_shape, cluster7_shape, cluster8_shape, cluster9_shape, cluster10_shape, cluster11_shape]




node_k = 5
node_n = 2
ni_k = node_k
ni_n = node_n
cfg_topology = "mesh"
algorithm_dict = {'allgather': 'hierarchicalring', 'allreduce': 'hierarchicalring', 'reducescatter': 'hierarchicalring', 'reducelocal': 'base', 'alltoall': 'hierarchicalring', 'ordertoorder': 'hierarchicalring', 'pointtopoint': 'base', 'manytomanymulticast': 'alpa'}

node_network, noc_network = build(
    node_k=node_k, node_n=node_k, ni_k=ni_k, ni_n=ni_k,
    cfg_topology=cfg_topology,
    cfg_filepath=os.path.join(file_path, "../cfg/dojo.cfg")
)

tp = tensor_pipeline(
    topology=("mesh2d"),
    algorithm=algorithm_dict
)

allreduce_api = allreduce(
    topology="mesh2d",
    algorithm=algorithm_dict["allreduce"]
)

reducescatter_api = reducescatter(
    topology="mesh2d",
    algorithm=algorithm_dict["reducescatter"]
)

allgather_api = allgather(
    topology="mesh2d",
    algorithm=algorithm_dict["allgather"]
)

pointtopoint_api = pointtopoint(
    topology="mesh2d",
    algorithm=algorithm_dict["pointtopoint"]
)

manytomany_api = manytomanymulticast(
    topology="mesh2d",
    algorithm=algorithm_dict["manytomanymulticast"]
)


initial_event_tag = 0
initial_cluster0_dependency_list = [[[] for _ in range(cluster0_shape[1])] for _ in range(cluster0_shape[0])]
initial_cluster1_dependency_list = [[[] for _ in range(cluster1_shape[1])] for _ in range(cluster1_shape[0])]
initial_cluster2_dependency_list = [[[] for _ in range(cluster2_shape[1])] for _ in range(cluster2_shape[0])]
initial_cluster3_dependency_list = [[[] for _ in range(cluster3_shape[1])] for _ in range(cluster3_shape[0])]
initial_cluster4_dependency_list = [[[] for _ in range(cluster4_shape[1])] for _ in range(cluster4_shape[0])]
initial_cluster5_dependency_list = [[[] for _ in range(cluster5_shape[1])] for _ in range(cluster5_shape[0])]
initial_cluster6_dependency_list = [[[] for _ in range(cluster6_shape[1])] for _ in range(cluster6_shape[0])]
initial_cluster7_dependency_list = [[[] for _ in range(cluster7_shape[1])] for _ in range(cluster7_shape[0])]
initial_cluster8_dependency_list = [[[] for _ in range(cluster8_shape[1])] for _ in range(cluster8_shape[0])]
initial_cluster9_dependency_list = [[[] for _ in range(cluster9_shape[1])] for _ in range(cluster9_shape[0])]
initial_cluster10_dependency_list = [[[] for _ in range(cluster10_shape[1])] for _ in range(cluster10_shape[0])]
initial_cluster11_dependency_list = [[[] for _ in range(cluster11_shape[1])] for _ in range(cluster11_shape[0])]

initial_cluster_dependency_lists = [initial_cluster0_dependency_list, initial_cluster1_dependency_list, initial_cluster2_dependency_list, initial_cluster3_dependency_list, initial_cluster4_dependency_list, initial_cluster5_dependency_list, initial_cluster6_dependency_list, initial_cluster7_dependency_list, initial_cluster8_dependency_list, initial_cluster9_dependency_list, initial_cluster10_dependency_list, initial_cluster11_dependency_list]
'''

dojo1_part = '''for cluster_idx, cluster_step in enumerate([4, 5, 6]):
    for layer_step in range(2):'''

dojo1_cal = '''intra_0_time = segment_time(finished_events_list, record_event_tags, [-1], [0])
intra_1_time = segment_time(finished_events_list, record_event_tags, [0], [1])
intra_2_time = segment_time(finished_events_list, record_event_tags, [2], [3])
intra_3_time = segment_time(finished_events_list, record_event_tags, [5], [6])
inter_0_time = segment_time(finished_events_list, record_event_tags, [1], [2])
inter_1_time = segment_time(finished_events_list, record_event_tags, [4], [5])

whole_time = intra_0_time + intra_2_time * (9 - 2) + intra_3_time + intra_1_time * (18 - 9) + inter_0_time * (9 - 2) + inter_1_time
'''




dojo2_mapping = '''clusters_number = 12

cluster0_nodes_list = [(0, 0), (0, 1)]
cluster1_nodes_list = [(1, 0), (1, 1)]
cluster2_nodes_list = [(2, 0), (2, 1)]
cluster3_nodes_list = [(3, 0), (3, 1)]
cluster4_nodes_list = [(4, 0), (4, 1)]
cluster5_nodes_list = [(4, 2), (4, 3)]
cluster6_nodes_list = [(3, 2), (3, 3)]
cluster7_nodes_list = [(2, 2), (2, 3)]
cluster8_nodes_list = [(1, 2), (1, 3)]
cluster9_nodes_list = [(0, 2), (0, 3)]
cluster10_nodes_list = [(0, 4), (1, 4)]
cluster11_nodes_list = [(2, 4), (3, 4)]

cluster0_shape = [1, 2]
cluster1_shape = [1, 2]
cluster2_shape = [1, 2]
cluster3_shape = [1, 2]
cluster4_shape = [1, 2]
cluster5_shape = [1, 2]
cluster6_shape = [1, 2]
cluster7_shape = [1, 2]
cluster8_shape = [1, 2]
cluster9_shape = [1, 2]
cluster10_shape = [2, 1]
cluster11_shape = [2, 1]

cluster_nodes_lists = [cluster0_nodes_list, cluster1_nodes_list, cluster2_nodes_list, cluster3_nodes_list, cluster4_nodes_list, cluster5_nodes_list, cluster6_nodes_list, cluster7_nodes_list, cluster8_nodes_list, cluster9_nodes_list, cluster10_nodes_list, cluster11_nodes_list]
cluster_shapes = [cluster0_shape, cluster1_shape, cluster2_shape, cluster3_shape, cluster4_shape, cluster5_shape, cluster6_shape, cluster7_shape, cluster8_shape, cluster9_shape, cluster10_shape, cluster11_shape]



node_k = 5
node_n = 2
ni_k = node_k
ni_n = node_n
cfg_topology = "mesh"
algorithm_dict = {'allgather': 'hierarchicalring', 'allreduce': 'hierarchicalring', 'reducescatter': 'hierarchicalring', 'reducelocal': 'base', 'alltoall': 'hierarchicalring', 'ordertoorder': 'hierarchicalring', 'pointtopoint': 'base', 'manytomanymulticast': 'alpa'}

node_network, noc_network = build(
    node_k=node_k, node_n=node_k, ni_k=ni_k, ni_n=ni_k,
    cfg_topology=cfg_topology,
    cfg_filepath=os.path.join(file_path, "../cfg/dojo.cfg")
)

tp = tensor_pipeline(
    topology=("mesh2d"),
    algorithm=algorithm_dict
)

allreduce_api = allreduce(
    topology="mesh2d",
    algorithm=algorithm_dict["allreduce"]
)

reducescatter_api = reducescatter(
    topology="mesh2d",
    algorithm=algorithm_dict["reducescatter"]
)

allgather_api = allgather(
    topology="mesh2d",
    algorithm=algorithm_dict["allgather"]
)

pointtopoint_api = pointtopoint(
    topology="mesh2d",
    algorithm=algorithm_dict["pointtopoint"]
)

manytomany_api = manytomanymulticast(
    topology="mesh2d",
    algorithm=algorithm_dict["manytomanymulticast"]
)


initial_event_tag = 0
initial_cluster0_dependency_list = [[[] for _ in range(cluster0_shape[1])] for _ in range(cluster0_shape[0])]
initial_cluster1_dependency_list = [[[] for _ in range(cluster1_shape[1])] for _ in range(cluster1_shape[0])]
initial_cluster2_dependency_list = [[[] for _ in range(cluster2_shape[1])] for _ in range(cluster2_shape[0])]
initial_cluster3_dependency_list = [[[] for _ in range(cluster3_shape[1])] for _ in range(cluster3_shape[0])]
initial_cluster4_dependency_list = [[[] for _ in range(cluster4_shape[1])] for _ in range(cluster4_shape[0])]
initial_cluster5_dependency_list = [[[] for _ in range(cluster5_shape[1])] for _ in range(cluster5_shape[0])]
initial_cluster6_dependency_list = [[[] for _ in range(cluster6_shape[1])] for _ in range(cluster6_shape[0])]
initial_cluster7_dependency_list = [[[] for _ in range(cluster7_shape[1])] for _ in range(cluster7_shape[0])]
initial_cluster8_dependency_list = [[[] for _ in range(cluster8_shape[1])] for _ in range(cluster8_shape[0])]
initial_cluster9_dependency_list = [[[] for _ in range(cluster9_shape[1])] for _ in range(cluster9_shape[0])]
initial_cluster10_dependency_list = [[[] for _ in range(cluster10_shape[1])] for _ in range(cluster10_shape[0])]
initial_cluster11_dependency_list = [[[] for _ in range(cluster11_shape[1])] for _ in range(cluster11_shape[0])]

initial_cluster_dependency_lists = [initial_cluster0_dependency_list, initial_cluster1_dependency_list, initial_cluster2_dependency_list, initial_cluster3_dependency_list, initial_cluster4_dependency_list, initial_cluster5_dependency_list, initial_cluster6_dependency_list, initial_cluster7_dependency_list, initial_cluster8_dependency_list, initial_cluster9_dependency_list, initial_cluster10_dependency_list, initial_cluster11_dependency_list]
'''

dojo2_part = '''for cluster_idx, cluster_step in enumerate([9, 10, 11]):
    for layer_step in range(2):'''

dojo2_cal = '''intra_0_time = segment_time(finished_events_list, record_event_tags, [-1], [0])
intra_1_time = segment_time(finished_events_list, record_event_tags, [0], [1])
intra_2_time = segment_time(finished_events_list, record_event_tags, [2], [3])
intra_3_time = segment_time(finished_events_list, record_event_tags, [5], [6])
inter_0_time = segment_time(finished_events_list, record_event_tags, [1], [2])
inter_1_time = segment_time(finished_events_list, record_event_tags, [4], [5])

whole_time = intra_0_time + intra_2_time * (3 - 2) + intra_3_time + intra_1_time * (6 - 3) + inter_0_time * (3 - 2) + inter_1_time
'''




tpuv3_mapping = '''clusters_number = 8

cluster0_nodes_list = [(0, 0), (0, 1)]
cluster1_nodes_list = [(1, 0), (1, 1)]
cluster2_nodes_list = [(2, 0), (2, 1)]
cluster3_nodes_list = [(3, 0), (3, 1)]
cluster4_nodes_list = [(3, 2), (3, 3)]
cluster5_nodes_list = [(2, 2), (2, 3)]
cluster6_nodes_list = [(1, 2), (1, 3)]
cluster7_nodes_list = [(0, 2), (0, 3)]

cluster0_shape = [1, 2]
cluster1_shape = [1, 2]
cluster2_shape = [1, 2]
cluster3_shape = [1, 2]
cluster4_shape = [1, 2]
cluster5_shape = [1, 2]
cluster6_shape = [1, 2]
cluster7_shape = [1, 2]

cluster_nodes_lists = [cluster0_nodes_list, cluster1_nodes_list, cluster2_nodes_list, cluster3_nodes_list, cluster4_nodes_list, cluster5_nodes_list, cluster6_nodes_list, cluster7_nodes_list]
cluster_shapes = [cluster0_shape, cluster1_shape, cluster2_shape, cluster3_shape, cluster4_shape, cluster5_shape, cluster6_shape, cluster7_shape]



node_k = 4
node_n = 2
ni_k = node_k
ni_n = node_n
cfg_topology = "torus"
algorithm_dict = {'allgather': 'hierarchicalring', 'allreduce': 'hierarchicalring', 'reducescatter': 'hierarchicalring', 'reducelocal': 'base', 'alltoall': 'hierarchicalring', 'ordertoorder': 'hierarchicalring', 'pointtopoint': 'base', 'manytomanymulticast': 'alpa'}

node_network, noc_network = build(
    node_k=node_k, node_n=node_k, ni_k=ni_k, ni_n=ni_k,
    cfg_topology=cfg_topology,
    cfg_filepath=os.path.join(file_path, "../cfg/tpuv3.cfg")
)

tp = tensor_pipeline(
    topology=("torus2d"),
    algorithm=algorithm_dict
)

allreduce_api = allreduce(
    topology="torus2d",
    algorithm=algorithm_dict["allreduce"]
)

reducescatter_api = reducescatter(
    topology="torus2d",
    algorithm=algorithm_dict["reducescatter"]
)

allgather_api = allgather(
    topology="torus2d",
    algorithm=algorithm_dict["allgather"]
)

pointtopoint_api = pointtopoint(
    topology="torus2d",
    algorithm=algorithm_dict["pointtopoint"]
)

manytomany_api = manytomanymulticast(
    topology="torus2d",
    algorithm=algorithm_dict["manytomanymulticast"]
)


initial_event_tag = 0
initial_cluster0_dependency_list = [[[] for _ in range(cluster0_shape[1])] for _ in range(cluster0_shape[0])]
initial_cluster1_dependency_list = [[[] for _ in range(cluster1_shape[1])] for _ in range(cluster1_shape[0])]
initial_cluster2_dependency_list = [[[] for _ in range(cluster2_shape[1])] for _ in range(cluster2_shape[0])]
initial_cluster3_dependency_list = [[[] for _ in range(cluster3_shape[1])] for _ in range(cluster3_shape[0])]
initial_cluster4_dependency_list = [[[] for _ in range(cluster4_shape[1])] for _ in range(cluster4_shape[0])]
initial_cluster5_dependency_list = [[[] for _ in range(cluster5_shape[1])] for _ in range(cluster5_shape[0])]
initial_cluster6_dependency_list = [[[] for _ in range(cluster6_shape[1])] for _ in range(cluster6_shape[0])]
initial_cluster7_dependency_list = [[[] for _ in range(cluster7_shape[1])] for _ in range(cluster7_shape[0])]

initial_cluster_dependency_lists = [initial_cluster0_dependency_list, initial_cluster1_dependency_list, initial_cluster2_dependency_list, initial_cluster3_dependency_list, initial_cluster4_dependency_list, initial_cluster5_dependency_list, initial_cluster6_dependency_list, initial_cluster7_dependency_list]
'''

tpuv3_part = '''for cluster_idx, cluster_step in enumerate([2, 3, 4]):
    for layer_step in range(2):'''

tpuv3_cal = '''intra_0_time = segment_time(finished_events_list, record_event_tags, [-1], [0])
intra_1_time = segment_time(finished_events_list, record_event_tags, [0], [1])
intra_2_time = segment_time(finished_events_list, record_event_tags, [2], [3])
intra_3_time = segment_time(finished_events_list, record_event_tags, [5], [6])
inter_0_time = segment_time(finished_events_list, record_event_tags, [1], [2])
inter_1_time = segment_time(finished_events_list, record_event_tags, [4], [5])

whole_time = intra_0_time + intra_2_time * (clusters_number - 2) + intra_3_time + intra_1_time * (model_config['n_layer'] - clusters_number) + inter_0_time * (clusters_number - 2) + inter_1_time
'''




a100_scaling_mapping = '''clusters_number = 8

cluster0_nodes_list = [(0, 0), (0, 1), (0, 2), (0, 3), (0, 4), (0, 5), (0, 6), (0, 7)]
cluster1_nodes_list = [(0, 8), (0, 9), (0, 10), (0, 11), (0, 12), (0, 13), (0, 14), (0, 15)]
cluster2_nodes_list = [(0, 16), (0, 17), (0, 18), (0, 19), (0, 20), (0, 21), (0, 22), (0, 23)]
cluster3_nodes_list = [(0, 24), (0, 25), (0, 26), (0, 27), (0, 28), (0, 29), (0, 30), (0, 31)]
cluster4_nodes_list = [(1, 0), (1, 1), (1, 2), (1, 3), (1, 4), (1, 5), (1, 6), (1, 7)]
cluster5_nodes_list = [(1, 8), (1, 9), (1, 10), (1, 11), (1, 12), (1, 13), (1, 14), (1, 15)]
cluster6_nodes_list = [(1, 16), (1, 17), (1, 18), (1, 19), (1, 20), (1, 21), (1, 22), (1, 23)]
cluster7_nodes_list = [(1, 24), (1, 25), (1, 26), (1, 27), (1, 28), (1, 29), (1, 30), (1, 31)]

cluster0_shape = [1, 8]
cluster1_shape = [1, 8]
cluster2_shape = [1, 8]
cluster3_shape = [1, 8]
cluster4_shape = [1, 8]
cluster5_shape = [1, 8]
cluster6_shape = [1, 8]
cluster7_shape = [1, 8]

cluster_nodes_lists = [cluster0_nodes_list, cluster1_nodes_list, cluster2_nodes_list, cluster3_nodes_list, cluster4_nodes_list, cluster5_nodes_list, cluster6_nodes_list, cluster7_nodes_list]
cluster_shapes = [cluster0_shape, cluster1_shape, cluster2_shape, cluster3_shape, cluster4_shape, cluster5_shape, cluster6_shape, cluster7_shape]



node_k = 32
node_n = 2
ni_k = node_k
ni_n = node_n
cfg_topology = "dgx2"
algorithm_dict = {'allgather': 'havlingdoubling', 'allreduce': 'havlingdoubling', 'reducescatter': 'havlingdoubling', 'reducelocal': 'base', 'alltoall': 'havlingdoubling', 'ordertoorder': 'havlingdoubling', 'pointtopoint': 'base', 'manytomanymulticast': 'alpa'}

node_network, noc_network = build(
    node_k=node_k, node_n=node_n, ni_k=ni_k, ni_n=ni_n,
    cfg_topology=cfg_topology,
    cfg_filepath=os.path.join(file_path, "../cfg/a100_scaling.cfg")
)

tp = tensor_pipeline(
    topology=("dgx2"),
    algorithm=algorithm_dict
)

allreduce_api = allreduce(
    topology="dgx2",
    algorithm=algorithm_dict["allreduce"]
)

reducescatter_api = reducescatter(
    topology="dgx2",
    algorithm=algorithm_dict["reducescatter"]
)

allgather_api = allgather(
    topology="dgx2",
    algorithm=algorithm_dict["allgather"]
)

pointtopoint_api = pointtopoint(
    topology="dgx2",
    algorithm=algorithm_dict["pointtopoint"]
)

manytomany_api = manytomanymulticast(
    topology="dgx2",
    algorithm=algorithm_dict["manytomanymulticast"]
)


initial_event_tag = 0
initial_cluster0_dependency_list = [[[] for _ in range(cluster0_shape[1])] for _ in range(cluster0_shape[0])]
initial_cluster1_dependency_list = [[[] for _ in range(cluster1_shape[1])] for _ in range(cluster1_shape[0])]
initial_cluster2_dependency_list = [[[] for _ in range(cluster2_shape[1])] for _ in range(cluster2_shape[0])]
initial_cluster3_dependency_list = [[[] for _ in range(cluster3_shape[1])] for _ in range(cluster3_shape[0])]
initial_cluster4_dependency_list = [[[] for _ in range(cluster4_shape[1])] for _ in range(cluster4_shape[0])]
initial_cluster5_dependency_list = [[[] for _ in range(cluster5_shape[1])] for _ in range(cluster5_shape[0])]
initial_cluster6_dependency_list = [[[] for _ in range(cluster6_shape[1])] for _ in range(cluster6_shape[0])]
initial_cluster7_dependency_list = [[[] for _ in range(cluster7_shape[1])] for _ in range(cluster7_shape[0])]

initial_cluster_dependency_lists = [initial_cluster0_dependency_list, initial_cluster1_dependency_list, initial_cluster2_dependency_list, initial_cluster3_dependency_list, initial_cluster4_dependency_list, initial_cluster5_dependency_list, initial_cluster6_dependency_list, initial_cluster7_dependency_list]
'''

a100_scaling_part = '''for cluster_idx, cluster_step in enumerate([2, 3, 4]):
    for layer_step in range(2):'''

a100_scaling_cal = '''intra_0_time = segment_time(finished_events_list, record_event_tags, [-1], [0])
intra_1_time = segment_time(finished_events_list, record_event_tags, [0], [1])
intra_2_time = segment_time(finished_events_list, record_event_tags, [2], [3])
intra_3_time = segment_time(finished_events_list, record_event_tags, [5], [6])
inter_0_time = segment_time(finished_events_list, record_event_tags, [1], [2])
inter_1_time = segment_time(finished_events_list, record_event_tags, [4], [5])

whole_time = intra_0_time + intra_2_time * (clusters_number - 2) + intra_3_time + intra_1_time * (model_config['n_layer'] - clusters_number) + inter_0_time * (clusters_number - 2) + inter_1_time
'''




dojo_scaling_mapping = '''clusters_number = 8

cluster0_nodes_list = [(0, 0), (0, 1), (0, 2), (0, 3), (1, 0), (1, 1), (1, 2), (1, 3)]
cluster1_nodes_list = [(2, 0), (2, 1), (2, 2), (2, 3), (3, 0), (3, 1), (3, 2), (3, 3)]
cluster2_nodes_list = [(4, 0), (4, 1), (4, 2), (4, 3), (5, 0), (5, 1), (5, 2), (5, 3)]
cluster3_nodes_list = [(6, 0), (6, 1), (6, 2), (6, 3), (7, 0), (7, 1), (7, 2), (7, 3)]
cluster4_nodes_list = [(6, 4), (6, 5), (6, 6), (6, 7), (7, 4), (7, 5), (7, 6), (7, 7)]
cluster5_nodes_list = [(4, 4), (4, 5), (4, 6), (4, 7), (5, 4), (5, 5), (5, 6), (5, 7)]
cluster6_nodes_list = [(2, 4), (2, 5), (2, 6), (2, 7), (3, 4), (3, 5), (3, 6), (3, 7)]
cluster7_nodes_list = [(0, 4), (0, 5), (0, 6), (0, 7), (1, 4), (1, 5), (1, 6), (1, 7)]

cluster0_shape = [2, 4]
cluster1_shape = [2, 4]
cluster2_shape = [2, 4]
cluster3_shape = [2, 4]
cluster4_shape = [2, 4]
cluster5_shape = [2, 4]
cluster6_shape = [2, 4]
cluster7_shape = [2, 4]

cluster_nodes_lists = [cluster0_nodes_list, cluster1_nodes_list, cluster2_nodes_list, cluster3_nodes_list, cluster4_nodes_list, cluster5_nodes_list, cluster6_nodes_list, cluster7_nodes_list]
cluster_shapes = [cluster0_shape, cluster1_shape, cluster2_shape, cluster3_shape, cluster4_shape, cluster5_shape, cluster6_shape, cluster7_shape]



node_k = 8
node_n = 2
ni_k = node_k
ni_n = node_n
cfg_topology = "mesh"
algorithm_dict = {'allgather': 'hierarchicalring', 'allreduce': 'hierarchicalring', 'reducescatter': 'hierarchicalring', 'reducelocal': 'base', 'alltoall': 'hierarchicalring', 'ordertoorder': 'hierarchicalring', 'pointtopoint': 'base', 'manytomanymulticast': 'alpa'}

node_network, noc_network = build(
    node_k=node_k, node_n=node_k, ni_k=ni_k, ni_n=ni_k,
    cfg_topology=cfg_topology,
    cfg_filepath=os.path.join(file_path, "../cfg/dojo_scaling.cfg")
)

tp = tensor_pipeline(
    topology=("mesh2d"),
    algorithm=algorithm_dict
)

allreduce_api = allreduce(
    topology="mesh2d",
    algorithm=algorithm_dict["allreduce"]
)

reducescatter_api = reducescatter(
    topology="mesh2d",
    algorithm=algorithm_dict["reducescatter"]
)

allgather_api = allgather(
    topology="mesh2d",
    algorithm=algorithm_dict["allgather"]
)

pointtopoint_api = pointtopoint(
    topology="mesh2d",
    algorithm=algorithm_dict["pointtopoint"]
)

manytomany_api = manytomanymulticast(
    topology="mesh2d",
    algorithm=algorithm_dict["manytomanymulticast"]
)


initial_event_tag = 0
initial_cluster0_dependency_list = [[[] for _ in range(cluster0_shape[1])] for _ in range(cluster0_shape[0])]
initial_cluster1_dependency_list = [[[] for _ in range(cluster1_shape[1])] for _ in range(cluster1_shape[0])]
initial_cluster2_dependency_list = [[[] for _ in range(cluster2_shape[1])] for _ in range(cluster2_shape[0])]
initial_cluster3_dependency_list = [[[] for _ in range(cluster3_shape[1])] for _ in range(cluster3_shape[0])]
initial_cluster4_dependency_list = [[[] for _ in range(cluster4_shape[1])] for _ in range(cluster4_shape[0])]
initial_cluster5_dependency_list = [[[] for _ in range(cluster5_shape[1])] for _ in range(cluster5_shape[0])]
initial_cluster6_dependency_list = [[[] for _ in range(cluster6_shape[1])] for _ in range(cluster6_shape[0])]
initial_cluster7_dependency_list = [[[] for _ in range(cluster7_shape[1])] for _ in range(cluster7_shape[0])]

initial_cluster_dependency_lists = [initial_cluster0_dependency_list, initial_cluster1_dependency_list, initial_cluster2_dependency_list, initial_cluster3_dependency_list, initial_cluster4_dependency_list, initial_cluster5_dependency_list, initial_cluster6_dependency_list, initial_cluster7_dependency_list]

'''

dojo_scaling_part = '''for cluster_idx, cluster_step in enumerate([3, 4, 5]):
    for layer_step in range(2):'''

dojo_scaling_cal = '''intra_0_time = segment_time(finished_events_list, record_event_tags, [-1], [0])
intra_1_time = segment_time(finished_events_list, record_event_tags, [0], [1])
intra_2_time = segment_time(finished_events_list, record_event_tags, [2], [3])
intra_3_time = segment_time(finished_events_list, record_event_tags, [5], [6])
inter_0_time = segment_time(finished_events_list, record_event_tags, [1], [2])
inter_1_time = segment_time(finished_events_list, record_event_tags, [4], [5])

whole_time = intra_0_time + intra_2_time * (clusters_number - 2) + intra_3_time + intra_1_time * (model_config['n_layer'] - clusters_number) + inter_0_time * (clusters_number - 2) + inter_1_time
'''




tpuv3_scaling_mapping = '''clusters_number = 8

cluster0_nodes_list = [(0, 0), (0, 1), (0, 2), (0, 3), (0, 4), (0, 5), (0, 6), (0, 7)]
cluster1_nodes_list = [(1, 0), (1, 1), (1, 2), (1, 3), (1, 4), (1, 5), (1, 6), (1, 7)]
cluster2_nodes_list = [(2, 0), (2, 1), (2, 2), (2, 3), (2, 4), (2, 5), (2, 6), (2, 7)]
cluster3_nodes_list = [(3, 0), (3, 1), (3, 2), (3, 3), (3, 4), (3, 5), (3, 6), (3, 7)]
cluster4_nodes_list = [(4, 0), (4, 1), (4, 2), (4, 3), (4, 4), (4, 5), (4, 6), (4, 7)]
cluster5_nodes_list = [(5, 0), (5, 1), (5, 2), (5, 3), (5, 4), (5, 5), (5, 6), (5, 7)]
cluster6_nodes_list = [(6, 0), (6, 1), (6, 2), (6, 3), (6, 4), (6, 5), (6, 6), (6, 7)]
cluster7_nodes_list = [(7, 0), (7, 1), (7, 2), (7, 3), (7, 4), (7, 5), (7, 6), (7, 7)]

cluster0_shape = [1, 8]
cluster1_shape = [1, 8]
cluster2_shape = [1, 8]
cluster3_shape = [1, 8]
cluster4_shape = [1, 8]
cluster5_shape = [1, 8]
cluster6_shape = [1, 8]
cluster7_shape = [1, 8]

cluster_nodes_lists = [cluster0_nodes_list, cluster1_nodes_list, cluster2_nodes_list, cluster3_nodes_list, cluster4_nodes_list, cluster5_nodes_list, cluster6_nodes_list, cluster7_nodes_list]
cluster_shapes = [cluster0_shape, cluster1_shape, cluster2_shape, cluster3_shape, cluster4_shape, cluster5_shape, cluster6_shape, cluster7_shape]



node_k = 8
node_n = 2
ni_k = node_k
ni_n = node_n
cfg_topology = "torus"
algorithm_dict = {'allgather': 'hierarchicalring', 'allreduce': 'hierarchicalring', 'reducescatter': 'hierarchicalring', 'reducelocal': 'base', 'alltoall': 'hierarchicalring', 'ordertoorder': 'hierarchicalring', 'pointtopoint': 'base', 'manytomanymulticast': 'alpa'}

node_network, noc_network = build(
    node_k=node_k, node_n=node_k, ni_k=ni_k, ni_n=ni_k,
    cfg_topology=cfg_topology,
    cfg_filepath=os.path.join(file_path, "../cfg/tpuv3_scaling.cfg")
)

tp = tensor_pipeline(
    topology=("torus2d"),
    algorithm=algorithm_dict
)

allreduce_api = allreduce(
    topology="torus2d",
    algorithm=algorithm_dict["allreduce"]
)

reducescatter_api = reducescatter(
    topology="torus2d",
    algorithm=algorithm_dict["reducescatter"]
)

allgather_api = allgather(
    topology="torus2d",
    algorithm=algorithm_dict["allgather"]
)

pointtopoint_api = pointtopoint(
    topology="torus2d",
    algorithm=algorithm_dict["pointtopoint"]
)

manytomany_api = manytomanymulticast(
    topology="torus2d",
    algorithm=algorithm_dict["manytomanymulticast"]
)


initial_event_tag = 0
initial_cluster0_dependency_list = [[[] for _ in range(cluster0_shape[1])] for _ in range(cluster0_shape[0])]
initial_cluster1_dependency_list = [[[] for _ in range(cluster1_shape[1])] for _ in range(cluster1_shape[0])]
initial_cluster2_dependency_list = [[[] for _ in range(cluster2_shape[1])] for _ in range(cluster2_shape[0])]
initial_cluster3_dependency_list = [[[] for _ in range(cluster3_shape[1])] for _ in range(cluster3_shape[0])]
initial_cluster4_dependency_list = [[[] for _ in range(cluster4_shape[1])] for _ in range(cluster4_shape[0])]
initial_cluster5_dependency_list = [[[] for _ in range(cluster5_shape[1])] for _ in range(cluster5_shape[0])]
initial_cluster6_dependency_list = [[[] for _ in range(cluster6_shape[1])] for _ in range(cluster6_shape[0])]
initial_cluster7_dependency_list = [[[] for _ in range(cluster7_shape[1])] for _ in range(cluster7_shape[0])]

initial_cluster_dependency_lists = [initial_cluster0_dependency_list, initial_cluster1_dependency_list, initial_cluster2_dependency_list, initial_cluster3_dependency_list, initial_cluster4_dependency_list, initial_cluster5_dependency_list, initial_cluster6_dependency_list, initial_cluster7_dependency_list]
'''

tpuv3_scaling_part = '''for cluster_idx, cluster_step in enumerate([2, 3]):
    for layer_step in range(2):'''

tpuv3_scaling_cal = '''intra_0_time = segment_time(finished_events_list, record_event_tags, [-1], [0])
intra_1_time = segment_time(finished_events_list, record_event_tags, [0], [1])
intra_2_time = segment_time(finished_events_list, record_event_tags, [2], [3])
inter_0_time = segment_time(finished_events_list, record_event_tags, [1], [2])

whole_time = intra_0_time + intra_2_time * ((model_config['n_layer'] // 2) - clusters_number - 1) + (intra_1_time + inter_0_time) * (clusters_number - 1)
'''




# mapping_dict = {'tpuv3': tpuv3_mapping}
# part_dict = {'tpuv3': tpuv3_part}
# cal_dict = {'tpuv3': tpuv3_cal}
mapping_dict = {'a100': a100_mapping, 'dojo1': dojo1_mapping, 'dojo2': dojo2_mapping, 'tpuv3': tpuv3_mapping, 'a100_scaling': a100_scaling_mapping, 'dojo_scaling': dojo_scaling_mapping, 'tpuv3_scaling': tpuv3_scaling_mapping}
part_dict = {'a100': a100_part, 'dojo1': dojo1_part, 'dojo2': dojo2_part, 'tpuv3': tpuv3_part, 'a100_scaling': a100_scaling_part, 'dojo_scaling': dojo_scaling_part, 'tpuv3_scaling': tpuv3_scaling_part}
cal_dict = {'a100': a100_cal, 'dojo1': dojo1_cal, 'dojo2': dojo2_cal, 'tpuv3': tpuv3_cal, 'a100_scaling': a100_scaling_cal, 'dojo_scaling': dojo_scaling_cal, 'tpuv3_scaling': tpuv3_scaling_cal}

