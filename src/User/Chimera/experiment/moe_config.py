# File name  :    moe_config.py
# Author     :    xiaocuicui
# Time       :    2025/02/17 13:38:54
# Version    :    V1.0
# Abstract   :        

import os
import sys
file_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(file_path, '../'))

a100_mapping = '''clusters_number = 4

cluster0_nodes_list = [(0, 0), (0, 1), (0, 2), (0, 3)]
cluster1_nodes_list = [(0, 4), (0, 5), (0, 6), (0, 7)]
cluster2_nodes_list = [(1, 0), (1, 1), (1, 2), (1, 3)]
cluster3_nodes_list = [(1, 4), (1, 5), (1, 6), (1, 7)]

cluster0_shape = [1, 4]
cluster1_shape = [1, 4]
cluster2_shape = [1, 4]
cluster3_shape = [1, 4]

cluster0_dataparallel = [2, 1]
cluster1_dataparallel = [2, 1]
cluster2_dataparallel = [2, 1]
cluster3_dataparallel = [2, 1]

cluster0_sub0_nodes_list = [(0, 0), (0, 1)]
cluster0_sub1_nodes_list = [(0, 2), (0, 3)]
cluster1_sub0_nodes_list = [(0, 4), (0, 5)]
cluster1_sub1_nodes_list = [(0, 6), (0, 7)]
cluster2_sub0_nodes_list = [(1, 0), (1, 1)]
cluster2_sub1_nodes_list = [(1, 2), (1, 3)]
cluster3_sub0_nodes_list = [(1, 4), (1, 5)]
cluster3_sub1_nodes_list = [(1, 6), (1, 7)]

cluster0_sub0_shape = [1, 2]
cluster0_sub1_shape = [1, 2]
cluster1_sub0_shape = [1, 2]
cluster1_sub1_shape = [1, 2]
cluster2_sub0_shape = [1, 2]
cluster2_sub1_shape = [1, 2]
cluster3_sub0_shape = [1, 2]
cluster3_sub1_shape = [1, 2]

cluster_nodes_lists = [cluster0_nodes_list, cluster1_nodes_list, cluster2_nodes_list, cluster3_nodes_list]
cluster_shapes = [cluster0_shape, cluster1_shape, cluster2_shape, cluster3_shape]
cluster_dataparallels = [cluster0_dataparallel, cluster1_dataparallel, cluster2_dataparallel, cluster3_dataparallel]

cluster_sub0_nodes_lists = [cluster0_sub0_nodes_list, cluster1_sub0_nodes_list, cluster2_sub0_nodes_list, cluster3_sub0_nodes_list]
cluster_sub1_nodes_lists = [cluster0_sub1_nodes_list, cluster1_sub1_nodes_list, cluster2_sub1_nodes_list, cluster3_sub1_nodes_list]

cluster_sub0_shapes = [cluster0_sub0_shape, cluster1_sub0_shape, cluster2_sub0_shape, cluster3_sub0_shape]
cluster_sub1_shapes = [cluster0_sub1_shape, cluster1_sub1_shape, cluster2_sub1_shape, cluster3_sub1_shape]


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

se = sequence_expert(
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

alltoall_api = alltoall(
    topology="dgx2",
    algorithm=algorithm_dict["alltoall"]
)

ordertoorder_api = ordertoorder(
    topology="dgx2",
    algorithm=algorithm_dict["ordertoorder"]
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

initial_cluster0_sub0_dependency_list = [[[] for _ in range(cluster0_sub0_shape[1])] for _ in range(cluster0_sub0_shape[0])]
initial_cluster0_sub1_dependency_list = [[[] for _ in range(cluster0_sub1_shape[1])] for _ in range(cluster0_sub1_shape[0])]
initial_cluster1_sub0_dependency_list = [[[] for _ in range(cluster1_sub0_shape[1])] for _ in range(cluster1_sub0_shape[0])]
initial_cluster1_sub1_dependency_list = [[[] for _ in range(cluster1_sub1_shape[1])] for _ in range(cluster1_sub1_shape[0])]
initial_cluster2_sub0_dependency_list = [[[] for _ in range(cluster2_sub0_shape[1])] for _ in range(cluster2_sub0_shape[0])]
initial_cluster2_sub1_dependency_list = [[[] for _ in range(cluster2_sub1_shape[1])] for _ in range(cluster2_sub1_shape[0])]
initial_cluster3_sub0_dependency_list = [[[] for _ in range(cluster3_sub0_shape[1])] for _ in range(cluster3_sub0_shape[0])]
initial_cluster3_sub1_dependency_list = [[[] for _ in range(cluster3_sub1_shape[1])] for _ in range(cluster3_sub1_shape[0])]

initial_cluster_dependency_lists = [initial_cluster0_dependency_list, initial_cluster1_dependency_list, initial_cluster2_dependency_list, initial_cluster3_dependency_list]
initial_cluster_sub0_dependency_lists = [initial_cluster0_sub0_dependency_list, initial_cluster1_sub0_dependency_list, initial_cluster2_sub0_dependency_list, initial_cluster3_sub0_dependency_list]
initial_cluster_sub1_dependency_lists = [initial_cluster0_sub1_dependency_list, initial_cluster1_sub1_dependency_list, initial_cluster2_sub1_dependency_list, initial_cluster3_sub1_dependency_list]
'''

a100_part = '''for cluster_idx, cluster_step in enumerate([0, 1, 2]):
    for layer_step in range(2):'''

a100_cal = '''intra_0_time = segment_time(finished_events_list, record_event_tags, [-1], [0])
intra_1_time = segment_time(finished_events_list, record_event_tags, [0], [1])
intra_2_time = segment_time(finished_events_list, record_event_tags, [2], [3])
intra_3_time = segment_time(finished_events_list, record_event_tags, [5], [6])
inter_0_time = segment_time(finished_events_list, record_event_tags, [1], [2])
inter_1_time = segment_time(finished_events_list, record_event_tags, [4], [5])

whole_time = intra_0_time + intra_2_time * (clusters_number - 2) + intra_3_time + intra_1_time * ((model_config['n_layer'] // 2) - clusters_number) + inter_0_time * (clusters_number - 2) + inter_1_time
'''




dojo_mapping = '''clusters_number = 6

cluster0_nodes_list = [(0, 0), (0, 1), (0, 2), (0, 3)]
cluster1_nodes_list = [(1, 0), (1, 1), (1, 2), (1, 3)]
cluster2_nodes_list = [(2, 0), (2, 1), (2, 2), (2, 3)]
cluster3_nodes_list = [(3, 0), (3, 1), (3, 2), (3, 3)]
cluster4_nodes_list = [(4, 0), (4, 1), (4, 2), (4, 3)]
cluster5_nodes_list = [(1, 4), (2, 4), (3, 4), (4, 4)]

cluster0_shape = [1, 4]
cluster1_shape = [1, 4]
cluster2_shape = [1, 4]
cluster3_shape = [1, 4]
cluster4_shape = [1, 4]
cluster5_shape = [4, 1]

cluster0_dataparallel = [2, 1]
cluster1_dataparallel = [2, 1]
cluster2_dataparallel = [2, 1]
cluster3_dataparallel = [2, 1]
cluster4_dataparallel = [2, 1]
cluster5_dataparallel = [1, 2]

cluster0_sub0_nodes_list = [(0, 0), (0, 1)]
cluster0_sub1_nodes_list = [(0, 2), (0, 3)]
cluster1_sub0_nodes_list = [(1, 0), (1, 1)]
cluster1_sub1_nodes_list = [(1, 2), (1, 3)]
cluster2_sub0_nodes_list = [(2, 0), (2, 1)]
cluster2_sub1_nodes_list = [(2, 2), (2, 3)]
cluster3_sub0_nodes_list = [(3, 0), (3, 1)]
cluster3_sub1_nodes_list = [(3, 2), (3, 3)]
cluster4_sub0_nodes_list = [(4, 0), (4, 1)]
cluster4_sub1_nodes_list = [(4, 2), (4, 3)]
cluster5_sub0_nodes_list = [(1, 4), (2, 4)]
cluster5_sub1_nodes_list = [(3, 4), (4, 4)]

cluster0_sub0_shape = [1, 2]
cluster0_sub1_shape = [1, 2]
cluster1_sub0_shape = [1, 2]
cluster1_sub1_shape = [1, 2]
cluster2_sub0_shape = [1, 2]
cluster2_sub1_shape = [1, 2]
cluster3_sub0_shape = [1, 2]
cluster3_sub1_shape = [1, 2]
cluster4_sub0_shape = [1, 2]
cluster4_sub1_shape = [1, 2]
cluster5_sub0_shape = [2, 1]
cluster5_sub1_shape = [2, 1]

cluster_nodes_lists = [cluster0_nodes_list, cluster1_nodes_list, cluster2_nodes_list, cluster3_nodes_list, cluster4_nodes_list, cluster5_nodes_list]
cluster_shapes = [cluster0_shape, cluster1_shape, cluster2_shape, cluster3_shape, cluster4_shape, cluster5_shape]
cluster_dataparallels = [cluster0_dataparallel, cluster1_dataparallel, cluster2_dataparallel, cluster3_dataparallel, cluster4_dataparallel, cluster5_dataparallel]

cluster_sub0_nodes_lists = [cluster0_sub0_nodes_list, cluster1_sub0_nodes_list, cluster2_sub0_nodes_list, cluster3_sub0_nodes_list, cluster4_sub0_nodes_list, cluster5_sub0_nodes_list]
cluster_sub1_nodes_lists = [cluster0_sub1_nodes_list, cluster1_sub1_nodes_list, cluster2_sub1_nodes_list, cluster3_sub1_nodes_list, cluster4_sub1_nodes_list, cluster5_sub1_nodes_list]

cluster_sub0_shapes = [cluster0_sub0_shape, cluster1_sub0_shape, cluster2_sub0_shape, cluster3_sub0_shape, cluster4_sub0_shape, cluster5_sub0_shape]
cluster_sub1_shapes = [cluster0_sub1_shape, cluster1_sub1_shape, cluster2_sub1_shape, cluster3_sub1_shape, cluster4_sub1_shape, cluster5_sub1_shape]


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

se = sequence_expert(
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

alltoall_api = alltoall(
    topology="mesh2d",
    algorithm=algorithm_dict["alltoall"]
)

ordertoorder_api = ordertoorder(
    topology="mesh2d",
    algorithm=algorithm_dict["ordertoorder"]
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

initial_cluster0_sub0_dependency_list = [[[] for _ in range(cluster0_sub0_shape[1])] for _ in range(cluster0_sub0_shape[0])]
initial_cluster0_sub1_dependency_list = [[[] for _ in range(cluster0_sub1_shape[1])] for _ in range(cluster0_sub1_shape[0])]
initial_cluster1_sub0_dependency_list = [[[] for _ in range(cluster1_sub0_shape[1])] for _ in range(cluster1_sub0_shape[0])]
initial_cluster1_sub1_dependency_list = [[[] for _ in range(cluster1_sub1_shape[1])] for _ in range(cluster1_sub1_shape[0])]
initial_cluster2_sub0_dependency_list = [[[] for _ in range(cluster2_sub0_shape[1])] for _ in range(cluster2_sub0_shape[0])]
initial_cluster2_sub1_dependency_list = [[[] for _ in range(cluster2_sub1_shape[1])] for _ in range(cluster2_sub1_shape[0])]
initial_cluster3_sub0_dependency_list = [[[] for _ in range(cluster3_sub0_shape[1])] for _ in range(cluster3_sub0_shape[0])]
initial_cluster3_sub1_dependency_list = [[[] for _ in range(cluster3_sub1_shape[1])] for _ in range(cluster3_sub1_shape[0])]
initial_cluster4_sub0_dependency_list = [[[] for _ in range(cluster4_sub0_shape[1])] for _ in range(cluster4_sub0_shape[0])]
initial_cluster4_sub1_dependency_list = [[[] for _ in range(cluster4_sub1_shape[1])] for _ in range(cluster4_sub1_shape[0])]
initial_cluster5_sub0_dependency_list = [[[] for _ in range(cluster5_sub0_shape[1])] for _ in range(cluster5_sub0_shape[0])]
initial_cluster5_sub1_dependency_list = [[[] for _ in range(cluster5_sub1_shape[1])] for _ in range(cluster5_sub1_shape[0])]

initial_cluster_dependency_lists = [initial_cluster0_dependency_list, initial_cluster1_dependency_list, initial_cluster2_dependency_list, initial_cluster3_dependency_list, initial_cluster4_dependency_list, initial_cluster5_dependency_list]
initial_cluster_sub0_dependency_lists = [initial_cluster0_sub0_dependency_list, initial_cluster1_sub0_dependency_list, initial_cluster2_sub0_dependency_list, initial_cluster3_sub0_dependency_list, initial_cluster4_sub0_dependency_list, initial_cluster5_sub0_dependency_list]
initial_cluster_sub1_dependency_lists = [initial_cluster0_sub1_dependency_list, initial_cluster1_sub1_dependency_list, initial_cluster2_sub1_dependency_list, initial_cluster3_sub1_dependency_list, initial_cluster4_sub1_dependency_list, initial_cluster5_sub1_dependency_list]
'''

dojo_part = '''for cluster_idx, cluster_step in enumerate([3, 4, 5]):
    for layer_step in range(2):'''

dojo_cal = '''intra_0_time = segment_time(finished_events_list, record_event_tags, [-1], [0])
intra_1_time = segment_time(finished_events_list, record_event_tags, [0], [1])
intra_2_time = segment_time(finished_events_list, record_event_tags, [2], [3])
intra_3_time = segment_time(finished_events_list, record_event_tags, [5], [6])
inter_0_time = segment_time(finished_events_list, record_event_tags, [1], [2])
inter_1_time = segment_time(finished_events_list, record_event_tags, [4], [5])

whole_time = intra_0_time + intra_2_time * (clusters_number - 2) + intra_3_time + intra_1_time * (model_config['n_layer'] - clusters_number) + inter_0_time * (clusters_number - 2) + inter_1_time
'''




tpuv3_mapping = '''clusters_number = 4

cluster0_nodes_list = [(0, 0), (1, 0), (2, 0), (3, 0)]
cluster1_nodes_list = [(0, 1), (1, 1), (2, 1), (3, 1)]
cluster2_nodes_list = [(0, 2), (1, 2), (2, 2), (3, 2)]
cluster3_nodes_list = [(0, 3), (1, 3), (2, 3), (3, 3)]

cluster0_shape = [4, 1]
cluster1_shape = [4, 1]
cluster2_shape = [4, 1]
cluster3_shape = [4, 1]

cluster0_dataparallel = [1, 2]
cluster1_dataparallel = [1, 2]
cluster2_dataparallel = [1, 2]
cluster3_dataparallel = [1, 2]

cluster0_sub0_nodes_list = [(0, 0), (1, 0)]
cluster0_sub1_nodes_list = [(2, 0), (3, 0)]
cluster1_sub0_nodes_list = [(0, 1), (1, 1)]
cluster1_sub1_nodes_list = [(2, 1), (3, 1)]
cluster2_sub0_nodes_list = [(0, 2), (1, 2)]
cluster2_sub1_nodes_list = [(2, 2), (3, 2)]
cluster3_sub0_nodes_list = [(0, 3), (1, 3)]
cluster3_sub1_nodes_list = [(2, 3), (3, 3)]

cluster0_sub0_shape = [2, 1]
cluster0_sub1_shape = [2, 1]
cluster1_sub0_shape = [2, 1]
cluster1_sub1_shape = [2, 1]
cluster2_sub0_shape = [2, 1]
cluster2_sub1_shape = [2, 1]
cluster3_sub0_shape = [2, 1]
cluster3_sub1_shape = [2, 1]

cluster_nodes_lists = [cluster0_nodes_list, cluster1_nodes_list, cluster2_nodes_list, cluster3_nodes_list]
cluster_shapes = [cluster0_shape, cluster1_shape, cluster2_shape, cluster3_shape]
cluster_dataparallels = [cluster0_dataparallel, cluster1_dataparallel, cluster2_dataparallel, cluster3_dataparallel]

cluster_sub0_nodes_lists = [cluster0_sub0_nodes_list, cluster1_sub0_nodes_list, cluster2_sub0_nodes_list, cluster3_sub0_nodes_list]
cluster_sub1_nodes_lists = [cluster0_sub1_nodes_list, cluster1_sub1_nodes_list, cluster2_sub1_nodes_list, cluster3_sub1_nodes_list]

cluster_sub0_shapes = [cluster0_sub0_shape, cluster1_sub0_shape, cluster2_sub0_shape, cluster3_sub0_shape]
cluster_sub1_shapes = [cluster0_sub1_shape, cluster1_sub1_shape, cluster2_sub1_shape, cluster3_sub1_shape]


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

se = sequence_expert(
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

alltoall_api = alltoall(
    topology="torus2d",
    algorithm=algorithm_dict["alltoall"]
)

ordertoorder_api = ordertoorder(
    topology="torus2d",
    algorithm=algorithm_dict["ordertoorder"]
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

initial_cluster0_sub0_dependency_list = [[[] for _ in range(cluster0_sub0_shape[1])] for _ in range(cluster0_sub0_shape[0])]
initial_cluster0_sub1_dependency_list = [[[] for _ in range(cluster0_sub1_shape[1])] for _ in range(cluster0_sub1_shape[0])]
initial_cluster1_sub0_dependency_list = [[[] for _ in range(cluster1_sub0_shape[1])] for _ in range(cluster1_sub0_shape[0])]
initial_cluster1_sub1_dependency_list = [[[] for _ in range(cluster1_sub1_shape[1])] for _ in range(cluster1_sub1_shape[0])]
initial_cluster2_sub0_dependency_list = [[[] for _ in range(cluster2_sub0_shape[1])] for _ in range(cluster2_sub0_shape[0])]
initial_cluster2_sub1_dependency_list = [[[] for _ in range(cluster2_sub1_shape[1])] for _ in range(cluster2_sub1_shape[0])]
initial_cluster3_sub0_dependency_list = [[[] for _ in range(cluster3_sub0_shape[1])] for _ in range(cluster3_sub0_shape[0])]
initial_cluster3_sub1_dependency_list = [[[] for _ in range(cluster3_sub1_shape[1])] for _ in range(cluster3_sub1_shape[0])]

initial_cluster_dependency_lists = [initial_cluster0_dependency_list, initial_cluster1_dependency_list, initial_cluster2_dependency_list, initial_cluster3_dependency_list]
initial_cluster_sub0_dependency_lists = [initial_cluster0_sub0_dependency_list, initial_cluster1_sub0_dependency_list, initial_cluster2_sub0_dependency_list, initial_cluster3_sub0_dependency_list]
initial_cluster_sub1_dependency_lists = [initial_cluster0_sub1_dependency_list, initial_cluster1_sub1_dependency_list, initial_cluster2_sub1_dependency_list, initial_cluster3_sub1_dependency_list]
'''

tpuv3_part = '''for cluster_idx, cluster_step in enumerate([0, 1]):
    for layer_step in range(2):'''

tpuv3_cal = '''intra_0_time = segment_time(finished_events_list, record_event_tags, [-1], [0])
intra_1_time = segment_time(finished_events_list, record_event_tags, [0], [1])
intra_2_time = segment_time(finished_events_list, record_event_tags, [2], [3])
inter_0_time = segment_time(finished_events_list, record_event_tags, [1], [2])

whole_time = intra_0_time + intra_2_time * ((model_config['n_layer'] // 2) - clusters_number - 1) + (intra_1_time + inter_0_time) * (clusters_number - 1)
'''




a100_scaling_mapping = '''clusters_number = 4

cluster0_nodes_list = [(0, 0), (0, 1), (0, 2), (0, 3), (0, 4), (0, 5), (0, 6), (0, 7), (0, 8), (0, 9), (0, 10), (0, 11), (0, 12), (0, 13), (0, 14), (0, 15)]
cluster1_nodes_list = [(0, 16), (0, 17), (0, 18), (0, 19), (0, 20), (0, 21), (0, 22), (0, 23), (0, 24), (0, 25), (0, 26), (0, 27), (0, 28), (0, 29), (0, 30), (0, 31)]
cluster2_nodes_list = [(1, 0), (1, 1), (1, 2), (1, 3), (1, 4), (1, 5), (1, 6), (1, 7), (1, 8), (1, 9), (1, 10), (1, 11), (1, 12), (1, 13), (1, 14), (1, 15)]
cluster3_nodes_list = [(1, 16), (1, 17), (1, 18), (1, 19), (1, 20), (1, 21), (1, 22), (1, 23), (1, 24), (1, 25), (1, 26), (1, 27), (1, 28), (1, 29), (1, 30), (1, 31)]

cluster0_shape = [1, 16]
cluster1_shape = [1, 16]
cluster2_shape = [1, 16]
cluster3_shape = [1, 16]

cluster0_dataparallel = [2, 1]
cluster1_dataparallel = [2, 1]
cluster2_dataparallel = [2, 1]
cluster3_dataparallel = [2, 1]

cluster0_sub0_nodes_list = [(0, 0), (0, 1), (0, 2), (0, 3), (0, 4), (0, 5), (0, 6), (0, 7)]
cluster0_sub1_nodes_list = [(0, 8), (0, 9), (0, 10), (0, 11), (0, 12), (0, 13), (0, 14), (0, 15)]
cluster1_sub0_nodes_list = [(0, 16), (0, 17), (0, 18), (0, 19), (0, 20), (0, 21), (0, 22), (0, 23)]
cluster1_sub1_nodes_list = [(0, 24), (0, 25), (0, 26), (0, 27), (0, 28), (0, 29), (0, 30), (0, 31)]
cluster2_sub0_nodes_list = [(1, 0), (1, 1), (1, 2), (1, 3), (1, 4), (1, 5), (1, 6), (1, 7)]
cluster2_sub1_nodes_list = [(1, 8), (1, 9), (1, 10), (1, 11), (1, 12), (1, 13), (1, 14), (1, 15)]
cluster3_sub0_nodes_list = [(1, 16), (1, 17), (1, 18), (1, 19), (1, 20), (1, 21), (1, 22), (1, 23)]
cluster3_sub1_nodes_list = [(1, 24), (1, 25), (1, 26), (1, 27), (1, 28), (1, 29), (1, 30), (1, 31)]

cluster0_sub0_shape = [1, 8]
cluster0_sub1_shape = [1, 8]
cluster1_sub0_shape = [1, 8]
cluster1_sub1_shape = [1, 8]
cluster2_sub0_shape = [1, 8]
cluster2_sub1_shape = [1, 8]
cluster3_sub0_shape = [1, 8]
cluster3_sub1_shape = [1, 8]

cluster_nodes_lists = [cluster0_nodes_list, cluster1_nodes_list, cluster2_nodes_list, cluster3_nodes_list]
cluster_shapes = [cluster0_shape, cluster1_shape, cluster2_shape, cluster3_shape]
cluster_dataparallels = [cluster0_dataparallel, cluster1_dataparallel, cluster2_dataparallel, cluster3_dataparallel]

cluster_sub0_nodes_lists = [cluster0_sub0_nodes_list, cluster1_sub0_nodes_list, cluster2_sub0_nodes_list, cluster3_sub0_nodes_list]
cluster_sub1_nodes_lists = [cluster0_sub1_nodes_list, cluster1_sub1_nodes_list, cluster2_sub1_nodes_list, cluster3_sub1_nodes_list]

cluster_sub0_shapes = [cluster0_sub0_shape, cluster1_sub0_shape, cluster2_sub0_shape, cluster3_sub0_shape]
cluster_sub1_shapes = [cluster0_sub1_shape, cluster1_sub1_shape, cluster2_sub1_shape, cluster3_sub1_shape]

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

se = sequence_expert(
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

alltoall_api = alltoall(
    topology="dgx2",
    algorithm=algorithm_dict["alltoall"]
)

ordertoorder_api = ordertoorder(
    topology="dgx2",
    algorithm=algorithm_dict["ordertoorder"]
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

initial_cluster0_sub0_dependency_list = [[[] for _ in range(cluster0_sub0_shape[1])] for _ in range(cluster0_sub0_shape[0])]
initial_cluster0_sub1_dependency_list = [[[] for _ in range(cluster0_sub1_shape[1])] for _ in range(cluster0_sub1_shape[0])]
initial_cluster1_sub0_dependency_list = [[[] for _ in range(cluster1_sub0_shape[1])] for _ in range(cluster1_sub0_shape[0])]
initial_cluster1_sub1_dependency_list = [[[] for _ in range(cluster1_sub1_shape[1])] for _ in range(cluster1_sub1_shape[0])]
initial_cluster2_sub0_dependency_list = [[[] for _ in range(cluster2_sub0_shape[1])] for _ in range(cluster2_sub0_shape[0])]
initial_cluster2_sub1_dependency_list = [[[] for _ in range(cluster2_sub1_shape[1])] for _ in range(cluster2_sub1_shape[0])]
initial_cluster3_sub0_dependency_list = [[[] for _ in range(cluster3_sub0_shape[1])] for _ in range(cluster3_sub0_shape[0])]
initial_cluster3_sub1_dependency_list = [[[] for _ in range(cluster3_sub1_shape[1])] for _ in range(cluster3_sub1_shape[0])]
                                         
initial_cluster_dependency_lists = [initial_cluster0_dependency_list, initial_cluster1_dependency_list, initial_cluster2_dependency_list, initial_cluster3_dependency_list]
initial_cluster_sub0_dependency_lists = [initial_cluster0_sub0_dependency_list, initial_cluster1_sub0_dependency_list, initial_cluster2_sub0_dependency_list, initial_cluster3_sub0_dependency_list]
initial_cluster_sub1_dependency_lists = [initial_cluster0_sub1_dependency_list, initial_cluster1_sub1_dependency_list, initial_cluster2_sub1_dependency_list, initial_cluster3_sub1_dependency_list]
'''

a100_scaling_part = '''for cluster_idx, cluster_step in enumerate([0, 1, 2]):
    for layer_step in range(2):'''

a100_scaling_cal = '''intra_0_time = segment_time(finished_events_list, record_event_tags, [-1], [0])
intra_1_time = segment_time(finished_events_list, record_event_tags, [0], [1])
intra_2_time = segment_time(finished_events_list, record_event_tags, [2], [3])
intra_3_time = segment_time(finished_events_list, record_event_tags, [5], [6])
inter_0_time = segment_time(finished_events_list, record_event_tags, [1], [2])
inter_1_time = segment_time(finished_events_list, record_event_tags, [4], [5])

whole_time = intra_0_time + intra_2_time * (clusters_number - 2) + intra_3_time + intra_1_time * (model_config['n_layer'] - clusters_number) + inter_0_time * (clusters_number - 2) + inter_1_time
'''




dojo_scaling_mapping = '''clusters_number = 4

cluster0_nodes_list = [(0, 0), (0, 1), (0, 2), (0, 3), (1, 0), (1, 1), (1, 2), (1, 3), (2, 0), (2, 1), (2, 2), (2, 3), (3, 0), (3, 1), (3, 2), (3, 3)]
cluster1_nodes_list = [(0, 4), (0, 5), (0, 6), (0, 7), (1, 4), (1, 5), (1, 6), (1, 7), (2, 4), (2, 5), (2, 6), (2, 7), (3, 4), (3, 5), (3, 6), (3, 7)]
cluster2_nodes_list = [(4, 4), (4, 5), (4, 6), (4, 7), (5, 4), (5, 5), (5, 6), (5, 7), (6, 4), (6, 5), (6, 6), (6, 7), (7, 4), (7, 5), (7, 6), (7, 7)]
cluster3_nodes_list = [(4, 0), (4, 1), (4, 2), (4, 3), (5, 0), (5, 1), (5, 2), (5, 3), (6, 0), (6, 1), (6, 2), (6, 3), (7, 0), (7, 1), (7, 2), (7, 3)]

cluster0_shape = [4, 4]
cluster1_shape = [4, 4]
cluster2_shape = [4, 4]
cluster3_shape = [4, 4]

cluster0_dataparallel = [2, 1]
cluster1_dataparallel = [2, 1]
cluster2_dataparallel = [2, 1]
cluster3_dataparallel = [2, 1]

cluster0_sub0_nodes_list = [(0, 0), (0, 1), (1, 0), (1, 1), (2, 0), (2, 1), (3, 0), (3, 1)]
cluster0_sub1_nodes_list = [(0, 2), (0, 3), (1, 2), (1, 3), (2, 2), (2, 3), (3, 2), (3, 3)]
cluster1_sub0_nodes_list = [(0, 4), (0, 5), (1, 4), (1, 5), (2, 4), (2, 5), (3, 4), (3, 5)]
cluster1_sub1_nodes_list = [(0, 6), (0, 7), (1, 6), (1, 7), (2, 6), (2, 7), (3, 6), (3, 7)]
cluster2_sub0_nodes_list = [(4, 4), (4, 5), (5, 4), (5, 5), (6, 4), (6, 5), (7, 4), (7, 5)]
cluster2_sub1_nodes_list = [(4, 6), (4, 7), (5, 6), (5, 7), (6, 6), (6, 7), (7, 6), (7, 7)]
cluster3_sub0_nodes_list = [(4, 0), (4, 1), (5, 0), (5, 1), (6, 0), (6, 1), (7, 0), (7, 1)]
cluster3_sub1_nodes_list = [(4, 2), (4, 3), (5, 2), (5, 3), (6, 2), (6, 3), (7, 2), (7, 3)]

cluster0_sub0_shape = [4, 2]
cluster0_sub1_shape = [4, 2]
cluster1_sub0_shape = [4, 2]
cluster1_sub1_shape = [4, 2]
cluster2_sub0_shape = [4, 2]
cluster2_sub1_shape = [4, 2]
cluster3_sub0_shape = [4, 2]
cluster3_sub1_shape = [4, 2]

cluster_nodes_lists = [cluster0_nodes_list, cluster1_nodes_list, cluster2_nodes_list, cluster3_nodes_list]
cluster_shapes = [cluster0_shape, cluster1_shape, cluster2_shape, cluster3_shape]
cluster_dataparallels = [cluster0_dataparallel, cluster1_dataparallel, cluster2_dataparallel, cluster3_dataparallel]

cluster_sub0_nodes_lists = [cluster0_sub0_nodes_list, cluster1_sub0_nodes_list, cluster2_sub0_nodes_list, cluster3_sub0_nodes_list]
cluster_sub1_nodes_lists = [cluster0_sub1_nodes_list, cluster1_sub1_nodes_list, cluster2_sub1_nodes_list, cluster3_sub1_nodes_list]

cluster_sub0_shapes = [cluster0_sub0_shape, cluster1_sub0_shape, cluster2_sub0_shape, cluster3_sub0_shape]
cluster_sub1_shapes = [cluster0_sub1_shape, cluster1_sub1_shape, cluster2_sub1_shape, cluster3_sub1_shape]

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

se = sequence_expert(
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

alltoall_api = alltoall(
    topology="mesh2d",
    algorithm=algorithm_dict["alltoall"]
)

ordertoorder_api = ordertoorder(
    topology="mesh2d",
    algorithm=algorithm_dict["ordertoorder"]
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

initial_cluster0_sub0_dependency_list = [[[] for _ in range(cluster0_sub0_shape[1])] for _ in range(cluster0_sub0_shape[0])]
initial_cluster0_sub1_dependency_list = [[[] for _ in range(cluster0_sub1_shape[1])] for _ in range(cluster0_sub1_shape[0])]
initial_cluster1_sub0_dependency_list = [[[] for _ in range(cluster1_sub0_shape[1])] for _ in range(cluster1_sub0_shape[0])]
initial_cluster1_sub1_dependency_list = [[[] for _ in range(cluster1_sub1_shape[1])] for _ in range(cluster1_sub1_shape[0])]
initial_cluster2_sub0_dependency_list = [[[] for _ in range(cluster2_sub0_shape[1])] for _ in range(cluster2_sub0_shape[0])]
initial_cluster2_sub1_dependency_list = [[[] for _ in range(cluster2_sub1_shape[1])] for _ in range(cluster2_sub1_shape[0])]
initial_cluster3_sub0_dependency_list = [[[] for _ in range(cluster3_sub0_shape[1])] for _ in range(cluster3_sub0_shape[0])]
initial_cluster3_sub1_dependency_list = [[[] for _ in range(cluster3_sub1_shape[1])] for _ in range(cluster3_sub1_shape[0])]

initial_cluster_dependency_lists = [initial_cluster0_dependency_list, initial_cluster1_dependency_list, initial_cluster2_dependency_list, initial_cluster3_dependency_list]
initial_cluster_sub0_dependency_lists = [initial_cluster0_sub0_dependency_list, initial_cluster1_sub0_dependency_list, initial_cluster2_sub0_dependency_list, initial_cluster3_sub0_dependency_list]
initial_cluster_sub1_dependency_lists = [initial_cluster0_sub1_dependency_list, initial_cluster1_sub1_dependency_list, initial_cluster2_sub1_dependency_list, initial_cluster3_sub1_dependency_list]
'''

dojo_scaling_part = '''for cluster_idx, cluster_step in enumerate([0, 1]):
    for layer_step in range(2):'''

dojo_scaling_cal = '''intra_0_time = segment_time(finished_events_list, record_event_tags, [-1], [0])
intra_1_time = segment_time(finished_events_list, record_event_tags, [0], [1])
intra_2_time = segment_time(finished_events_list, record_event_tags, [2], [3])
inter_0_time = segment_time(finished_events_list, record_event_tags, [1], [2])

whole_time = intra_0_time + intra_2_time * ((model_config['n_layer'] // 2) - clusters_number - 1) + (intra_1_time + inter_0_time) * (clusters_number - 1)
'''




tpuv3_scaling_mapping = '''clusters_number = 4

cluster0_nodes_list = [(0, 0), (0, 1), (0, 2), (0, 3), (0, 4), (0, 5), (0, 6), (0, 7), (1, 0), (1, 1), (1, 2), (1, 3), (1, 4), (1, 5), (1, 6), (1, 7)]
cluster1_nodes_list = [(2, 0), (2, 1), (2, 2), (2, 3), (2, 4), (2, 5), (2, 6), (2, 7), (3, 0), (3, 1), (3, 2), (3, 3), (3, 4), (3, 5), (3, 6), (3, 7)]
cluster2_nodes_list = [(4, 0), (4, 1), (4, 2), (4, 3), (4, 4), (4, 5), (4, 6), (4, 7), (5, 0), (5, 1), (5, 2), (5, 3), (5, 4), (5, 5), (5, 6), (5, 7)]
cluster3_nodes_list = [(6, 0), (6, 1), (6, 2), (6, 3), (6, 4), (6, 5), (6, 6), (6, 7), (7, 0), (7, 1), (7, 2), (7, 3), (7, 4), (7, 5), (7, 6), (7, 7)]

cluster0_shape = [2, 8]
cluster1_shape = [2, 8]
cluster2_shape = [2, 8]
cluster3_shape = [2, 8]

cluster0_dataparallel = [1, 2]
cluster1_dataparallel = [1, 2]
cluster2_dataparallel = [1, 2]
cluster3_dataparallel = [1, 2]

cluster0_sub0_nodes_list = [(0, 0), (0, 1), (0, 2), (0, 3), (0, 4), (0, 5), (0, 6), (0, 7)]
cluster0_sub1_nodes_list = [(1, 0), (1, 1), (1, 2), (1, 3), (1, 4), (1, 5), (1, 6), (1, 7)]
cluster1_sub0_nodes_list = [(2, 0), (2, 1), (2, 2), (2, 3), (2, 4), (2, 5), (2, 6), (2, 7)]
cluster1_sub1_nodes_list = [(3, 0), (3, 1), (3, 2), (3, 3), (3, 4), (3, 5), (3, 6), (3, 7)]
cluster2_sub0_nodes_list = [(4, 0), (4, 1), (4, 2), (4, 3), (4, 4), (4, 5), (4, 6), (4, 7)]
cluster2_sub1_nodes_list = [(5, 0), (5, 1), (5, 2), (5, 3), (5, 4), (5, 5), (5, 6), (5, 7)]
cluster3_sub0_nodes_list = [(6, 0), (6, 1), (6, 2), (6, 3), (6, 4), (6, 5), (6, 6), (6, 7)]
cluster3_sub1_nodes_list = [(7, 0), (7, 1), (7, 2), (7, 3), (7, 4), (7, 5), (7, 6), (7, 7)]

cluster0_sub0_shape = [1, 8]
cluster0_sub1_shape = [1, 8]
cluster1_sub0_shape = [1, 8]
cluster1_sub1_shape = [1, 8]
cluster2_sub0_shape = [1, 8]
cluster2_sub1_shape = [1, 8]
cluster3_sub0_shape = [1, 8]
cluster3_sub1_shape = [1, 8]

cluster_nodes_lists = [cluster0_nodes_list, cluster1_nodes_list, cluster2_nodes_list, cluster3_nodes_list]
cluster_shapes = [cluster0_shape, cluster1_shape, cluster2_shape, cluster3_shape]
cluster_dataparallels = [cluster0_dataparallel, cluster1_dataparallel, cluster2_dataparallel, cluster3_dataparallel]

cluster_sub0_nodes_lists = [cluster0_sub0_nodes_list, cluster1_sub0_nodes_list, cluster2_sub0_nodes_list, cluster3_sub0_nodes_list]
cluster_sub1_nodes_lists = [cluster0_sub1_nodes_list, cluster1_sub1_nodes_list, cluster2_sub1_nodes_list, cluster3_sub1_nodes_list]

cluster_sub0_shapes = [cluster0_sub0_shape, cluster1_sub0_shape, cluster2_sub0_shape, cluster3_sub0_shape]
cluster_sub1_shapes = [cluster0_sub1_shape, cluster1_sub1_shape, cluster2_sub1_shape, cluster3_sub1_shape]

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

se = sequence_expert(
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

alltoall_api = alltoall(
    topology="torus2d",
    algorithm=algorithm_dict["alltoall"]
)

ordertoorder_api = ordertoorder(
    topology="torus2d",
    algorithm=algorithm_dict["ordertoorder"]
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

initial_cluster0_sub0_dependency_list = [[[] for _ in range(cluster0_sub0_shape[1])] for _ in range(cluster0_sub0_shape[0])]
initial_cluster0_sub1_dependency_list = [[[] for _ in range(cluster0_sub1_shape[1])] for _ in range(cluster0_sub1_shape[0])]
initial_cluster1_sub0_dependency_list = [[[] for _ in range(cluster1_sub0_shape[1])] for _ in range(cluster1_sub0_shape[0])]
initial_cluster1_sub1_dependency_list = [[[] for _ in range(cluster1_sub1_shape[1])] for _ in range(cluster1_sub1_shape[0])]
initial_cluster2_sub0_dependency_list = [[[] for _ in range(cluster2_sub0_shape[1])] for _ in range(cluster2_sub0_shape[0])]
initial_cluster2_sub1_dependency_list = [[[] for _ in range(cluster2_sub1_shape[1])] for _ in range(cluster2_sub1_shape[0])]
initial_cluster3_sub0_dependency_list = [[[] for _ in range(cluster3_sub0_shape[1])] for _ in range(cluster3_sub0_shape[0])]
initial_cluster3_sub1_dependency_list = [[[] for _ in range(cluster3_sub1_shape[1])] for _ in range(cluster3_sub1_shape[0])]

initial_cluster_dependency_lists = [initial_cluster0_dependency_list, initial_cluster1_dependency_list, initial_cluster2_dependency_list, initial_cluster3_dependency_list]                                      
initial_cluster_sub0_dependency_lists = [initial_cluster0_sub0_dependency_list, initial_cluster1_sub0_dependency_list, initial_cluster2_sub0_dependency_list, initial_cluster3_sub0_dependency_list]
initial_cluster_sub1_dependency_lists = [initial_cluster0_sub1_dependency_list, initial_cluster1_sub1_dependency_list, initial_cluster2_sub1_dependency_list, initial_cluster3_sub1_dependency_list]
'''

tpuv3_scaling_part = '''for cluster_idx, cluster_step in enumerate([0, 1]):
    for layer_step in range(2):'''

tpuv3_scaling_cal = '''intra_0_time = segment_time(finished_events_list, record_event_tags, [-1], [0])
intra_1_time = segment_time(finished_events_list, record_event_tags, [0], [1])
intra_2_time = segment_time(finished_events_list, record_event_tags, [2], [3])
inter_0_time = segment_time(finished_events_list, record_event_tags, [1], [2])

whole_time = intra_0_time + intra_2_time * ((model_config['n_layer'] // 2) - clusters_number - 1) + (intra_1_time + inter_0_time) * (clusters_number - 1)
'''




a100_expand_mapping = '''clusters_number = 8

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

cluster0_dataparallel = [2, 1]
cluster1_dataparallel = [2, 1]
cluster2_dataparallel = [2, 1]
cluster3_dataparallel = [2, 1]
cluster4_dataparallel = [2, 1]
cluster5_dataparallel = [2, 1]
cluster6_dataparallel = [2, 1]
cluster7_dataparallel = [2, 1]

cluster0_sub0_nodes_list = [(0, 0), (0, 1), (0, 2), (0, 3)]
cluster0_sub1_nodes_list = [(0, 4), (0, 5), (0, 6), (0, 7)]
cluster1_sub0_nodes_list = [(0, 8), (0, 9), (0, 10), (0, 11)]
cluster1_sub1_nodes_list = [(0, 12), (0, 13), (0, 14), (0, 15)]
cluster2_sub0_nodes_list = [(0, 16), (0, 17), (0, 18), (0, 19)]
cluster2_sub1_nodes_list = [(0, 20), (0, 21), (0, 22), (0, 23)]
cluster3_sub0_nodes_list = [(0, 24), (0, 25), (0, 26), (0, 27)]
cluster3_sub1_nodes_list = [(0, 28), (0, 29), (0, 30), (0, 31)]
cluster4_sub0_nodes_list = [(1, 0), (1, 1), (1, 2), (1, 3)]
cluster4_sub1_nodes_list = [(1, 4), (1, 5), (1, 6), (1, 7)]
cluster5_sub0_nodes_list = [(1, 8), (1, 9), (1, 10), (1, 11)]
cluster5_sub1_nodes_list = [(1, 12), (1, 13), (1, 14), (1, 15)]
cluster6_sub0_nodes_list = [(1, 16), (1, 17), (1, 18), (1, 19)]
cluster6_sub1_nodes_list = [(1, 20), (1, 21), (1, 22), (1, 23)]
cluster7_sub0_nodes_list = [(1, 24), (1, 25), (1, 26), (1, 27)]
cluster7_sub1_nodes_list = [(1, 28), (1, 29), (1, 30), (1, 31)]

cluster0_sub0_shape = [1, 4]
cluster0_sub1_shape = [1, 4]
cluster1_sub0_shape = [1, 4]
cluster1_sub1_shape = [1, 4]
cluster2_sub0_shape = [1, 4]
cluster2_sub1_shape = [1, 4]
cluster3_sub0_shape = [1, 4]
cluster3_sub1_shape = [1, 4]
cluster4_sub0_shape = [1, 4]
cluster4_sub1_shape = [1, 4]
cluster5_sub0_shape = [1, 4]
cluster5_sub1_shape = [1, 4]
cluster6_sub0_shape = [1, 4]
cluster6_sub1_shape = [1, 4]
cluster7_sub0_shape = [1, 4]
cluster7_sub1_shape = [1, 4]

cluster_nodes_lists = [cluster0_nodes_list, cluster1_nodes_list, cluster2_nodes_list, cluster3_nodes_list, cluster4_nodes_list, cluster5_nodes_list, cluster6_nodes_list, cluster7_nodes_list]
cluster_shapes = [cluster0_shape, cluster1_shape, cluster2_shape, cluster3_shape, cluster4_shape, cluster5_shape, cluster6_shape, cluster7_shape]
cluster_dataparallels = [cluster0_dataparallel, cluster1_dataparallel, cluster2_dataparallel, cluster3_dataparallel, cluster4_dataparallel, cluster5_dataparallel, cluster6_dataparallel, cluster7_dataparallel]

cluster_sub0_nodes_lists = [cluster0_sub0_nodes_list, cluster1_sub0_nodes_list, cluster2_sub0_nodes_list, cluster3_sub0_nodes_list, cluster4_sub0_nodes_list, cluster5_sub0_nodes_list, cluster6_sub0_nodes_list, cluster7_sub0_nodes_list]
cluster_sub1_nodes_lists = [cluster0_sub1_nodes_list, cluster1_sub1_nodes_list, cluster2_sub1_nodes_list, cluster3_sub1_nodes_list, cluster4_sub1_nodes_list, cluster5_sub1_nodes_list, cluster6_sub1_nodes_list, cluster7_sub1_nodes_list]

cluster_sub0_shapes = [cluster0_sub0_shape, cluster1_sub0_shape, cluster2_sub0_shape, cluster3_sub0_shape, cluster4_sub0_shape, cluster5_sub0_shape, cluster6_sub0_shape, cluster7_sub0_shape]
cluster_sub1_shapes = [cluster0_sub1_shape, cluster1_sub1_shape, cluster2_sub1_shape, cluster3_sub1_shape, cluster4_sub1_shape, cluster5_sub1_shape, cluster6_sub1_shape, cluster7_sub1_shape]

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

se = sequence_expert(
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

alltoall_api = alltoall(
    topology="dgx2",
    algorithm=algorithm_dict["alltoall"]
)

ordertoorder_api = ordertoorder(
    topology="dgx2",
    algorithm=algorithm_dict["ordertoorder"]
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

initial_cluster0_sub0_dependency_list = [[[] for _ in range(cluster0_sub0_shape[1])] for _ in range(cluster0_sub0_shape[0])]
initial_cluster0_sub1_dependency_list = [[[] for _ in range(cluster0_sub1_shape[1])] for _ in range(cluster0_sub1_shape[0])]
initial_cluster1_sub0_dependency_list = [[[] for _ in range(cluster1_sub0_shape[1])] for _ in range(cluster1_sub0_shape[0])]
initial_cluster1_sub1_dependency_list = [[[] for _ in range(cluster1_sub1_shape[1])] for _ in range(cluster1_sub1_shape[0])]
initial_cluster2_sub0_dependency_list = [[[] for _ in range(cluster2_sub0_shape[1])] for _ in range(cluster2_sub0_shape[0])]
initial_cluster2_sub1_dependency_list = [[[] for _ in range(cluster2_sub1_shape[1])] for _ in range(cluster2_sub1_shape[0])]
initial_cluster3_sub0_dependency_list = [[[] for _ in range(cluster3_sub0_shape[1])] for _ in range(cluster3_sub0_shape[0])]
initial_cluster3_sub1_dependency_list = [[[] for _ in range(cluster3_sub1_shape[1])] for _ in range(cluster3_sub1_shape[0])]
initial_cluster4_sub0_dependency_list = [[[] for _ in range(cluster4_sub0_shape[1])] for _ in range(cluster4_sub0_shape[0])]
initial_cluster4_sub1_dependency_list = [[[] for _ in range(cluster4_sub1_shape[1])] for _ in range(cluster4_sub1_shape[0])]
initial_cluster5_sub0_dependency_list = [[[] for _ in range(cluster5_sub0_shape[1])] for _ in range(cluster5_sub0_shape[0])]
initial_cluster5_sub1_dependency_list = [[[] for _ in range(cluster5_sub1_shape[1])] for _ in range(cluster5_sub1_shape[0])]
initial_cluster6_sub0_dependency_list = [[[] for _ in range(cluster6_sub0_shape[1])] for _ in range(cluster6_sub0_shape[0])]
initial_cluster6_sub1_dependency_list = [[[] for _ in range(cluster6_sub1_shape[1])] for _ in range(cluster6_sub1_shape[0])]
initial_cluster7_sub0_dependency_list = [[[] for _ in range(cluster7_sub0_shape[1])] for _ in range(cluster7_sub0_shape[0])]
initial_cluster7_sub1_dependency_list = [[[] for _ in range(cluster7_sub1_shape[1])] for _ in range(cluster7_sub1_shape[0])]
                                         
initial_cluster_dependency_lists = [initial_cluster0_dependency_list, initial_cluster1_dependency_list, initial_cluster2_dependency_list, initial_cluster3_dependency_list, initial_cluster4_dependency_list, initial_cluster5_dependency_list, initial_cluster6_dependency_list, initial_cluster7_dependency_list]
initial_cluster_sub0_dependency_lists = [initial_cluster0_sub0_dependency_list, initial_cluster1_sub0_dependency_list, initial_cluster2_sub0_dependency_list, initial_cluster3_sub0_dependency_list, initial_cluster4_sub0_dependency_list, initial_cluster5_sub0_dependency_list, initial_cluster6_sub0_dependency_list, initial_cluster7_sub0_dependency_list]
initial_cluster_sub1_dependency_lists = [initial_cluster0_sub1_dependency_list, initial_cluster1_sub1_dependency_list, initial_cluster2_sub1_dependency_list, initial_cluster3_sub1_dependency_list, initial_cluster4_sub1_dependency_list, initial_cluster5_sub1_dependency_list, initial_cluster6_sub1_dependency_list, initial_cluster7_sub1_dependency_list]
'''

a100_expand_part = '''for cluster_idx, cluster_step in enumerate([2, 3, 4]):
    for layer_step in range(2):'''

a100_expand_cal = '''intra_0_time = segment_time(finished_events_list, record_event_tags, [-1], [0])
intra_1_time = segment_time(finished_events_list, record_event_tags, [0], [1])
intra_2_time = segment_time(finished_events_list, record_event_tags, [2], [3])
intra_3_time = segment_time(finished_events_list, record_event_tags, [5], [6])
inter_0_time = segment_time(finished_events_list, record_event_tags, [1], [2])
inter_1_time = segment_time(finished_events_list, record_event_tags, [4], [5])

whole_time = intra_0_time + intra_2_time * (clusters_number - 2) + intra_3_time + intra_1_time * (model_config['n_layer'] - clusters_number) + inter_0_time * (clusters_number - 2) + inter_1_time
'''




dojo_expand_mapping = '''clusters_number = 8

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

cluster0_dataparallel = [2, 1]
cluster1_dataparallel = [2, 1]
cluster2_dataparallel = [2, 1]
cluster3_dataparallel = [2, 1]
cluster4_dataparallel = [2, 1]
cluster5_dataparallel = [2, 1]
cluster6_dataparallel = [2, 1]
cluster7_dataparallel = [2, 1]

cluster0_sub0_nodes_list = [(0, 0), (0, 1), (0, 2), (0, 3)]
cluster0_sub1_nodes_list = [(0, 4), (0, 5), (0, 6), (0, 7)]
cluster1_sub0_nodes_list = [(1, 0), (1, 1), (1, 2), (1, 3)]
cluster1_sub1_nodes_list = [(1, 4), (1, 5), (1, 6), (1, 7)]
cluster2_sub0_nodes_list = [(2, 0), (2, 1), (2, 2), (2, 3)]
cluster2_sub1_nodes_list = [(2, 4), (2, 5), (2, 6), (2, 7)]
cluster3_sub0_nodes_list = [(3, 0), (3, 1), (3, 2), (3, 3)]
cluster3_sub1_nodes_list = [(3, 4), (3, 5), (3, 6), (3, 7)]
cluster4_sub0_nodes_list = [(4, 0), (4, 1), (4, 2), (4, 3)]
cluster4_sub1_nodes_list = [(4, 4), (4, 5), (4, 6), (4, 7)]
cluster5_sub0_nodes_list = [(5, 0), (5, 1), (5, 2), (5, 3)]
cluster5_sub1_nodes_list = [(5, 4), (5, 5), (5, 6), (5, 7)]
cluster6_sub0_nodes_list = [(6, 0), (6, 1), (6, 2), (6, 3)]
cluster6_sub1_nodes_list = [(6, 4), (6, 5), (6, 6), (6, 7)]
cluster7_sub0_nodes_list = [(7, 0), (7, 1), (7, 2), (7, 3)]
cluster7_sub1_nodes_list = [(7, 4), (7, 5), (7, 6), (7, 7)]

cluster0_sub0_shape = [1, 4]
cluster0_sub1_shape = [1, 4]
cluster1_sub0_shape = [1, 4]
cluster1_sub1_shape = [1, 4]
cluster2_sub0_shape = [1, 4]
cluster2_sub1_shape = [1, 4]
cluster3_sub0_shape = [1, 4]
cluster3_sub1_shape = [1, 4]
cluster4_sub0_shape = [1, 4]
cluster4_sub1_shape = [1, 4]
cluster5_sub0_shape = [1, 4]
cluster5_sub1_shape = [1, 4]
cluster6_sub0_shape = [1, 4]
cluster6_sub1_shape = [1, 4]
cluster7_sub0_shape = [1, 4]
cluster7_sub1_shape = [1, 4]

cluster_nodes_lists = [cluster0_nodes_list, cluster1_nodes_list, cluster2_nodes_list, cluster3_nodes_list, cluster4_nodes_list, cluster5_nodes_list, cluster6_nodes_list, cluster7_nodes_list]
cluster_shapes = [cluster0_shape, cluster1_shape, cluster2_shape, cluster3_shape, cluster4_shape, cluster5_shape, cluster6_shape, cluster7_shape]
cluster_dataparallels = [cluster0_dataparallel, cluster1_dataparallel, cluster2_dataparallel, cluster3_dataparallel, cluster4_dataparallel, cluster5_dataparallel, cluster6_dataparallel, cluster7_dataparallel]

cluster_sub0_nodes_lists = [cluster0_sub0_nodes_list, cluster1_sub0_nodes_list, cluster2_sub0_nodes_list, cluster3_sub0_nodes_list, cluster4_sub0_nodes_list, cluster5_sub0_nodes_list, cluster6_sub0_nodes_list, cluster7_sub0_nodes_list]
cluster_sub1_nodes_lists = [cluster0_sub1_nodes_list, cluster1_sub1_nodes_list, cluster2_sub1_nodes_list, cluster3_sub1_nodes_list, cluster4_sub1_nodes_list, cluster5_sub1_nodes_list, cluster6_sub1_nodes_list, cluster7_sub1_nodes_list]

cluster_sub0_shapes = [cluster0_sub0_shape, cluster1_sub0_shape, cluster2_sub0_shape, cluster3_sub0_shape, cluster4_sub0_shape, cluster5_sub0_shape, cluster6_sub0_shape, cluster7_sub0_shape]
cluster_sub1_shapes = [cluster0_sub1_shape, cluster1_sub1_shape, cluster2_sub1_shape, cluster3_sub1_shape, cluster4_sub1_shape, cluster5_sub1_shape, cluster6_sub1_shape, cluster7_sub1_shape]

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

se = sequence_expert(
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

alltoall_api = alltoall(
    topology="mesh2d",
    algorithm=algorithm_dict["alltoall"]
)

ordertoorder_api = ordertoorder(
    topology="mesh2d",
    algorithm=algorithm_dict["ordertoorder"]
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

initial_cluster0_sub0_dependency_list = [[[] for _ in range(cluster0_sub0_shape[1])] for _ in range(cluster0_sub0_shape[0])]
initial_cluster0_sub1_dependency_list = [[[] for _ in range(cluster0_sub1_shape[1])] for _ in range(cluster0_sub1_shape[0])]
initial_cluster1_sub0_dependency_list = [[[] for _ in range(cluster1_sub0_shape[1])] for _ in range(cluster1_sub0_shape[0])]
initial_cluster1_sub1_dependency_list = [[[] for _ in range(cluster1_sub1_shape[1])] for _ in range(cluster1_sub1_shape[0])]
initial_cluster2_sub0_dependency_list = [[[] for _ in range(cluster2_sub0_shape[1])] for _ in range(cluster2_sub0_shape[0])]
initial_cluster2_sub1_dependency_list = [[[] for _ in range(cluster2_sub1_shape[1])] for _ in range(cluster2_sub1_shape[0])]
initial_cluster3_sub0_dependency_list = [[[] for _ in range(cluster3_sub0_shape[1])] for _ in range(cluster3_sub0_shape[0])]
initial_cluster3_sub1_dependency_list = [[[] for _ in range(cluster3_sub1_shape[1])] for _ in range(cluster3_sub1_shape[0])]
initial_cluster4_sub0_dependency_list = [[[] for _ in range(cluster4_sub0_shape[1])] for _ in range(cluster4_sub0_shape[0])]
initial_cluster4_sub1_dependency_list = [[[] for _ in range(cluster4_sub1_shape[1])] for _ in range(cluster4_sub1_shape[0])]
initial_cluster5_sub0_dependency_list = [[[] for _ in range(cluster5_sub0_shape[1])] for _ in range(cluster5_sub0_shape[0])]
initial_cluster5_sub1_dependency_list = [[[] for _ in range(cluster5_sub1_shape[1])] for _ in range(cluster5_sub1_shape[0])]
initial_cluster6_sub0_dependency_list = [[[] for _ in range(cluster6_sub0_shape[1])] for _ in range(cluster6_sub0_shape[0])]
initial_cluster6_sub1_dependency_list = [[[] for _ in range(cluster6_sub1_shape[1])] for _ in range(cluster6_sub1_shape[0])]
initial_cluster7_sub0_dependency_list = [[[] for _ in range(cluster7_sub0_shape[1])] for _ in range(cluster7_sub0_shape[0])]
initial_cluster7_sub1_dependency_list = [[[] for _ in range(cluster7_sub1_shape[1])] for _ in range(cluster7_sub1_shape[0])]
                                         
initial_cluster_dependency_lists = [initial_cluster0_dependency_list, initial_cluster1_dependency_list, initial_cluster2_dependency_list, initial_cluster3_dependency_list, initial_cluster4_dependency_list, initial_cluster5_dependency_list, initial_cluster6_dependency_list, initial_cluster7_dependency_list]
initial_cluster_sub0_dependency_lists = [initial_cluster0_sub0_dependency_list, initial_cluster1_sub0_dependency_list, initial_cluster2_sub0_dependency_list, initial_cluster3_sub0_dependency_list, initial_cluster4_sub0_dependency_list, initial_cluster5_sub0_dependency_list, initial_cluster6_sub0_dependency_list, initial_cluster7_sub0_dependency_list]
initial_cluster_sub1_dependency_lists = [initial_cluster0_sub1_dependency_list, initial_cluster1_sub1_dependency_list, initial_cluster2_sub1_dependency_list, initial_cluster3_sub1_dependency_list, initial_cluster4_sub1_dependency_list, initial_cluster5_sub1_dependency_list, initial_cluster6_sub1_dependency_list, initial_cluster7_sub1_dependency_list]
'''

dojo_expand_part = '''for cluster_idx, cluster_step in enumerate([0, 1]):
    for layer_step in range(2):'''

dojo_expand_cal = '''intra_0_time = segment_time(finished_events_list, record_event_tags, [-1], [0])
intra_1_time = segment_time(finished_events_list, record_event_tags, [0], [1])
intra_2_time = segment_time(finished_events_list, record_event_tags, [2], [3])
inter_0_time = segment_time(finished_events_list, record_event_tags, [1], [2])

whole_time = intra_0_time + intra_2_time * (model_config['n_layer'] - clusters_number - 1) + (intra_1_time + inter_0_time) * (clusters_number - 1)
'''




tpuv3_expand_mapping = '''clusters_number = 8

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

cluster0_dataparallel = [2, 1]
cluster1_dataparallel = [2, 1]
cluster2_dataparallel = [2, 1]
cluster3_dataparallel = [2, 1]
cluster4_dataparallel = [2, 1]
cluster5_dataparallel = [2, 1]
cluster6_dataparallel = [2, 1]
cluster7_dataparallel = [2, 1]

cluster0_sub0_nodes_list = [(0, 0), (0, 1), (0, 2), (0, 3)]
cluster0_sub1_nodes_list = [(0, 4), (0, 5), (0, 6), (0, 7)]
cluster1_sub0_nodes_list = [(1, 0), (1, 1), (1, 2), (1, 3)]
cluster1_sub1_nodes_list = [(1, 4), (1, 5), (1, 6), (1, 7)]
cluster2_sub0_nodes_list = [(2, 0), (2, 1), (2, 2), (2, 3)]
cluster2_sub1_nodes_list = [(2, 4), (2, 5), (2, 6), (2, 7)]
cluster3_sub0_nodes_list = [(3, 0), (3, 1), (3, 2), (3, 3)]
cluster3_sub1_nodes_list = [(3, 4), (3, 5), (3, 6), (3, 7)]
cluster4_sub0_nodes_list = [(4, 0), (4, 1), (4, 2), (4, 3)]
cluster4_sub1_nodes_list = [(4, 4), (4, 5), (4, 6), (4, 7)]
cluster5_sub0_nodes_list = [(5, 0), (5, 1), (5, 2), (5, 3)]
cluster5_sub1_nodes_list = [(5, 4), (5, 5), (5, 6), (5, 7)]
cluster6_sub0_nodes_list = [(6, 0), (6, 1), (6, 2), (6, 3)]
cluster6_sub1_nodes_list = [(6, 4), (6, 5), (6, 6), (6, 7)]
cluster7_sub0_nodes_list = [(7, 0), (7, 1), (7, 2), (7, 3)]
cluster7_sub1_nodes_list = [(7, 4), (7, 5), (7, 6), (7, 7)]

cluster0_sub0_shape = [1, 4]
cluster0_sub1_shape = [1, 4]
cluster1_sub0_shape = [1, 4]
cluster1_sub1_shape = [1, 4]
cluster2_sub0_shape = [1, 4]
cluster2_sub1_shape = [1, 4]
cluster3_sub0_shape = [1, 4]
cluster3_sub1_shape = [1, 4]
cluster4_sub0_shape = [1, 4]
cluster4_sub1_shape = [1, 4]
cluster5_sub0_shape = [1, 4]
cluster5_sub1_shape = [1, 4]
cluster6_sub0_shape = [1, 4]
cluster6_sub1_shape = [1, 4]
cluster7_sub0_shape = [1, 4]
cluster7_sub1_shape = [1, 4]

cluster_nodes_lists = [cluster0_nodes_list, cluster1_nodes_list, cluster2_nodes_list, cluster3_nodes_list, cluster4_nodes_list, cluster5_nodes_list, cluster6_nodes_list, cluster7_nodes_list]
cluster_shapes = [cluster0_shape, cluster1_shape, cluster2_shape, cluster3_shape, cluster4_shape, cluster5_shape, cluster6_shape, cluster7_shape]
cluster_dataparallels = [cluster0_dataparallel, cluster1_dataparallel, cluster2_dataparallel, cluster3_dataparallel, cluster4_dataparallel, cluster5_dataparallel, cluster6_dataparallel, cluster7_dataparallel]

cluster_sub0_nodes_lists = [cluster0_sub0_nodes_list, cluster1_sub0_nodes_list, cluster2_sub0_nodes_list, cluster3_sub0_nodes_list, cluster4_sub0_nodes_list, cluster5_sub0_nodes_list, cluster6_sub0_nodes_list, cluster7_sub0_nodes_list]
cluster_sub1_nodes_lists = [cluster0_sub1_nodes_list, cluster1_sub1_nodes_list, cluster2_sub1_nodes_list, cluster3_sub1_nodes_list, cluster4_sub1_nodes_list, cluster5_sub1_nodes_list, cluster6_sub1_nodes_list, cluster7_sub1_nodes_list]

cluster_sub0_shapes = [cluster0_sub0_shape, cluster1_sub0_shape, cluster2_sub0_shape, cluster3_sub0_shape, cluster4_sub0_shape, cluster5_sub0_shape, cluster6_sub0_shape, cluster7_sub0_shape]
cluster_sub1_shapes = [cluster0_sub1_shape, cluster1_sub1_shape, cluster2_sub1_shape, cluster3_sub1_shape, cluster4_sub1_shape, cluster5_sub1_shape, cluster6_sub1_shape, cluster7_sub1_shape]

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

se = sequence_expert(
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

alltoall_api = alltoall(
    topology="torus2d",
    algorithm=algorithm_dict["alltoall"]
)

ordertoorder_api = ordertoorder(
    topology="torus2d",
    algorithm=algorithm_dict["ordertoorder"]
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

initial_cluster0_sub0_dependency_list = [[[] for _ in range(cluster0_sub0_shape[1])] for _ in range(cluster0_sub0_shape[0])]
initial_cluster0_sub1_dependency_list = [[[] for _ in range(cluster0_sub1_shape[1])] for _ in range(cluster0_sub1_shape[0])]
initial_cluster1_sub0_dependency_list = [[[] for _ in range(cluster1_sub0_shape[1])] for _ in range(cluster1_sub0_shape[0])]
initial_cluster1_sub1_dependency_list = [[[] for _ in range(cluster1_sub1_shape[1])] for _ in range(cluster1_sub1_shape[0])]
initial_cluster2_sub0_dependency_list = [[[] for _ in range(cluster2_sub0_shape[1])] for _ in range(cluster2_sub0_shape[0])]
initial_cluster2_sub1_dependency_list = [[[] for _ in range(cluster2_sub1_shape[1])] for _ in range(cluster2_sub1_shape[0])]
initial_cluster3_sub0_dependency_list = [[[] for _ in range(cluster3_sub0_shape[1])] for _ in range(cluster3_sub0_shape[0])]
initial_cluster3_sub1_dependency_list = [[[] for _ in range(cluster3_sub1_shape[1])] for _ in range(cluster3_sub1_shape[0])]
initial_cluster4_sub0_dependency_list = [[[] for _ in range(cluster4_sub0_shape[1])] for _ in range(cluster4_sub0_shape[0])]
initial_cluster4_sub1_dependency_list = [[[] for _ in range(cluster4_sub1_shape[1])] for _ in range(cluster4_sub1_shape[0])]
initial_cluster5_sub0_dependency_list = [[[] for _ in range(cluster5_sub0_shape[1])] for _ in range(cluster5_sub0_shape[0])]
initial_cluster5_sub1_dependency_list = [[[] for _ in range(cluster5_sub1_shape[1])] for _ in range(cluster5_sub1_shape[0])]
initial_cluster6_sub0_dependency_list = [[[] for _ in range(cluster6_sub0_shape[1])] for _ in range(cluster6_sub0_shape[0])]
initial_cluster6_sub1_dependency_list = [[[] for _ in range(cluster6_sub1_shape[1])] for _ in range(cluster6_sub1_shape[0])]
initial_cluster7_sub0_dependency_list = [[[] for _ in range(cluster7_sub0_shape[1])] for _ in range(cluster7_sub0_shape[0])]
initial_cluster7_sub1_dependency_list = [[[] for _ in range(cluster7_sub1_shape[1])] for _ in range(cluster7_sub1_shape[0])]
                                         
initial_cluster_dependency_lists = [initial_cluster0_dependency_list, initial_cluster1_dependency_list, initial_cluster2_dependency_list, initial_cluster3_dependency_list, initial_cluster4_dependency_list, initial_cluster5_dependency_list, initial_cluster6_dependency_list, initial_cluster7_dependency_list]
initial_cluster_sub0_dependency_lists = [initial_cluster0_sub0_dependency_list, initial_cluster1_sub0_dependency_list, initial_cluster2_sub0_dependency_list, initial_cluster3_sub0_dependency_list, initial_cluster4_sub0_dependency_list, initial_cluster5_sub0_dependency_list, initial_cluster6_sub0_dependency_list, initial_cluster7_sub0_dependency_list]
initial_cluster_sub1_dependency_lists = [initial_cluster0_sub1_dependency_list, initial_cluster1_sub1_dependency_list, initial_cluster2_sub1_dependency_list, initial_cluster3_sub1_dependency_list, initial_cluster4_sub1_dependency_list, initial_cluster5_sub1_dependency_list, initial_cluster6_sub1_dependency_list, initial_cluster7_sub1_dependency_list]
'''

tpuv3_expand_part = '''for cluster_idx, cluster_step in enumerate([0, 1]):
    for layer_step in range(2):'''

tpuv3_expand_cal = '''intra_0_time = segment_time(finished_events_list, record_event_tags, [-1], [0])
intra_1_time = segment_time(finished_events_list, record_event_tags, [0], [1])
intra_2_time = segment_time(finished_events_list, record_event_tags, [2], [3])
inter_0_time = segment_time(finished_events_list, record_event_tags, [1], [2])

whole_time = intra_0_time + intra_2_time * (model_config['n_layer'] - clusters_number - 1) + (intra_1_time + inter_0_time) * (clusters_number - 1)
'''




# mapping_dict = {'tpuv3': tpuv3_mapping}
# part_dict = {'tpuv3': tpuv3_part}
# cal_dict = {'tpuv3': tpuv3_cal}
mapping_dict = {'a100': a100_mapping, 'dojo': dojo_mapping, 'tpuv3': tpuv3_mapping, 'a100_scaling': a100_scaling_mapping, 'dojo_scaling': dojo_scaling_mapping, 'tpuv3_scaling': tpuv3_scaling_mapping, 'a100_expand': a100_expand_mapping, 'dojo_expand': dojo_expand_mapping, 'tpuv3_expand': tpuv3_expand_mapping}
part_dict = {'a100': a100_part, 'dojo': dojo_part, 'tpuv3': tpuv3_part, 'a100_scaling': a100_scaling_part, 'dojo_scaling': dojo_scaling_part, 'tpuv3_scaling': tpuv3_scaling_part, 'a100_expand': a100_expand_part, 'dojo_expand': dojo_expand_part, 'tpuv3_expand': tpuv3_expand_part}
cal_dict = {'a100': a100_cal, 'dojo': dojo_cal, 'tpuv3': tpuv3_cal, 'a100_scaling': a100_scaling_cal, 'dojo_scaling': dojo_scaling_cal, 'tpuv3_scaling': tpuv3_scaling_cal, 'a100_expand': a100_expand_cal, 'dojo_expand': dojo_expand_cal, 'tpuv3_expand': tpuv3_expand_cal}
