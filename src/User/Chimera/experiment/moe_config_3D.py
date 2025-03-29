# File name  :    moe_config.py
# Author     :    xiaocuicui
# Time       :    2025/02/17 13:38:54
# Version    :    V1.0
# Abstract   :        

import os
import sys
file_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(file_path, '../'))


tpuv4_mapping = '''clusters_number = 2

cluster0_nodes_list = [(0, 0, 0), (0, 0, 1), (0, 1, 0), (0, 1, 1)]
cluster1_nodes_list = [(1, 0, 0), (1, 0, 1), (1, 1, 0), (1, 1, 1)]

cluster0_shape = [1, 2, 2]
cluster1_shape = [1, 2, 2]

cluster0_dataparallel = [2, 1, 1]
cluster1_dataparallel = [2, 1, 1]

cluster0_sub0_nodes_list = [(0, 0, 0), (0, 1, 0)]
cluster0_sub1_nodes_list = [(0, 0, 1), (0, 1, 1)]
cluster1_sub0_nodes_list = [(1, 0, 0), (1, 1, 0)]
cluster1_sub1_nodes_list = [(1, 0, 1), (1, 1, 1)]

cluster0_sub0_shape = [1, 2, 1]
cluster0_sub1_shape = [1, 2, 1]
cluster1_sub0_shape = [1, 2, 1]
cluster1_sub1_shape = [1, 2, 1]

cluster_nodes_lists = [cluster0_nodes_list, cluster1_nodes_list]
cluster_shapes = [cluster0_shape, cluster1_shape]
cluster_dataparallels = [cluster0_dataparallel, cluster1_dataparallel]

cluster_sub0_nodes_lists = [cluster0_sub0_nodes_list, cluster1_sub0_nodes_list]
cluster_sub1_nodes_lists = [cluster0_sub1_nodes_list, cluster1_sub1_nodes_list]

cluster_sub0_shapes = [cluster0_sub0_shape, cluster1_sub0_shape]
cluster_sub1_shapes = [cluster0_sub1_shape, cluster1_sub1_shape]


node_k = 2
node_n = 3
ni_k = node_k
ni_n = node_n
cfg_topology = "torus"
algorithm_dict = {'allgather': 'hierarchicalring', 'allreduce': 'hierarchicalring', 'reducescatter': 'hierarchicalring', 'reducelocal': 'base', 'alltoall': 'hierarchicalring', 'ordertoorder': 'hierarchicalring', 'pointtopoint': 'base', 'manytomanymulticast': 'alpa'}

node_network, noc_network = build_3D(
    node_k=node_k, node_n=node_n, ni_k=ni_k, ni_n=ni_n,
    cfg_topology=cfg_topology,
    cfg_filepath=os.path.join(file_path, "../cfg/tpuv4.cfg")
)

se = sequence_expert(
    topology=("torus3d"),
    algorithm=algorithm_dict
)

allreduce_api = allreduce(
    topology="torus3d",
    algorithm=algorithm_dict["allreduce"]
)

reducescatter_api = reducescatter(
    topology="torus3d",
    algorithm=algorithm_dict["reducescatter"]
)

allgather_api = allgather(
    topology="torus3d",
    algorithm=algorithm_dict["allgather"]
)

alltoall_api = alltoall(
    topology="torus3d",
    algorithm=algorithm_dict["alltoall"]
)

ordertoorder_api = ordertoorder(
    topology="torus3d",
    algorithm=algorithm_dict["ordertoorder"]
)

pointtopoint_api = pointtopoint(
    topology="torus3d",
    algorithm=algorithm_dict["pointtopoint"]
)

manytomany_api = manytomanymulticast(
    topology="torus3d",
    algorithm=algorithm_dict["manytomanymulticast"]
)


initial_event_tag = 0

initial_cluster0_dependency_list = [[[[] for _ in range(cluster0_shape[2])] for _ in range(cluster0_shape[1])] for _ in range(cluster0_shape[0])]
initial_cluster1_dependency_list = [[[[] for _ in range(cluster1_shape[2])] for _ in range(cluster1_shape[1])] for _ in range(cluster1_shape[0])]

initial_cluster0_sub0_dependency_list = [[[[] for _ in range(cluster0_sub0_shape[2])] for _ in range(cluster0_sub0_shape[1])] for _ in range(cluster0_sub0_shape[0])]
initial_cluster0_sub1_dependency_list = [[[[] for _ in range(cluster0_sub1_shape[2])] for _ in range(cluster0_sub1_shape[1])] for _ in range(cluster0_sub1_shape[0])]
initial_cluster1_sub0_dependency_list = [[[[] for _ in range(cluster1_sub0_shape[2])] for _ in range(cluster1_sub0_shape[1])] for _ in range(cluster1_sub0_shape[0])]
initial_cluster1_sub1_dependency_list = [[[[] for _ in range(cluster1_sub1_shape[2])] for _ in range(cluster1_sub1_shape[1])] for _ in range(cluster1_sub1_shape[0])]

initial_cluster_dependency_lists = [initial_cluster0_dependency_list, initial_cluster1_dependency_list]
initial_cluster_sub0_dependency_lists = [initial_cluster0_sub0_dependency_list, initial_cluster1_sub0_dependency_list]
initial_cluster_sub1_dependency_lists = [initial_cluster0_sub1_dependency_list, initial_cluster1_sub1_dependency_list]
'''

tpuv4_part = '''for cluster_idx, cluster_step in enumerate([0, 1]):
    for layer_step in range(2):'''

tpuv4_cal = '''intra_0_time = segment_time_3D(finished_events_list, record_event_tags, [-1], [0])
intra_1_time = segment_time_3D(finished_events_list, record_event_tags, [0], [1])
intra_2_time = segment_time_3D(finished_events_list, record_event_tags, [2], [3])
inter_0_time = segment_time_3D(finished_events_list, record_event_tags, [1], [2])

whole_time = intra_0_time + intra_2_time * ((model_config['n_layer'] // 2) - clusters_number - 1) + (intra_1_time + inter_0_time) * (clusters_number - 1)
'''




tpuv4_scaling_mapping = '''clusters_number = 8

cluster0_nodes_list = [(0, 0, 0), (0, 0, 1), (0, 0, 2), (0, 0, 3), (0, 1, 0), (0, 1, 1), (0, 1, 2), (0, 1, 3)]
cluster1_nodes_list = [(0, 2, 0), (0, 2, 1), (0, 2, 2), (0, 2, 3), (0, 3, 0), (0, 3, 1), (0, 3, 2), (0, 3, 3)]
cluster2_nodes_list = [(1, 0, 0), (1, 0, 1), (1, 0, 2), (1, 0, 3), (1, 1, 0), (1, 1, 1), (1, 1, 2), (1, 1, 3)]
cluster3_nodes_list = [(1, 2, 0), (1, 2, 1), (1, 2, 2), (1, 2, 3), (1, 3, 0), (1, 3, 1), (1, 3, 2), (1, 3, 3)]
cluster4_nodes_list = [(2, 0, 0), (2, 0, 1), (2, 0, 2), (2, 0, 3), (2, 1, 0), (2, 1, 1), (2, 1, 2), (2, 1, 3)]
cluster5_nodes_list = [(2, 2, 0), (2, 2, 1), (2, 2, 2), (2, 2, 3), (2, 3, 0), (2, 3, 1), (2, 3, 2), (2, 3, 3)]
cluster6_nodes_list = [(3, 0, 0), (3, 0, 1), (3, 0, 2), (3, 0, 3), (3, 1, 0), (3, 1, 1), (3, 1, 2), (3, 1, 3)]
cluster7_nodes_list = [(3, 2, 0), (3, 2, 1), (3, 2, 2), (3, 2, 3), (3, 3, 0), (3, 3, 1), (3, 3, 2), (3, 3, 3)]

cluster0_shape = [1, 2, 4]
cluster1_shape = [1, 2, 4]
cluster2_shape = [1, 2, 4]
cluster3_shape = [1, 2, 4]
cluster4_shape = [1, 2, 4]
cluster5_shape = [1, 2, 4]
cluster6_shape = [1, 2, 4]
cluster7_shape = [1, 2, 4]

cluster0_dataparallel = [2, 1, 1]
cluster1_dataparallel = [2, 1, 1]
cluster2_dataparallel = [2, 1, 1]
cluster3_dataparallel = [2, 1, 1]
cluster4_dataparallel = [2, 1, 1]
cluster5_dataparallel = [2, 1, 1]
cluster6_dataparallel = [2, 1, 1]
cluster7_dataparallel = [2, 1, 1]

cluster0_sub0_nodes_list = [(0, 0, 0), (0, 0, 1), (0, 1, 0), (0, 1, 1)]
cluster0_sub1_nodes_list = [(0, 0, 2), (0, 0, 3), (0, 1, 2), (0, 1, 3)]
cluster1_sub0_nodes_list = [(0, 2, 0), (0, 2, 1), (0, 3, 0), (0, 3, 1)]
cluster1_sub1_nodes_list = [(0, 2, 2), (0, 2, 3), (0, 3, 2), (0, 3, 3)]
cluster2_sub0_nodes_list = [(1, 0, 0), (1, 0, 1), (1, 1, 0), (1, 1, 1)]
cluster2_sub1_nodes_list = [(1, 0, 2), (1, 0, 3), (1, 1, 2), (1, 1, 3)]
cluster3_sub0_nodes_list = [(1, 2, 0), (1, 2, 1), (1, 3, 0), (1, 3, 1)]
cluster3_sub1_nodes_list = [(1, 2, 2), (1, 2, 3), (1, 3, 2), (1, 3, 3)]
cluster4_sub0_nodes_list = [(2, 0, 0), (2, 0, 1), (2, 1, 0), (2, 1, 1)]
cluster4_sub1_nodes_list = [(2, 0, 2), (2, 0, 3), (2, 1, 2), (2, 1, 3)]
cluster5_sub0_nodes_list = [(2, 2, 0), (2, 2, 1), (2, 3, 0), (2, 3, 1)]
cluster5_sub1_nodes_list = [(2, 2, 2), (2, 2, 3), (2, 3, 2), (2, 3, 3)]
cluster6_sub0_nodes_list = [(3, 0, 0), (3, 0, 1), (3, 1, 0), (3, 1, 1)]
cluster6_sub1_nodes_list = [(3, 0, 2), (3, 0, 3), (3, 1, 2), (3, 1, 3)]
cluster7_sub0_nodes_list = [(3, 2, 0), (3, 2, 1), (3, 3, 0), (3, 3, 1)]
cluster7_sub1_nodes_list = [(3, 2, 2), (3, 2, 3), (3, 3, 2), (3, 3, 3)]

cluster0_sub0_shape = [1, 2, 2]
cluster0_sub1_shape = [1, 2, 2]
cluster1_sub0_shape = [1, 2, 2]
cluster1_sub1_shape = [1, 2, 2]
cluster2_sub0_shape = [1, 2, 2]
cluster2_sub1_shape = [1, 2, 2]
cluster3_sub0_shape = [1, 2, 2]
cluster3_sub1_shape = [1, 2, 2]
cluster4_sub0_shape = [1, 2, 2]
cluster4_sub1_shape = [1, 2, 2]
cluster5_sub0_shape = [1, 2, 2]
cluster5_sub1_shape = [1, 2, 2]
cluster6_sub0_shape = [1, 2, 2]
cluster6_sub1_shape = [1, 2, 2]
cluster7_sub0_shape = [1, 2, 2]
cluster7_sub1_shape = [1, 2, 2]

cluster_nodes_lists = [cluster0_nodes_list, cluster1_nodes_list, cluster2_nodes_list, cluster3_nodes_list, cluster4_nodes_list, cluster5_nodes_list, cluster6_nodes_list, cluster7_nodes_list]
cluster_shapes = [cluster0_shape, cluster1_shape, cluster2_shape, cluster3_shape, cluster4_shape, cluster5_shape, cluster6_shape, cluster7_shape]
cluster_dataparallels = [cluster0_dataparallel, cluster1_dataparallel, cluster2_dataparallel, cluster3_dataparallel, cluster4_dataparallel, cluster5_dataparallel, cluster6_dataparallel, cluster7_dataparallel]

cluster_sub0_nodes_lists = [cluster0_sub0_nodes_list, cluster1_sub0_nodes_list, cluster2_sub0_nodes_list, cluster3_sub0_nodes_list, cluster4_sub0_nodes_list, cluster5_sub0_nodes_list, cluster6_sub0_nodes_list, cluster7_sub0_nodes_list]
cluster_sub1_nodes_lists = [cluster0_sub1_nodes_list, cluster1_sub1_nodes_list, cluster2_sub1_nodes_list, cluster3_sub1_nodes_list, cluster4_sub1_nodes_list, cluster5_sub1_nodes_list, cluster6_sub1_nodes_list, cluster7_sub1_nodes_list]

cluster_sub0_shapes = [cluster0_sub0_shape, cluster1_sub0_shape, cluster2_sub0_shape, cluster3_sub0_shape, cluster4_sub0_shape, cluster5_sub0_shape, cluster6_sub0_shape, cluster7_sub0_shape]
cluster_sub1_shapes = [cluster0_sub1_shape, cluster1_sub1_shape, cluster2_sub1_shape, cluster3_sub1_shape, cluster4_sub1_shape, cluster5_sub1_shape, cluster6_sub1_shape, cluster7_sub1_shape]

node_k = 4
node_n = 3
ni_k = node_k
ni_n = node_n
cfg_topology = "torus"
algorithm_dict = {'allgather': 'hierarchicalring', 'allreduce': 'hierarchicalring', 'reducescatter': 'hierarchicalring', 'reducelocal': 'base', 'alltoall': 'hierarchicalring', 'ordertoorder': 'hierarchicalring', 'pointtopoint': 'base', 'manytomanymulticast': 'alpa'}

node_network, noc_network = build_3D(
    node_k=node_k, node_n=node_n, ni_k=ni_k, ni_n=ni_n, 
    cfg_topology=cfg_topology, 
    cfg_filepath=os.path.join(file_path, "../cfg/tpuv4_scaling.cfg")
)

se = sequence_expert(
    topology=("torus3d"), 
    algorithm=algorithm_dict
)

allreduce_api = allreduce(
    topology="torus3d", 
    algorithm=algorithm_dict["allreduce"]
)

reducescatter_api = reducescatter(
    topology="torus3d",
    algorithm=algorithm_dict["reducescatter"]
)

allgather_api = allgather(
    topology="torus3d",
    algorithm=algorithm_dict["allgather"]
)

alltoall_api = alltoall(
    topology="torus3d",
    algorithm=algorithm_dict["alltoall"]
)

ordertoorder_api = ordertoorder(
    topology="torus3d",
    algorithm=algorithm_dict["ordertoorder"]
)

pointtopoint_api = pointtopoint(
    topology="torus3d",
    algorithm=algorithm_dict["pointtopoint"]
)

manytomany_api = manytomanymulticast(
    topology="torus3d",
    algorithm=algorithm_dict["manytomanymulticast"]
)

initial_event_tag = 0

initial_cluster0_dependency_list = [[[[] for _ in range(cluster0_shape[2])] for _ in range(cluster0_shape[1])] for _ in range(cluster0_shape[0])]
initial_cluster1_dependency_list = [[[[] for _ in range(cluster1_shape[2])] for _ in range(cluster1_shape[1])] for _ in range(cluster0_shape[0])]
initial_cluster2_dependency_list = [[[[] for _ in range(cluster2_shape[2])] for _ in range(cluster2_shape[1])] for _ in range(cluster0_shape[0])]
initial_cluster3_dependency_list = [[[[] for _ in range(cluster3_shape[2])] for _ in range(cluster3_shape[1])] for _ in range(cluster0_shape[0])]
initial_cluster4_dependency_list = [[[[] for _ in range(cluster4_shape[2])] for _ in range(cluster4_shape[1])] for _ in range(cluster0_shape[0])]
initial_cluster5_dependency_list = [[[[] for _ in range(cluster5_shape[2])] for _ in range(cluster5_shape[1])] for _ in range(cluster0_shape[0])]
initial_cluster6_dependency_list = [[[[] for _ in range(cluster6_shape[2])] for _ in range(cluster6_shape[1])] for _ in range(cluster0_shape[0])]
initial_cluster7_dependency_list = [[[[] for _ in range(cluster7_shape[2])] for _ in range(cluster7_shape[1])] for _ in range(cluster0_shape[0])]

initial_cluster0_sub0_dependency_list = [[[[] for _ in range(cluster0_sub0_shape[2])] for _ in range(cluster0_sub0_shape[1])] for _ in range(cluster0_sub0_shape[0])]
initial_cluster0_sub1_dependency_list = [[[[] for _ in range(cluster0_sub1_shape[2])] for _ in range(cluster0_sub1_shape[1])] for _ in range(cluster0_sub0_shape[0])]
initial_cluster1_sub0_dependency_list = [[[[] for _ in range(cluster1_sub0_shape[2])] for _ in range(cluster1_sub0_shape[1])] for _ in range(cluster1_sub0_shape[0])]
initial_cluster1_sub1_dependency_list = [[[[] for _ in range(cluster1_sub1_shape[2])] for _ in range(cluster1_sub1_shape[1])] for _ in range(cluster1_sub0_shape[0])]
initial_cluster2_sub0_dependency_list = [[[[] for _ in range(cluster2_sub0_shape[2])] for _ in range(cluster2_sub0_shape[1])] for _ in range(cluster2_sub0_shape[0])]
initial_cluster2_sub1_dependency_list = [[[[] for _ in range(cluster2_sub1_shape[2])] for _ in range(cluster2_sub1_shape[1])] for _ in range(cluster2_sub0_shape[0])]
initial_cluster3_sub0_dependency_list = [[[[] for _ in range(cluster3_sub0_shape[2])] for _ in range(cluster3_sub0_shape[1])] for _ in range(cluster3_sub0_shape[0])]
initial_cluster3_sub1_dependency_list = [[[[] for _ in range(cluster3_sub1_shape[2])] for _ in range(cluster3_sub1_shape[1])] for _ in range(cluster3_sub0_shape[0])]
initial_cluster4_sub0_dependency_list = [[[[] for _ in range(cluster4_sub0_shape[2])] for _ in range(cluster4_sub0_shape[1])] for _ in range(cluster4_sub0_shape[0])]
initial_cluster4_sub1_dependency_list = [[[[] for _ in range(cluster4_sub1_shape[2])] for _ in range(cluster4_sub1_shape[1])] for _ in range(cluster4_sub0_shape[0])]
initial_cluster5_sub0_dependency_list = [[[[] for _ in range(cluster5_sub0_shape[2])] for _ in range(cluster5_sub0_shape[1])] for _ in range(cluster5_sub0_shape[0])]
initial_cluster5_sub1_dependency_list = [[[[] for _ in range(cluster5_sub1_shape[2])] for _ in range(cluster5_sub1_shape[1])] for _ in range(cluster5_sub0_shape[0])]
initial_cluster6_sub0_dependency_list = [[[[] for _ in range(cluster6_sub0_shape[2])] for _ in range(cluster6_sub0_shape[1])] for _ in range(cluster6_sub0_shape[0])]
initial_cluster6_sub1_dependency_list = [[[[] for _ in range(cluster6_sub1_shape[2])] for _ in range(cluster6_sub1_shape[1])] for _ in range(cluster6_sub0_shape[0])]
initial_cluster7_sub0_dependency_list = [[[[] for _ in range(cluster7_sub0_shape[2])] for _ in range(cluster7_sub0_shape[1])] for _ in range(cluster7_sub0_shape[0])]
initial_cluster7_sub1_dependency_list = [[[[] for _ in range(cluster7_sub1_shape[2])] for _ in range(cluster7_sub1_shape[1])] for _ in range(cluster7_sub0_shape[0])]

initial_cluster_dependency_lists = [initial_cluster0_dependency_list, initial_cluster1_dependency_list, initial_cluster2_dependency_list, initial_cluster3_dependency_list, initial_cluster4_dependency_list, initial_cluster5_dependency_list, initial_cluster6_dependency_list, initial_cluster7_dependency_list]                                     
initial_cluster_sub0_dependency_lists = [initial_cluster0_sub0_dependency_list, initial_cluster1_sub0_dependency_list, initial_cluster2_sub0_dependency_list, initial_cluster3_sub0_dependency_list, initial_cluster4_sub0_dependency_list, initial_cluster5_sub0_dependency_list, initial_cluster6_sub0_dependency_list, initial_cluster7_sub0_dependency_list]
initial_cluster_sub1_dependency_lists = [initial_cluster0_sub1_dependency_list, initial_cluster1_sub1_dependency_list, initial_cluster2_sub1_dependency_list, initial_cluster3_sub1_dependency_list, initial_cluster4_sub1_dependency_list, initial_cluster5_sub1_dependency_list, initial_cluster6_sub1_dependency_list, initial_cluster7_sub1_dependency_list]
'''

tpuv4_scaling_part = '''for cluster_idx, cluster_step in enumerate([2, 3, 4]):
    for layer_step in range(2):'''

tpuv4_scaling_cal = '''intra_0_time = segment_time_3D(finished_events_list, record_event_tags, [-1], [0])
intra_1_time = segment_time_3D(finished_events_list, record_event_tags, [0], [1])
intra_2_time = segment_time_3D(finished_events_list, record_event_tags, [2], [3])
intra_3_time = segment_time_3D(finished_events_list, record_event_tags, [5], [6])
inter_0_time = segment_time_3D(finished_events_list, record_event_tags, [1], [2])
inter_1_time = segment_time_3D(finished_events_list, record_event_tags, [4], [5])

whole_time = intra_0_time + intra_2_time * (clusters_number - 2) + intra_3_time + intra_1_time * (model_config['n_layer'] - clusters_number) + inter_0_time * (clusters_number - 2) + inter_1_time
'''




# mapping_dict = {'tpuv4': tpuv4_mapping}
# part_dict = {'tpuv4': tpuv4_part}
# cal_dict = {'tpuv4': tpuv4_cal}
mapping_dict = {'tpuv4': tpuv4_mapping, 'tpuv4_scaling': tpuv4_scaling_mapping}
part_dict = {'tpuv4': tpuv4_part, 'tpuv4_scaling': tpuv4_scaling_part}
cal_dict = {'tpuv4': tpuv4_cal, 'tpuv4_scaling': tpuv4_scaling_cal}
