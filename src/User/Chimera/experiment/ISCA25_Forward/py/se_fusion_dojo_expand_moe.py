
import os
import sys
file_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(file_path, '../../../../../computation/'))
sys.path.append(os.path.join(file_path, '../../../../../computation/gpt2/'))
sys.path.append(os.path.join(file_path, '../../../parallism/'))
sys.path.append(os.path.join(file_path, '../../../../../communication/collective_communication/'))
sys.path.append(os.path.join(file_path, '../../../../../components/'))
sys.path.append(os.path.join(file_path, '../../../../../utils/'))


from compute_backward import compute_2d_base
from moe_backward import embedding, attention, mlp, lm_head, moe
from sequence_expert import sequence_expert
from allreduce import allreduce
from reducescatter import reducescatter
from allgather import allgather
from alltoall import alltoall, ordertoorder
from multicast import pointtopoint, manytomanymulticast
from Launcher import build, launch
from utils import get_gpt2_config



computation = 128 * 1024
reduction_cores = 1 / 256


data_bytes = 2
batch_size = 1
sequence_length = 256

model_name = 'deepspeedmoe-1.3b'
config_path = os.path.join(file_path, '../../../../../computation/gpt2/input/deepspeedmoe-1.3b-config.json')
model_config = get_gpt2_config(model_name, config_path)



pass_bytes = data_bytes * batch_size * sequence_length * model_config['n_embd']

embedding_layer = embedding(databytes=data_bytes, vocab_size=model_config['vocab_size'], hidden_states=model_config['n_embd'])
embedding_layer.cal(batch_size=batch_size, sequence_length=sequence_length)
embedding_bytes = embedding_layer.cal_reorder

lm_head_layer = lm_head(databytes=data_bytes, vocab_size=model_config['vocab_size'], hidden_states=model_config['n_embd'])
lm_head_layer.cal(batch_size=batch_size, sequence_length=sequence_length)
lm_head_bytes = lm_head_layer.cal_tensor

attention_layer = attention(databytes=data_bytes, hidden_states=model_config['n_embd'], num_heads=model_config['n_head'], attn_head_size=model_config['n_embd'] // model_config['n_head'], use_cache=False)
attention_layer.cal(batch_size=batch_size, sequence_length=sequence_length)
attention_bytes = attention_layer.cal_tensor

mlp_layer = mlp(databytes=data_bytes, hidden_states=model_config['n_embd'])
mlp_layer.cal(batch_size=batch_size, sequence_length=sequence_length)
mlp_bytes = mlp_layer.cal_tensor

moe_layer = moe(databytes=data_bytes, hidden_states=model_config['n_embd'])
moe_layer.cal(batch_size=batch_size, sequence_length=sequence_length, topk=model_config['top_k'])
moe_bytes = moe_layer.cal_tensor
clusters_number = 8

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
cluster_layers_number = model_config['n_layer'] // clusters_number
record_event_tags = []
for cluster_idx, cluster_step in enumerate([0, 1]):
    for layer_step in range(2):
        if cluster_idx == 0:

            if layer_step == 0:

                cluster_sub0_allgather_event_tag, cluster_sub0_allgather_dependency_list = allgather_api.cal_time(
                    whole_nodes=node_network, current_event_tag=initial_event_tag, current_dependency_list=initial_cluster_sub0_dependency_lists[cluster_step],
                    source_nodes_coordinates_list=cluster_sub0_nodes_lists[cluster_step],
                    source_x_number=cluster_sub0_shapes[cluster_step][1], source_y_number=cluster_sub0_shapes[cluster_step][0],
                    topology_x_limitation=node_k, topology_y_limitation=node_k,
                    message_flits=pass_bytes,
                    reduction=reduction_cores
                )

                cluster_sub1_allgather_event_tag, cluster_sub1_allgather_dependency_list = allgather_api.cal_time(
                    whole_nodes=node_network, current_event_tag=cluster_sub0_allgather_event_tag, current_dependency_list=initial_cluster_sub1_dependency_lists[cluster_step],
                    source_nodes_coordinates_list=cluster_sub0_nodes_lists[cluster_step],
                    source_x_number=cluster_sub1_shapes[cluster_step][1], source_y_number=cluster_sub1_shapes[cluster_step][0],
                    topology_x_limitation=node_k, topology_y_limitation=node_k,
                    message_flits=pass_bytes,
                    reduction=reduction_cores
                )

            else:

                cluster_sub0_allgather_event_tag, cluster_sub0_allgather_dependency_list = allgather_api.cal_time(
                    whole_nodes=node_network, current_event_tag=cluster_se_event_tag, current_dependency_list=cluster_sub0_se_dependency_list,
                    source_nodes_coordinates_list=cluster_sub0_nodes_lists[cluster_step],
                    source_x_number=cluster_sub0_shapes[cluster_step][1], source_y_number=cluster_sub0_shapes[cluster_step][0],
                    topology_x_limitation=node_k, topology_y_limitation=node_k,
                    message_flits=pass_bytes,
                    reduction=reduction_cores
                )

                cluster_sub1_allgather_event_tag, cluster_sub1_allgather_dependency_list = allgather_api.cal_time(
                    whole_nodes=node_network, current_event_tag=cluster_sub0_allgather_event_tag, current_dependency_list=cluster_sub1_se_dependency_list,
                    source_nodes_coordinates_list=cluster_sub1_nodes_lists[cluster_step],
                    source_x_number=cluster_sub1_shapes[cluster_step][1], source_y_number=cluster_sub1_shapes[cluster_step][0],
                    topology_x_limitation=node_k, topology_y_limitation=node_k,
                    message_flits=pass_bytes,
                    reduction=reduction_cores
                )

        else:

            if layer_step == 0:
                cluster_sub0_allgather_event_tag = cluster_sub1_pointtopoint_event_tag
                cluster_sub0_allgather_dependency_list = cluster_sub0_pointtopoint_dependency_list
                cluster_sub1_allgather_dependency_list = cluster_sub1_pointtopoint_dependency_list

            else:

                cluster_sub0_allgather_event_tag, cluster_sub0_allgather_dependency_list = allgather_api.cal_time(
                    whole_nodes=node_network, current_event_tag=cluster_se_event_tag, current_dependency_list=cluster_sub0_se_dependency_list,
                    source_nodes_coordinates_list=cluster_sub0_nodes_lists[cluster_step],
                    source_x_number=cluster_sub0_shapes[cluster_step][1], source_y_number=cluster_sub0_shapes[cluster_step][0],
                    topology_x_limitation=node_k, topology_y_limitation=node_k,
                    message_flits=pass_bytes,
                    reduction=reduction_cores
                )

                cluster_sub1_allgather_event_tag, cluster_sub1_allgather_dependency_list = allgather_api.cal_time(
                    whole_nodes=node_network, current_event_tag=cluster_sub0_allgather_event_tag, current_dependency_list=cluster_sub1_se_dependency_list,
                    source_nodes_coordinates_list=cluster_sub1_nodes_lists[cluster_step],
                    source_x_number=cluster_sub1_shapes[cluster_step][1], source_y_number=cluster_sub1_shapes[cluster_step][0],
                    topology_x_limitation=node_k, topology_y_limitation=node_k,
                    message_flits=pass_bytes,
                    reduction=reduction_cores
                )


        cluster_sub0_attention_event_tag, cluster_sub0_attention_dependency_list = compute_2d_base(
            whole_nodes=node_network, current_event_tag=cluster_sub1_allgather_event_tag, current_dependency_list=cluster_sub0_allgather_dependency_list,
            source_nodes_coordinates_list=cluster_sub0_nodes_lists[cluster_step],
            source_x_number=cluster_sub0_shapes[cluster_step][1], source_y_number=cluster_sub0_shapes[cluster_step][0],
            topology_x_limitation=node_k, topology_y_limitation=node_k,
            message_flits=attention_bytes/cluster_sub0_shapes[cluster_step][0]/cluster_sub0_shapes[cluster_step][1],
            reduction=computation
        )

        cluster_sub1_attention_event_tag, cluster_sub1_attention_dependency_list = compute_2d_base(
            whole_nodes=node_network, current_event_tag=cluster_sub0_attention_event_tag, current_dependency_list=cluster_sub1_allgather_dependency_list,
            source_nodes_coordinates_list=cluster_sub1_nodes_lists[cluster_step],
            source_x_number=cluster_sub1_shapes[cluster_step][1], source_y_number=cluster_sub1_shapes[cluster_step][0],
            topology_x_limitation=node_k, topology_y_limitation=node_k,
            message_flits=attention_bytes/cluster_sub1_shapes[cluster_step][0]/cluster_sub1_shapes[cluster_step][1],
            reduction=computation
        )

        cluster_sub0_reducescatter_event_tag, cluster_sub0_reducescatter_dependency_list = reducescatter_api.cal_time(
            whole_nodes=node_network, current_event_tag=cluster_sub1_attention_event_tag, current_dependency_list=cluster_sub0_attention_dependency_list,
            source_nodes_coordinates_list=cluster_sub0_nodes_lists[cluster_step],
            source_x_number=cluster_sub0_shapes[cluster_step][1], source_y_number=cluster_sub0_shapes[cluster_step][0],
            topology_x_limitation=node_k, topology_y_limitation=node_k,
            message_flits=pass_bytes,
            reduction=reduction_cores
        )

        cluster_sub1_reducescatter_event_tag, cluster_sub1_reducescatter_dependency_list = reducescatter_api.cal_time(
            whole_nodes=node_network, current_event_tag=cluster_sub0_reducescatter_event_tag, current_dependency_list=cluster_sub1_attention_dependency_list,
            source_nodes_coordinates_list=cluster_sub1_nodes_lists[cluster_step],
            source_x_number=cluster_sub1_shapes[cluster_step][1], source_y_number=cluster_sub1_shapes[cluster_step][0],
            topology_x_limitation=node_k, topology_y_limitation=node_k,
            message_flits=pass_bytes,
            reduction=reduction_cores
        )

        cluster_sub0_allgather_event_tag, cluster_sub0_allgather_dependency_list = allgather_api.cal_time(
            whole_nodes=node_network, current_event_tag=cluster_sub1_reducescatter_event_tag, current_dependency_list=cluster_sub0_reducescatter_dependency_list,
            source_nodes_coordinates_list=cluster_sub0_nodes_lists[cluster_step],
            source_x_number=cluster_sub0_shapes[cluster_step][1], source_y_number=cluster_sub0_shapes[cluster_step][0],
            topology_x_limitation=node_k, topology_y_limitation=node_k,
            message_flits=pass_bytes,
            reduction=reduction_cores
        )

        cluster_sub1_allgather_event_tag, cluster_sub1_allgather_dependency_list = allgather_api.cal_time(
            whole_nodes=node_network, current_event_tag=cluster_sub0_allgather_event_tag, current_dependency_list=cluster_sub1_reducescatter_dependency_list,
            source_nodes_coordinates_list=cluster_sub1_nodes_lists[cluster_step],
            source_x_number=cluster_sub1_shapes[cluster_step][1], source_y_number=cluster_sub1_shapes[cluster_step][0],
            topology_x_limitation=node_k, topology_y_limitation=node_k,
            message_flits=pass_bytes,
            reduction=reduction_cores
        )

        cluster_sub0_mlp_event_tag, cluster_sub0_mlp_dependency_list = compute_2d_base(
            whole_nodes=node_network, current_event_tag=cluster_sub1_allgather_event_tag, current_dependency_list=cluster_sub0_allgather_dependency_list,
            source_nodes_coordinates_list=cluster_sub0_nodes_lists[cluster_step],
            source_x_number=cluster_sub0_shapes[cluster_step][1], source_y_number=cluster_sub0_shapes[cluster_step][0],
            topology_x_limitation=node_k, topology_y_limitation=node_k,
            message_flits=mlp_bytes/cluster_sub0_shapes[cluster_step][0]/cluster_sub0_shapes[cluster_step][1],
            reduction=computation
        )

        cluster_sub1_mlp_event_tag, cluster_sub1_mlp_dependency_list = compute_2d_base(
            whole_nodes=node_network, current_event_tag=cluster_sub0_mlp_event_tag, current_dependency_list=cluster_sub1_allgather_dependency_list,
            source_nodes_coordinates_list=cluster_sub1_nodes_lists[cluster_step],
            source_x_number=cluster_sub1_shapes[cluster_step][1], source_y_number=cluster_sub1_shapes[cluster_step][0],
            topology_x_limitation=node_k, topology_y_limitation=node_k,
            message_flits=mlp_bytes/cluster_sub1_shapes[cluster_step][0]/cluster_sub1_shapes[cluster_step][1],
            reduction=computation
        )

        cluster_sub0_reducescatter_event_tag, cluster_sub0_reducescatter_dependency_list = reducescatter_api.cal_time(
            whole_nodes=node_network, current_event_tag=cluster_sub1_mlp_event_tag, current_dependency_list=cluster_sub0_mlp_dependency_list,
            source_nodes_coordinates_list=cluster_sub0_nodes_lists[cluster_step],
            source_x_number=cluster_sub0_shapes[cluster_step][1], source_y_number=cluster_sub0_shapes[cluster_step][0],
            topology_x_limitation=node_k, topology_y_limitation=node_k,
            message_flits=pass_bytes,
            reduction=reduction_cores
        )

        cluster_sub1_reducescatter_event_tag, cluster_sub1_reducescatter_dependency_list = reducescatter_api.cal_time(
            whole_nodes=node_network, current_event_tag=cluster_sub0_reducescatter_event_tag, current_dependency_list=cluster_sub1_mlp_dependency_list,
            source_nodes_coordinates_list=cluster_sub1_nodes_lists[cluster_step],
            source_x_number=cluster_sub1_shapes[cluster_step][1], source_y_number=cluster_sub1_shapes[cluster_step][0],
            topology_x_limitation=node_k, topology_y_limitation=node_k,
            message_flits=pass_bytes,
            reduction=reduction_cores
        )

        cluster_sub0_allgather_event_tag, cluster_sub0_allgather_dependency_list = allgather_api.cal_time(
            whole_nodes=node_network, current_event_tag=cluster_sub1_reducescatter_event_tag, current_dependency_list=cluster_sub0_reducescatter_dependency_list,
            source_nodes_coordinates_list=cluster_sub0_nodes_lists[cluster_step],
            source_x_number=cluster_sub0_shapes[cluster_step][1], source_y_number=cluster_sub0_shapes[cluster_step][0],
            topology_x_limitation=node_k, topology_y_limitation=node_k,
            message_flits=pass_bytes,
            reduction=reduction_cores
        )

        cluster_sub1_allgather_event_tag, cluster_sub1_allgather_dependency_list = allgather_api.cal_time(
            whole_nodes=node_network, current_event_tag=cluster_sub0_allgather_event_tag, current_dependency_list=cluster_sub1_reducescatter_dependency_list,
            source_nodes_coordinates_list=cluster_sub1_nodes_lists[cluster_step],
            source_x_number=cluster_sub1_shapes[cluster_step][1], source_y_number=cluster_sub1_shapes[cluster_step][0],
            topology_x_limitation=node_k, topology_y_limitation=node_k,
            message_flits=pass_bytes,
            reduction=reduction_cores
        )

        cluster_sub0_attention_event_tag, cluster_sub0_attention_dependency_list = compute_2d_base(
            whole_nodes=node_network, current_event_tag=cluster_sub1_allgather_event_tag, current_dependency_list=cluster_sub0_allgather_dependency_list,
            source_nodes_coordinates_list=cluster_sub0_nodes_lists[cluster_step],
            source_x_number=cluster_sub0_shapes[cluster_step][1], source_y_number=cluster_sub0_shapes[cluster_step][0],
            topology_x_limitation=node_k, topology_y_limitation=node_k,
            message_flits=attention_bytes/cluster_sub0_shapes[cluster_step][0]/cluster_sub0_shapes[cluster_step][1],
            reduction=computation
        )

        cluster_sub1_attention_event_tag, cluster_sub1_attention_dependency_list = compute_2d_base(
            whole_nodes=node_network, current_event_tag=cluster_sub0_attention_event_tag, current_dependency_list=cluster_sub1_allgather_dependency_list,
            source_nodes_coordinates_list=cluster_sub1_nodes_lists[cluster_step],
            source_x_number=cluster_sub1_shapes[cluster_step][1], source_y_number=cluster_sub1_shapes[cluster_step][0],
            topology_x_limitation=node_k, topology_y_limitation=node_k,
            message_flits=attention_bytes/cluster_sub1_shapes[cluster_step][0]/cluster_sub1_shapes[cluster_step][1],
            reduction=computation
        )

        cluster_sub0_reducescatter_event_tag, cluster_sub0_reducescatter_dependency_list = reducescatter_api.cal_time(
            whole_nodes=node_network, current_event_tag=cluster_sub1_attention_event_tag, current_dependency_list=cluster_sub0_attention_dependency_list,
            source_nodes_coordinates_list=cluster_sub0_nodes_lists[cluster_step],
            source_x_number=cluster_sub0_shapes[cluster_step][1], source_y_number=cluster_sub0_shapes[cluster_step][0],
            topology_x_limitation=node_k, topology_y_limitation=node_k,
            message_flits=pass_bytes,
            reduction=reduction_cores
        )

        cluster_sub1_reducescatter_event_tag, cluster_sub1_reducescatter_dependency_list = reducescatter_api.cal_time(
            whole_nodes=node_network, current_event_tag=cluster_sub0_reducescatter_event_tag, current_dependency_list=cluster_sub1_attention_dependency_list,
            source_nodes_coordinates_list=cluster_sub1_nodes_lists[cluster_step],
            source_x_number=cluster_sub1_shapes[cluster_step][1], source_y_number=cluster_sub1_shapes[cluster_step][0],
            topology_x_limitation=node_k, topology_y_limitation=node_k,
            message_flits=pass_bytes,
            reduction=reduction_cores
        )

        cluster_reducescatter_event_tag = cluster_sub1_reducescatter_event_tag
        cluster_reducescatter_events = []
        for dependency_y in cluster_sub0_reducescatter_dependency_list:
            for dependency_x in dependency_y:
                for dependency_event in dependency_x:
                    if dependency_event not in cluster_reducescatter_events:
                        cluster_reducescatter_events.append(dependency_event)
        for dependency_y in cluster_sub1_reducescatter_dependency_list:
            for dependency_x in dependency_y:
                for dependency_event in dependency_x:
                    if dependency_event not in cluster_reducescatter_events:
                        cluster_reducescatter_events.append(dependency_event)
        cluster_reducescatter_dependency_list = [[cluster_reducescatter_events for _ in range(cluster_shapes[cluster_step][1])] for _ in range(cluster_shapes[cluster_step][0])]

        cluster_se_event_tag, cluster_se_dependency_list = se.fusion(
            whole_nodes=node_network, current_event_tag=cluster_reducescatter_event_tag, current_dependency_list=cluster_reducescatter_dependency_list,
            source_nodes_coordinates_list=cluster_nodes_lists[cluster_step],
            source_x_number=cluster_shapes[cluster_step][1], source_y_number=cluster_shapes[cluster_step][0],
            topology_x_limitation=node_k, topology_y_limitation=node_k,
            data_parallelism_degree=cluster_dataparallels[cluster_step], top_k=model_config['top_k'],
            message_flits=pass_bytes,
            reduction=reduction_cores
        )

        cluster_moe_event_tag, cluster_moe_dependency_list = compute_2d_base(
            whole_nodes=node_network, current_event_tag=cluster_se_event_tag, current_dependency_list=cluster_se_dependency_list,
            source_nodes_coordinates_list=cluster_nodes_lists[cluster_step],
            source_x_number=cluster_shapes[cluster_step][1], source_y_number=cluster_shapes[cluster_step][0],
            topology_x_limitation=node_k, topology_y_limitation=node_k,
            message_flits=moe_bytes*cluster_dataparallels[cluster_step][0]*cluster_dataparallels[cluster_step][1]/cluster_shapes[cluster_step][0]/cluster_shapes[cluster_step][1],
            reduction=computation
        )

        cluster_se_event_tag, cluster_se_dependency_list = se.fusion(
            whole_nodes=node_network, current_event_tag=cluster_moe_event_tag, current_dependency_list=cluster_moe_dependency_list,
            source_nodes_coordinates_list=cluster_nodes_lists[cluster_step],
            source_x_number=cluster_shapes[cluster_step][1], source_y_number=cluster_shapes[cluster_step][0],
            topology_x_limitation=node_k, topology_y_limitation=node_k,
            data_parallelism_degree=cluster_dataparallels[cluster_step], top_k=model_config['top_k'],
            message_flits=pass_bytes,
            reduction=reduction_cores
        )
        record_event_tags.append(cluster_se_dependency_list)

        cluster_sub0_se_events = []
        for dependency_y in cluster_se_dependency_list:
            for dependency_x in dependency_y:
                for dependency_event in dependency_x:
                    if dependency_event not in cluster_sub0_se_events:
                        cluster_sub0_se_events.append(dependency_event)
        cluster_sub0_se_dependency_list = [[cluster_sub0_se_events for _ in range(cluster_sub0_shapes[cluster_step][1])] for _ in range(cluster_sub0_shapes[cluster_step][0])]

        cluster_sub1_se_events = []
        for dependency_y in cluster_se_dependency_list:
            for dependency_x in dependency_y:
                for dependency_event in dependency_x:
                    if dependency_event not in cluster_sub1_se_events:
                        cluster_sub1_se_events.append(dependency_event)
        cluster_sub1_se_dependency_list = [[cluster_sub1_se_events for _ in range(cluster_sub1_shapes[cluster_step][1])] for _ in range(cluster_sub1_shapes[cluster_step][0])]

    if cluster_step < clusters_number - 1:

        cluster_sub0_pointtopoint_event_tag, cluster_sub0_pointtopoint_dependency_list = manytomany_api.cal_time(
            whole_nodes=node_network, current_event_tag=cluster_se_event_tag, current_dependency_list=cluster_se_dependency_list,
            source_nodes_coordinates_list=cluster_sub0_nodes_lists[cluster_step],
            target_nodes_coordinates_list=cluster_sub0_nodes_lists[cluster_step+1],
            source_x_number=cluster_sub0_shapes[cluster_step][1], source_y_number=cluster_sub0_shapes[cluster_step][0],
            target_x_number=cluster_sub0_shapes[cluster_step+1][1], target_y_number=cluster_sub0_shapes[cluster_step+1][0],
            topology_x_limitation=node_k, topology_y_limitation=node_k,
            message_flits=pass_bytes,
            reduction=reduction_cores
        )

        cluster_sub1_pointtopoint_event_tag, cluster_sub1_pointtopoint_dependency_list = manytomany_api.cal_time(
            whole_nodes=node_network, current_event_tag=cluster_sub0_pointtopoint_event_tag, current_dependency_list=cluster_se_dependency_list,
            source_nodes_coordinates_list=cluster_sub1_nodes_lists[cluster_step],
            target_nodes_coordinates_list=cluster_sub1_nodes_lists[cluster_step+1],
            source_x_number=cluster_sub1_shapes[cluster_step][1], source_y_number=cluster_sub1_shapes[cluster_step][0],
            target_x_number=cluster_sub1_shapes[cluster_step+1][1], target_y_number=cluster_sub1_shapes[cluster_step+1][0],
            topology_x_limitation=node_k, topology_y_limitation=node_k,
            message_flits=pass_bytes,
            reduction=reduction_cores
        )
        
        cluster_pointtopoint_events = []
        for dependency_y in cluster_sub0_pointtopoint_dependency_list:
            for dependency_x in dependency_y:
                for dependency_event in dependency_x:
                    if dependency_event not in cluster_pointtopoint_events:
                        cluster_pointtopoint_events.append(dependency_event)
        for dependency_y in cluster_sub1_pointtopoint_dependency_list:
            for dependency_x in dependency_y:
                for dependency_event in dependency_x:
                    if dependency_event not in cluster_pointtopoint_events:
                        cluster_pointtopoint_events.append(dependency_event)
        cluster_pointtopoint_dependency_list = [[cluster_pointtopoint_events for _ in range(cluster_shapes[cluster_step+1][1])] for _ in range(cluster_shapes[cluster_step+1][0])]
        record_event_tags.append(cluster_pointtopoint_dependency_list)

        for sub0_dependency_y in cluster_sub0_pointtopoint_dependency_list:
            for sub0_dependency_x in dependency_y:
                for sub0_dependency_event in dependency_x:

                    for sub1_dependency_y_idx, sub1_dependency_y in enumerate(cluster_sub1_pointtopoint_dependency_list):
                        for sub1_dependency_x_idx, sub1_dependency_x in enumerate(sub1_dependency_y):

                            if dependency_event not in cluster_sub1_pointtopoint_dependency_list[sub1_dependency_y_idx][sub1_dependency_x_idx]:
                                cluster_sub1_pointtopoint_dependency_list[sub1_dependency_y_idx][sub1_dependency_x_idx].append(sub0_dependency_event)

        for sub1_dependency_y in cluster_sub1_pointtopoint_dependency_list:
            for sub1_dependency_x in dependency_y:
                for sub1_dependency_event in dependency_x:

                    for sub0_dependency_y_idx, sub0_dependency_y in enumerate(cluster_sub0_pointtopoint_dependency_list):
                        for sub0_dependency_x_idx, sub0_dependency_x in enumerate(sub0_dependency_y):

                            if dependency_event not in cluster_sub0_pointtopoint_dependency_list[sub0_dependency_y_idx][sub0_dependency_x_idx]:
                                cluster_sub0_pointtopoint_dependency_list[sub0_dependency_y_idx][sub0_dependency_x_idx].append(sub1_dependency_event)

import pickle
with open(os.path.join(file_path,"../txt/"+os.path.splitext(os.path.basename(__file__))[0]+'_record.pkl'), 'wb') as f:
    pickle.dump(record_event_tags, f)





run_cost = launch(
    whole_nodes=node_network, booksim_net=noc_network, print_flag=True, booksim2_flit_units=256
)

import pickle
with open(os.path.join(file_path,"../txt/"+os.path.splitext(os.path.basename(__file__))[0]+'.pkl'), 'wb') as f:
    pickle.dump(node_network.finished_events_list, f)


finished_events_list = pickle.load(open(os.path.join(file_path,"../txt/"+os.path.splitext(os.path.basename(__file__))[0]+'.pkl'), 'rb'))
record_event_tags = pickle.load(open(os.path.join(file_path,"../txt/"+os.path.splitext(os.path.basename(__file__))[0]+'_record.pkl'), 'rb'))

sys.path.append(os.path.join(file_path, '../../'))
from finished_segments import segment_time
intra_0_time = segment_time(finished_events_list, record_event_tags, [-1], [0])
intra_1_time = segment_time(finished_events_list, record_event_tags, [0], [1])
intra_2_time = segment_time(finished_events_list, record_event_tags, [2], [3])
inter_0_time = segment_time(finished_events_list, record_event_tags, [1], [2])

whole_time = intra_0_time + intra_2_time * (model_config['n_layer'] - clusters_number - 1) + (intra_1_time + inter_0_time) * (clusters_number - 1)
print(whole_time)
with open(os.path.join(file_path,"../txt/"+os.path.splitext(os.path.basename(__file__))[0]+'.txt'), 'w') as f:
    f.write(str(whole_time))
