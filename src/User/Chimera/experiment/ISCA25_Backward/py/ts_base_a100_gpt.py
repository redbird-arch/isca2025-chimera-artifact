
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
from gpt2_backward import embedding, attention, mlp, lm_head, attention_input_backward, attention_weight_backward, mlp_input_backward, mlp_weight_backward
from tensor_pipeline import tensor_pipeline
from allreduce import allreduce
from reducescatter import reducescatter
from allgather import allgather
from multicast import pointtopoint, manytomanymulticast
from Launcher import build, launch
from utils import get_gpt2_config

import math



computation = 128*1024
reduction_cores = 1 / 256


data_bytes = 4
batch_size = 1
sequence_length = 256

model_name = 'gpt2-medium'
config_path = os.path.join(file_path, '../../../../../computation/gpt2/input/gpt2-medium-config.json')
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

attention_input_backward_layer = attention_input_backward(databytes=data_bytes, hidden_states=model_config['n_embd'], num_heads=model_config['n_head'], attn_head_size=model_config['n_embd'] // model_config['n_head'], use_cache=False)
attention_input_backward_layer.cal(batch_size=batch_size, sequence_length=sequence_length)
attention_input_backward_bytes = attention_input_backward_layer.cal_tensor

attention_weight_backward_layer = attention_weight_backward(databytes=data_bytes, hidden_states=model_config['n_embd'], num_heads=model_config['n_head'], attn_head_size=model_config['n_embd'] // model_config['n_head'], use_cache=False)
attention_weight_backward_layer.cal(batch_size=batch_size, sequence_length=sequence_length)
attention_weight_backward_bytes = attention_weight_backward_layer.cal_tensor

mlp_layer = mlp(databytes=data_bytes, hidden_states=model_config['n_embd'])
mlp_layer.cal(batch_size=batch_size, sequence_length=sequence_length)
mlp_bytes = mlp_layer.cal_tensor

mlp_input_backward_layer = mlp_input_backward(databytes=data_bytes, hidden_states=model_config['n_embd'])
mlp_input_backward_layer.cal(batch_size=batch_size, sequence_length=sequence_length)
mlp_input_backward_bytes = mlp_input_backward_layer.cal_tensor

mlp_weight_backward_layer = mlp_weight_backward(databytes=data_bytes, hidden_states=model_config['n_embd'])
mlp_weight_backward_layer.cal(batch_size=batch_size, sequence_length=sequence_length)
mlp_weight_backward_bytes = mlp_weight_backward_layer.cal_tensor
clusters_number = 8

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
cluster_layers_number = model_config['n_layer'] // clusters_number
record_event_tags = []
for cluster_idx, cluster_step in enumerate([2, 3, 4]):
    for layer_step in range(2):
        if cluster_idx == 0:
            if layer_step == 0:
                cluster_allreduce_event_tag, cluster_allreduce_dependency_list = allreduce_api.cal_time(
                    whole_nodes=node_network, current_event_tag=initial_event_tag, current_dependency_list=initial_cluster_dependency_lists[0],
                    source_nodes_coordinates_list=cluster_nodes_lists[cluster_step],
                    source_x_number=cluster_shapes[cluster_step][1], source_y_number=cluster_shapes[cluster_step][0],
                    topology_x_limitation=node_k, topology_y_limitation=node_n,
                    message_flits=pass_bytes,
                    reduction=reduction_cores
                )
            else:
                cluster_allreduce_event_tag, cluster_allreduce_dependency_list = allreduce_api.cal_time(
                    whole_nodes=node_network, current_event_tag=cluster_attention_weight_event_tag, current_dependency_list=cluster_attention_weight_dependency_list,
                    source_nodes_coordinates_list=cluster_nodes_lists[cluster_step],
                    source_x_number=cluster_shapes[cluster_step][1], source_y_number=cluster_shapes[cluster_step][0],
                    topology_x_limitation=node_k, topology_y_limitation=node_n,
                    message_flits=pass_bytes,
                    reduction=reduction_cores
                )
        else:
            if layer_step == 0:
                cluster_allreduce_event_tag, cluster_allreduce_dependency_list = allreduce_api.cal_time(
                    whole_nodes=node_network, current_event_tag=cluster_pointtopoint_event_tag, current_dependency_list=cluster_pointtopoint_dependency_list,
                    source_nodes_coordinates_list=cluster_nodes_lists[cluster_step],
                    source_x_number=cluster_shapes[cluster_step][1], source_y_number=cluster_shapes[cluster_step][0],
                    topology_x_limitation=node_k, topology_y_limitation=node_n,
                    message_flits=pass_bytes,
                    reduction=reduction_cores
                )
            else:
                cluster_allreduce_event_tag, cluster_allreduce_dependency_list = allreduce_api.cal_time(
                    whole_nodes=node_network, current_event_tag=cluster_attention_weight_event_tag, current_dependency_list=cluster_attention_weight_dependency_list,
                    source_nodes_coordinates_list=cluster_nodes_lists[cluster_step],
                    source_x_number=cluster_shapes[cluster_step][1], source_y_number=cluster_shapes[cluster_step][0],
                    topology_x_limitation=node_k, topology_y_limitation=node_n,
                    message_flits=pass_bytes,
                    reduction=reduction_cores
                )

        if layer_step == 0:
            cluster_mlp_input_event_tag, cluster_mlp_input_dependency_list = compute_2d_base(
                whole_nodes=node_network, current_event_tag=cluster_allreduce_event_tag, current_dependency_list=cluster_allreduce_dependency_list,
                source_nodes_coordinates_list=cluster_nodes_lists[cluster_step],
                source_x_number=cluster_shapes[cluster_step][1], source_y_number=cluster_shapes[cluster_step][0],
                topology_x_limitation=node_k, topology_y_limitation=node_n,
                message_flits=mlp_input_backward_bytes/cluster_shapes[cluster_step][0]/cluster_shapes[cluster_step][1],
                reduction=computation
            )
        else:
            cluster_weight_allreduce_dependency_list = []
            for sublist1, sublist2 in zip(cluster_attention_weight_dependency_list, cluster_allreduce_dependency_list):
                merged_sublist = []
                for pair1, pair2 in zip(sublist1, sublist2):
                    merged_sublist.append(pair1 + pair2)
                cluster_weight_allreduce_dependency_list.append(merged_sublist)

            cluster_mlp_input_event_tag, cluster_mlp_input_dependency_list = compute_2d_base(
                whole_nodes=node_network, current_event_tag=cluster_allreduce_event_tag, current_dependency_list=cluster_weight_allreduce_dependency_list,
                source_nodes_coordinates_list=cluster_nodes_lists[cluster_step],
                source_x_number=cluster_shapes[cluster_step][1], source_y_number=cluster_shapes[cluster_step][0],
                topology_x_limitation=node_k, topology_y_limitation=node_n,
                message_flits=mlp_input_backward_bytes/cluster_shapes[cluster_step][0]/cluster_shapes[cluster_step][1],
                reduction=computation
            )

        cluster_mlp_weight_event_tag, cluster_mlp_weight_dependency_list = compute_2d_base(
            whole_nodes=node_network, current_event_tag=cluster_mlp_input_event_tag, current_dependency_list=cluster_mlp_input_dependency_list,
            source_nodes_coordinates_list=cluster_nodes_lists[cluster_step],
            source_x_number=cluster_shapes[cluster_step][1], source_y_number=cluster_shapes[cluster_step][0],
            topology_x_limitation=node_k, topology_y_limitation=node_n,
            message_flits=mlp_weight_backward_bytes/cluster_shapes[cluster_step][0]/cluster_shapes[cluster_step][1],
            reduction=computation
        )

        cluster_reducescatter_event_tag, cluster_reducescatter_dependency_list = reducescatter_api.cal_time(
            whole_nodes=node_network, current_event_tag=cluster_mlp_weight_event_tag, current_dependency_list=cluster_mlp_input_dependency_list,
            source_nodes_coordinates_list=cluster_nodes_lists[cluster_step],
            source_x_number=cluster_shapes[cluster_step][1], source_y_number=cluster_shapes[cluster_step][0],
            topology_x_limitation=node_k, topology_y_limitation=node_n,
            message_flits=pass_bytes,
            reduction=reduction_cores
        )

        cluster_allreduce_event_tag, cluster_allreduce_dependency_list = allreduce_api.cal_time(
            whole_nodes=node_network, current_event_tag=cluster_reducescatter_event_tag, current_dependency_list=cluster_reducescatter_dependency_list,
            source_nodes_coordinates_list=cluster_nodes_lists[cluster_step],
            source_x_number=cluster_shapes[cluster_step][1], source_y_number=cluster_shapes[cluster_step][0],
            topology_x_limitation=node_k, topology_y_limitation=node_n,
            message_flits=pass_bytes,
            reduction=reduction_cores
        )

        cluster_weight_allreduce_dependency_list = []
        for sublist1, sublist2 in zip(cluster_mlp_weight_dependency_list, cluster_allreduce_dependency_list):
            merged_sublist = []
            for pair1, pair2 in zip(sublist1, sublist2):
                merged_sublist.append(pair1 + pair2)
            cluster_weight_allreduce_dependency_list.append(merged_sublist)

        cluster_attention_input_event_tag, cluster_attention_input_dependency_list = compute_2d_base(
            whole_nodes=node_network, current_event_tag=cluster_allreduce_event_tag, current_dependency_list=cluster_weight_allreduce_dependency_list,
            source_nodes_coordinates_list=cluster_nodes_lists[cluster_step],
            source_x_number=cluster_shapes[cluster_step][1], source_y_number=cluster_shapes[cluster_step][0],
            topology_x_limitation=node_k, topology_y_limitation=node_n,
            message_flits=attention_input_backward_bytes/cluster_shapes[cluster_step][0]/cluster_shapes[cluster_step][1],
            reduction=computation
        )

        cluster_attention_weight_event_tag, cluster_attention_weight_dependency_list = compute_2d_base(
            whole_nodes=node_network, current_event_tag=cluster_attention_input_event_tag, current_dependency_list=cluster_attention_input_dependency_list,
            source_nodes_coordinates_list=cluster_nodes_lists[cluster_step],
            source_x_number=cluster_shapes[cluster_step][1], source_y_number=cluster_shapes[cluster_step][0],
            topology_x_limitation=node_k, topology_y_limitation=node_n,
            message_flits=attention_weight_backward_bytes/cluster_shapes[cluster_step][0]/cluster_shapes[cluster_step][1],
            reduction=computation
        )
        record_event_tags.append(cluster_attention_weight_dependency_list)

    if cluster_step < clusters_number - 1:
        cluster_pointtopoint_event_tag, cluster_pointtopoint_dependency_list = pointtopoint_api.cal_time(
            whole_nodes=node_network, current_event_tag=cluster_attention_weight_event_tag, current_dependency_list=cluster_attention_weight_dependency_list,
            source_nodes_coordinates_list=cluster_nodes_lists[cluster_step],
            target_nodes_coordinates_list=cluster_nodes_lists[cluster_step+1],
            source_x_number=cluster_shapes[cluster_step][1], source_y_number=cluster_shapes[cluster_step][0],
            target_x_number=cluster_shapes[cluster_step+1][1], target_y_number=cluster_shapes[cluster_step+1][0],
            topology_x_limitation=node_k, topology_y_limitation=node_n,
            message_flits=pass_bytes,
            reduction=reduction_cores
        )
        record_event_tags.append(cluster_pointtopoint_dependency_list)

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
intra_3_time = segment_time(finished_events_list, record_event_tags, [5], [6])
inter_0_time = segment_time(finished_events_list, record_event_tags, [1], [2])
inter_1_time = segment_time(finished_events_list, record_event_tags, [4], [5])

whole_time = intra_0_time + intra_2_time * (clusters_number - 2) + intra_3_time + intra_1_time * (model_config['n_layer'] - clusters_number) + inter_0_time * (clusters_number - 2) + inter_1_time
print(whole_time)
with open(os.path.join(file_path,"../txt/"+os.path.splitext(os.path.basename(__file__))[0]+'.txt'), 'w') as f:
    f.write(str(whole_time))
