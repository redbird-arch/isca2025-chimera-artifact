# File name  :    te_fusion_moe.py
# Author     :    xiaocuicui
# Time       :    2025/02/17 12:27:19
# Version    :    V1.0
# Abstract   :        

import os
import sys
file_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(file_path)
sys.path.append(os.path.join(file_path, '../'))

from moe_config import mapping_dict, part_dict, cal_dict

target_folder = os.path.join(file_path, './ISCA25_Backward')
target_folder_cfg = os.path.join(file_path, './ISCA25_Backward/cfg')
target_folder_txt = os.path.join(file_path, './ISCA25_Backward/txt')
target_folder_py = os.path.join(file_path, './ISCA25_Backward/py')
if not os.path.exists(target_folder):
    os.makedirs(target_folder)
if not os.path.exists(target_folder_cfg):
    os.makedirs(target_folder_cfg)
if not os.path.exists(target_folder_txt):
    os.makedirs(target_folder_txt)
if not os.path.exists(target_folder_py):
    os.makedirs(target_folder_py)


# hardware_config = 'tpuv3'
# mapping = mapping_dict[hardware_config]
# part = part_dict[hardware_config]
# cal = cal_dict[hardware_config]
hardware_configs = ['a100', 'dojo', 'tpuv3', 'a100_scaling', 'dojo_scaling', 'tpuv3_scaling', 'a100_expand', 'dojo_expand', 'tpuv3_expand']
for hardware_config in hardware_configs:
    mapping = mapping_dict[hardware_config]
    part = part_dict[hardware_config]
    cal = cal_dict[hardware_config]

    code = ''''''
    code += '''
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
from moe_backward import embedding, attention, mlp, lm_head, moe, attention_input_backward, attention_weight_backward, mlp_input_backward, mlp_weight_backward, moe_input_backward, moe_weight_backward
from sequence_expert import sequence_expert
from tensor_expert import tensor_expert_half
from tensor_pipeline import tensor_pipeline
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

moe_layer = moe(databytes=data_bytes, hidden_states=model_config['n_embd'])
moe_layer.cal(batch_size=batch_size, sequence_length=sequence_length, topk=model_config['top_k'])
moe_bytes = moe_layer.cal_tensor

moe_input_backward_layer = moe_input_backward(databytes=data_bytes, hidden_states=model_config['n_embd'])
moe_input_backward_layer.cal(batch_size=batch_size, sequence_length=sequence_length, topk=model_config['top_k'])
moe_input_backward_bytes = moe_input_backward_layer.cal_tensor

moe_weight_backward_layer = moe_weight_backward(databytes=data_bytes, hidden_states=model_config['n_embd'])
moe_weight_backward_layer.cal(batch_size=batch_size, sequence_length=sequence_length, topk=model_config['top_k'])
moe_weight_backward_bytes = moe_weight_backward_layer.cal_tensor
'''


    code += mapping

    code += '''cluster_layers_number = model_config['n_layer'] // clusters_number
record_event_tags = []
'''

    code += part

    code += '''
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
                    source_nodes_coordinates_list=cluster_sub1_nodes_lists[cluster_step],
                    source_x_number=cluster_sub1_shapes[cluster_step][1], source_y_number=cluster_sub1_shapes[cluster_step][0],
                    topology_x_limitation=node_k, topology_y_limitation=node_k,
                    message_flits=pass_bytes,
                    reduction=reduction_cores
                )
            else:

                cluster_alltoall_event_tag, cluster_alltoall_dependency_list = alltoall_api.cal_time(
                    whole_nodes=node_network, current_event_tag=cluster_sub0_attention_weight_event_tag, current_dependency_list=cluster_attention_input_dependency_list,
                    source_nodes_coordinates_list=cluster_nodes_lists[cluster_step],
                    source_x_number=cluster_shapes[cluster_step][1], source_y_number=cluster_shapes[cluster_step][0],
                    topology_x_limitation=node_k, topology_y_limitation=node_k,
                    message_flits=pass_bytes*model_config['top_k']*cluster_dataparallels[cluster_step][0]*cluster_dataparallels[cluster_step][1],
                    reduction=reduction_cores
                )

                cluster_sub0_allgather_event_tag, cluster_sub0_allgather_dependency_list = allgather_api.cal_time(
                    whole_nodes=node_network, current_event_tag=cluster_alltoall_event_tag, current_dependency_list=cluster_alltoall_dependency_list,
                    source_nodes_coordinates_list=cluster_sub0_nodes_lists[cluster_step],
                    source_x_number=cluster_sub0_shapes[cluster_step][1], source_y_number=cluster_sub0_shapes[cluster_step][0],
                    topology_x_limitation=node_k, topology_y_limitation=node_k,
                    message_flits=pass_bytes,
                    reduction=reduction_cores
                )

                cluster_sub1_allgather_event_tag, cluster_sub1_allgather_dependency_list = allgather_api.cal_time(
                    whole_nodes=node_network, current_event_tag=cluster_sub0_allgather_event_tag, current_dependency_list=cluster_alltoall_dependency_list,
                    source_nodes_coordinates_list=cluster_sub1_nodes_lists[cluster_step],
                    source_x_number=cluster_sub1_shapes[cluster_step][1], source_y_number=cluster_sub1_shapes[cluster_step][0],
                    topology_x_limitation=node_k, topology_y_limitation=node_k,
                    message_flits=pass_bytes,
                    reduction=reduction_cores
                )
        else:
            if layer_step == 0:
                cluster_sub0_allgather_event_tag, cluster_sub0_allgather_dependency_list = allgather_api.cal_time(
                    whole_nodes=node_network, current_event_tag=cluster_manytomany_event_tag, current_dependency_list=cluster_sub0_manytomany_dependency_list,
                    source_nodes_coordinates_list=cluster_sub0_nodes_lists[cluster_step],
                    source_x_number=cluster_sub0_shapes[cluster_step][1], source_y_number=cluster_sub0_shapes[cluster_step][0],
                    topology_x_limitation=node_k, topology_y_limitation=node_k,
                    message_flits=pass_bytes,
                    reduction=reduction_cores
                )

                cluster_sub1_allgather_event_tag, cluster_sub1_allgather_dependency_list = allgather_api.cal_time(
                    whole_nodes=node_network, current_event_tag=cluster_sub0_allgather_event_tag, current_dependency_list=cluster_sub1_manytomany_dependency_list,
                    source_nodes_coordinates_list=cluster_sub1_nodes_lists[cluster_step],
                    source_x_number=cluster_sub1_shapes[cluster_step][1], source_y_number=cluster_sub1_shapes[cluster_step][0],
                    topology_x_limitation=node_k, topology_y_limitation=node_k,
                    message_flits=pass_bytes,
                    reduction=reduction_cores
                )

            else:

                cluster_alltoall_event_tag, cluster_alltoall_dependency_list = alltoall_api.cal_time(
                    whole_nodes=node_network, current_event_tag=cluster_sub0_attention_weight_event_tag, current_dependency_list=cluster_attention_input_dependency_list,
                    source_nodes_coordinates_list=cluster_nodes_lists[cluster_step],
                    source_x_number=cluster_shapes[cluster_step][1], source_y_number=cluster_shapes[cluster_step][0],
                    topology_x_limitation=node_k, topology_y_limitation=node_k,
                    message_flits=pass_bytes*model_config['top_k']*cluster_dataparallels[cluster_step][0]*cluster_dataparallels[cluster_step][1],
                    reduction=reduction_cores
                )


                cluster_sub0_allgather_event_tag, cluster_sub0_allgather_dependency_list = allgather_api.cal_time(
                    whole_nodes=node_network, current_event_tag=cluster_alltoall_event_tag, current_dependency_list=cluster_alltoall_dependency_list,
                    source_nodes_coordinates_list=cluster_sub0_nodes_lists[cluster_step],
                    source_x_number=cluster_sub0_shapes[cluster_step][1], source_y_number=cluster_sub0_shapes[cluster_step][0],
                    topology_x_limitation=node_k, topology_y_limitation=node_k,
                    message_flits=pass_bytes,
                    reduction=reduction_cores
                )

                cluster_sub1_allgather_event_tag, cluster_sub1_allgather_dependency_list = allgather_api.cal_time(
                    whole_nodes=node_network, current_event_tag=cluster_sub0_allgather_event_tag, current_dependency_list=cluster_alltoall_dependency_list,
                    source_nodes_coordinates_list=cluster_sub1_nodes_lists[cluster_step],
                    source_x_number=cluster_sub1_shapes[cluster_step][1], source_y_number=cluster_sub1_shapes[cluster_step][0],
                    topology_x_limitation=node_k, topology_y_limitation=node_k,
                    message_flits=pass_bytes,
                    reduction=reduction_cores
                )

        cluster_allgather_events = []
        for dependency_y in cluster_sub1_allgather_dependency_list:
            for dependency_x in dependency_y:
                for dependency_event in dependency_x:
                    if dependency_event not in cluster_allgather_events:
                        cluster_allgather_events.append(dependency_event)
        for dependency_y in cluster_sub0_allgather_dependency_list:
            for dependency_x in dependency_y:
                for dependency_event in dependency_x:
                    if dependency_event not in cluster_allgather_events:
                        cluster_allgather_events.append(dependency_event)
        cluster_allgather_dependency_list = [[cluster_allgather_events for _ in range(cluster_shapes[cluster_step][1])] for _ in range(cluster_shapes[cluster_step][0])]

        if layer_step == 0:
            cluster_moe_input_event_tag, cluster_moe_input_dependency_list = compute_2d_base(
                whole_nodes=node_network, current_event_tag=cluster_sub1_allgather_event_tag, current_dependency_list=cluster_allgather_dependency_list,
                source_nodes_coordinates_list=cluster_nodes_lists[cluster_step],
                source_x_number=cluster_shapes[cluster_step][1], source_y_number=cluster_shapes[cluster_step][0],
                topology_x_limitation=node_k, topology_y_limitation=node_k,
                message_flits=moe_input_backward_bytes/cluster_shapes[cluster_step][0]/cluster_shapes[cluster_step][1],
                reduction=computation
            )
        else:
            cluster_weight_allgather_dependency_list = []
            for sublist1, sublist2 in zip(cluster_attention_weight_dependency_list, cluster_allgather_dependency_list):
                merged_sublist = []
                for pair1, pair2 in zip(sublist1, sublist2):
                    merged_sublist.append(pair1 + pair2)
                cluster_weight_allgather_dependency_list.append(merged_sublist)

            cluster_moe_input_event_tag, cluster_moe_input_dependency_list = compute_2d_base(
                whole_nodes=node_network, current_event_tag=cluster_sub1_allgather_event_tag, current_dependency_list=cluster_weight_allgather_dependency_list,
                source_nodes_coordinates_list=cluster_nodes_lists[cluster_step],
                source_x_number=cluster_shapes[cluster_step][1], source_y_number=cluster_shapes[cluster_step][0],
                topology_x_limitation=node_k, topology_y_limitation=node_k,
                message_flits=moe_input_backward_bytes/cluster_shapes[cluster_step][0]/cluster_shapes[cluster_step][1],
                reduction=computation
            )

        cluster_moe_weight_event_tag, cluster_moe_weight_dependency_list = compute_2d_base(
            whole_nodes=node_network, current_event_tag=cluster_moe_input_event_tag, current_dependency_list=cluster_moe_input_dependency_list,
            source_nodes_coordinates_list=cluster_nodes_lists[cluster_step],
            source_x_number=cluster_shapes[cluster_step][1], source_y_number=cluster_shapes[cluster_step][0],
            topology_x_limitation=node_k, topology_y_limitation=node_k,
            message_flits=moe_weight_backward_bytes/cluster_shapes[cluster_step][0]/cluster_shapes[cluster_step][1],
            reduction=computation
        )

        cluster_alltoall_event_tag, cluster_alltoall_dependency_list = alltoall_api.cal_time(
            whole_nodes=node_network, current_event_tag=cluster_moe_weight_event_tag, current_dependency_list=cluster_moe_input_dependency_list,
            source_nodes_coordinates_list=cluster_nodes_lists[cluster_step],
            source_x_number=cluster_shapes[cluster_step][1], source_y_number=cluster_shapes[cluster_step][0],
            topology_x_limitation=node_k, topology_y_limitation=node_k,
            message_flits=pass_bytes*model_config['top_k']*cluster_dataparallels[cluster_step][0]*cluster_dataparallels[cluster_step][1],
            reduction=reduction_cores
        )

        cluster_sub1_allgather_event_tag, cluster_sub1_allgather_dependency_list = allgather_api.cal_time(
            whole_nodes=node_network, current_event_tag=cluster_alltoall_event_tag, current_dependency_list=cluster_alltoall_dependency_list,
            source_nodes_coordinates_list=cluster_sub1_nodes_lists[cluster_step],
            source_x_number=cluster_sub1_shapes[cluster_step][1], source_y_number=cluster_sub1_shapes[cluster_step][0],
            topology_x_limitation=node_k, topology_y_limitation=node_k,
            message_flits=pass_bytes,
            reduction=reduction_cores
        )

        cluster_sub0_allgather_event_tag, cluster_sub0_allgather_dependency_list = allgather_api.cal_time(
            whole_nodes=node_network, current_event_tag=cluster_sub1_allgather_event_tag, current_dependency_list=cluster_alltoall_dependency_list,
            source_nodes_coordinates_list=cluster_sub0_nodes_lists[cluster_step],
            source_x_number=cluster_sub0_shapes[cluster_step][1], source_y_number=cluster_sub0_shapes[cluster_step][0],
            topology_x_limitation=node_k, topology_y_limitation=node_k,
            message_flits=pass_bytes,
            reduction=reduction_cores
        )

        cluster_allgather_events = []
        for dependency_y in cluster_sub0_allgather_dependency_list:
            for dependency_x in dependency_y:
                for dependency_event in dependency_x:
                    if dependency_event not in cluster_allgather_events:
                        cluster_allgather_events.append(dependency_event)
        for dependency_y in cluster_sub1_allgather_dependency_list:
            for dependency_x in dependency_y:
                for dependency_event in dependency_x:
                    if dependency_event not in cluster_allgather_events:
                        cluster_allgather_events.append(dependency_event)
        cluster_allgather_dependency_list = [[cluster_allgather_events for _ in range(cluster_shapes[cluster_step][1])] for _ in range(cluster_shapes[cluster_step][0])]

        cluster_weight_allgather_dependency_list = []
        for sublist1, sublist2 in zip(cluster_moe_weight_dependency_list, cluster_allgather_dependency_list):
            merged_sublist = []
            for pair1, pair2 in zip(sublist1, sublist2):
                merged_sublist.append(pair1 + pair2)
            cluster_weight_allgather_dependency_list.append(merged_sublist)

        cluster_sub1_attention_input_event_tag, cluster_sub1_attention_input_dependency_list = compute_2d_base(
            whole_nodes=node_network, current_event_tag=cluster_sub0_allgather_event_tag, current_dependency_list=cluster_weight_allgather_dependency_list,
            source_nodes_coordinates_list=cluster_sub1_nodes_lists[cluster_step],
            source_x_number=cluster_sub1_shapes[cluster_step][1], source_y_number=cluster_sub1_shapes[cluster_step][0],
            topology_x_limitation=node_k, topology_y_limitation=node_k,
            message_flits=attention_input_backward_bytes/cluster_sub1_shapes[cluster_step][0]/cluster_sub1_shapes[cluster_step][1],
            reduction=computation
        )

        cluster_sub0_attention_input_event_tag, cluster_sub0_attention_input_dependency_list = compute_2d_base(
            whole_nodes=node_network, current_event_tag=cluster_sub1_attention_input_event_tag, current_dependency_list=cluster_weight_allgather_dependency_list,
            source_nodes_coordinates_list=cluster_sub0_nodes_lists[cluster_step],
            source_x_number=cluster_sub0_shapes[cluster_step][1], source_y_number=cluster_sub0_shapes[cluster_step][0],
            topology_x_limitation=node_k, topology_y_limitation=node_k,
            message_flits=attention_input_backward_bytes/cluster_sub0_shapes[cluster_step][0]/cluster_sub0_shapes[cluster_step][1],
            reduction=computation
        )

        cluster_sub1_attention_weight_event_tag, cluster_sub1_attention_weight_dependency_list = compute_2d_base(
            whole_nodes=node_network, current_event_tag=cluster_sub0_attention_input_event_tag, current_dependency_list=cluster_sub1_attention_input_dependency_list,
            source_nodes_coordinates_list=cluster_sub1_nodes_lists[cluster_step],
            source_x_number=cluster_sub1_shapes[cluster_step][1], source_y_number=cluster_sub1_shapes[cluster_step][0],
            topology_x_limitation=node_k, topology_y_limitation=node_k,
            message_flits=attention_weight_backward_bytes/cluster_sub1_shapes[cluster_step][0]/cluster_sub1_shapes[cluster_step][1],
            reduction=computation
        )

        cluster_sub0_attention_weight_event_tag, cluster_sub0_attention_weight_dependency_list = compute_2d_base(
            whole_nodes=node_network, current_event_tag=cluster_sub1_attention_weight_event_tag, current_dependency_list=cluster_sub0_attention_input_dependency_list,
            source_nodes_coordinates_list=cluster_sub0_nodes_lists[cluster_step],
            source_x_number=cluster_sub0_shapes[cluster_step][1], source_y_number=cluster_sub0_shapes[cluster_step][0],
            topology_x_limitation=node_k, topology_y_limitation=node_k,
            message_flits=attention_weight_backward_bytes/cluster_sub0_shapes[cluster_step][0]/cluster_sub0_shapes[cluster_step][1],
            reduction=computation
        )

        cluster_sub1_reducescatter_event_tag, cluster_sub1_reducescatter_dependency_list = reducescatter_api.cal_time(
            whole_nodes=node_network, current_event_tag=cluster_sub0_attention_weight_event_tag, current_dependency_list=cluster_sub1_attention_input_dependency_list,
            source_nodes_coordinates_list=cluster_sub1_nodes_lists[cluster_step],
            source_x_number=cluster_sub1_shapes[cluster_step][1], source_y_number=cluster_sub1_shapes[cluster_step][0],
            topology_x_limitation=node_k, topology_y_limitation=node_k,
            message_flits=pass_bytes,
            reduction=reduction_cores
        )

        cluster_sub0_reducescatter_event_tag, cluster_sub0_reducescatter_dependency_list = reducescatter_api.cal_time(
            whole_nodes=node_network, current_event_tag=cluster_sub1_reducescatter_event_tag, current_dependency_list=cluster_sub0_attention_input_dependency_list,
            source_nodes_coordinates_list=cluster_sub0_nodes_lists[cluster_step],
            source_x_number=cluster_sub0_shapes[cluster_step][1], source_y_number=cluster_sub0_shapes[cluster_step][0],
            topology_x_limitation=node_k, topology_y_limitation=node_k,
            message_flits=pass_bytes,
            reduction=reduction_cores
        )

        cluster_sub1_allgather_event_tag, cluster_sub1_allgather_dependency_list = allgather_api.cal_time(
            whole_nodes=node_network, current_event_tag=cluster_sub0_reducescatter_event_tag, current_dependency_list=cluster_sub1_reducescatter_dependency_list,
            source_nodes_coordinates_list=cluster_sub1_nodes_lists[cluster_step],
            source_x_number=cluster_sub1_shapes[cluster_step][1], source_y_number=cluster_sub1_shapes[cluster_step][0],
            topology_x_limitation=node_k, topology_y_limitation=node_k,
            message_flits=pass_bytes,
            reduction=reduction_cores
        )

        cluster_sub0_allgather_event_tag, cluster_sub0_allgather_dependency_list = allgather_api.cal_time(
            whole_nodes=node_network, current_event_tag=cluster_sub1_allgather_event_tag, current_dependency_list=cluster_sub0_reducescatter_dependency_list,
            source_nodes_coordinates_list=cluster_sub0_nodes_lists[cluster_step],
            source_x_number=cluster_sub0_shapes[cluster_step][1], source_y_number=cluster_sub0_shapes[cluster_step][0],
            topology_x_limitation=node_k, topology_y_limitation=node_k,
            message_flits=pass_bytes,
            reduction=reduction_cores
        )

        cluster_allgather_events = []
        for dependency_y in cluster_sub1_allgather_dependency_list:
            for dependency_x in dependency_y:
                for dependency_event in dependency_x:
                    if dependency_event not in cluster_allgather_events:
                        cluster_allgather_events.append(dependency_event)
        for dependency_y in cluster_sub0_allgather_dependency_list:
            for dependency_x in dependency_y:
                for dependency_event in dependency_x:
                    if dependency_event not in cluster_allgather_events:
                        cluster_allgather_events.append(dependency_event)
        cluster_allgather_dependency_list = [[cluster_allgather_events for _ in range(cluster_shapes[cluster_step][1])] for _ in range(cluster_shapes[cluster_step][0])]

        cluster_weight_events = []
        for dependency_y in cluster_sub1_attention_weight_dependency_list:
            for dependency_x in dependency_y:
                for dependency_event in dependency_x:
                    if dependency_event not in cluster_weight_events:
                        cluster_weight_events.append(dependency_event)
        for dependency_y in cluster_sub0_attention_weight_dependency_list:
            for dependency_x in dependency_y:
                for dependency_event in dependency_x:
                    if dependency_event not in cluster_weight_events:
                        cluster_weight_events.append(dependency_event)
        cluster_weight_dependency_list = [[cluster_weight_events for _ in range(cluster_shapes[cluster_step][1])] for _ in range(cluster_shapes[cluster_step][0])]

        cluster_weight_allgather_dependency_list = []
        for sublist1, sublist2 in zip(cluster_weight_dependency_list, cluster_allgather_dependency_list):
            merged_sublist = []
            for pair1, pair2 in zip(sublist1, sublist2):
                merged_sublist.append(pair1 + pair2)
            cluster_weight_allgather_dependency_list.append(merged_sublist)

        cluster_sub1_mlp_input_event_tag, cluster_sub1_mlp_input_dependency_list = compute_2d_base(
            whole_nodes=node_network, current_event_tag=cluster_sub0_allgather_event_tag, current_dependency_list=cluster_weight_allgather_dependency_list,
            source_nodes_coordinates_list=cluster_sub1_nodes_lists[cluster_step],
            source_x_number=cluster_sub1_shapes[cluster_step][1], source_y_number=cluster_sub1_shapes[cluster_step][0],
            topology_x_limitation=node_k, topology_y_limitation=node_k,
            message_flits=mlp_input_backward_bytes/cluster_sub1_shapes[cluster_step][0]/cluster_sub1_shapes[cluster_step][1],
            reduction=computation
        )

        cluster_sub0_mlp_input_event_tag, cluster_sub0_mlp_input_dependency_list = compute_2d_base(
            whole_nodes=node_network, current_event_tag=cluster_sub1_mlp_input_event_tag, current_dependency_list=cluster_weight_allgather_dependency_list,
            source_nodes_coordinates_list=cluster_sub0_nodes_lists[cluster_step],
            source_x_number=cluster_sub0_shapes[cluster_step][1], source_y_number=cluster_sub0_shapes[cluster_step][0],
            topology_x_limitation=node_k, topology_y_limitation=node_k,
            message_flits=mlp_input_backward_bytes/cluster_sub0_shapes[cluster_step][0]/cluster_sub0_shapes[cluster_step][1],
            reduction=computation
        )

        cluster_sub1_mlp_weight_event_tag, cluster_sub1_mlp_weight_dependency_list = compute_2d_base(
            whole_nodes=node_network, current_event_tag=cluster_sub0_mlp_input_event_tag, current_dependency_list=cluster_sub1_mlp_input_dependency_list,
            source_nodes_coordinates_list=cluster_sub1_nodes_lists[cluster_step],
            source_x_number=cluster_sub1_shapes[cluster_step][1], source_y_number=cluster_sub1_shapes[cluster_step][0],
            topology_x_limitation=node_k, topology_y_limitation=node_k,
            message_flits=mlp_weight_backward_bytes/cluster_sub1_shapes[cluster_step][0]/cluster_sub1_shapes[cluster_step][1],
            reduction=computation
        )

        cluster_sub0_mlp_weight_event_tag, cluster_sub0_mlp_weight_dependency_list = compute_2d_base(
            whole_nodes=node_network, current_event_tag=cluster_sub1_mlp_weight_event_tag, current_dependency_list=cluster_sub0_mlp_input_dependency_list,
            source_nodes_coordinates_list=cluster_sub0_nodes_lists[cluster_step],
            source_x_number=cluster_sub0_shapes[cluster_step][1], source_y_number=cluster_sub0_shapes[cluster_step][0],
            topology_x_limitation=node_k, topology_y_limitation=node_k,
            message_flits=mlp_weight_backward_bytes/cluster_sub0_shapes[cluster_step][0]/cluster_sub0_shapes[cluster_step][1],
            reduction=computation
        )

        cluster_sub1_reducescatter_event_tag, cluster_sub1_reducescatter_dependency_list = reducescatter_api.cal_time(
            whole_nodes=node_network, current_event_tag=cluster_sub0_mlp_weight_event_tag, current_dependency_list=cluster_sub1_mlp_input_dependency_list,
            source_nodes_coordinates_list=cluster_sub1_nodes_lists[cluster_step],
            source_x_number=cluster_sub1_shapes[cluster_step][1], source_y_number=cluster_sub1_shapes[cluster_step][0],
            topology_x_limitation=node_k, topology_y_limitation=node_k,
            message_flits=pass_bytes,
            reduction=reduction_cores
        )

        cluster_sub0_reducescatter_event_tag, cluster_sub0_reducescatter_dependency_list = reducescatter_api.cal_time(
            whole_nodes=node_network, current_event_tag=cluster_sub1_reducescatter_event_tag, current_dependency_list=cluster_sub0_mlp_input_dependency_list,
            source_nodes_coordinates_list=cluster_sub0_nodes_lists[cluster_step],
            source_x_number=cluster_sub0_shapes[cluster_step][1], source_y_number=cluster_sub0_shapes[cluster_step][0],
            topology_x_limitation=node_k, topology_y_limitation=node_k,
            message_flits=pass_bytes,
            reduction=reduction_cores
        )

        cluster_sub1_allgather_event_tag, cluster_sub1_allgather_dependency_list = allgather_api.cal_time(
            whole_nodes=node_network, current_event_tag=cluster_sub0_reducescatter_event_tag, current_dependency_list=cluster_sub1_reducescatter_dependency_list,
            source_nodes_coordinates_list=cluster_sub1_nodes_lists[cluster_step],
            source_x_number=cluster_sub1_shapes[cluster_step][1], source_y_number=cluster_sub1_shapes[cluster_step][0],
            topology_x_limitation=node_k, topology_y_limitation=node_k,
            message_flits=pass_bytes,
            reduction=reduction_cores
        )

        cluster_sub0_allgather_event_tag, cluster_sub0_allgather_dependency_list = allgather_api.cal_time(
            whole_nodes=node_network, current_event_tag=cluster_sub1_allgather_event_tag, current_dependency_list=cluster_sub0_reducescatter_dependency_list,
            source_nodes_coordinates_list=cluster_sub0_nodes_lists[cluster_step],
            source_x_number=cluster_sub0_shapes[cluster_step][1], source_y_number=cluster_sub0_shapes[cluster_step][0],
            topology_x_limitation=node_k, topology_y_limitation=node_k,
            message_flits=pass_bytes,
            reduction=reduction_cores
        )

        cluster_allgather_events = []
        for dependency_y in cluster_sub1_allgather_dependency_list:
            for dependency_x in dependency_y:
                for dependency_event in dependency_x:
                    if dependency_event not in cluster_allgather_events:
                        cluster_allgather_events.append(dependency_event)
        for dependency_y in cluster_sub0_allgather_dependency_list:
            for dependency_x in dependency_y:
                for dependency_event in dependency_x:
                    if dependency_event not in cluster_allgather_events:
                        cluster_allgather_events.append(dependency_event)
        cluster_allgather_dependency_list = [[cluster_allgather_events for _ in range(cluster_shapes[cluster_step][1])]
        for _ in range(cluster_shapes[cluster_step][0])]

        cluster_weight_evnets = []
        for dependency_y in cluster_sub1_mlp_weight_dependency_list:
            for dependency_x in dependency_y:
                for dependency_event in dependency_x:
                    if dependency_event not in cluster_weight_evnets:
                        cluster_weight_evnets.append(dependency_event)
        for dependency_y in cluster_sub0_mlp_weight_dependency_list:
            for dependency_x in dependency_y:
                for dependency_event in dependency_x:
                    if dependency_event not in cluster_weight_evnets:
                        cluster_weight_evnets.append(dependency_event)
        cluster_weight_dependency_list = [[cluster_weight_evnets for _ in range(cluster_shapes[cluster_step][1])] for _ in range(cluster_shapes[cluster_step][0])]

        cluster_weight_allgather_dependency_list = []
        for sublist1, sublist2 in zip(cluster_weight_dependency_list, cluster_allgather_dependency_list):
            merged_sublist = []
            for pair1, pair2 in zip(sublist1, sublist2):
                merged_sublist.append(pair1 + pair2)
            cluster_weight_allgather_dependency_list.append(merged_sublist)

        cluster_sub1_attention_input_event_tag, cluster_sub1_attention_input_dependency_list = compute_2d_base(
            whole_nodes=node_network, current_event_tag=cluster_sub0_allgather_event_tag, current_dependency_list=cluster_weight_allgather_dependency_list,
            source_nodes_coordinates_list=cluster_sub1_nodes_lists[cluster_step],
            source_x_number=cluster_sub1_shapes[cluster_step][1], source_y_number=cluster_sub1_shapes[cluster_step][0],
            topology_x_limitation=node_k, topology_y_limitation=node_k,
            message_flits=attention_input_backward_bytes/cluster_sub1_shapes[cluster_step][0]/cluster_sub1_shapes[cluster_step][1],
            reduction=computation
        )

        cluster_sub0_attention_input_event_tag, cluster_sub0_attention_input_dependency_list = compute_2d_base(
            whole_nodes=node_network, current_event_tag=cluster_sub1_attention_input_event_tag, current_dependency_list=cluster_weight_allgather_dependency_list,
            source_nodes_coordinates_list=cluster_sub0_nodes_lists[cluster_step],
            source_x_number=cluster_sub0_shapes[cluster_step][1], source_y_number=cluster_sub0_shapes[cluster_step][0],
            topology_x_limitation=node_k, topology_y_limitation=node_k,
            message_flits=attention_input_backward_bytes/cluster_sub0_shapes[cluster_step][0]/cluster_sub0_shapes[cluster_step][1],
            reduction=computation
        )

        cluster_attention_input_events = []
        for dependency_y in cluster_sub1_attention_input_dependency_list:
            for dependency_x in dependency_y:
                for dependency_event in dependency_x:
                    if dependency_event not in cluster_attention_input_events:
                        cluster_attention_input_events.append(dependency_event)
        for dependency_y in cluster_sub0_attention_input_dependency_list:
            for dependency_x in dependency_y:
                for dependency_event in dependency_x:
                    if dependency_event not in cluster_attention_input_events:
                        cluster_attention_input_events.append(dependency_event)
        cluster_attention_input_dependency_list = [[cluster_attention_input_events for _ in range(cluster_shapes[cluster_step][1])] for _ in range(cluster_shapes[cluster_step][0])]

        cluster_sub1_attention_weight_event_tag, cluster_sub1_attention_weight_dependency_list = compute_2d_base(
            whole_nodes=node_network, current_event_tag=cluster_sub0_attention_input_event_tag, current_dependency_list=cluster_sub1_attention_input_dependency_list,
            source_nodes_coordinates_list=cluster_sub1_nodes_lists[cluster_step],
            source_x_number=cluster_sub1_shapes[cluster_step][1], source_y_number=cluster_sub1_shapes[cluster_step][0],
            topology_x_limitation=node_k, topology_y_limitation=node_k,
            message_flits=attention_weight_backward_bytes/cluster_sub1_shapes[cluster_step][0]/cluster_sub1_shapes[cluster_step][1],
            reduction=computation
        )

        cluster_sub0_attention_weight_event_tag, cluster_sub0_attention_weight_dependency_list = compute_2d_base(
            whole_nodes=node_network, current_event_tag=cluster_sub1_attention_weight_event_tag, current_dependency_list=cluster_sub0_attention_input_dependency_list,
            source_nodes_coordinates_list=cluster_sub0_nodes_lists[cluster_step],
            source_x_number=cluster_sub0_shapes[cluster_step][1], source_y_number=cluster_sub0_shapes[cluster_step][0],
            topology_x_limitation=node_k, topology_y_limitation=node_k,
            message_flits=attention_weight_backward_bytes/cluster_sub0_shapes[cluster_step][0]/cluster_sub0_shapes[cluster_step][1],
            reduction=computation
        )

        cluster_attention_weight_events = []
        for dependency_y in cluster_sub1_attention_weight_dependency_list:
            for dependency_x in dependency_y:
                for dependency_event in dependency_x:
                    if dependency_event not in cluster_attention_weight_events:
                        cluster_attention_weight_events.append(dependency_event)
        for dependency_y in cluster_sub0_attention_weight_dependency_list:
            for dependency_x in dependency_y:
                for dependency_event in dependency_x:
                    if dependency_event not in cluster_attention_weight_events:
                        cluster_attention_weight_events.append(dependency_event)
        cluster_attention_weight_dependency_list = [[cluster_attention_weight_events for _ in range(cluster_shapes[cluster_step][1])] for _ in range(cluster_shapes[cluster_step][0])]
        record_event_tags.append(cluster_attention_weight_dependency_list)

    if cluster_step < clusters_number - 1:

        cluster_manytomany_event_tag, cluster_manytomany_dependency_list = manytomany_api.cal_time(
            whole_nodes=node_network, current_event_tag=cluster_sub0_attention_weight_event_tag, current_dependency_list=cluster_attention_weight_dependency_list,
            source_nodes_coordinates_list=cluster_nodes_lists[cluster_step],
            target_nodes_coordinates_list=cluster_nodes_lists[cluster_step+1],
            source_x_number=cluster_shapes[cluster_step][1], source_y_number=cluster_shapes[cluster_step][0],
            target_x_number=cluster_shapes[cluster_step+1][1], target_y_number=cluster_shapes[cluster_step+1][0],
            topology_x_limitation=node_k, topology_y_limitation=node_k,
            message_flits=pass_bytes*model_config['top_k']*cluster_dataparallels[cluster_step][0]*cluster_dataparallels[cluster_step][1]/cluster_shapes[cluster_step][0]/cluster_shapes[cluster_step][1],
            reduction=reduction_cores
        )
        record_event_tags.append(cluster_manytomany_dependency_list)

        cluster_manytomany_events = []
        for dependency_y in cluster_manytomany_dependency_list:
            for dependency_x in dependency_y:
                for dependency_event in dependency_x:
                    if dependency_event not in cluster_manytomany_events:
                        cluster_manytomany_events.append(dependency_event)

        cluster_sub0_manytomany_dependency_list = [[cluster_manytomany_events for _ in range(cluster_sub0_shapes[cluster_step+1][1])] for _ in range(cluster_sub0_shapes[cluster_step+1][0])]
        cluster_sub1_manytomany_dependency_list = [[cluster_manytomany_events for _ in range(cluster_sub1_shapes[cluster_step+1][1])] for _ in range(cluster_sub1_shapes[cluster_step+1][0])]

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
'''

    code += cal

    code += '''print(whole_time)
with open(os.path.join(file_path,"../txt/"+os.path.splitext(os.path.basename(__file__))[0]+'.txt'), 'w') as f:
    f.write(str(whole_time))
'''



    file_path = os.path.join(target_folder_py, 'te_fusion_' + hardware_config + '_moe.py')
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(code)



