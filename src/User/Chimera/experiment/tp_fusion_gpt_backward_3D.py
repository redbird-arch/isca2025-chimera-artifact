# File name  :    tp_fusion_gpt.py
# Author     :    xiaocuicui
# Time       :    2025/02/17 12:27:19
# Version    :    V1.0
# Abstract   :        

import os
import sys
file_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(file_path)
sys.path.append(os.path.join(file_path, '../'))

from gpt_config_3D import mapping_dict, part_dict, cal_dict

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


# hardware_config = 'tpuv4'
# mapping = mapping_dict[hardware_config]
# part = part_dict[hardware_config]
# cal = cal_dict[hardware_config]
hardware_configs = ['tpuv4', 'tpuv4_scaling']
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
sys.path.append(os.path.join(file_path, '../../../parallism_3D/'))
sys.path.append(os.path.join(file_path, '../../../../../communication/collective_communication/'))
sys.path.append(os.path.join(file_path, '../../../../../components/'))
sys.path.append(os.path.join(file_path, '../../../../../utils/'))


from compute_backward import compute_3d_base
from gpt2_backward import embedding, attention, mlp, lm_head, attention_input_backward, attention_weight_backward, mlp_input_backward, mlp_weight_backward
from tensor_pipeline import tensor_pipeline
from allreduce import allreduce
from reducescatter import reducescatter
from allgather import allgather
from multicast import pointtopoint, manytomanymulticast
from Launcher import build_3D, launch
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
'''


    code += mapping

    code += '''cluster_layers_number = model_config['n_layer'] // clusters_number
record_event_tags = []
'''

    code += part

    code += '''
        if cluster_idx == 0:
            if layer_step == 0:
                cluster_allgather_event_tag, cluster_allgather_dependency_list = allgather_api.cal_time(
                    whole_nodes=node_network, current_event_tag=initial_event_tag, current_dependency_list=initial_cluster_dependency_lists[0],
                    source_nodes_coordinates_list=cluster_nodes_lists[cluster_step],
                    source_x_number=cluster_shapes[cluster_step][2], source_y_number=cluster_shapes[cluster_step][1], source_z_number=cluster_shapes[cluster_step][0],
                    topology_x_limitation=node_k, topology_y_limitation=node_k, topology_z_limitation=node_k,
                    message_flits=pass_bytes,
                    reduction=reduction_cores
                )
            else:
                cluster_allgather_event_tag, cluster_allgather_dependency_list = allgather_api.cal_time(
                    whole_nodes=node_network, current_event_tag=cluster_attention_weight_event_tag, current_dependency_list=cluster_attention_weight_dependency_list,
                    source_nodes_coordinates_list=cluster_nodes_lists[cluster_step],
                    source_x_number=cluster_shapes[cluster_step][2], source_y_number=cluster_shapes[cluster_step][1], source_z_number=cluster_shapes[cluster_step][0],
                    topology_x_limitation=node_k, topology_y_limitation=node_k, topology_z_limitation=node_k,
                    message_flits=pass_bytes,
                    reduction=reduction_cores
                )
        else:
            if layer_step == 0:
                cluster_allgather_event_tag, cluster_allgather_dependency_list = allgather_api.cal_time(
                    whole_nodes=node_network, current_event_tag=cluster_allgather_event_tag, current_dependency_list=cluster_allgather_dependency_list,
                    source_nodes_coordinates_list=cluster_nodes_lists[cluster_step],
                    source_x_number=cluster_shapes[cluster_step][2], source_y_number=cluster_shapes[cluster_step][1], source_z_number=cluster_shapes[cluster_step][0],
                    topology_x_limitation=node_k, topology_y_limitation=node_k, topology_z_limitation=node_k,
                    message_flits=pass_bytes,
                    reduction=reduction_cores
                )
            else:
                cluster_allgather_event_tag, cluster_allgather_dependency_list = allgather_api.cal_time(
                    whole_nodes=node_network, current_event_tag=cluster_attention_weight_event_tag, current_dependency_list=cluster_attention_weight_dependency_list,
                    source_nodes_coordinates_list=cluster_nodes_lists[cluster_step],
                    source_x_number=cluster_shapes[cluster_step][2], source_y_number=cluster_shapes[cluster_step][1], source_z_number=cluster_shapes[cluster_step][0],
                    topology_x_limitation=node_k, topology_y_limitation=node_k, topology_z_limitation=node_k,
                    message_flits=pass_bytes,
                    reduction=reduction_cores
                )

        if layer_step == 0:
            cluster_mlp_input_event_tag, cluster_mlp_input_dependency_list = compute_3d_base(
                whole_nodes=node_network, current_event_tag=cluster_allgather_event_tag, current_dependency_list=cluster_allgather_dependency_list,
                source_nodes_coordinates_list=cluster_nodes_lists[cluster_step],
                source_x_number=cluster_shapes[cluster_step][2], source_y_number=cluster_shapes[cluster_step][1], source_z_number=cluster_shapes[cluster_step][0],
                topology_x_limitation=node_k, topology_y_limitation=node_k, topology_z_limitation=node_k,
                message_flits=mlp_input_backward_bytes/cluster_shapes[cluster_step][0]/cluster_shapes[cluster_step][1]/cluster_shapes[cluster_step][2],
                reduction=computation
            )
        else:
            cluster_weight_allgather_dependency_list = []
            for sublist1, sublist2 in zip(cluster_attention_weight_dependency_list, cluster_allgather_dependency_list):
                merged_sublist = []
                for pair1, pair2 in zip(sublist1, sublist2):
                    merged_sublist.append(pair1 + pair2)
                cluster_weight_allgather_dependency_list.append(merged_sublist)

            cluster_mlp_input_event_tag, cluster_mlp_input_dependency_list = compute_3d_base(
                whole_nodes=node_network, current_event_tag=cluster_allgather_event_tag, current_dependency_list=cluster_weight_allgather_dependency_list,
                source_nodes_coordinates_list=cluster_nodes_lists[cluster_step],
                source_x_number=cluster_shapes[cluster_step][2], source_y_number=cluster_shapes[cluster_step][1], source_z_number=cluster_shapes[cluster_step][0],
                topology_x_limitation=node_k, topology_y_limitation=node_k, topology_z_limitation=node_k,
                message_flits=mlp_input_backward_bytes/cluster_shapes[cluster_step][0]/cluster_shapes[cluster_step][1]/cluster_shapes[cluster_step][2],
                reduction=computation
            )

        cluster_mlp_weight_event_tag, cluster_mlp_weight_dependency_list = compute_3d_base(
            whole_nodes=node_network, current_event_tag=cluster_mlp_input_event_tag, current_dependency_list=cluster_mlp_input_dependency_list,
            source_nodes_coordinates_list=cluster_nodes_lists[cluster_step],
            source_x_number=cluster_shapes[cluster_step][2], source_y_number=cluster_shapes[cluster_step][1], source_z_number=cluster_shapes[cluster_step][0],
            topology_x_limitation=node_k, topology_y_limitation=node_k, topology_z_limitation=node_k,
            message_flits=mlp_weight_backward_bytes/cluster_shapes[cluster_step][0]/cluster_shapes[cluster_step][1]/cluster_shapes[cluster_step][2],
            reduction=computation
        )

        cluster_reducescatter_event_tag, cluster_reducescatter_dependency_list = reducescatter_api.cal_time(
            whole_nodes=node_network, current_event_tag=cluster_mlp_weight_event_tag, current_dependency_list=cluster_mlp_input_dependency_list,
            source_nodes_coordinates_list=cluster_nodes_lists[cluster_step],
            source_x_number=cluster_shapes[cluster_step][2], source_y_number=cluster_shapes[cluster_step][1], source_z_number=cluster_shapes[cluster_step][0],
            topology_x_limitation=node_k, topology_y_limitation=node_k, topology_z_limitation=node_k,
            message_flits=pass_bytes,
            reduction=reduction_cores
        )

        cluster_allgather_event_tag, cluster_allgather_dependency_list = allgather_api.cal_time(
            whole_nodes=node_network, current_event_tag=cluster_reducescatter_event_tag, current_dependency_list=cluster_reducescatter_dependency_list,
            source_nodes_coordinates_list=cluster_nodes_lists[cluster_step],
            source_x_number=cluster_shapes[cluster_step][2], source_y_number=cluster_shapes[cluster_step][1], source_z_number=cluster_shapes[cluster_step][0],
            topology_x_limitation=node_k, topology_y_limitation=node_k, topology_z_limitation=node_k,
            message_flits=pass_bytes,
            reduction=reduction_cores
        )

        cluster_weight_allgather_dependency_list = []
        for sublist1, sublist2 in zip(cluster_mlp_weight_dependency_list, cluster_allgather_dependency_list):
            merged_sublist = []
            for pair1, pair2 in zip(sublist1, sublist2):
                merged_sublist.append(pair1 + pair2)
            cluster_weight_allgather_dependency_list.append(merged_sublist)

        cluster_attention_input_event_tag, cluster_attention_input_dependency_list = compute_3d_base(
            whole_nodes=node_network, current_event_tag=cluster_allgather_event_tag, current_dependency_list=cluster_weight_allgather_dependency_list,
            source_nodes_coordinates_list=cluster_nodes_lists[cluster_step],
            source_x_number=cluster_shapes[cluster_step][2], source_y_number=cluster_shapes[cluster_step][1], source_z_number=cluster_shapes[cluster_step][0],
            topology_x_limitation=node_k, topology_y_limitation=node_k, topology_z_limitation=node_k,
            message_flits=attention_input_backward_bytes/cluster_shapes[cluster_step][0]/cluster_shapes[cluster_step][1]/cluster_shapes[cluster_step][2],
            reduction=computation
        )

        cluster_attention_weight_event_tag, cluster_attention_weight_dependency_list = compute_3d_base(
            whole_nodes=node_network, current_event_tag=cluster_attention_input_event_tag, current_dependency_list=cluster_attention_input_dependency_list,
            source_nodes_coordinates_list=cluster_nodes_lists[cluster_step],
            source_x_number=cluster_shapes[cluster_step][2], source_y_number=cluster_shapes[cluster_step][1], source_z_number=cluster_shapes[cluster_step][0],
            topology_x_limitation=node_k, topology_y_limitation=node_k, topology_z_limitation=node_k,
            message_flits=attention_weight_backward_bytes/cluster_shapes[cluster_step][0]/cluster_shapes[cluster_step][1]/cluster_shapes[cluster_step][2],
            reduction=computation
        )

        record_event_tags.append(cluster_attention_weight_dependency_list)

    if cluster_step < clusters_number - 1:
        cluster_manytomany_event_tag, cluster_manytomany_dependency_list = manytomany_api.cal_time(
            whole_nodes=node_network, current_event_tag=cluster_attention_weight_event_tag, current_dependency_list=cluster_attention_weight_dependency_list,
            source_nodes_coordinates_list=cluster_nodes_lists[cluster_step],
            target_nodes_coordinates_list=cluster_nodes_lists[cluster_step+1],
            source_x_number=cluster_shapes[cluster_step][1], source_y_number=cluster_shapes[cluster_step][0],
            target_x_number=cluster_shapes[cluster_step+1][2], target_y_number=cluster_shapes[cluster_step+1][1], target_z_number=cluster_shapes[cluster_step+1][0],
            topology_x_limitation=node_k, topology_y_limitation=node_k, topology_z_limitation=node_k,
            message_flits=pass_bytes,
            reduction=reduction_cores
        )

        cluster_allgather_event_tag, cluster_allgather_dependency_list = allgather_api.cal_time(
            whole_nodes=node_network, current_event_tag=cluster_manytomany_event_tag, current_dependency_list=cluster_manytomany_dependency_list,
            source_nodes_coordinates_list=cluster_nodes_lists[cluster_step+1],
            source_x_number=cluster_shapes[cluster_step+1][2], source_y_number=cluster_shapes[cluster_step+1][1], source_z_number=cluster_shapes[cluster_step+1][0],
            topology_x_limitation=node_k, topology_y_limitation=node_k, topology_z_limitation=node_k,
            message_flits=pass_bytes,
            reduction=reduction_cores
        )
        record_event_tags.append(cluster_allgather_dependency_list)

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
from finished_segments import segment_time_3D
'''

    code += cal

    code += '''print(whole_time)
with open(os.path.join(file_path,"../txt/"+os.path.splitext(os.path.basename(__file__))[0]+'.txt'), 'w') as f:
    f.write(str(whole_time))
'''



    file_path = os.path.join(target_folder_py, 'tp_fusion_' + hardware_config + '_gpt.py')
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(code)



