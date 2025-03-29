# File name  :    scalesim_gemmtask.py
# Author     :    xiaocuicui
# Time       :    2024/08/01 13:28:39
# Version    :    V1.0
# Abstract   :        

import os
import sys
file_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(file_path, '../../utils/'))


from utils import get_gpt2_config


batch_size = 1
sequence_size = 128

model_name = 'deepspeedmoe-1.3b'
config_path = os.path.join(file_path, '../gpt2/input/deepspeedmoe-1.3b-config.json')                        
model_config = get_gpt2_config(model_name, config_path)
hidden_states=model_config['n_embd']

print(batch_size, sequence_size, hidden_states)

'''
M, N, K
(M, k) * (k, N) = (M, N)
'''
gpt_attention_tasks = [["Layer", "M", "N", "K"]]

# x q k v
layer_task = ["xqkv", sequence_size, hidden_states*3, hidden_states]
gpt_attention_tasks.append(layer_task)

# QKT
layer_task = ["QKT", sequence_size, sequence_size, hidden_states]
gpt_attention_tasks.append(layer_task)

# QKTV
layer_task = ["QKTV", sequence_size, hidden_states, sequence_size]
gpt_attention_tasks.append(layer_task)

# cproj
layer_task = ["cproj", sequence_size, hidden_states, hidden_states]
gpt_attention_tasks.append(layer_task)


with open(os.path.join(file_path, 'moe_1point3_128_attention.csv'), 'w', newline='') as file:
    for item in gpt_attention_tasks:
            line = ','.join(map(str, item)) + ','
            file.write(line + '\n') 


gpt_mlp_tasks = [["Layer", "M", "N", "K"]]

#cfc
layer_task = ["cfc", sequence_size, hidden_states*4, hidden_states]
gpt_mlp_tasks.append(layer_task)

#cproj
layer_task = ["cproj", sequence_size, hidden_states, hidden_states*4]
gpt_mlp_tasks.append(layer_task)


with open(os.path.join(file_path, 'moe_1point3_128_mlp.csv'), 'w', newline='') as file:
    for item in gpt_mlp_tasks:
            line = ','.join(map(str, item)) + ','
            file.write(line + '\n')


gpt_mlp_tasks = [["Layer", "M", "N", "K"]]
"""
top_2
"""
#cfc
layer_task = ["cfc", sequence_size, hidden_states*4, hidden_states]
gpt_mlp_tasks.append(layer_task)
layer_task = ["cfc", sequence_size, hidden_states*4, hidden_states]
gpt_mlp_tasks.append(layer_task)

#cproj
layer_task = ["cproj", sequence_size, hidden_states, hidden_states*4]
gpt_mlp_tasks.append(layer_task)
layer_task = ["cproj", sequence_size, hidden_states, hidden_states*4]
gpt_mlp_tasks.append(layer_task)


with open(os.path.join(file_path, 'moe_1point3_128_moe.csv'), 'w', newline='') as file:
    for item in gpt_mlp_tasks:
            line = ','.join(map(str, item)) + ','
            file.write(line + '\n')

