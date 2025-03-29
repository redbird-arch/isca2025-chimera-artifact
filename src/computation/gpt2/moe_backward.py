# File name  :    gpt2.py
# Author     :    junwei cui, le qin, weilin cai
# Time       :    2023/10/13 14:47:11
# Version    :    V1.0
# Abstract   :    This file analysis GPT-2 Model from <~/miniconda3/envs/env_name/lib/python3.8/site-packages/transformers/models/gpt2/modeling_gpt2.py> 


import os
import sys
file_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(file_path, '../../utils/'))
sys.path.append(os.path.join(file_path, '../'))


import utils
from get_csv_cycles import sum_total_cycles


def get_gpt_parameter(model_name):
    '''
    ~/miniconda3/envs/env_name/lib/python3.8/site-packages/transformers/models/gpt2/configuration_gpt2.py
    '''
    GPT2_PRETRAINED_CONFIG_ARCHIVE_MAP = {
    "gpt2": "https://huggingface.co/gpt2/resolve/main/config.json",
    "gpt2-medium": "https://huggingface.co/gpt2-medium/resolve/main/config.json",
    "gpt2-large": "https://huggingface.co/gpt2-large/resolve/main/config.json",
    "gpt2-xl": "https://huggingface.co/gpt2-xl/resolve/main/config.json",
    "distilgpt2": "https://huggingface.co/distilgpt2/resolve/main/config.json",
    }

    # https://huggingface.co/facebook/opt-350m
    # https://github.com/huggingface/transformers/blob/main/src/transformers/models/opt/configuration_opt.py
    OPT_PRETRAINED_CONFIG_ARCHIVE_MAP = {
    "facebook/opt-125m": "https://huggingface.co/facebook/opt-125m/blob/main/config.json",
    "facebook/opt-350m": "https://huggingface.co/facebook/opt-350m/blob/main/config.json",
    "facebook/opt-1.3b": "https://huggingface.co/facebook/opt-1.3b/blob/main/config.json",
    "facebook/opt-2.7b": "https://huggingface.co/facebook/opt-2.7b/blob/main/config.json",
    "facebook/opt-6.7b": "https://huggingface.co/facebook/opt-6.7b/blob/main/config.json",
    "facebook/opt-13b": "https://huggingface.co/facebook/opt-13b/blob/main/config.json",
    }

    GPT_NEO_PRETRAINED_CONFIG_ARCHIVE_MAP = {
        "EleutherAI/gpt-neo-1.3B": "https://huggingface.co/EleutherAI/gpt-neo-1.3B/resolve/main/config.json",
        "EleutherAI/gpt-neo-125m": "https://huggingface.co/EleutherAI/gpt-neo-125m/blob/main/config.json",
        # See all GPTNeo models at https://huggingface.co/models?filter=gpt_neo
    }    

    # model_name = 'gpt2-medium'
    config_path = os.path.join(file_path, f'./input/{model_name}-config.json')

    if os.path.exists(config_path) is False:
        utils.download_file(GPT2_PRETRAINED_CONFIG_ARCHIVE_MAP[model_name], config_path)
    else:
        pass
    model_config = utils.load_dict_json(config_path)

    '''
    n_ctx (:obj:`int`, `optional`, defaults to 1024):
            Dimensionality of the causal mask (usually same as n_positions).
    n_positions (:obj:`int`, `optional`, defaults to 1024):
            The maximum sequence length that this model might ever be used with. Typically set this to something large
            just in case (e.g., 512 or 1024 or 2048).
    n_embd (:obj:`int`, `optional`, defaults to 768):
            Dimensionality of the embeddings and hidden states.     
    n_head (:obj:`int`, `optional`, defaults to 12):
            Number of attention heads for each attention layer in the Transformer encoder.               
    n_layer (:obj:`int`, `optional`, defaults to 12):
            Number of hidden layers in the Transformer encoder.    
    '''

    sequece_length = model_config['n_positions']
    hidden_states = model_config['n_embd']
    num_heads = model_config['n_head']
    attn_head_size = hidden_states // num_heads
    num_layers = model_config['n_layer']

    return sequece_length, hidden_states, num_heads, attn_head_size, num_layers


'''
TODO: Adding finer-grained operators that have already been implemented on main branch (vocabulary and so on)
TODO: Considering input_databytes and output_databytes
'''


class attention():

    '''
    https://huggingface.co/transformers/v2.9.1/quickstart.html#using-the-past
    '''

    def __init__(self, databytes, hidden_states, num_heads, attn_head_size, use_cache):
        self.databytes = databytes
        self.hidden_states = hidden_states
        self.num_heads = num_heads
        self.attn_head_size = attn_head_size
        self.use_cache = use_cache

        # ops of tensor such as gemm
        self.cal_tensor = 0
        # ops of vector such as softmax
        # NOTE: this version just considered no extra weights
        self.cal_vector = 0
        # ops of reorder such as transpose
        self.cal_reorder = 0


    def cal(self, batch_size, sequence_length, scalesim=True):
        if scalesim:
            self.cal_tensor = sum_total_cycles(os.path.join(file_path, f'../backend/runfile/moe_1point3_256_attention_normal/GoogleTPU_v1_ws/COMPUTE_REPORT.csv'))
        else:
            self.cal_tensor = 0
            self.cal_vector = 0
            self.cal_reorder = 0

            self.batch_size = batch_size

            self.sequence_length = sequence_length # make sure it's the same with given value

            # LN
            self.cal_vector += self.batch_size * self.sequence_length * (self.hidden_states * 2 + self.hidden_states * 5 + self.hidden_states * 3)
            
            # q, k, v
            self.cal_tensor += self.batch_size * self.sequence_length * self.hidden_states * 2 * self.hidden_states * 3

            # Q * K^T
            self.cal_tensor += self.batch_size * self.sequence_length * self.hidden_states * 2 * self.sequence_length

            # mask
            self.cal_vector += self.batch_size * self.sequence_length * self.sequence_length

            '''
            TODO: should consider multi-head (reorder cost)
            '''
            # softmax
            self.cal_vector += self.batch_size * self.sequence_length * self.sequence_length * 3

            # softmax * V
            self.cal_tensor += self.batch_size * self.sequence_length * self.sequence_length * 2 * self.hidden_states

            # c_proj
            self.cal_tensor += self.batch_size * self.sequence_length * self.hidden_states * 2 * self.hidden_states
            self.cal_reorder += self.batch_size * self.sequence_length * self.hidden_states    # dropout

            # residual add
            self.cal_tensor += self.batch_size * self.sequence_length * self.hidden_states


class attention_input_backward():
    
    def __init__(self, databytes, hidden_states, num_heads, attn_head_size, use_cache):
        self.databytes = databytes
        self.hidden_states = hidden_states
        self.num_heads = num_heads
        self.attn_head_size = attn_head_size
        self.use_cache = use_cache

        # ops of tensor such as gemm
        self.cal_tensor = 0
        # ops of vector such as softmax gradients
        self.cal_vector = 0
        # ops of reorder such as transpose
        self.cal_reorder = 0

    def cal(self, batch_size, sequence_length, scalesim=True):
        if scalesim:
            self.cal_tensor = sum_total_cycles(os.path.join(file_path, f'../backend/runfile/moe_1point3_256_attention_input_backward/GoogleTPU_v1_ws/COMPUTE_REPORT.csv'))
        else:
            self.cal_tensor = 0
            self.cal_vector = 0
            self.cal_reorder = 0

            self.batch_size = batch_size

            self.sequence_length = sequence_length # make sure it's the same with given value

            # c_proj: ofmap_gradient * WT
            self.cal_tensor += self.batch_size * self.sequence_length * self.hidden_states * 2 * self.hidden_states

            # P = S * V
            # G_s = G * V^T s, h, s
            # G_V  = S^T * G s, s, h
            self.cal_tensor += self.batch_size * self.sequence_length * self.hidden_states * 2 * self.sequence_length
            self.cal_tensor += self.batch_size * self.sequence_length * self.sequence_length * 2 * self.hidden_states

            # softmax: ele-wise with cost of 4
            self.cal_vector += self.batch_size * self.sequence_length * self.sequence_length * 4

            # Q * K^T
            # G_q = G  * K s, s, h
            # G_k = Q ^ T * G h, s, s
            self.cal_tensor += self.batch_size * self.sequence_length * self.sequence_length * 2 * self.hidden_states
            self.cal_tensor += self.batch_size * self.hidden_states * self.sequence_length * 2 * self.sequence_length

            # QKV = x * qkv
            # G_x = G * qkv ^ T s, 3h, h
            self.cal_tensor += self.batch_size * self.sequence_length * self.hidden_states * 3 * 2 * self.hidden_states

            # LN: ele-wise with cost of 10
            self.cal_vector += self.batch_size * self.sequence_length * self.hidden_states * 10

            # Residual add
            self.cal_tensor += self.batch_size * self.sequence_length * self.hidden_states


class attention_weight_backward():

    def __init__(self, databytes, hidden_states, num_heads, attn_head_size, use_cache):
        self.databytes = databytes
        self.hidden_states = hidden_states
        self.num_heads = num_heads
        self.attn_head_size = attn_head_size
        self.use_cache = use_cache

        # ops of tensor such as gemm
        self.cal_tensor = 0
        # ops of vector such as softmax gradients
        self.cal_vector = 0
        # ops of reorder such as transpose
        self.cal_reorder = 0

    def cal(self, batch_size, sequence_length, scalesim=True):
        if scalesim:
            self.cal_tensor = sum_total_cycles(os.path.join(file_path, f'../backend/runfile/moe_1point3_256_attention_weight_backward/GoogleTPU_v1_ws/COMPUTE_REPORT.csv'))
        else:
            self.cal_tensor = 0
            self.cal_vector = 0
            self.cal_reorder = 0

            self.batch_size = batch_size

            self.sequence_length = sequence_length # make sure it's the same with given value

            # c_proj: input ^ T * ofmap_gradient
            self.cal_tensor += self.batch_size * self.hidden_states * self.sequence_length * 2 * self.hidden_states

            # QKV = x * qkv
            # G_qkv = x^ T * G h, s, 3h
            self.cal_tensor += self.batch_size * self.sequence_length * self.hidden_states * 2 * self.hidden_states * 3

            # LN
            self.cal_vector += self.batch_size * self.sequence_length * self.hidden_states * 2


class mlp():

    def __init__(self, databytes, hidden_states):
        self.databytes = databytes
        self.hidden_states = hidden_states

        # ops of tensor such as gemm
        self.cal_tensor = 0
        # ops of vector such as softmax
        # NOTE: this version just consider no extra weights
        self.cal_vector = 0
        # ops of reorder such as transpose
        self.cal_reorder = 0

        self.task = 1


    def cal(self, batch_size, sequence_length, scalesim=True):
        if scalesim:
            self.cal_tensor = sum_total_cycles(os.path.join(file_path, f'../backend/runfile/moe_1point3_256_mlp_normal/GoogleTPU_v1_ws/COMPUTE_REPORT.csv'))
        else:
            self.cal_tensor = 0
            self.cal_vector = 0
            self.cal_reorder = 0

            self.batch_size = batch_size
            self.sequence_length = sequence_length

            # LN
            self.cal_vector += self.batch_size * self.sequence_length * (self.hidden_states * 2 + self.hidden_states * 5 + self.hidden_states * 3)

            # c_fc
            self.cal_tensor += self.batch_size * self.sequence_length * self.hidden_states * 2 * self.hidden_states * 4

            # gelu
            self.cal_vector += self.batch_size * self.sequence_length * self.hidden_states * 4 * 8

            # c_proj
            self.cal_tensor += self.batch_size * self.sequence_length * self.hidden_states * 4 * 2 * self.hidden_states
            self.cal_reorder += self.batch_size * self.sequence_length * self.hidden_states    # dropout

            # residual add
            self.cal_tensor += self.batch_size * self.sequence_length * self.hidden_states


class mlp_input_backward():

    def __init__(self, databytes, hidden_states):
        self.databytes = databytes
        self.hidden_states = hidden_states

        # ops of tensor such as gemm
        self.cal_tensor = 0
        # ops of vector such as softmax gradients
        self.cal_vector = 0
        # ops of reorder such as transpose
        self.cal_reorder = 0


    def cal(self, batch_size, sequence_length, scalesim=True):
        if scalesim:
            self.cal_tensor = sum_total_cycles(os.path.join(file_path, f'../backend/runfile/moe_1point3_256_mlp_input_backward/GoogleTPU_v1_ws/COMPUTE_REPORT.csv'))
        else:
            self.cal_tensor = 0
            self.cal_vector = 0
            self.cal_reorder = 0

            self.batch_size = batch_size
            self.sequence_length = sequence_length

            # c_proj: ofmap_gradient * WT
            self.cal_tensor += self.batch_size * self.sequence_length * self.hidden_states * 2 * self.hidden_states * 4

            # GELU: ele-wise with cost of 10
            self.cal_vector += self.batch_size * self.sequence_length * self.hidden_states * 4 * 10

            # c_fc: ofmap_gradient * WT
            self.cal_tensor += self.batch_size * self.sequence_length * self.hidden_states * 4 * 2 * self.hidden_states
            
            # LN: ele-wise with cost of 10
            self.cal_vector += self.batch_size * self.sequence_length * self.hidden_states * 10

            # Residual add
            self.cal_tensor += self.batch_size * self.sequence_length * self.hidden_states

            # Gradient of the communication
            self.task = 0


class mlp_weight_backward():

    def __init__(self, databytes, hidden_states):
        self.databytes = databytes
        self.hidden_states = hidden_states

        # ops of tensor such as gemm
        self.cal_tensor = 0
        # ops of vector such as softmax gradients
        self.cal_vector = 0
        # ops of reorder such as transpose
        self.cal_reorder = 0

    def cal(self, batch_size, sequence_length, scalesim=True):
        if scalesim:
            self.cal_tensor = sum_total_cycles(os.path.join(file_path, f'../backend/runfile/moe_1point3_256_mlp_weight_backward/GoogleTPU_v1_ws/COMPUTE_REPORT.csv'))
        else:
            self.cal_tensor = 0
            self.cal_vector = 0
            self.cal_reorder = 0

            self.batch_size = batch_size
            self.sequence_length = sequence_length    

            # c_proj: input ^ T * ofmap_gradient
            self.cal_tensor += self.batch_size * self.hidden_states * self.sequence_length * 2 * self.hidden_states * 4

            # c_fc: input ^ T * ofmap_gradient
            self.cal_tensor += self.batch_size * self.hidden_states * 4 * self.sequence_length * 2 * self.hidden_states

            # LN
            self.cal_vector += self.batch_size * self.sequence_length * self.hidden_states * 2


class moe():

    def __init__(self, databytes, hidden_states):
        self.databytes = databytes
        self.hidden_states = hidden_states

        # ops of tensor such as gemm
        self.cal_tensor = 0
        # ops of vector such as softmax
        # NOTE: this version just consider no extra weights
        self.cal_vector = 0
        # ops of reorder such as transpose
        self.cal_reorder = 0


    def cal(self, batch_size, sequence_length, topk, scalesim=True):
        if scalesim:
            self.cal_tensor = sum_total_cycles(os.path.join(file_path, f'../backend/runfile/moe_1point3_256_moe_normal/GoogleTPU_v1_ws/COMPUTE_REPORT.csv'))
        else:
            self.cal_tensor = 0
            self.cal_vector = 0
            self.cal_reorder = 0

            self.batch_size = batch_size
            self.sequence_length = sequence_length
            self.topk = topk

            # LN
            self.cal_vector += self.batch_size * self.sequence_length * (self.hidden_states * 2 + self.hidden_states * 5 + self.hidden_states * 3)

            # c_fc
            self.cal_tensor += self.batch_size * self.sequence_length * self.hidden_states * 2 * self.hidden_states * 4

            # gelu
            self.cal_vector += self.batch_size * self.sequence_length * self.hidden_states * 4 * 8

            # c_proj
            self.cal_tensor += self.batch_size * self.sequence_length * self.hidden_states * 4 * 2 * self.hidden_states
            self.cal_reorder += self.batch_size * self.sequence_length * self.hidden_states    # dropout

            # residual add
            self.cal_tensor += self.batch_size * self.sequence_length * self.hidden_states

            self.cal_tensor *= self.topk
            self.cal_vector *= self.topk
            self.cal_reorder *= self.topk


class moe_input_backward():

    def __init__(self, databytes, hidden_states):
        self.databytes = databytes
        self.hidden_states = hidden_states

        # ops of tensor such as gemm
        self.cal_tensor = 0
        # ops of vector such as softmax
        # NOTE: this version just consider no extra weights
        self.cal_vector = 0
        # ops of reorder such as transpose
        self.cal_reorder = 0

    def cal(self, batch_size, sequence_length, topk, scalesim=True):
        if scalesim:
            self.cal_tensor = sum_total_cycles(os.path.join(file_path, f'../backend/runfile/moe_1point3_256_moe_input_backward/GoogleTPU_v1_ws/COMPUTE_REPORT.csv'))
        else:
            self.cal_tensor = 0
            self.cal_vector = 0
            self.cal_reorder = 0

            self.batch_size = batch_size
            self.sequence_length = sequence_length
            self.topk = topk

            # c_proj: ofmap_gradient * WT
            self.cal_tensor += self.batch_size * self.sequence_length * self.hidden_states * 2 * self.hidden_states * 4

            # GELU: ele-wise with cost of 10
            self.cal_vector += self.batch_size * self.sequence_length * self.hidden_states * 4 * 10

            # c_fc: ofmap_gradient * WT
            self.cal_tensor += self.batch_size * self.sequence_length * self.hidden_states * 4 * 2 * self.hidden_states
            
            # LN: ele-wise with cost of 10
            self.cal_vector += self.batch_size * self.sequence_length * self.hidden_states * 10

            # Residual add
            self.cal_tensor += self.batch_size * self.sequence_length * self.hidden_states

            self.cal_tensor *= self.topk
            self.cal_vector *= self.topk
            self.cal_reorder *= self.topk


class moe_weight_backward():
    
    def __init__(self, databytes, hidden_states):
        self.databytes = databytes
        self.hidden_states = hidden_states

        # ops of tensor such as gemm
        self.cal_tensor = 0
        # ops of vector such as softmax
        # NOTE: this version just consider no extra weights
        self.cal_vector = 0
        # ops of reorder such as transpose
        self.cal_reorder = 0


    def cal(self, batch_size, sequence_length, topk, scalesim=True):
        if scalesim:
            self.cal_tensor = sum_total_cycles(os.path.join(file_path, f'../backend/runfile/moe_1point3_256_moe_weight_backward/GoogleTPU_v1_ws/COMPUTE_REPORT.csv'))
        else:
            self.cal_tensor = 0
            self.cal_vector = 0
            self.cal_reorder = 0

            self.batch_size = batch_size
            self.sequence_length = sequence_length
            self.topk = topk

            # c_proj: input ^ T * ofmap_gradient
            self.cal_tensor += self.batch_size * self.hidden_states * self.sequence_length * 2 * self.hidden_states * 4

            # c_fc: input ^ T * ofmap_gradient
            self.cal_tensor += self.batch_size * self.hidden_states * 4 * self.sequence_length * 2 * self.hidden_states

            # LN
            self.cal_vector += self.batch_size * self.sequence_length * self.hidden_states * 2

            self.cal_tensor *= self.topk
            self.cal_vector *= self.topk
            self.cal_reorder *= self.topk


class embedding():

    def __init__(self, databytes, vocab_size, hidden_states):

        self.databytes = databytes
        self.vocab_size = vocab_size
        self.hidden_states = hidden_states

        # ops of tensor such as gemm
        self.cal_tensor = 0
        # ops of vector
        self.cal_vector = 0
        # ops of reorder such as transpose
        self.cal_reorder = 0


    def cal(self, batch_size, sequence_length):
        self.cal_tensor = 0
        self.cal_vector = 0
        self.cal_reorder = 0

        self.batch_size = batch_size
        self.sequence_length = sequence_length

        # embedding
        self.cal_reorder += self.batch_size * self.sequence_length * self.vocab_size # TODO: cost should be checked

        # communication


class lm_head():

    def __init__(self, databytes, vocab_size, hidden_states):
        self.databytes = databytes
        self.hidden_states = hidden_states
        self.vocab_size = vocab_size

        # ops of tensor such as gemm
        self.cal_tensor = 0
        # ops of vector such as softmax
        # NOTE: this version just consider no extra weights
        self.cal_vector = 0
        # ops of reorder such as transpose
        self.cal_reorder = 0

        self.task = 1


    def cal(self, batch_size, sequence_length):
        self.cal_tensor = 0
        self.cal_vector = 0
        self.cal_reorder = 0

        self.batch_size = batch_size
        self.sequence_length = sequence_length

        if self.task == 1:
            self.cal_tensor += self.batch_size * self.sequence_length * self.hidden_states * 2 * self.vocab_size

            # communication
            self.task = 0

        else:
            
            self.cal_tensor += self.batch_size * self.sequence_length * self.hidden_states * 2 * self.vocab_size

            # communication
            self.task = 0



if __name__ == '__main__':

    databytes = 2
    batch_size = 1
    use_cache = False

    model_name = 'gpt2-xl'
    # model_name = 'deepspeedmoe-6.7b'
    sequece_length, hidden_states, num_heads, attn_head_size, num_layers = get_gpt_parameter(model_name)
    # print(f"sequece_length: {sequece_length}")
    # print(f"hidden_states: {hidden_states}")
    # print(f"num_heads: {num_heads}")
    # print(f"attn_head_size: {attn_head_size}")
    # print(f"num_layers: {num_layers}")


    test_a = attention(databytes=databytes, hidden_states=hidden_states, num_heads=num_heads, attn_head_size=attn_head_size, use_cache=use_cache)
    test_a.cal(1, 50)
    print(test_a.cal_tensor)
    test_a.cal(1, 1)
    print(test_a.cal_tensor)

    test_b = mlp(databytes=databytes, hidden_states=hidden_states)
    test_b.cal(1, 50)
    print(test_b.cal_tensor)
    test_b.cal(1, 1)
    print(test_b.cal_tensor)

