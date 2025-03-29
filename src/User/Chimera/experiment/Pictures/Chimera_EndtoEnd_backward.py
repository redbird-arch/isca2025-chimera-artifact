import os
import sys
file_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(file_path)
sys.path.append(os.path.join(file_path, '../../../../utils/'))
sys.path.append(os.path.join(file_path, '../../../../computation/gpt2/'))

from utils import get_gpt2_config
from gpt2 import attention, mlp, moe

import matplotlib.pyplot as plt
import numpy as np
import math

compuatation = 128 * 1024
data_bytes = 4
batch_size = 1
sequence_length = 256

gpt_model_name = 'gpt2-medium'
gpt_config_path = os.path.join(file_path, '../../../../computation/gpt2/input/gpt2-medium-config.json')                           
gpt_model_config = get_gpt2_config(gpt_model_name, gpt_config_path)

moe_model_name = 'deepspeedmoe-1.3b'
moe_config_path = os.path.join(file_path, '../../../../computation/gpt2/input/deepspeedmoe-1.3b-config.json')                           
moe_model_config = get_gpt2_config(moe_model_name, moe_config_path)

data = np.zeros((2, 2, 4))

# Function to safely open file and return value or 0 if file does not exist
def safe_read(file_path):
    if os.path.exists(file_path):
        with open(file_path, 'r') as f:
            return int(f.read().strip())
    else:
        return 0

# Function to safely perform division, ensuring no division by zero
def safe_divide(numerator, denominator):
    if denominator == 0:
        return 0
    else:
        return numerator / denominator

# Modify all the file reading sections to use safe_read
gpt_sp_base = safe_read(os.path.join(file_path, "../ISCA25_Backward/txt/ts_base_dojo1_gpt.txt"))
gpt_sp_base += safe_read(os.path.join(file_path, "../ISCA25_Backward/txt/ts_base_dojo2_gpt.txt"))
gpt_sp_fuse = safe_read(os.path.join(file_path, "../ISCA25_Backward/txt/ts_fusion_dojo1_gpt.txt"))
gpt_sp_fuse += safe_read(os.path.join(file_path, "../ISCA25_Backward/txt/ts_fusion_dojo2_gpt.txt"))
data[0][0][0] = safe_divide(gpt_sp_base, gpt_sp_fuse)
print("Inference Speedup of gpt sp", "dojo", ":", data[0][0][0])

gpt_sp_base = safe_read(os.path.join(file_path, "../ISCA25_Backward/txt/ts_base_tpuv3_gpt.txt"))
gpt_sp_fuse = safe_read(os.path.join(file_path, "../ISCA25_Backward/txt/ts_fusion_tpuv3_gpt.txt"))
data[0][0][1] = safe_divide(gpt_sp_base, gpt_sp_fuse)
print("Inference Speedup of gpt sp", "tpuv3", ":", data[0][0][1])

gpt_sp_base = safe_read(os.path.join(file_path, "../ISCA25_Backward/txt/ts_base_tpuv4_gpt.txt"))
gpt_sp_fuse = safe_read(os.path.join(file_path, "../ISCA25_Backward/txt/ts_fusion_tpuv4_gpt.txt"))
data[0][0][2] = safe_divide(gpt_sp_base, gpt_sp_fuse)
print("Inference Speedup of gpt sp", "tpuv4", ":", data[0][0][2])

gpt_sp_base = safe_read(os.path.join(file_path, "../ISCA25_Backward/txt/ts_base_a100_gpt.txt"))
gpt_sp_fuse = safe_read(os.path.join(file_path, "../ISCA25_Backward/txt/ts_fusion_a100_gpt.txt"))
data[0][0][3] = safe_divide(gpt_sp_base, gpt_sp_fuse)
print("Inference Speedup of gpt sp", "a100", ":", data[0][0][3])


gpt_tp_base = safe_read(os.path.join(file_path, "../ISCA25_Backward/txt/tp_alpa_dojo1_gpt.txt"))
gpt_tp_base += safe_read(os.path.join(file_path, "../ISCA25_Backward/txt/tp_alpa_dojo2_gpt.txt"))
gpt_tp_fuse = safe_read(os.path.join(file_path, "../ISCA25_Backward/txt/tp_fusion_dojo1_gpt.txt"))
gpt_tp_fuse += safe_read(os.path.join(file_path, "../ISCA25_Backward/txt/tp_fusion_dojo2_gpt.txt"))
data[0][1][0] = safe_divide(gpt_tp_base, gpt_tp_fuse)
print("Inference Speedup of gpt tp", "dojo", ":", data[0][1][0])

gpt_tp_base = safe_read(os.path.join(file_path, "../ISCA25_Backward/txt/tp_alpa_tpuv3_gpt.txt"))
gpt_tp_fuse = safe_read(os.path.join(file_path, "../ISCA25_Backward/txt/tp_fusion_tpuv3_gpt.txt"))
data[0][1][1] = safe_divide(gpt_tp_base, gpt_tp_fuse)
print("Inference Speedup of gpt tp", "tpuv3", ":", data[0][1][1])

gpt_tp_base = safe_read(os.path.join(file_path, "../ISCA25_Backward/txt/tp_alpa_tpuv4_gpt.txt"))
gpt_tp_fuse = safe_read(os.path.join(file_path, "../ISCA25_Backward/txt/tp_fusion_tpuv4_gpt.txt"))
data[0][1][2] = safe_divide(gpt_tp_base, gpt_tp_fuse)
print("Inference Speedup of gpt tp", "tpuv4", ":", data[0][1][2])

gpt_tp_base = safe_read(os.path.join(file_path, "../ISCA25_Backward/txt/tp_alpa_a100_gpt.txt"))
gpt_tp_fuse = safe_read(os.path.join(file_path, "../ISCA25_Backward/txt/tp_fusion_a100_gpt.txt"))
data[0][1][3] = safe_divide(gpt_tp_base, gpt_tp_fuse)
print("Inference Speedup of gpt tp", "a100", ":", data[0][1][3])


moe_se_base = safe_read(os.path.join(file_path, "../ISCA25_Backward/txt/se_base_dojo_moe.txt"))
moe_se_fuse = safe_read(os.path.join(file_path, "../ISCA25_Backward/txt/se_fusion_dojo_moe.txt"))
data[1][0][0] = safe_divide(moe_se_base, moe_se_fuse)
print("Inference Speedup of moe se", "dojo", ":", data[1][0][0])

moe_se_base = safe_read(os.path.join(file_path, "../ISCA25_Backward/txt/se_base_tpuv3_moe.txt"))
moe_se_fuse = safe_read(os.path.join(file_path, "../ISCA25_Backward/txt/se_fusion_tpuv3_moe.txt"))
data[1][0][1] = safe_divide(moe_se_base, moe_se_fuse)
print("Inference Speedup of moe se", "tpuv3", ":", data[1][0][1])

moe_se_base = safe_read(os.path.join(file_path, "../ISCA25_Backward/txt/se_base_tpuv4_moe.txt"))
moe_se_fuse = safe_read(os.path.join(file_path, "../ISCA25_Backward/txt/se_fusion_tpuv4_moe.txt"))
data[1][0][2] = safe_divide(moe_se_base, moe_se_fuse)
print("Inference Speedup of moe se", "tpuv4", ":", data[1][0][2])

moe_se_base = safe_read(os.path.join(file_path, "../ISCA25_Backward/txt/se_base_a100_moe.txt"))
moe_se_fuse = safe_read(os.path.join(file_path, "../ISCA25_Backward/txt/se_fusion_a100_moe.txt"))
data[1][0][3] = safe_divide(moe_se_base, moe_se_fuse)
print("Inference Speedup of moe se", "a100", ":", data[1][0][3])


moe_te_base = safe_read(os.path.join(file_path, "../ISCA25_Backward/txt/te_base_dojo_moe.txt"))
moe_te_fuse = safe_read(os.path.join(file_path, "../ISCA25_Backward/txt/te_fusion_dojo_moe.txt"))
data[1][1][0] = safe_divide(moe_te_base, moe_te_fuse)
print("Inference Speedup of moe te", "dojo", ":", data[1][1][0])

moe_te_base = safe_read(os.path.join(file_path, "../ISCA25_Backward/txt/te_base_tpuv3_moe.txt"))
moe_te_fuse = safe_read(os.path.join(file_path, "../ISCA25_Backward/txt/te_fusion_tpuv3_moe.txt"))
data[1][1][1] = safe_divide(moe_te_base, moe_te_fuse)
print("Inference Speedup of moe te", "tpuv3", ":", data[1][1][1])

moe_te_base = safe_read(os.path.join(file_path, "../ISCA25_Backward/txt/te_base_tpuv4_moe.txt"))
moe_te_fuse = safe_read(os.path.join(file_path, "../ISCA25_Backward/txt/te_fusion_tpuv4_moe.txt"))
data[1][1][2] = safe_divide(moe_te_base, moe_te_fuse)
print("Inference Speedup of moe te", "tpuv4", ":", data[1][1][2])

moe_te_base = safe_read(os.path.join(file_path, "../ISCA25_Backward/txt/te_base_a100_moe.txt"))
moe_te_fuse = safe_read(os.path.join(file_path, "../ISCA25_Backward/txt/te_fusion_a100_moe.txt"))
data[1][1][3] = safe_divide(moe_te_base, moe_te_fuse)
print("Inference Speedup of moe te", "a100", ":", data[1][1][3])

np.save('./endtoend_pic.npy', data)

# plt.rcParams['font.sans-serif'] = ['Times New Roman']

major_categories = ['GPT2-medium', 'DeepSpeedMoE-1.3B']
minor_categories = ['Hybrid Para 1', 'Hybrid Para 2', 'Hybrid Para 3', 'Hybrid Para 4']
groups = ['2D-Mesh', '2D-Torus', '3D-Torus', 'Fat Tree']

fig, ax = plt.subplots(figsize=(12, 4))

group_width = 0.2
inter_group_space = 0.15  
intra_group_space = 0.05

colors = ['#ef476f', '#ffd166', '#06d6a0', '#118ab2']

hatches = ['\\', '..', '+', 'x']  
hatch_labels = ['\\\\\\', '...', '+++', 'xxx']  
hatch_colors = ['black', 'black', "black", "black"]  

start_positions = [0, len(groups) * (2 * group_width + intra_group_space) + inter_group_space]

for i, major in enumerate(data):  
    for j, minor in enumerate(major):  
        x_positions = np.arange(len(groups)) * (2 * group_width + intra_group_space) + start_positions[i] + j * group_width
        for k, value in enumerate(minor):
            ax.bar(x_positions[k], value, width=group_width, color=colors[i*2+j], hatch=hatches[k], edgecolor=hatch_colors[k], label=f'{minor_categories[j]} {groups[k]}' if i == 0 and j == 0 and k == 0 else '')
            ax.text(x_positions[k], value, f'{value:.2f}', ha='center', va='bottom', fontsize=16)

# ax.set_title('Hierarchical Bar Chart Example')
# ax.set_xlabel('Model')
ax.set_ylabel('Speedup', fontsize=18)

xticks_positions = [start_positions[i] + 2.6 * group_width + intra_group_space for i in range(len(data))]
xticks_labels = major_categories

ax.set_xticks(xticks_positions)
ax.set_xticklabels(xticks_labels, fontsize=18)

ax.xaxis.set_ticks_position('none')
# ax.yaxis.set_ticks_position('none')

max_value = np.max(data)
ax.set_ylim(0, max_value * 1.24)
plt.yticks(fontsize=14)

from matplotlib.patches import Patch
from matplotlib.lines import Line2D

legend_elements_color = [Patch(facecolor=colors[0], label=minor_categories[0]),
                         Patch(facecolor=colors[1], label=minor_categories[1]),
                         Patch(facecolor=colors[2], label=minor_categories[2]),
                         Patch(facecolor=colors[3], label=minor_categories[3])]
legend_elements_hatch = [Patch(facecolor='white', hatch=hatch_labels[i], edgecolor=hatch_colors[i], label=groups[i]) for i in range(len(hatches))]

legend_color = ax.legend(handles=legend_elements_color, ncol=4, loc='upper left', bbox_to_anchor=(0,1.16), borderpad=0.1, fontsize=18, labelspacing=0.1, columnspacing=0.5, handlelength=1)
ax.add_artist(legend_color)  # Add the first legend manually
ax.legend(handles=legend_elements_hatch, ncol=4, loc='upper left', bbox_to_anchor=(0.02,1.02), borderpad=0.1, fontsize=18, labelspacing=0.1, columnspacing=0.5, handlelength=1)

# plt.savefig(os.path.join(file_path, '../Pictures/Chimera_EndtoEnd_backward' + '.png'), format='png', dpi=300)
# plt.savefig(os.path.join(file_path, '../Pictures/Chimera_EndtoEnd_backward' + '.svg'), format='svg', dpi=800)
plt.savefig(os.path.join(file_path, '../Pictures/Chimera_EndtoEnd_backward' + '.pdf'), format='pdf', dpi=800)


gpt_speedup = data[0][0][0] + data[0][0][1] + data[0][0][2] + data[0][0][3] + data[0][1][0] + data[0][1][1] + data[0][1][2] + data[0][1][3]
gpt_speedup /= 8
moe_speedup = data[1][0][0] + data[1][0][1] + data[1][0][2] + data[1][0][3] + data[1][1][0] + data[1][1][1] + data[1][1][2] + data[1][1][3]
moe_speedup /= 8
print("Average Inference Speedup of GPT2 is", gpt_speedup)
print("Average Inference Speedup of MoE is", moe_speedup)

