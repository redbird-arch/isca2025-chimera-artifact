import os
import sys
import matplotlib.pyplot as plt
import numpy as np

# Define the directory containing the files
file_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(file_path, '../'))
directory_path = os.path.join(file_path, "../ISCA25_Synthetic/txt")

pattern_strategy = {
    "tp": ["alpa", "fusion"],
    "te": ["base", "fusion"],
    "ts": ["base", "megatron"],
    "pe": ["base", "fusion"],
    "ps": ["base", "fusion"],
    "se": ["base", "fusion"]
}

# Function to read the numeric value from each file
def read_file_data(file_path):
    with open(file_path, 'r') as file:
        data = file.read().strip()
    return float(data)

# Create the plot
fig, axs = plt.subplots(2, 3, figsize=(20, 12))  # 2 rows, 3 columns
axs = axs.flatten()  # Flatten to easily loop over the axes

bw_labels = ['Mesh-Baseline', 'Torus-Baseline', '3D-Torus-Baseline', 'DGX2-Baseline', 'Mesh-Fusion', 'Torus-Fusion', '3D-Torus-Baseline', 'DGX2-Fusion']
fill_colors = ['#4cc9f0', '#ef767a', '#fcbf49', '#49beaa', '#456990', '#ef5b5b', '#f77f00', '#20a39e']
line_colors = ['#4cc9f0', '#ef767a', '#fcbf49', '#49beaa', '#456990', '#ef5b5b', '#f77f00', '#20a39e']
linestyles = ['--', '--', '--', '--', '-', '-', '-', '-']
markers = ['s', '^', 'o', 'D', 's', '^', 'o', 'D']

# Store the plot data for legend
plot_data = []

subname_dict = {
    'ts': 'TP+SP',
    'tp': 'TP+PP',
    'te': 'TP+EP',
    'pe': 'PP+EP',
    'ps': 'SP+PP',
    'se': 'SP+EP'
}

for idx, pattern_name in enumerate(['ts', 'tp', 'te', 'pe', 'ps', 'se']):
    # List all files in the directory that start with the pattern name
    pattern_files = [f for f in os.listdir(directory_path) if f.startswith(f"{pattern_name}_")]

    # Sort the files based on the numeric value after the last underscore in the filename
    pattern_files_sorted = sorted(pattern_files, key=lambda x: int(x.split('_')[-1].split('.')[0]))

    # Categorize files into different groups based on their content descriptors
    pattern_base_mesh = [f for f in pattern_files_sorted if f"{pattern_strategy[pattern_name][0]}_mesh" in f]
    pattern_base_torus = [f for f in pattern_files_sorted if f"{pattern_strategy[pattern_name][0]}_torus_" in f]
    pattern_base_torus3d = [f for f in pattern_files_sorted if f"{pattern_strategy[pattern_name][0]}_torus3d_" in f]
    pattern_base_dgx2 = [f for f in pattern_files_sorted if f"{pattern_strategy[pattern_name][0]}_dgx2" in f]
    pattern_fusion_mesh = [f for f in pattern_files_sorted if f"{pattern_strategy[pattern_name][1]}_mesh" in f]
    pattern_fusion_torus = [f for f in pattern_files_sorted if f"{pattern_strategy[pattern_name][1]}_torus_" in f]
    pattern_fusion_torus3d = [f for f in pattern_files_sorted if f"{pattern_strategy[pattern_name][1]}_torus3d_" in f]
    pattern_fusion_dgx2 = [f for f in pattern_files_sorted if f"{pattern_strategy[pattern_name][1]}_dgx2" in f]

    # Gather and sort data from each group
    data_base_mesh = [read_file_data(os.path.join(directory_path, f)) for f in pattern_base_mesh]
    data_base_torus = [read_file_data(os.path.join(directory_path, f)) for f in pattern_base_torus]
    data_base_torus3d = [read_file_data(os.path.join(directory_path, f)) for f in pattern_base_torus3d]
    data_base_dgx2 = [read_file_data(os.path.join(directory_path, f)) for f in pattern_base_dgx2]
    data_fusion_mesh = [read_file_data(os.path.join(directory_path, f)) for f in pattern_fusion_mesh]
    data_fusion_torus = [read_file_data(os.path.join(directory_path, f)) for f in pattern_fusion_torus]
    data_fusion_torus3d = [read_file_data(os.path.join(directory_path, f)) for f in pattern_fusion_torus3d]
    data_fusion_dgx2 = [read_file_data(os.path.join(directory_path, f)) for f in pattern_fusion_dgx2]
    max_length = 14
    all_data = [data_base_mesh, data_base_torus, data_base_torus3d, data_base_dgx2, data_fusion_mesh, data_fusion_torus, data_fusion_torus3d, data_fusion_dgx2]
    all_data_padded = [data[:max_length] for data in all_data]

    # Print Base/Fusion results (Base divided by Fusion) and mean values
    print(f"\n{pattern_name.upper()} Data (Base/Fusion):")
    for topology, base_data, fusion_data in zip(
            ['Mesh', 'Torus', 'Torus3D', 'DGX2'],
            [data_base_mesh, data_base_torus, data_base_torus3d, data_base_dgx2],
            [data_fusion_mesh, data_fusion_torus, data_fusion_torus3d, data_fusion_dgx2]):
        base_fusion_ratio = np.divide(base_data, fusion_data, where=fusion_data != 0)  # Base / Fusion
        mean_base_fusion = np.mean(base_fusion_ratio)  # Mean of Base/Fusion ratio
        max_base_fusion = np.max(base_fusion_ratio)  # Max of Base/Fusion ratio
        min_base_fusion = np.min(base_fusion_ratio)  # Min of Base/Fusion ratio
        print(f"  {topology} - Base/Fusion: {base_fusion_ratio}, Mean: {mean_base_fusion:.4f}", f"Max: {max_base_fusion:.4f}", f"Min: {min_base_fusion:.4f}")

    # Data size array and labels
    Data_Size = np.array([4096, 8192, 16384, 32768, 65536, 131072, 262144, 524288, 1048567, 2097152, 4194304, 8388608, 16777216, 33554432])
    Data_Size_Labels = ['4KB', '8KB', '16KB', '32KB', '64KB', '128KB', '256KB', '512KB', '1MB', '2MB', '4MB', '8MB', '16MB', '32MB']

    # Bandwidth calculation
    bandwidths = [np.divide(Data_Size, data, where=data != 0, out=np.zeros_like(data)) for data in all_data_padded]

    # Plotting each pattern_name data in its respective subplot
    ax = axs[idx]  # Get the subplot axis
    for bw, fill_color, line_color, linestyle, marker in zip(bandwidths, fill_colors, line_colors, linestyles, markers):
        ax.plot(Data_Size, bw, color=line_color, markerfacecolor=fill_color, linestyle=linestyle, marker=marker, markersize=11, markeredgewidth=2, linewidth=2)

    # Set labels and titles for each subplot
    ax.set_xlabel('Message Size', fontsize=20)
    ax.set_ylabel('Bandwidth (GB/s)', fontsize=22)
    ax.set_xscale('log')
    ax.tick_params(axis='x', labelsize=18, rotation=30)
    ax.tick_params(axis='y', labelsize=20)
    ax.set_xticks(Data_Size[::2])
    ax.set_xticklabels(Data_Size_Labels[::2])

    y_upper_limit = ax.get_ylim()[1]
    if y_upper_limit > 100:
        ax.set_yticks(np.arange(0, y_upper_limit + 10, 15))
    elif y_upper_limit > 55:
        ax.set_yticks(np.arange(0, y_upper_limit + 5, 10))
    else:
        ax.set_yticks(np.arange(0, y_upper_limit + 5, 5))

    ax.set_title(f'{subname_dict[pattern_name].upper()}', fontsize=24)
    ax.grid(True)

# Add a global legend above the first row of subplots
legend_order = ['Mesh-Baseline', 'Torus-Baseline', '3D-Torus-Baseline', 'DGX2-Baseline',
                'Mesh-Fusion', 'Torus-Fusion', '3D-Torus-Fusion', 'DGX2-Fusion']
# fig.legend(legend_order, loc='upper center', ncol=4, fontsize=20, frameon=True)

# Adjust the layout and save the figure
plt.tight_layout()
plt.subplots_adjust(top=0.88)  # Make room for the global legend
# plt.savefig('Chimera_Synthetic_All_Patterns.png', dpi=400)
plt.savefig('Chimera_Synthetic_All_Patterns.pdf')
# plt.show()
