
import matplotlib.pyplot as plt
import numpy as np
import re

##################################################################################### TP+EP

file_path = '../../../../../Real_Perf/TP+EP.txt'

# Lists to store extracted times
baseline_times = []
fusion_times = []

# Reading the file and extracting times
with open(file_path, 'r') as file:
    content = file.read()
    # Regular expression to capture the timing information
    pattern = r"Original Time:(\d+\.\d+)\nFusion Time: (\d+\.\d+)"
    matches = re.findall(pattern, content)
    for match in matches:
        baseline_times.append(float(match[0]))
        fusion_times.append(float(match[1]))

speedup = [baseline / fusion for baseline, fusion in zip(baseline_times, fusion_times)]

# Calculate minimum, maximum, and average speedup
speedup_min = min(speedup)
speedup_max = max(speedup)
speedup_avg = sum(speedup) / len(speedup)

# Print the results
print("########## TP+EP ##########:")
print("Speedup Min:", speedup_min)
print("Speedup Max:", speedup_max)
print("Speedup Avg:", speedup_avg)


# Create the plot
fig, axs = plt.subplots(figsize=(16, 9))

baseline_position = 0.5
fusion_position = 0.85
# Create violin plots for baseline and fusion on separate axes
violin_parts_baseline = axs.violinplot(baseline_times, positions=[baseline_position], vert=True, widths=0.3)
violin_parts_fusion = axs.violinplot(fusion_times, positions=[fusion_position], vert=True, widths=0.3)

# Adjust properties for baseline violin plot
for pc in violin_parts_baseline['bodies']:
    pc.set_facecolor('#4cc9f0')
    pc.set_edgecolor('#118ab2')
    pc.set_alpha(0.6)

# Adjust properties for fusion violin plot
for pc in violin_parts_fusion['bodies']:
    pc.set_facecolor('#fcbf49')
    pc.set_edgecolor('#f77f00')
    pc.set_alpha(0.6)

# Set axes appearance
axs.set_xticks([baseline_position, fusion_position])  # Position for baseline and fusion
axs.set_xticklabels(['Baseline', 'Fusion'], fontsize=44)  # Labeling axes

# Calculate the min and max values for y-axis
y_min = 0
interval = 0.5
y_max = np.ceil(max(max(baseline_times), max(fusion_times)) * 10) / 10 + 0.3
axs.set_ylim(y_min, y_max)
axs.set_yticks(np.arange(y_min, y_max, interval))
axs.set_yticklabels([f'{tick:.1f}' for tick in np.arange(y_min, y_max, interval)], fontsize=36)

# Annotate min and max values on the plot directly at their positions
min_baseline = np.floor(min(baseline_times) * 100) / 100
max_baseline = np.ceil(max(baseline_times) * 100) / 100
min_fusion = np.floor(min(fusion_times) * 100) / 100
max_fusion = np.ceil(max(fusion_times) * 100) / 100
annotate_size = 38
axs.annotate(f'{min_baseline:.2f}', xy=(baseline_position, min_baseline), xytext=(-45, -35), textcoords='offset points', fontsize=annotate_size)
axs.annotate(f'{max_baseline:.2f}', xy=(baseline_position, max_baseline), xytext=(-45, 9), textcoords='offset points', fontsize=annotate_size)
axs.annotate(f'{min_fusion}', xy=(fusion_position, min_fusion), xytext=(-45, -35), textcoords='offset points', fontsize=annotate_size)
axs.annotate(f'{max_fusion}', xy=(fusion_position, max_fusion), xytext=(-45, 9), textcoords='offset points', fontsize=annotate_size)

# Set title for the plot
axs.set_title('TP+EP: B=128, S=8192, H=2048', fontsize=43)
axs.set_ylabel('Communication Time (s)', fontsize=36)

# plt.savefig('TP+EP.png', dpi=400)
plt.savefig('TP+EP.pdf')


##################################################################################### TP+PP

file_path = '../../../../../Real_Perf/TP+PP.txt'

# Lists to store extracted times
baseline_times = []
fusion_times = []

# Reading the file and extracting times
with open(file_path, 'r') as file:
    content = file.read()
    # Regular expression to capture the timing information
    pattern = r"Original Time:(\d+\.\d+)\nFusion Time: (\d+\.\d+)"
    matches = re.findall(pattern, content)
    for match in matches:
        baseline_times.append(float(match[0]))
        fusion_times.append(float(match[1]))

speedup = [baseline / fusion for baseline, fusion in zip(baseline_times, fusion_times)]

# Calculate minimum, maximum, and average speedup
speedup_min = min(speedup)
speedup_max = max(speedup)
speedup_avg = sum(speedup) / len(speedup)

# Print the results
print("########## TP+PP ##########:")
print("Speedup Min:", speedup_min)
print("Speedup Max:", speedup_max)
print("Speedup Avg:", speedup_avg)


# Create the plot
fig, axs = plt.subplots(figsize=(16, 9))

baseline_position = 0.5
fusion_position = 0.85
# Create violin plots for baseline and fusion on separate axes
violin_parts_baseline = axs.violinplot(baseline_times, positions=[baseline_position], vert=True, widths=0.3)
violin_parts_fusion = axs.violinplot(fusion_times, positions=[fusion_position], vert=True, widths=0.3)

# Adjust properties for baseline violin plot
for pc in violin_parts_baseline['bodies']:
    pc.set_facecolor('#4cc9f0')
    pc.set_edgecolor('#118ab2')
    pc.set_alpha(0.6)

# Adjust properties for fusion violin plot
for pc in violin_parts_fusion['bodies']:
    pc.set_facecolor('#fcbf49')
    pc.set_edgecolor('#f77f00')
    pc.set_alpha(0.6)

# Set axes appearance
axs.set_xticks([baseline_position, fusion_position])  # Position for baseline and fusion
axs.set_xticklabels(['Baseline', 'Fusion'], fontsize=44)  # Labeling axes

# Calculate the min and max values for y-axis
y_min = 0
interval = 0.5
y_max = np.ceil(max(max(baseline_times), max(fusion_times)) * 10) / 10 + 0.3
axs.set_ylim(y_min, y_max)
axs.set_yticks(np.arange(y_min, y_max, interval))
axs.set_yticklabels([f'{tick:.1f}' for tick in np.arange(y_min, y_max, interval)], fontsize=36)

# Annotate min and max values on the plot directly at their positions
min_baseline = np.floor(min(baseline_times) * 100) / 100
max_baseline = np.ceil(max(baseline_times) * 100) / 100
min_fusion = np.floor(min(fusion_times) * 100) / 100
max_fusion = np.ceil(max(fusion_times) * 100) / 100
annotate_size = 38
axs.annotate(f'{min_baseline}', xy=(baseline_position, min_baseline), xytext=(-45, -35), textcoords='offset points', fontsize=annotate_size)
axs.annotate(f'{max_baseline}', xy=(baseline_position, max_baseline), xytext=(-45, 9), textcoords='offset points', fontsize=annotate_size)
axs.annotate(f'{min_fusion}', xy=(fusion_position, min_fusion), xytext=(-45, -35), textcoords='offset points', fontsize=annotate_size)
axs.annotate(f'{max_fusion}', xy=(fusion_position, max_fusion), xytext=(-45, 9), textcoords='offset points', fontsize=annotate_size)

# Set title for the plot
axs.set_title('TP+PP: B=64, S=8192, H=2048', fontsize=43)
axs.set_ylabel('Communication Time (s)', fontsize=36)

# plt.savefig('TP+PP.png', dpi=400)
plt.savefig('TP+PP.pdf')


##################################################################################### TP+SP

file_path = '../../../../../Real_Perf/TP+SP.txt'

# Lists to store extracted times
baseline_times = []
fusion_times = []

# Reading the file and extracting times
with open(file_path, 'r') as file:
    content = file.read()
    # Regular expression to capture the timing information
    pattern = r"Original Time:(\d+\.\d+)\nFusion Time: (\d+\.\d+)"
    matches = re.findall(pattern, content)
    for match in matches:
        baseline_times.append(float(match[0]))
        fusion_times.append(float(match[1]))

speedup = [baseline / fusion for baseline, fusion in zip(baseline_times, fusion_times)]

# Calculate minimum, maximum, and average speedup
speedup_min = min(speedup)
speedup_max = max(speedup)
speedup_avg = sum(speedup) / len(speedup)

# Print the results
print("########## TP+SP ##########:")
print("Speedup Min:", speedup_min)
print("Speedup Max:", speedup_max)
print("Speedup Avg:", speedup_avg)

# Create the plot
fig, axs = plt.subplots(figsize=(16, 9))

baseline_position = 0.5
fusion_position = 0.85
# Create violin plots for baseline and fusion on separate axes
violin_parts_baseline = axs.violinplot(baseline_times, positions=[baseline_position], vert=True, widths=0.3)
violin_parts_fusion = axs.violinplot(fusion_times, positions=[fusion_position], vert=True, widths=0.3)

# Adjust properties for baseline violin plot
for pc in violin_parts_baseline['bodies']:
    pc.set_facecolor('#4cc9f0')
    pc.set_edgecolor('#118ab2')
    pc.set_alpha(0.6)

# Adjust properties for fusion violin plot
for pc in violin_parts_fusion['bodies']:
    pc.set_facecolor('#fcbf49')
    pc.set_edgecolor('#f77f00')
    pc.set_alpha(0.6)

# Set axes appearance
axs.set_xticks([baseline_position, fusion_position])  # Position for baseline and fusion
axs.set_xticklabels(['Baseline', 'Fusion'], fontsize=44)  # Labeling axes

# Calculate the min and max values for y-axis
y_min = 0
interval = 0.5
y_max = np.ceil(max(max(baseline_times), max(fusion_times)) * 10) / 10 + 0.3
axs.set_ylim(y_min, y_max)
axs.set_yticks(np.arange(y_min, y_max, interval))
axs.set_yticklabels([f'{tick:.1f}' for tick in np.arange(y_min, y_max, interval)], fontsize=36)

# Annotate min and max values on the plot directly at their positions
min_baseline = np.floor(min(baseline_times) * 100) / 100
max_baseline = np.ceil(max(baseline_times) * 100) / 100
min_fusion = np.floor(min(fusion_times) * 100) / 100
max_fusion = np.ceil(max(fusion_times) * 100) / 100
annotate_size = 38
axs.annotate(f'{min_baseline}', xy=(baseline_position, min_baseline), xytext=(-45, -35), textcoords='offset points', fontsize=annotate_size)
axs.annotate(f'{max_baseline}', xy=(baseline_position, max_baseline), xytext=(-45, 9), textcoords='offset points', fontsize=annotate_size)
axs.annotate(f'{min_fusion}', xy=(fusion_position, min_fusion), xytext=(-45, -35), textcoords='offset points', fontsize=annotate_size)
axs.annotate(f'{max_fusion}', xy=(fusion_position, max_fusion), xytext=(-45, 9), textcoords='offset points', fontsize=annotate_size)

# Set title for the plot
axs.set_title('TP+SP: B=128, S=8192, H=2048', fontsize=43)
axs.set_ylabel('Communication Time (s)', fontsize=36)

# plt.savefig('TP+SP.png', dpi=400)
plt.savefig('TP+SP.pdf')



##################################################################################### SP+EP

file_path = '../../../../../Real_Perf/SP+EP.txt'

# Lists to store extracted times
baseline_times = []
fusion_times = []

# Reading the file and extracting times
with open(file_path, 'r') as file:
    content = file.read()
    # Regular expression to capture the timing information
    pattern = r"AllGather_All2All_All2All_All2All Time:(\d+\.\d+)\nFusion Time: (\d+\.\d+)"
    matches = re.findall(pattern, content)
    for match in matches:
        baseline_times.append(float(match[0]))
        fusion_times.append(float(match[1]))

speedup = [baseline / fusion for baseline, fusion in zip(baseline_times, fusion_times)]

# Calculate minimum, maximum, and average speedup
speedup_min = min(speedup)
speedup_max = max(speedup)
speedup_avg = sum(speedup) / len(speedup)

# Print the results
print("########## SP+EP ##########:")
print("Speedup Min:", speedup_min)
print("Speedup Max:", speedup_max)
print("Speedup Avg:", speedup_avg)


# Create the plot
fig, axs = plt.subplots(figsize=(16, 9))

baseline_position = 0.5
fusion_position = 0.85
# Create violin plots for baseline and fusion on separate axes
violin_parts_baseline = axs.violinplot(baseline_times, positions=[baseline_position], vert=True, widths=0.3)
violin_parts_fusion = axs.violinplot(fusion_times, positions=[fusion_position], vert=True, widths=0.3)

# Adjust properties for baseline violin plot
for pc in violin_parts_baseline['bodies']:
    pc.set_facecolor('#4cc9f0')
    pc.set_edgecolor('#118ab2')
    pc.set_alpha(0.6)

# Adjust properties for fusion violin plot
for pc in violin_parts_fusion['bodies']:
    pc.set_facecolor('#fcbf49')
    pc.set_edgecolor('#f77f00')
    pc.set_alpha(0.6)

# Set axes appearance
axs.set_xticks([baseline_position, fusion_position])  # Position for baseline and fusion
axs.set_xticklabels(['Baseline', 'Fusion'], fontsize=44)  # Labeling axes

# Calculate the min and max values for y-axis
y_min = 0
interval = 0.5
y_max = np.ceil(max(max(baseline_times), max(fusion_times)) * 10) / 10 + 0.3
axs.set_ylim(y_min, y_max)
axs.set_yticks(np.arange(y_min, y_max, interval))
axs.set_yticklabels([f'{tick:.1f}' for tick in np.arange(y_min, y_max, interval)], fontsize=36)

# Annotate min and max values on the plot directly at their positions
min_baseline = np.floor(min(baseline_times) * 100) / 100
max_baseline = np.ceil(max(baseline_times) * 100) / 100
min_fusion = np.floor(min(fusion_times) * 100) / 100
max_fusion = np.ceil(max(fusion_times) * 100) / 100
annotate_size = 38
axs.annotate(f'{min_baseline}', xy=(baseline_position, min_baseline), xytext=(-45, -35), textcoords='offset points', fontsize=annotate_size)
axs.annotate(f'{max_baseline}', xy=(baseline_position, max_baseline), xytext=(-45, 9), textcoords='offset points', fontsize=annotate_size)
axs.annotate(f'{min_fusion}', xy=(fusion_position, min_fusion), xytext=(-45, -35), textcoords='offset points', fontsize=annotate_size)
axs.annotate(f'{max_fusion}', xy=(fusion_position, max_fusion), xytext=(-45, 9), textcoords='offset points', fontsize=annotate_size)

# Set title for the plot
axs.set_title('SP+EP: B=128, S=8192, H=2048', fontsize=43)
axs.set_ylabel('Communication Time (s)', fontsize=36)

# plt.savefig('SP+EP.png', dpi=400)
plt.savefig('SP+EP.pdf')


##################################################################################### PP+EP

file_path = '../../../../../Real_Perf/PP+EP.txt'

# Lists to store extracted times
baseline_times = []
fusion_times = []

# Reading the file and extracting times
with open(file_path, 'r') as file:
    content = file.read()
    # Regular expression to capture the timing information
    pattern = r"P2P_All2All Time:(\d+\.\d+)\nFused_Multicast Time: (\d+\.\d+)"
    matches = re.findall(pattern, content)
    for match in matches:
        baseline_times.append(float(match[0]))
        fusion_times.append(float(match[1]))

speedup = [baseline / fusion for baseline, fusion in zip(baseline_times, fusion_times)]

# Calculate minimum, maximum, and average speedup
speedup_min = min(speedup)
speedup_max = max(speedup)
speedup_avg = sum(speedup) / len(speedup)

# Print the results
print("########## PP+EP ##########:")
print("Speedup Min:", speedup_min)
print("Speedup Max:", speedup_max)
print("Speedup Avg:", speedup_avg)


# Create the plot
fig, axs = plt.subplots(figsize=(16, 9))

baseline_position = 0.5
fusion_position = 0.85
# Create violin plots for baseline and fusion on separate axes
violin_parts_baseline = axs.violinplot(baseline_times, positions=[baseline_position], vert=True, widths=0.3)
violin_parts_fusion = axs.violinplot(fusion_times, positions=[fusion_position], vert=True, widths=0.3)

# Adjust properties for baseline violin plot
for pc in violin_parts_baseline['bodies']:
    pc.set_facecolor('#4cc9f0')
    pc.set_edgecolor('#118ab2')
    pc.set_alpha(0.6)

# Adjust properties for fusion violin plot
for pc in violin_parts_fusion['bodies']:
    pc.set_facecolor('#fcbf49')
    pc.set_edgecolor('#f77f00')
    pc.set_alpha(0.6)

# Set axes appearance
axs.set_xticks([baseline_position, fusion_position])  # Position for baseline and fusion
axs.set_xticklabels(['Baseline', 'Fusion'], fontsize=44)  # Labeling axes

# Calculate the min and max values for y-axis
y_min = 0
interval = 0.2
y_max = np.ceil(max(max(baseline_times), max(fusion_times)) * 10) / 10 + 0.3
axs.set_ylim(y_min, y_max)
axs.set_yticks(np.arange(y_min, y_max, interval))
axs.set_yticklabels([f'{tick:.1f}' for tick in np.arange(y_min, y_max, interval)], fontsize=36)

# Annotate min and max values on the plot directly at their positions
min_baseline = np.floor(min(baseline_times) * 100) / 100
max_baseline = np.ceil(max(baseline_times) * 100) / 100
min_fusion = np.floor(min(fusion_times) * 100) / 100
max_fusion = np.ceil(max(fusion_times) * 100) / 100
annotate_size = 38
axs.annotate(f'{min_baseline}', xy=(baseline_position, min_baseline), xytext=(-45, -35), textcoords='offset points', fontsize=annotate_size)
axs.annotate(f'{max_baseline}', xy=(baseline_position, max_baseline), xytext=(-45, 9), textcoords='offset points', fontsize=annotate_size)
axs.annotate(f'{min_fusion}', xy=(fusion_position, min_fusion), xytext=(-45, -35), textcoords='offset points', fontsize=annotate_size)
axs.annotate(f'{max_fusion}', xy=(fusion_position, max_fusion), xytext=(-45, 9), textcoords='offset points', fontsize=annotate_size)

# Set title for the plot
axs.set_title('PP+EP: B=64, S=8192, H=2048', fontsize=43)
axs.set_ylabel('Communication Time (s)', fontsize=36)

# plt.savefig('PP+EP.png', dpi=400)
plt.savefig('PP+EP.pdf')


##################################################################################### PP+SP

file_path = '../../../../../Real_Perf/PP+SP.txt'

# Lists to store extracted times
baseline_times = []
fusion_times = []

# Reading the file and extracting times
with open(file_path, 'r') as file:
    content = file.read()
    # Regular expression to capture the timing information
    pattern = r"Original Time:(\d+\.\d+)\nFusion Time: (\d+\.\d+)"
    matches = re.findall(pattern, content)
    for match in matches:
        baseline_times.append(float(match[0]))
        fusion_times.append(float(match[1]))

# Create the plot
fig, axs = plt.subplots(figsize=(16, 9))

baseline_position = 0.5
fusion_position = 0.85
# Create violin plots for baseline and fusion on separate axes
violin_parts_baseline = axs.violinplot(baseline_times, positions=[baseline_position], vert=True, widths=0.3)
violin_parts_fusion = axs.violinplot(fusion_times, positions=[fusion_position], vert=True, widths=0.3)

speedup = [baseline / fusion for baseline, fusion in zip(baseline_times, fusion_times)]

# Calculate minimum, maximum, and average speedup
speedup_min = min(speedup)
speedup_max = max(speedup)
speedup_avg = sum(speedup) / len(speedup)

# Print the results
print("########## PP+SP ##########:")
print("Speedup Min:", speedup_min)
print("Speedup Max:", speedup_max)
print("Speedup Avg:", speedup_avg)


# Adjust properties for baseline violin plot
for pc in violin_parts_baseline['bodies']:
    pc.set_facecolor('#4cc9f0')
    pc.set_edgecolor('#118ab2')
    pc.set_alpha(0.6)

# Adjust properties for fusion violin plot
for pc in violin_parts_fusion['bodies']:
    pc.set_facecolor('#fcbf49')
    pc.set_edgecolor('#f77f00')
    pc.set_alpha(0.6)

# Set axes appearance
axs.set_xticks([baseline_position, fusion_position])  # Position for baseline and fusion
axs.set_xticklabels(['Baseline', 'Fusion'], fontsize=44)  # Labeling axes

# Calculate the min and max values for y-axis
y_min = 0
interval = 0.5
y_max = np.ceil(max(max(baseline_times), max(fusion_times)) * 10) / 10 + 0.3
axs.set_ylim(y_min, y_max)
axs.set_yticks(np.arange(y_min, y_max, interval))
axs.set_yticklabels([f'{tick:.1f}' for tick in np.arange(y_min, y_max, interval)], fontsize=36)

# Annotate min and max values on the plot directly at their positions
min_baseline = np.floor(min(baseline_times) * 100) / 100
max_baseline = np.ceil(max(baseline_times) * 100) / 100
min_fusion = np.floor(min(fusion_times) * 100) / 100
max_fusion = np.ceil(max(fusion_times) * 100) / 100
annotate_size = 38
axs.annotate(f'{min_baseline:.2f}', xy=(baseline_position, min_baseline), xytext=(-45, -35), textcoords='offset points', fontsize=annotate_size)
axs.annotate(f'{max_baseline:.2f}', xy=(baseline_position, max_baseline), xytext=(-45, 9), textcoords='offset points', fontsize=annotate_size)
axs.annotate(f'{min_fusion:.2f}', xy=(fusion_position, min_fusion), xytext=(-45, -35), textcoords='offset points', fontsize=annotate_size)
axs.annotate(f'{max_fusion:.2f}', xy=(fusion_position, max_fusion), xytext=(-45, 9), textcoords='offset points', fontsize=annotate_size)

# Set title for the plot
axs.set_title('SP+PP: B=112, S=8192, H=2048', fontsize=43)
axs.set_ylabel('Communication Time (s)', fontsize=36)

# plt.savefig('PP+SP.png', dpi=400)
plt.savefig('PP+SP.pdf')
