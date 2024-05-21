import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
from matplotlib.patches import Patch
from matplotlib.font_manager import FontProperties
import tikzplotlib
from math import sqrt

def tikzplotlib_fix_ncols(obj):
    if hasattr(obj, "_ncols"):
        obj._ncol = obj._ncols
    for child in obj.get_children():
        tikzplotlib_fix_ncols(child)

def calculate_norm_price(prices, nash_price, monopoly_price):
    return [(price - nash_price) / (monopoly_price - nash_price) for price in prices]








config = {
    "Standard Bertrand": {
        "nash_price": 0,
        "monopoly_price": 0.5,
        "algorithms": {
            "TQL": {"folder_num": 0, "base_path": r"", "file_prefix": ""},
            "DQN": {"folder_num": 0, "base_path": r"", "file_prefix": ""},
            "PPO_C": {"folder_num": 0, "base_path": r"", "file_prefix": ""},
            "PPO_D": {"folder_num": 0, "base_path": r"", "file_prefix": ""},
            "SAC":{"folder_num": 0, "base_path": r"", "file_prefix": ""}
        }
    }
}



# Now focusing only on L10k results, hence simplifying the dictionary keys
rpdi_values = {algo: [] for algo in config["Standard Bertrand"]["algorithms"]}

model_info = config["Standard Bertrand"]
nash_price = model_info["nash_price"]
monopoly_price = model_info["monopoly_price"]

for algo, algo_info in model_info['algorithms'].items():
    base_path = algo_info['base_path']
    file_prefix = algo_info['file_prefix']
    
    for folder in range(1, algo_info['folder_num'] + 1):
        folder_path = os.path.join(base_path, f"{file_prefix}_{folder}")
        for file_name in os.listdir(folder_path):
            if "actions_0.csv" in file_name or "actions_1.csv" in file_name:
                file_path = os.path.join(folder_path, file_name)
                data = pd.read_csv(file_path)
                last_10000_values = data.iloc[-10000:, -1]
                normalized_last = calculate_norm_price(last_10000_values, nash_price, monopoly_price)
                rpdi_values[algo].extend(normalized_last)





algo_name_mapping = {
    "PPO_C": "PPO-C",
    "PPO_D": "PPO-D"

}

# Drawing the boxplot
fig, ax = plt.subplots(figsize=(10, 10 / sqrt(2)))  # Aspect ratio adjustment
boxes = ax.boxplot([rpdi_values[key] for key in rpdi_values.keys()], patch_artist=True, positions=np.arange(1, len(rpdi_values) + 1), showfliers=False)

# Set box face color to transparent and edge color for visibility
for box in boxes['boxes']:
    box.set_facecolor('none')  # Make box transparent
    box.set_edgecolor('black')  # Box edge color for visibility

# Adjust median line visibility and width
for median in boxes['medians']:
    median.set_color('black')  # Set median line color for visibility
    median.set_linewidth(3)  # Increase median line width for emphasis

# Correctly display algorithm names on the x-axis and increase font size
x_labels = [algo_name_mapping.get(key, key) for key in rpdi_values.keys()]
ax.set_xticklabels(x_labels, rotation=45, ha="right", fontsize=16)
ax.set_ylabel('RPDI', fontsize=16)

plt.tight_layout()

# Save and show plot adjustments
plt.savefig("RPDI_final_transparent_boxes.svg", format='svg', bbox_inches='tight')
plt.savefig("RPDI_final_transparent_boxes.pdf", format='pdf', bbox_inches='tight')
plt.savefig("RPDI_final_transparent_boxes.png", format='png', dpi=600, bbox_inches='tight')

fig = plt.gcf()
tikzplotlib_fix_ncols(fig)
tikzplotlib.save("RPDI_final_transparent_boxes.tex")

plt.show()
