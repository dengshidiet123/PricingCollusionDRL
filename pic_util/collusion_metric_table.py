import pandas as pd
import numpy as np
import os


import pandas as pd
import numpy as np
import os

# Define the calculate_delta function
def calculate_delta(mean_reward, nash_profit, monopoly_profit):
    return (mean_reward - nash_profit) / (monopoly_profit - nash_profit)

# Initialize lists to store delta values
delta_0_values = []
delta_1_values = []


# ### calvano
# nash_profit = 0.2229 # Define the Nash profit value
# monopoly_profit = 0.3375 # Define the Monopoly profit value

### simple or edgeworth
nash_profit = 0 # Define the Nash profit value
monopoly_profit = 0.125 # Define the Monopoly profit value

base_path = r""
file_prefix = ""

folder_num =50
# Loop through each folder
for i in range(1, folder_num+1): # Assuming folders are numbered consecutively from DQN_1 to DQN_22
    folder_path = f"{base_path}{file_prefix}_{i}"
    for file_name in os.listdir(folder_path):
        if "rewards_0.csv" in file_name or "rewards_1.csv" in file_name:
            file_path = os.path.join(folder_path, file_name)
            data = pd.read_csv(file_path)
            last_10000_values = data.iloc[-10000:, -1] # Assuming the values are in the last column
            mean_value = last_10000_values.mean()
            delta = calculate_delta(mean_value, nash_profit, monopoly_profit)
            
            if "rewards_0.csv" in file_name:
                delta_0_values.append(delta)
            else: # "rewards_1.csv"
                delta_1_values.append(delta)

# Calculate statistics for delta_0 and delta_1
def calculate_statistics(values):
    return {
        'mean': np.mean(values),
        'std': np.std(values),
        'min': np.min(values),
        '25%': np.percentile(values, 25),
        '50%': np.median(values),
        '75%': np.percentile(values, 75),
        'max': np.max(values)
    }

delta_0_stats = calculate_statistics(delta_0_values)
delta_1_stats = calculate_statistics(delta_1_values)

delta_bar_values = [(item_0 + item_1) / 2 for item_0, item_1 in zip(delta_0_values, delta_1_values)]
delta_bar_stats = calculate_statistics(delta_bar_values)


# # Output the results
# print("Delta_0 Statistics:", delta_0_stats)
# print("Delta_1 Statistics:", delta_1_stats)
# print("Delta_bar Statistics:", delta_bar_stats)

# Define the file path for the output text file
output_file_path = os.path.join(base_path, "collusion_metric.txt")

# Prepare the text to be written to the file
output_text = "Delta_0 :\n" + str(delta_0_values) + "\n\n"
output_text += "Delta_0 Statistics:\n" + str(delta_0_stats) + "\n\n"
output_text += "Delta_1 :\n" + str(delta_1_values) + "\n\n"
output_text += "Delta_1 Statistics:\n" + str(delta_1_stats) + "\n\n"
output_text += "Delta_bar :\n" + str(delta_bar_values) + "\n\n"
output_text += "Delta_bar Statistics:\n" + str(delta_bar_stats)

# Write the statistics to the text file
with open(output_file_path, "w") as file:
    file.write(output_text)

print(f"Collusion metrics have been saved to {output_file_path}")




