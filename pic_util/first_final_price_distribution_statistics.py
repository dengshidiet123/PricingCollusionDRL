import pandas as pd
import numpy as np
import os


import pandas as pd
import numpy as np
import os

# Define the calculate_delta function
def calculate_delta(prices, nash_price, monopoly_price):
    return [(price - nash_price) / (monopoly_price - nash_price) for price in prices]

# Initialize lists to store delta values
first_normalized_pirce_values_0 = []
last_normalized_price_values_0 = []

first_normalized_pirce_values_1 = []
last_normalized_price_values_1 = []


### calvano
nash_price = 1.4729 # Define the Nash profit value
monopoly_price = 1.9249 # Define the Monopoly profit value

# ### simple or edgeworth
# nash_price = 0 # Define the Nash profit value
# monopoly_price = 0.5 # Define the Monopoly profit value

base_path = r""
file_prefix = ""

folder_num =25
# Loop through each folder
for i in range(1, folder_num+1): # Assuming folders are numbered consecutively from DQN_1 to DQN_22
    folder_path = f"{base_path}{file_prefix}_{i}"
    for file_name in os.listdir(folder_path):
        if "actions_0.csv" in file_name or "actions_1.csv" in file_name:
            file_path = os.path.join(folder_path, file_name)
            data = pd.read_csv(file_path)
            first_10000_values = data.iloc[0:10000, -1] # Assuming the values are in the last column
            last_10000_values = data.iloc[-10000:, -1] # Assuming the values are in the last column
            # mean_value = last_10000_values.mean()
            normalized_first = calculate_delta(first_10000_values, nash_price, monopoly_price)
            normalized_last = calculate_delta(last_10000_values, nash_price, monopoly_price)
            
            if "actions_0.csv" in file_name:
                first_normalized_pirce_values_0.extend(normalized_first)
                last_normalized_price_values_0.extend(normalized_last)
            else: # "rewards_1.csv"
                first_normalized_pirce_values_1.extend(normalized_first)
                last_normalized_price_values_1.extend(normalized_last)

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

stats_first_0 = calculate_statistics(first_normalized_pirce_values_0)
stats_last_0 = calculate_statistics(last_normalized_price_values_0)

stats_first_1 = calculate_statistics(first_normalized_pirce_values_1)
stats_last_1 = calculate_statistics(last_normalized_price_values_1)

bar_first_values = [(item_0 + item_1) / 2 for item_0, item_1 in zip(first_normalized_pirce_values_0 , first_normalized_pirce_values_1)]

bar_last_values = [(item_0 + item_1) / 2 for item_0, item_1 in zip(last_normalized_price_values_0 , last_normalized_price_values_1)]

bar_first_stats = calculate_statistics(bar_first_values)

bar_last_stats = calculate_statistics(bar_last_values)


# # Output the results
# print("Delta_0 Statistics:", delta_0_stats)
# print("Delta_1 Statistics:", delta_1_stats)
# print("Delta_bar Statistics:", delta_bar_stats)

# Define the file path for the output text file
output_file_path = os.path.join(base_path, "first_final_1w_price_distribution.txt")

# Prepare the text to be written to the file
output_text = "stats_first_0 :\n" + str(stats_first_0) + "\n\n"
output_text += "stats_last_0:\n" + str(stats_last_0) + "\n\n"
output_text += "stats_first_1 :\n" + str(stats_first_1) + "\n\n"
output_text += "stats_last_1:\n" + str(stats_last_1) + "\n\n"
output_text += "bar_first_stats :\n" + str(bar_first_stats) + "\n\n"
output_text += "bar_last_stats:\n" + str(bar_last_stats)

# Write the statistics to the text file
with open(output_file_path, "w") as file:
    file.write(output_text)

print(f"Collusion metrics have been saved to {output_file_path}")




