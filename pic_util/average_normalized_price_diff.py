import pandas as pd
import numpy as np
import os

# Function to calculate the absolute price difference normalized by the price range
def calculate_delta(price_0, price_1, nash_price, monopoly_price):
    return np.abs(price_0 - price_1) / (monopoly_price - nash_price)

# Function to calculate the average price deviation normalized by the price range
def calculate_ave_price_delta(price_0, price_1, nash_price, monopoly_price):
    return (price_0 + price_1 - 2 * nash_price) / (2 * (monopoly_price - nash_price))


# Helper function to process files within a single folder
def process_single_folder(folder_path, nash_price, monopoly_price):
    diff_delta_values = []
    ave_delta_values = []
    try:
        for file_name in os.listdir(folder_path):
            if "actions_0.csv" in file_name:
                data = pd.read_csv(os.path.join(folder_path, file_name))
                last_10000_values_0 = data.iloc[-10000:, -1]

            if "actions_1.csv" in file_name:
                data = pd.read_csv(os.path.join(folder_path, file_name))
                last_10000_values_1 = data.iloc[-10000:, -1]

        diff_delta_values.append(calculate_delta(last_10000_values_0, last_10000_values_1, nash_price, monopoly_price))
        ave_delta_values.append(calculate_ave_price_delta(last_10000_values_0, last_10000_values_1, nash_price, monopoly_price))
    except Exception as e:
        print(f"Error processing folder {folder_path}: {e}")

    return diff_delta_values, ave_delta_values

# Function to process each folder and compute deltas
def process_folders(base_path, file_prefix, folder_num, nash_price, monopoly_price, num_bins):
    diff_normalized_price_values = []
    ave_normalized_price_values = []
    
    for i in range(1, folder_num + 1):
        folder_path = os.path.join(base_path, f"{file_prefix}_{i}")
        diff_delta_values, ave_delta_values = process_single_folder(folder_path, nash_price, monopoly_price)
        
        # Calculate and store averages for each folder
        diff_normalized_price_values.append(np.mean(diff_delta_values))
        ave_normalized_price_values.append(np.mean(ave_delta_values))

    # Analyze the distribution of values
    bin_counts, _ = np.histogram(diff_normalized_price_values, bins=num_bins)
    percentages = bin_counts / len(diff_normalized_price_values) * 100
    
    return diff_normalized_price_values, ave_normalized_price_values, percentages


# Function to create folder classifications based on specified bins
def classify_folders(diff_values, ave_values, diff_bin_range, ave_bin_ranges):
    folder_classifications = {range_label: {"folders": {}, "percentage": {}} for range_label in ave_bin_ranges}
    total_items = len(diff_values)
    
    for range_label, sub_ranges in ave_bin_ranges.items():
        folder_classifications[range_label]["folders"] = {sub_label: [] for sub_label in sub_ranges}
        folder_classifications[range_label]["percentage"] = {sub_label: 0 for sub_label in sub_ranges}

        count = 0
        for i, value in enumerate(diff_values):
            if diff_bin_range[0] <= value < diff_bin_range[1]:
                count +=1
                ave_value = ave_values[i]
                for sub_label, sub_range in sub_ranges.items():
                    if sub_range[0] <= ave_value < sub_range[1]:
                        folder_classifications[range_label]["folders"][sub_label].append(i + 1)
                        folder_classifications[range_label]["percentage"][sub_label] += 1

    print("diff_bin_range[0]: ", diff_bin_range[0])
    print("diff_bin_range[1]: ", diff_bin_range[1])
    print("count: ", count)
    print("total_items: ", total_items)
    
    for key, info in folder_classifications.items():
        for sub_key in info["percentage"]:
            info["percentage"][sub_key] = info["percentage"][sub_key] / total_items * 100

    return folder_classifications


def process_and_save_results(base_path, file_prefix, folder_num, nash_price, monopoly_price, num_bins, ave_bin_ranges,output_filename):
    diff_values, ave_values, percentages = process_folders(base_path, file_prefix, folder_num, nash_price, monopoly_price, num_bins)
    classifications = classify_folders(diff_values, ave_values, (num_bins[0], num_bins[1]), ave_bin_ranges)
    bin_counts, _ = np.histogram(diff_values, bins=num_bins)

    output_file_path = os.path.join(base_path, output_filename)
    with open(output_file_path, "w") as f:
        # Output differences and averages for each folder
        for i in range(folder_num):
            f.write(f"Folder {i + 1}: Diff: {diff_values[i]:.4f}  Ave: {ave_values[i]:.4f}\n")

        # Output distribution analysis
        f.write("\n\nDiff Normalized Price Distribution Analysis:\n")
        f.write("Bin Range\tPercentage\tCount\tFolders\n")
        for i in range(len(num_bins) - 1):
            bin_range = f"{num_bins[i]:.1f}-{num_bins[i+1]:.1f}"
            count = bin_counts[i]
            percentage = percentages[i]
            f.write(f"{bin_range}\t{percentage:.2f}%\t{count}\t")
            folders_in_bin = [str(idx + 1) for idx, value in enumerate(diff_values) if num_bins[i] <= value < num_bins[i + 1]]
            f.write(", ".join(folders_in_bin))
            f.write("\n")

        # Output classification of folders based on average normalized prices
        f.write("\n\nClassification of Folders Based on Ave Normalized Price in 0-0.2 Diff Bin:\n")
        for range_label, info in classifications.items():
            f.write(f"\n{range_label} Diff Bin Analysis:\n")
            for sub_label, sub_info in info["folders"].items():
                f.write(f"Percentage ({sub_label}): {classifications[range_label]['percentage'][sub_label]:.2f}%\n")
                f.write(f"Folders ({sub_label}): {', '.join(map(str, sub_info))}\n")

    print(f"Results have been saved to {output_file_path}")




# # # simple or edge
nash_price = 0
monopoly_price = 0.5


folder_num = 0
base_path = r""
file_prefix = "DQN"
num_bins = [0, 0.2, 0.6, 1.0]


##### 0.05 case
ave_bin_ranges = {
    "0-0.2": {
        "0-0.05": (0, 0.05),
        "0.05-0.2": (0.05, 0.2),
        "0.2-0.6": (0.2, 0.6),
        "0.6-1.0": (0.6, 1.0)
    }
}

output_filename = "0.05_average_normalized_price_diff_results.txt"
# Call function with parameters
process_and_save_results(base_path, file_prefix, folder_num, nash_price, monopoly_price, num_bins, ave_bin_ranges, output_filename)

##### 0.10 case
ave_bin_ranges = {
    "0-0.2": {
        "0-0.1": (0, 0.1),
        "0.1-0.2": (0.1, 0.2),
        "0.2-0.6": (0.2, 0.6),
        "0.6-1.0": (0.6, 1.0)
    }
}

output_filename = "0.1_average_normalized_price_diff_results.txt"

# Call function with parameters
process_and_save_results(base_path, file_prefix, folder_num, nash_price, monopoly_price, num_bins, ave_bin_ranges, output_filename)
