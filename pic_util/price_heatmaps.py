import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os




config = {}




def collect_data(base_folder, folder_num, file_prefix, timesteps):
    all_data = []
    for run_id in range(1, folder_num + 1):
        folder_path = os.path.join(base_folder, f"{file_prefix}_{run_id}")
        for file_name in os.listdir(folder_path):
            if "actions_0.csv" in file_name or "actions_1.csv" in file_name:
                file_path = os.path.join(folder_path, file_name)
                data = pd.read_csv(file_path, header=None)[0]
                all_data.append(data[-timesteps:].values)
    return np.array(all_data)

def plot_price_heatmaps(config, timesteps, nash_price, monopoly_price):

    fig, axs = plt.subplots(1, 5, figsize=(32, 6.5), sharey=True)  
    # value_range = (0, 0.6)
    value_range = (1.4, 2.0)

    titles = ['(a) TQL', '(b) DQN', '(c) PPO-C', '(d) PPO-D', '(e) SAC']

    for i, (algo, algo_info) in enumerate(config["Standard Bertrand"]["algorithms"].items()):
        all_data = collect_data(algo_info["base_path"], algo_info["folder_num"], algo_info["file_prefix"], timesteps)
        df = pd.DataFrame({'x': all_data[:, ::2].flatten(), 'y': all_data[:, 1::2].flatten()})
        x_bins = np.linspace(value_range[0], value_range[1], 15)
        y_bins = np.linspace(value_range[0], value_range[1], 15)
        heatmap_data, _, _ = np.histogram2d(df['x'], df['y'], bins=[x_bins, y_bins])
        total_data_points = 2 * timesteps * algo_info["folder_num"]
        heatmap_data = heatmap_data / total_data_points

        mesh = axs[i].pcolormesh(x_bins, y_bins, heatmap_data.T, cmap="Reds", shading='auto')
        axs[i].set_title(titles[i], fontsize=32, pad=20)
        axs[i].set_xlabel("Agent 0 Price", fontsize=32)
        
        if i == 0:
            axs[i].set_ylabel("Agent 1 Price", fontsize=32)
        
        # axs[i].set_xticks([0, 0.2, 0.4, 0.6])
        axs[i].set_xticks([1.4, 1.6, 1.8, 2.0])
        # axs[i].set_yticks([0.2, 0.4, 0.6])
        axs[i].set_yticks([1.6, 1.8, 2.0])
        axs[i].tick_params(axis='both', which='major', labelsize=32)


        axs[i].scatter(nash_price, nash_price, color='blue', s=200, label='Nash Price')
        axs[i].scatter(monopoly_price, monopoly_price, color='black', s=200, marker='x', label='Monopoly Price')
        # axs[i].text(-0.1, nash_price, 'Nash', color='blue', fontsize=32, verticalalignment='center', horizontalalignment='left')
        axs[i].text(1.5, nash_price, 'Nash', color='blue', fontsize=32, verticalalignment='center', horizontalalignment='left')
        axs[i].text(monopoly_price, monopoly_price, 'Monopoly', color='black', fontsize=32, verticalalignment='bottom', horizontalalignment='right')

    plt.subplots_adjust(bottom=0.2)
    cbar_ax = fig.add_axes([0.12, -0.05, 0.76, 0.02])
    cbar = fig.colorbar(mesh, cax=cbar_ax, orientation='horizontal')


    cbar.set_label('Occurrence Ratio', fontsize=32)
    cbar.ax.tick_params(labelsize=32)

    plt.tight_layout()
    plt.savefig("price_heatmaps_with_markers.png", format='png', dpi=600, bbox_inches='tight')
    plt.show()




timesteps = 10000  
nash_price = 1.4729 
monopoly_price = 1.9249  
plot_price_heatmaps(config, timesteps, nash_price, monopoly_price)