import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from datetime import datetime
import seaborn as sns
import csv

def plot_tabular_ql(env, actions_agent_0, actions_agent_1, rewards_agent_0, rewards_agent_1, seed):
    # Define a custom color palette
    custom_palette = ['#1f77b4',  # Vivid blue
                      '#ff7f0e',  # Vivid orange
                      '#2ca02c',  # Vivid green
                      '#9467bd',  # Vivid purple
                      '#d62728',  # Vivid red
                      '#8c564b',  # Brown
                      '#e377c2',  # Pink
                      '#7f7f7f',  # Gray
                      '#bcbd22',  # Olive
                      '#17becf']  # Cyan

    # Set seaborn style
    sns.set_style("whitegrid")

    # Extract relevant prices and profits from the environment
    nash_price = env.nash
    monopoly_price = env.monopoly
    nash_profit = env.nash_profit
    monopoly_profit = env.monopoly_profit

    # Define y-axis limits for price and profit plots
    price_min = env.low_price
    price_max = env.high_price
    price_relaxation = (price_max - price_min) / 10
    price_ylim = [price_min - price_relaxation, price_max + price_relaxation]

    profit_min = 0
    profit_max = 2 * monopoly_profit
    profit_relaxation = (profit_max - profit_min) / 10
    profit_ylim = [profit_min - profit_relaxation, profit_max + profit_relaxation]

    window_size_1 = 100  # Window size for moving average
    window_size_2 = 1000  # Window size for larger moving average

    # Plot smoothed actions with moving average window size 100
    fig1, ax1 = plt.subplots(figsize=(10, 6))
    smoothed_actions0 = pd.Series(actions_agent_0).rolling(window=window_size_1).mean()
    ax1.plot(smoothed_actions0, label='agent0', color=custom_palette[0 % len(custom_palette)], linestyle='-')
    smoothed_actions1 = pd.Series(actions_agent_1).rolling(window=window_size_1).mean()
    ax1.plot(smoothed_actions1, label='agent1', color=custom_palette[1 % len(custom_palette)], linestyle='-')

    ax1.axhline(nash_price, color='#d62728', linestyle='--', linewidth=2.5, label='Nash Price')  # Red for Nash
    ax1.axhline(monopoly_price, color='black', linestyle='-.', linewidth=2.5, label='Monopoly Price')  # Black for Monopoly
    ax1.set_xlabel('Timesteps')
    ax1.set_ylabel('Actions')
    ax1.legend(loc="upper left")
    ax1.set_ylim(price_ylim)  

    # Plot smoothed actions with moving average window size 1000
    fig5, ax5 = plt.subplots(figsize=(10, 6))
    smoothed_actions0 = pd.Series(actions_agent_0).rolling(window=window_size_2).mean()
    ax5.plot(smoothed_actions0, label='agent0', color=custom_palette[0 % len(custom_palette)], linestyle='-')
    smoothed_actions1 = pd.Series(actions_agent_1).rolling(window=window_size_2).mean()
    ax5.plot(smoothed_actions1, label='agent1', color=custom_palette[1 % len(custom_palette)], linestyle='-')

    ax5.axhline(nash_price, color='#d62728', linestyle='--', linewidth=2.5, label='Nash Price')  # Red for Nash
    ax5.axhline(monopoly_price, color='black', linestyle='-.', linewidth=2.5, label='Monopoly Price')  # Black for Monopoly
    ax5.set_xlabel('Timesteps')
    ax5.set_ylabel('Actions')
    ax5.legend(loc="upper left")
    ax5.set_ylim(price_ylim)  

    # Plot the final 100 actions
    fig4, ax4 = plt.subplots(figsize=(10, 6))
    final_actions0 = actions_agent_0[-100:]
    ax4.plot(final_actions0, label='agent0', color=custom_palette[0 % len(custom_palette)], linestyle='-')
    final_actions1 = actions_agent_1[-100:]
    ax4.plot(final_actions1, label='agent1', color=custom_palette[1 % len(custom_palette)], linestyle='-')

    ax4.axhline(nash_price, color='#d62728', linestyle='--', linewidth=2.5, label='Nash Price')  # Red for Nash
    ax4.axhline(monopoly_price, color='black', linestyle='-.', linewidth=2.5, label='Monopoly Price')  # Black for Monopoly
    ax4.set_xlabel('Timesteps')
    ax4.set_ylabel('Final 100 Actions')
    ax4.legend(loc="upper left")
    ax4.set_ylim(price_ylim)  

    # Plot smoothed rewards with moving average window size 1000
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    smoothed_reward0 = pd.Series(rewards_agent_0).rolling(window=window_size_2).mean()
    ax2.plot(smoothed_reward0, label='agent0', color=custom_palette[0 % len(custom_palette)], linestyle='-')
    smoothed_rewards1 = pd.Series(rewards_agent_1).rolling(window=window_size_2).mean()
    ax2.plot(smoothed_rewards1, label='agent1', color=custom_palette[1 % len(custom_palette)], linestyle='-')

    ax2.axhline(nash_profit, color='#d62728', linestyle='--', linewidth=2.5, label='Nash Profit')  # Red for Nash
    ax2.axhline(monopoly_profit, color='black', linestyle='-.', linewidth=2.5, label='Monopoly Profit')  # Black for Monopoly
    ax2.set_xlabel('Timesteps')
    ax2.set_ylabel('Rewards')
    ax2.legend(loc="upper left")
    # ax2.set_ylim(profit_ylim)  

    # Plot the final 100 rewards
    fig6, ax6 = plt.subplots(figsize=(10, 6))
    final_rewards0 = rewards_agent_0[-100:]
    ax6.plot(final_rewards0, label='agent0', color=custom_palette[0 % len(custom_palette)], linestyle='-')
    final_rewards1 = rewards_agent_1[-100:]
    ax6.plot(final_rewards1, label='agent1', color=custom_palette[1 % len(custom_palette)], linestyle='-')

    ax6.axhline(nash_profit, color='#d62728', linestyle='--', linewidth=2.5, label='Nash Profit')  # Red for Nash
    ax6.axhline(monopoly_profit, color='black', linestyle='-.', linewidth=2.5, label='Monopoly Profit')  # Black for Monopoly
    ax6.set_xlabel('Timesteps')
    ax6.set_ylabel('Final 100 Profits')
    ax6.legend(loc="upper left")

    # Plot metric values for both agents
    fig3, ax3 = plt.subplots(figsize=(10, 6))
    metrics0 = [(reward - nash_profit) / (monopoly_profit - nash_profit) for reward in rewards_agent_0]
    smoothed_metrics0 = pd.Series(metrics0).rolling(window=window_size_1).mean()
    ax3.plot(smoothed_metrics0, label='agent_0 Metric', color=custom_palette[0 % len(custom_palette)], linestyle='-')
    global_mean0 = np.mean(metrics0)
    ax3.axhline(global_mean0, color=custom_palette[(0 + 2) % len(custom_palette)], linestyle='-.', linewidth=2.5,
                label=f'agent_0 Global Mean: {global_mean0:.2f}')

    metrics1 = [(reward - nash_profit) / (monopoly_profit - nash_profit) for reward in rewards_agent_1]
    smoothed_metrics1 = pd.Series(metrics1).rolling(window=window_size_1).mean()
    ax3.plot(smoothed_metrics1, label='agent_1 Metric', color=custom_palette[1 % len(custom_palette)], linestyle='-')
    global_mean1 = np.mean(metrics1)
    ax3.axhline(global_mean1, color=custom_palette[(1 + 2) % len(custom_palette)], linestyle='-.', linewidth=2.5,
                label=f'agent_1 Global Mean: {global_mean1:.2f}')

    ax3.set_xlabel('Timesteps')
    ax3.set_ylabel('Metric Value')
    ax3.legend(loc="upper left")

    # Ensure the plots directory exists
    folder_name = f'QL_{env.environment_type}_{seed}'
    plot_dir = os.path.join("logs", folder_name)
    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)
    current_time = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    current_time_1 = datetime.now().strftime('%Y-%m-%d')

    # Save plots to files
    fig1.savefig(os.path.join(plot_dir, f"QL_actions.png"), dpi=300)  # Higher dpi for better resolution
    fig2.savefig(os.path.join(plot_dir, f"QL_rewards.png"), dpi=300)
    fig3.savefig(os.path.join(plot_dir, f"QL_metrics.png"), dpi=300)
    fig4.savefig(os.path.join(plot_dir, f"QL_final_100_actions.png"), dpi=300)
    fig5.savefig(os.path.join(plot_dir, f"QL_actions_MA1000.png"), dpi=300)  # Higher dpi for better resolution
    fig6.savefig(os.path.join(plot_dir, f"QL_final_100_profits.png"), dpi=300)

    # Close plots to free memory
    plt.close('all')

    # Save data to CSV files
    action0_file = os.path.join(plot_dir, f"QL_{current_time_1}_actions_0.csv")
    action1_file = os.path.join(plot_dir, f"QL_{current_time_1}_actions_1.csv")
    reward0_file = os.path.join(plot_dir, f"QL_{current_time_1}_rewards_0.csv")
    reward1_file = os.path.join(plot_dir, f"QL_{current_time_1}_rewards_1.csv")

    # Write actions for agent 0 to CSV
    with open(action0_file, 'w', newline='') as file:
        writer = csv.writer(file)
        actions = np.array(actions_agent_0)
        for item in actions:
            writer.writerow([item])

    # Write actions for agent 1 to CSV
    with open(action1_file, 'w', newline='') as file:
        writer = csv.writer(file)
        actions = np.array(actions_agent_1)
        for item in actions:
            writer.writerow([item])

    # Write rewards for agent 0 to CSV
    with open(reward0_file, 'w', newline='') as file:
        writer = csv.writer(file)
        rewards = np.array(rewards_agent_0)
        for item in rewards:
            writer.writerow([item])

    # Write rewards for agent 1 to CSV
    with open(reward1_file, 'w', newline='') as file:
        writer = csv.writer(file)
        rewards = np.array(rewards_agent_1)
        for item in rewards:
            writer.writerow([item])
