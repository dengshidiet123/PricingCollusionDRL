import pandas as pd
from ray.rllib.evaluation.episode_v2 import EpisodeV2
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from typing import Dict
import os
import matplotlib.pyplot as plt
from ray.rllib.algorithms.callbacks import DefaultCallbacks
from ray.rllib.env import BaseEnv
from ray.rllib.evaluation import RolloutWorker
from ray.rllib.policy import Policy
import seaborn as sns
from datetime import datetime
import csv

# Custom callback class for custom metrics and visualizations
class CustomMetricCallback(DefaultCallbacks):
    palette = sns.color_palette("husl", 5)

    ##################################
    def __init__(self, algorithm_name, environment_config):
        super().__init__()
        self.algorithm_name = algorithm_name
        self.environment_config = environment_config
        self.run_id = self.environment_config["run_id"]
    #####################################

    def on_episode_start(
            self,
            *,
            worker: RolloutWorker,
            base_env: BaseEnv,
            policies: Dict[str, Policy],
            episode: EpisodeV2,
            env_index: int,
            **kwargs
    ):
        # Initialize storage for actions and rewards at the start of an episode
        for agent_id in ["agent0", "agent1"]:
            episode.user_data[f'{agent_id}_actions'] = []
            episode.user_data[f'{agent_id}_rewards'] = []
        print(episode.user_data)
        print(episode.custom_metrics)

    def on_episode_step(
            self,
            *,
            worker: RolloutWorker,
            base_env: BaseEnv,
            policies: Dict[str, Policy],
            episode: EpisodeV2,
            env_index: int,
            **kwargs
    ):
        # Store actions and rewards for each agent at each step
        for agent_id in episode.get_agents():
            last_info = episode.last_info_for(agent_id)
            action = last_info["action"]
            reward = last_info["reward"]
            episode.user_data[f'{agent_id}_actions'].append(action)
            episode.user_data[f'{agent_id}_rewards'].append(reward)

    def on_episode_end(
            self,
            *,
            episode: EpisodeV2,
            **kwargs,
    ):
        # Define custom color palette
        custom_palette = ['#1f77b4', # Vivid blue
                          '#ff7f0e', # Vivid orange
                          '#2ca02c', # Vivid green
                          '#9467bd', # Vivid purple
                          '#d62728', # Vivid red
                          '#8c564b', # Brown
                          '#e377c2', # Pink
                          '#7f7f7f', # Gray
                          '#bcbd22', # Olive
                          '#17becf'] # Cyan
        sns.set_style("whitegrid")

        # Retrieve relevant metrics from the last step
        nash_price = episode.last_info_for("agent0")["nash"]
        monopoly_price = episode.last_info_for("agent0")["monopoly"]
        nash_profit = episode.last_info_for("agent0")["nash_profit"]
        monopoly_profit = episode.last_info_for("agent0")["monopoly_profit"]
        price_min = episode.last_info_for("agent0")["low_price"]
        price_max = episode.last_info_for("agent0")["high_price"]
        profit_min = episode.last_info_for("agent0")["low_profit"]
        profit_max = episode.last_info_for("agent0")["high_profit"]

        price_relaxation = (price_max - price_min) / 10
        price_ylim = [price_min - price_relaxation, price_max + price_relaxation]
        profit_relaxation = (profit_max - profit_min) / 10
        profit_ylim = [profit_min - profit_relaxation, profit_max + profit_relaxation]

        window_size_1 = 100
        window_size_2 = 1000

        # Plot smoothed actions with moving average window size 100
        fig1, ax1 = plt.subplots(figsize=(10, 6))
        for agent_id in episode.get_agents():
            actions = episode.user_data[f'{agent_id}_actions']
            smoothed_actions = pd.Series(actions).rolling(window=window_size_1).mean()
            color_index = int(agent_id[-1]) % len(custom_palette)
            ax1.plot(smoothed_actions, label=f'{agent_id}', color=custom_palette[color_index], linestyle='-')
        ax1.axhline(nash_price, color='#d62728', linestyle='--', linewidth=2.5, label='Nash Price')
        ax1.axhline(monopoly_price, color='black', linestyle='-.', linewidth=2.5, label='Monopoly Price')
        ax1.set_xlabel('Timesteps')
        ax1.set_ylabel('Actions')
        ax1.legend(loc="upper left")
        ax1.set_ylim(price_ylim)

        # Plot smoothed actions with moving average window size 1000
        fig5, ax5 = plt.subplots(figsize=(10, 6))
        for agent_id in episode.get_agents():
            actions = episode.user_data[f'{agent_id}_actions']
            smoothed_actions = pd.Series(actions).rolling(window=window_size_2).mean()
            color_index = int(agent_id[-1]) % len(custom_palette)
            ax5.plot(smoothed_actions, label=f'{agent_id}', color=custom_palette[color_index], linestyle='-')
        ax5.axhline(nash_price, color='#d62728', linestyle='--', linewidth=2.5, label='Nash Price')
        ax5.axhline(monopoly_price, color='black', linestyle='-.', linewidth=2.5, label='Monopoly Price')
        ax5.set_xlabel('Timesteps')
        ax5.set_ylabel('Actions')
        ax5.legend(loc="upper left")
        ax5.set_ylim(price_ylim)

        # Plot the final 100 actions
        fig4, ax4 = plt.subplots(figsize=(10, 6))
        for agent_id in episode.get_agents():
            actions = episode.user_data[f'{agent_id}_actions']
            final_actions = actions[-100:]
            color_index = int(agent_id[-1]) % len(custom_palette)
            ax4.plot(final_actions, label=f'{agent_id}', color=custom_palette[color_index], linestyle='-')
        ax4.axhline(nash_price, color='#d62728', linestyle='--', linewidth=2.5, label='Nash Price')
        ax4.axhline(monopoly_price, color='black', linestyle='-.', linewidth=2.5, label='Monopoly Price')
        ax4.set_xlabel('Timesteps')
        ax4.set_ylabel('Final 100 Actions')
        ax4.legend(loc="upper left")
        ax4.set_ylim(price_ylim)

        # Plot smoothed rewards with moving average window size 1000
        fig2, ax2 = plt.subplots(figsize=(10, 6))
        for agent_id in episode.get_agents():
            rewards = episode.user_data[f'{agent_id}_rewards']
            smoothed_rewards = pd.Series(rewards).rolling(window=window_size_2).mean()
            color_index = int(agent_id[-1]) % len(custom_palette)
            ax2.plot(smoothed_rewards, label=f'{agent_id}', color=custom_palette[color_index], linestyle='-')
        ax2.axhline(nash_profit, color='#d62728', linestyle='--', linewidth=2.5, label='Nash Profit')
        ax2.axhline(monopoly_profit, color='black', linestyle='-.', linewidth=2.5, label='Monopoly Profit')
        ax2.set_xlabel('Timesteps')
        ax2.set_ylabel('Rewards')
        ax2.legend(loc="upper left")

        # Plot the final 100 rewards
        fig6, ax6 = plt.subplots(figsize=(10, 6))
        for agent_id in episode.get_agents():
            rewards = episode.user_data[f'{agent_id}_rewards']
            final_rewards = rewards[-100:]
            color_index = int(agent_id[-1]) % len(custom_palette)
            ax6.plot(final_rewards, label=f'{agent_id}', color=custom_palette[color_index], linestyle='-')
        ax6.axhline(nash_profit, color='#d62728', linestyle='--', linewidth=2.5, label='Nash Profit')
        ax6.axhline(monopoly_profit, color='black', linestyle='-.', linewidth=2.5, label='Monopoly Profit')
        ax6.set_xlabel('Timesteps')
        ax6.set_ylabel('Final 100 Profits')
        ax6.legend(loc="upper left")

        # Ensure the plots directory exists
        folder_name = datetime.now().strftime("%Y%m%d_%H%M%S")
        plot_dir = os.path.join("logs", f"{self.environment_config['environment_type']}_{self.algorithm_name}_{self.run_id}", folder_name)
        if not os.path.exists(plot_dir):
            os.makedirs(plot_dir)

        current_time = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        current_time_1 = datetime.now().strftime('%Y-%m-%d')

        # Plot metrics for both agents
        fig3, ax3 = plt.subplots(figsize=(10, 6))
        for agent_id in episode.get_agents():
            rewards = episode.user_data[f'{agent_id}_rewards']
            metrics = [(reward - nash_profit) / (monopoly_profit - nash_profit) for reward in rewards]
            smoothed_metrics = pd.Series(metrics).rolling(window=window_size_1).mean()
            color_index = int(agent_id[-1]) % len(custom_palette)
            ax3.plot(smoothed_metrics, label=f'{agent_id} Metric', color=custom_palette[color_index], linestyle='-')
            global_mean = np.mean(metrics)
            ax3.axhline(global_mean, color=custom_palette[(color_index + 2) % len(custom_palette)], linestyle='-.', linewidth=2.5, label=f'{agent_id} Global Mean: {global_mean:.2f}')
        ax3.set_xlabel('Timesteps')
        ax3.set_ylabel('Metric Value')
        ax3.legend(loc="upper left")

        # Plot the final 1000 actions
        fig7, ax7 = plt.subplots(figsize=(10, 6))
        for agent_id in episode.get_agents():
            actions = episode.user_data[f'{agent_id}_actions']
            final_actions = actions[-1000:]
            color_index = int(agent_id[-1]) % len(custom_palette)
            ax7.plot(final_actions, label=f'{agent_id}', color=custom_palette[color_index], linestyle='-')
        ax7.axhline(nash_price, color='#d62728', linestyle='--', linewidth=2.5, label='Nash Price')
        ax7.axhline(monopoly_price, color='black', linestyle='-.', linewidth=2.5, label='Monopoly Price')
        ax7.set_xlabel('Timesteps')
        ax7.set_ylabel('Final 1000 Actions')
        ax7.legend(loc="upper left")
        ax7.set_ylim(price_ylim)

        # Plot the final 1000 rewards
        fig8, ax8 = plt.subplots(figsize=(10, 6))
        for agent_id in episode.get_agents():
            rewards = episode.user_data[f'{agent_id}_rewards']
            final_rewards = rewards[-1000:]
            color_index = int(agent_id[-1]) % len(custom_palette)
            ax8.plot(final_rewards, label=f'{agent_id}', color=custom_palette[color_index], linestyle='-')
        ax8.axhline(nash_profit, color='#d62728', linestyle='--', linewidth=2.5, label='Nash Profit')
        ax8.axhline(monopoly_profit, color='black', linestyle='-.', linewidth=2.5, label='Monopoly Profit')
        ax8.set_xlabel('Timesteps')
        ax8.set_ylabel('Final 1000 Profits')
        ax8.legend(loc="upper left")

        # Save the figures to files
        fig1.savefig(os.path.join(plot_dir, f"{self.algorithm_name}_{current_time}_actions.png"), dpi=300)
        fig2.savefig(os.path.join(plot_dir, f"{self.algorithm_name}_{current_time}_rewards.png"), dpi=300)
        fig3.savefig(os.path.join(plot_dir, f"{self.algorithm_name}_{current_time}_metrics.png"), dpi=300)
        fig4.savefig(os.path.join(plot_dir, f"{self.algorithm_name}_{current_time}_final_100_actions.png"), dpi=300)
        fig5.savefig(os.path.join(plot_dir, f"{self.algorithm_name}_{current_time}_actions_MA1000.png"), dpi=300)
        fig6.savefig(os.path.join(plot_dir, f"{self.algorithm_name}_{current_time}_final_100_profits.png"), dpi=300)
        fig7.savefig(os.path.join(plot_dir, f"{self.algorithm_name}_{current_time}_final_1000_actions.png"), dpi=300)
        fig8.savefig(os.path.join(plot_dir, f"{self.algorithm_name}_{current_time}_final_1000_profits.png"), dpi=300)

        # Close figures to free up memory
        plt.close(fig1)
        plt.close(fig2)
        plt.close(fig3)
        plt.close(fig4)
        plt.close(fig5)
        plt.close(fig6)
        plt.close(fig7)
        plt.close(fig8)

        # Save actions and rewards to CSV files
        action0_file = os.path.join(plot_dir, f"{self.algorithm_name}_actions_0.csv")
        action1_file = os.path.join(plot_dir, f"{self.algorithm_name}_actions_1.csv")
        reward0_file = os.path.join(plot_dir, f"{self.algorithm_name}_rewards_0.csv")
        reward1_file = os.path.join(plot_dir, f"{self.algorithm_name}_rewards_1.csv")
        
        with open(action0_file, 'w', newline='') as file:
            writer = csv.writer(file)
            actions = episode.user_data['agent0_actions']
            actions = np.array(actions)
            for item in actions:
                writer.writerow([item])

        with open(action1_file, 'w', newline='') as file:
            writer = csv.writer(file)
            actions = episode.user_data['agent1_actions']
            actions = np.array(actions)
            for item in actions:
                writer.writerow([item])

        with open(reward0_file, 'w', newline='') as file:
            writer = csv.writer(file)
            rewards = episode.user_data['agent0_rewards']
            rewards = np.array(rewards)
            for item in rewards:
                writer.writerow([item])

        with open(reward1_file, 'w', newline='') as file:
            writer = csv.writer(file)
            rewards = episode.user_data['agent1_rewards']
            rewards = np.array(rewards)
            for item in rewards:
                writer.writerow([item])

####################################



# Function to create a custom callback
def create_callback(algorithm_name, environment_config):
    def callback(*args, **kwargs):
        return CustomMetricCallback(algorithm_name, environment_config, *args, **kwargs)
    return callback




