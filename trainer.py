from datetime import datetime
import numpy as np
import ray
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.algorithms.sac import SACConfig, SACTorchPolicy
from ray.rllib.algorithms.dqn.dqn import DQNConfig
from ray.tune import register_env
from ray.tune.logger import pretty_print
from custom_callbacks_lrz import create_callback
from constant import LOG_EXPORT_PATH, DEFAULT_TRAINING_CONFIGS
from exceptions import AlgorithmNotIntegratedError
import warnings
import torch  # or import tensorflow as tf

warnings.filterwarnings('ignore')


def run_multi_agent(environment, environment_config: dict, algorithm_name: str, training_config: dict = None, verbose=1):
    # Initialize training configuration if not provided
    if not training_config:
        training_config = DEFAULT_TRAINING_CONFIGS.get(algorithm_name)

    # Check if GPU is available and initialize Ray accordingly
    use_gpu = torch.cuda.is_available()
    if use_gpu:
        print(f"CUDA is available. Number of GPUs: {torch.cuda.device_count()}")
        print("GPU Name:", torch.cuda.get_device_name(0))
        ray.init(num_gpus=1)
    else:
        print("CUDA is not available. Running on CPU.")
        ray.init()

    # Register the environment
    register_env("bertrand", lambda env_config: environment(env_config))
    max_steps = environment_config.get("max_steps", 10000)
    
    # Training parameters
    train_batch_size_input = training_config.get("train_batch_size", 250)  # Total samples collected before an update
    sgd_minibatch_size = training_config.get("sgd_minibatch_size", 16)  # Samples used in each mini-batch for SGD
    num_sgd_iter = training_config.get("num_sgd_iter", 2)  # Number of SGD iterations per update
    rollout_fragment_length = training_config.get("rollout_fragment_length", 256)  # Length of each rollout fragment
    number_of_episode = training_config.get("number_of_episode", 1)  # Number of episodes to train
    training_intensity = training_config.get("training_intensity", train_batch_size_input * 1.5)  # Training intensity
    
    print(f"Running with {training_config} for {algorithm_name}.")
    callback_factory = create_callback(algorithm_name, environment_config)  # Create training callbacks

    # Configure the algorithm based on the provided algorithm name
    if algorithm_name == "PPO":
        algo = (
            PPOConfig()
                .training(train_batch_size=train_batch_size_input, sgd_minibatch_size=sgd_minibatch_size, num_sgd_iter=num_sgd_iter, clip_param=0.2)
                .rollouts(num_rollout_workers=1, rollout_fragment_length=rollout_fragment_length)
                .resources(num_gpus=1 if use_gpu else 0)
                .framework("torch")  # Use PyTorch framework
                .callbacks(callback_factory)
                .environment(env="bertrand", env_config=environment_config)
                .multi_agent(
                    policies=["agent0", "agent1"],
                    policy_mapping_fn=lambda agent_id, *args, **kwargs: agent_id)
                .build()
        )
    elif algorithm_name == "SAC":
        algo = (
            SACConfig()
                .training(training_intensity=training_intensity)
                .rollouts(num_rollout_workers=1)
                .resources(num_gpus=1 if use_gpu else 0)
                .framework("torch")  # Use PyTorch framework
                .callbacks(callback_factory)
                .environment(env="bertrand", env_config=environment_config)
                .multi_agent(
                    policies=["agent0", "agent1"],
                    policy_mapping_fn=lambda agent_id, *args, **kwargs: agent_id)
                .build()
        )
    elif algorithm_name == "DQN":
        config = DQNConfig()
        config.framework(framework="torch")
        config.training(training_intensity=training_intensity)
        config.rollouts(num_rollout_workers=1)
        config.resources(num_gpus=1 if use_gpu else 0)
        config.callbacks(callback_factory)
        config.environment(env="bertrand", env_config=environment_config)
        config.multi_agent(
            policies=["agent0", "agent1"],
            policy_mapping_fn=lambda agent_id, *args, **kwargs: agent_id
        )

        # # Configure the model parameters
        # config.model = {
        #     "fcnet_hiddens": [32, 32],  # Neural network structure
        #     "fcnet_activation": "relu",
        #     "max_seq_len": 20  # Maximum sequence length
        # }

        # # Set up the exploration configuration
        # config["evaluation_config"]["explore"] = True
        # config["evaluation_config"]["exploration_config"] = {
        #     "type": "EpsilonGreedy",  # Epsilon Greedy exploration strategy
        #     "initial_epsilon": 1.0,   # Initial probability of exploration
        #     "final_epsilon": 0,       # Final probability of exploration
        #     "epsilon_timesteps": 50000  # Timesteps over which epsilon value is reduced
        # }

        algo = config.build()

    else:
        raise AlgorithmNotIntegratedError("Please check your algorithm selection.")

    # Main training loop
    while True:
        result = algo.train()
        if verbose > 0:
            print("Last iteration number finished:", result["training_iteration"])
            print("Total timesteps completed:", result["timesteps_total"])
            print("Episode Finished:", result["episodes_total"])
            print("Average Reward:", result.get("episode_reward_mean", "Not available"))
            learner_info = result.get('info', {}).get('learner', {})
        
        # Break the loop if the required number of episodes is reached
        if result["episodes_total"] >= number_of_episode:
            break

    # Shutdown Ray
    ray.shutdown()
