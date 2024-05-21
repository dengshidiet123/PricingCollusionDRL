from pathlib import Path

PROJECT_PATH: Path = Path().parent.resolve()
EXPORT_PATH = PROJECT_PATH / "export"
PLOT_EXPORT_PATH = EXPORT_PATH / "plots"
LOG_EXPORT_PATH = EXPORT_PATH / "logs"



SAC_TRAINING_CONFIG = {"number_of_episode": 1, "num_sgd_iter": 1, "sgd_minibatch_size": 64,
                       "train_batch_size_input": 256, "training_intensity": 256 * 1.5}




PPO_TRAINING_CONFIG = {"number_of_episode": 1, "num_sgd_iter": 4,
                       "sgd_minibatch_size": 64, "train_batch_size": 1024}



DQN_TRAINING_CONFIG = {"number_of_episode": 1, "train_batch_size": 1024, "rollout_fragment_length": 512}




DEFAULT_TRAINING_CONFIGS = {"SAC": SAC_TRAINING_CONFIG, 
                            "PPO": PPO_TRAINING_CONFIG, 
                            "DQN": DQN_TRAINING_CONFIG, }