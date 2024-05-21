# Standard library imports
import sys
import random

# Add the parent directory to the system path
sys.path.insert(0, '..')

# Custom module imports
from trainer import run_multi_agent
from combined_environments import BertrandEnvironment, tabular_q_learning_cal




###### DQN, PPO, SAC

# run_id = 1
# environment_config = {"environment_type" : "calvano", "max_steps" : 1000, "k": 1, "space_type": "continuous", "run_id": run_id, "seed" : 1}
# training_config = {}
# run_multi_agent(environment = BertrandEnvironment, 
#             environment_config=environment_config,
#             algorithm_name="SAC",
#             training_config=training_config,
#             verbose=1)





##### TQL

# environment_type = "edgeworth"
# run_id = 1
# environment_config = {"environment_type" : environment_type, "k": 1, "space_type": "discrete", "run_id": run_id}
# env = BertrandEnvironment(environment_config)
# observation_agent_type = "both"
# observation_size = 1
# seed = 1
# alpha_factor = 0.125
# beta_factor = 0.95
# updated_env, log_result = tabular_q_learning_cal(env, 2000, f"type_{observation_agent_type}_size_{observation_size}_seed_{seed}", seed=seed,
#                                                 alpha_factor=alpha_factor,beta_factor=beta_factor)


