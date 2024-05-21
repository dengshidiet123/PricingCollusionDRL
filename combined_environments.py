# Standard library imports
import os
import random
from datetime import datetime
import csv

# Third-party imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import optimize
from gymnasium.spaces import Box, Dict, Discrete, Tuple, MultiDiscrete
from ray.rllib.env import MultiAgentEnv

# Local application imports
from plot_tabular_ql import plot_tabular_ql





class BertrandEnvironment(MultiAgentEnv):

    def demand_edgeworth(self, p, c, agent_idx):
        """
        Compute the demand for Edgeworth type environment.
        :param p: List of prices for both agents.
        :param c: List of costs for both agents.
        :param agent_idx: Index of the agent (0 or 1).
        :return: Demand for the specified agent.
        """
        if p[0] < p[1]:
            if agent_idx == 0:
                d0 = min(0.6, 1 - p[0])
                return d0
            elif agent_idx == 1:
                d1 = max(0, min(0.6, 0.4 - p[1]))
                return d1
        elif p[0] == p[1]:
            d = (1 - p[0]) / 2
            return d  
        else:
            if agent_idx == 0:
                d0 = max(0, min(0.6, 0.4 - p[0]))
                return d0
            elif agent_idx == 1:
                d1 = min(0.6, 1 - p[1])
                return d1

    def demand_calvano(self, a, p, mu, agent_idx):
        """
        Compute the demand for Calvano type environment.
        :param a: List of base attractiveness for both agents.
        :param p: List of prices for both agents.
        :param mu: Smoothing parameter.
        :param agent_idx: Index of the agent (0 or 1).
        :return: Demand for the specified agent.
        """
        return np.exp((a[agent_idx] - p[agent_idx]) / mu) / (np.sum(np.exp((a - p) / mu)) + np.exp(self.a_0 / mu))

    def demand_simple(self, p, agent_idx):
        """
        Compute the demand for Simple type environment.
        :param p: List of prices for both agents.
        :param agent_idx: Index of the agent (0 or 1).
        :return: Demand for the specified agent.
        """
        if agent_idx == 0:
            if p[0] < p[1]:
                return self.market_size - p[0]
            elif p[0] == p[1]:
                return 0.5 * (self.market_size - p[0])
            else:
                return 0
        elif agent_idx == 1:
            if p[0] < p[1]:
                return 0
            elif p[0] == p[1]:
                return 0.5 * (self.market_size - p[1])
            else:
                return self.market_size - p[1]

    def __init__(self, config=None):
        super(BertrandEnvironment, self).__init__()

        if config is None:
            config = {}

        self.environment_type = config.get("environment_type", "simple")  # Default to 'simple' environment

        # Initialize common parameters
        self.num_agents = 2
        self._agent_ids = ["agent0", "agent1"]
        self.agents = ["agent0", "agent1"]
        self.space_type = config.get("space_type", "discrete")  # 'discrete' or 'continuous'
        self.run_id = config.get("run_id", 0)

        # Additional parameters
        self.k = config.get("k", 1)  # Default value is 1
        self.xi = config.get("xi", 0.1)
        self.seed = config.get("seed", None)  # Seed parameter

        # Parameters for Simple Bertrand & Edgeworth environments
        self.mc = config.get("mc", 0) 
        self.market_size = config.get("market_size", 1)   
        self.min_price = config.get("min_price", 0.0) 
        self.max_price = config.get("max_price", 1.0)

        # Parameters for Calvano Bertrand environment
        self.c_i = config.get("c_i", 1) if self.environment_type == "calvano" else None
        self.a = np.array([self.c_i + config.get("a_minus_c_i", 1)] * self.num_agents) if self.environment_type == "calvano" else None
        self.a_0 = config.get("a_0", 0) if self.environment_type == "calvano" else None
        self.mu = config.get("mu", 0.25) if self.environment_type == "calvano" else None

        # Parameters for Bertrand Edgeworth environment
        self.c1 = config.get("c_1", 0.6) if self.environment_type == "edgeworth" else None
        self.c2 = config.get("c_2", 0.6) if self.environment_type == "edgeworth" else None

        # Further initialization based on environment type
        if self.environment_type == "simple":
            self.demand = self.demand_simple

            # SimpleBertrand specific initialization
            self.nash = self.mc
            print('Nash Price:', self.nash)

            self.monopoly = float(self.market_size / 2)
            print('Monopoly Price:', self.monopoly)

            self.nash_profit = (self.nash - self.mc) * self.demand([self.nash, self.nash], 0)
            print('Nash Profit:', self.nash_profit)
            self.monopoly_profit = (self.monopoly - self.mc) * self.demand([self.monopoly, self.monopoly], 0)
            print('Monopoly Profit:', self.monopoly_profit)

            self.low_price = self.min_price
            self.high_price = self.max_price

        elif self.environment_type == "calvano":
            self.demand = self.demand_calvano

            # CalvanoBertrand specific initialization
            # Nash Equilibrium Price
            def nash_func(p):
                g_x = np.exp(self.a_0 / self.mu)
                for i in range(self.num_agents):
                    g_x += np.exp((self.a[i] - p[i]) / self.mu)
                function_list = []
                for i in range(self.num_agents):
                    f_x = np.exp((self.a[i] - p[i]) / self.mu)
                    f_x_d = f_x * (-1 / self.mu)
                    g_x_d = f_x_d
                    demand = f_x / g_x
                    demand_derivative = (f_x_d * g_x - g_x_d * f_x) / (g_x * g_x)
                    function_list.append(demand + (p[i] - self.c_i) * demand_derivative)
                return function_list

            # Finding Nash price
            nash_sol = optimize.root(nash_func, np.array([1.5, 1.5]))
            self.nash = nash_sol.x[0]
            print('Nash Price:', self.nash)

            # Finding Monopoly Price
            def monopoly_func(p):
                """Maximize total profit."""
                return -(p[0] - self.c_i) * self.demand(self.a, p, self.mu, 0)

            monopoly_sol = optimize.minimize(monopoly_func, np.array([0]))
            self.monopoly = monopoly_sol.x[0]
            print('Monopoly Price:', self.monopoly)

            self.nash_profit = (self.nash - self.c_i) * self.demand(self.a, [self.nash, self.nash], self.mu, 0)
            print('Nash Profit:', self.nash_profit)
            self.monopoly_profit = (self.monopoly - self.c_i) * self.demand(self.a, [self.monopoly, self.monopoly], self.mu, 0)
            print('Monopoly Profit:', self.monopoly_profit)

            self.low_price = self.nash - self.xi * (self.monopoly - self.nash)
            print("Low price: ", self.low_price)
            self.high_price = self.monopoly + self.xi * (self.monopoly - self.nash)
            print("High price: ", self.high_price)

        elif self.environment_type == "edgeworth":
            self.demand = self.demand_edgeworth

            # EdgeworthBertrand specific initialization
            self.nash = self.mc
            print('Nash Price:', self.nash)

            self.monopoly = float(self.market_size / 2)
            print('Monopoly Price:', self.monopoly)

            self.nash_profit = (self.nash - self.mc) * self.demand([self.nash, self.nash], [self.c1, self.c2], 0)
            print('Nash Profit:', self.nash_profit)
            self.monopoly_profit = (self.monopoly - self.mc) * self.demand([self.monopoly, self.monopoly], [self.c1, self.c2], 0)
            print('Monopoly Profit:', self.monopoly_profit)

            self.low_price = self.min_price
            self.high_price = self.max_price

        self.reward_range = (-float('inf'), float('inf'))
        self.current_step = 0
        self.max_steps = config.get("max_steps", 10000)
        self.action_history = {}
        if self.space_type == "discrete":
            # Discrete action space setup
            self.m = config.get("m", 15)
            self.action_price_space = np.linspace(self.low_price, self.high_price, self.m)
            print("action choices: ", len(self.action_price_space))
            print(self.action_price_space)
            shape = tuple([self.m] * (2 * self.k + 1))
            self.q_table_agent_0 = np.zeros(shape)
            self.q_table_agent_1 = np.zeros(shape)

            self.observation_space = Dict(
                {
                    "agent0": MultiDiscrete([self.m] * (2 * self.k)),
                    "agent1": MultiDiscrete([self.m] * (2 * self.k))
                }
            )

            self.action_space = Dict(
                {
                    "agent0": Discrete(self.m),
                    "agent1": Discrete(self.m),
                }
            )

        elif self.space_type == "continuous":
            # Continuous action space setup
            self.observation_space = Dict(
                {
                    "agent0": Box(low=self.low_price, high=self.high_price, shape=(2*self.k,)),
                    "agent1": Box(low=self.low_price, high=self.high_price, shape=(2*self.k,)),
                }
            )

            self.action_space = Dict(
                {
                    "agent0": Box(low=self.low_price, high=self.high_price, shape=(1,)),
                    "agent1": Box(low=self.low_price, high=self.high_price, shape=(1,)),
                }
            )

            print(self.action_space["agent0"])
            print(self.action_space["agent0"])

        self.reset(seed=self.seed)

    def closest_to_nash(self):
        """
        For continuous action spaces, find or generate an action close to the Nash price.
        :return: Action value close to the Nash price.
        """
        if self.space_type == "continuous":
            return max(self.low_price, min(self.high_price, self.nash))
        else:
            differences = np.abs(self.action_price_space - self.nash)
            closest_index = np.argmin(differences)
            return closest_index

    def step(self, actions_dict: dict):
        """
        Execute one step in the environment.
        :param actions_dict: Dictionary of actions for both agents.
        :return: observation, rewards, terminated, truncated, info
        """
        if self.space_type == "discrete":
            # Update price history
            for agent_id in self._agent_ids:
                self.price_history[agent_id] = np.roll(self.price_history[agent_id], -1)
                self.price_history[agent_id][-1] = actions_dict[agent_id]

            # Create observation containing price history for both agents
            combined_history = np.concatenate([self.price_history['agent0'], self.price_history['agent1']])
            observation = {agent_id: combined_history for agent_id in self._agent_ids}

            self.current_step += 1
            price_agent0 = self.action_price_space[actions_dict["agent0"]]
            price_agent1 = self.action_price_space[actions_dict["agent1"]]

            self.price_agent0 = price_agent0
            self.price_agent1 = price_agent1

            self.prices = [self.price_agent0, self.price_agent1]
            for i, agent in enumerate(self.agents):
                self.action_history[agent].append(self.prices[i])

            rewards = {}

            if self.environment_type == "simple":
                # SimpleBertrand step logic
                for i, agent in enumerate(self.agents):
                    rewards[agent] = (self.prices[i] - self.mc) * self.demand(self.prices, i)
            elif self.environment_type == "calvano":
                # CalvanoBertrand step logic
                for i, agent in enumerate(self.agents):
                    rewards[agent] = (self.prices[i] - self.c_i) * self.demand(self.a, self.prices, self.mu, i)
            elif self.environment_type == "edgeworth":
                for i, agent in enumerate(self.agents):
                    rewards[agent] = (self.prices[i] - self.mc) * self.demand(self.prices, [self.c1, self.c2], i)
        
        elif self.space_type == "continuous":
            # Update price history
            for agent_id in self._agent_ids:
                self.price_history[agent_id] = np.roll(self.price_history[agent_id], -1)
                self.price_history[agent_id][-1] = actions_dict[agent_id][0]
                   
            # Create observation containing price history for both agents
            combined_history = np.concatenate([self.price_history['agent0'], self.price_history['agent1']])
            observation = {agent_id: combined_history for agent_id in self._agent_ids}

            self.current_step += 1
            self.price_agent0 = actions_dict["agent0"][0]
            self.price_agent1 = actions_dict["agent1"][0]
            self.prices = [self.price_agent0, self.price_agent1]

            for i, agent in enumerate(self.agents):
                self.action_history[agent].append(self.prices[i])

            rewards = {}

            if self.environment_type == "simple":
                # SimpleBertrand step logic
                for i, agent in enumerate(self.agents):
                    rewards[agent] = (self.prices[i] - self.mc) * self.demand(self.prices, i)
            elif self.environment_type == "calvano":
                for i, agent in enumerate(self.agents):
                    rewards[agent] = (self.prices[i] - self.c_i) * self.demand(self.a, self.prices, self.mu, i)
            elif self.environment_type == "edgeworth":
                for i, agent in enumerate(self.agents):
                    rewards[agent] = (self.prices[i] - self.mc) * self.demand(self.prices, [self.c1, self.c2], i)

        done = self.current_step >= self.max_steps
        terminated = {}
        truncated = {}
        for i, agent in enumerate(self.agents):
            terminated[agent] = done
            truncated[agent] = done
        terminated["__all__"] = done
        truncated["__all__"] = done
        info = dict(zip(
            self.agents,
            [
                {
                    "action": self.prices[i], 
                    "reward": rewards[agent], 
                    "nash": self.nash, 
                    "monopoly": self.monopoly, 
                    "nash_profit": self.nash_profit, 
                    "monopoly_profit": self.monopoly_profit, 
                    "low_price": self.low_price, 
                    "high_price": self.high_price, 
                    "low_profit": 0, 
                    "high_profit": 2 * self.monopoly_profit
                }
                for i, agent in enumerate(self.agents)
            ]
        ))

        return observation, rewards, terminated, truncated, info
    
    def render(self, mode='human'):
        """
        Render the environment (optional implementation).
        """
        pass

    def reset(self, *, seed=None, options=None):
        """
        Reset the environment to the initial state.
        :param seed: Random seed for reproducibility.
        :param options: Additional options for reset (not used here).
        :return: Initial observation and an empty info dictionary.
        """
        self.current_step = 0

        if self.space_type == "discrete":
            if seed is not None:
                np.random.seed(seed)  # Initialize random seed

            # Reset price history
            self.price_history = {}
            for agent_id in self._agent_ids:
                self.price_history[agent_id] = [self.action_space[agent_id].sample() for _ in range(self.k)]

            # Create initial observation
            combined_history = np.concatenate([self.price_history['agent0'], self.price_history['agent1']])
            observation = {agent_id: np.array(combined_history).reshape(-1) for agent_id in self._agent_ids}

            agent0_random = self.action_space["agent0"].sample()
            agent1_random = self.action_space["agent1"].sample()

            agent0_random_price = self.action_price_space[agent0_random]
            agent1_random_price = self.action_price_space[agent1_random]

            self.action_history["agent0"] = [agent0_random_price]
            self.action_history["agent1"] = [agent1_random_price]

            return observation, {}

        elif self.space_type == "continuous": 
            if seed is not None:
                np.random.seed(seed)  # Initialize random seed

            # Reset price history
            self.price_history = {}
            for agent_id in self._agent_ids:
                self.price_history[agent_id] = [self.action_space[agent_id].sample()[0] for _ in range(self.k)]

            # Create initial observation
            combined_history = np.concatenate([self.price_history['agent0'], self.price_history['agent1']])
            observation = {agent_id: np.array(combined_history).reshape(-1) for agent_id in self._agent_ids}

            agent0_random = self.action_space["agent0"].sample()
            agent1_random = self.action_space["agent1"].sample()

            self.action_history["agent0"] = [agent0_random[0]]
            self.action_history["agent1"] = [agent1_random[0]]

            return observation, {}





def tabular_q_learning_cal(env: BertrandEnvironment, num_iteration, export_name: str, seed,
                           alpha_factor=0.05, beta_factor=1):
    # Initialize parameters
    alpha = alpha_factor  # Learning rate
    gamma = 0.95  # Discount factor
    beta = beta_factor * 0.00001  # Factor for decreasing epsilon
    epsilon = 1.0  # Initial exploration-exploitation parameter
    convergence_criteria = 100000  # Criteria for convergence
    convergence_step = 0  # Step counter for convergence
    policy_changed = 0  # Counter for policy changes
    q_value_change_agent0 = []  # List to track Q-value changes for agent 0
    q_value_change_agent1 = []  # List to track Q-value changes for agent 1
    state, _ = env.reset()  # Initialize the environment and get initial state
    state_agent_0 = state.get("agent0")
    state_agent_1 = state.get("agent1")
    rewards_agent_0 = []  # List to track rewards for agent 0
    actions_agent_0 = []  # List to track actions for agent 0
    rewards_agent_1 = []  # List to track rewards for agent 1
    actions_agent_1 = []  # List to track actions for agent 1

    # Set random seed for reproducibility
    np.random.seed(seed=seed)

    # Main loop for Q-learning iterations
    for iteration in range(num_iteration):
        # Determine best actions before update
        best_action_before_update_agent0 = np.argmax(env.q_table_agent_0[tuple(state_agent_0) + (slice(None),)])
        best_action_before_update_agent1 = np.argmax(env.q_table_agent_1[tuple(state_agent_1) + (slice(None),)])

        # Epsilon-greedy action selection for agent 0
        if np.random.rand() < epsilon:
            action_agent0 = env.action_space["agent0"].sample()  # Explore: random action
        else:
            action_agent0 = best_action_before_update_agent0  # Exploit: best known action

        # Epsilon-greedy action selection for agent 1
        if np.random.rand() < epsilon:
            action_agent1 = env.action_space["agent1"].sample()  # Explore: random action
        else:
            action_agent1 = best_action_before_update_agent1  # Exploit: best known action

        # Take action and observe results
        next_state, reward_agent, reward_competitor, done, _ = env.step({"agent0": action_agent0, "agent1": action_agent1})
        next_state_agent_0 = list(next_state.get("agent0"))
        next_state_agent_1 = list(next_state.get("agent1"))

        # Get rewards for each agent
        reward_agent_0 = reward_agent.get("agent0")
        reward_agent_1 = reward_agent.get("agent1")

        # Record rewards and actions
        rewards_agent_0.append(reward_agent_0)
        rewards_agent_1.append(reward_agent_1)
        actions_agent_0.append(env.action_price_space[action_agent0])
        actions_agent_1.append(env.action_price_space[action_agent1])

        # Get Q-values before update
        qvalue_agent0_before = env.q_table_agent_0[tuple(state_agent_0) + (action_agent0,)]
        qvalue_agent1_before = env.q_table_agent_1[tuple(state_agent_1) + (action_agent1,)]

        # Q-learning update for agent 0
        best_q_agent0 = np.max(env.q_table_agent_0[tuple(next_state_agent_0) + (slice(None),)])
        env.q_table_agent_0[tuple(state_agent_0) + (action_agent0,)] += alpha * (
                reward_agent_0 + gamma * best_q_agent0 - env.q_table_agent_0[tuple(state_agent_0) + (action_agent0,)])

        # Q-learning update for agent 1
        best_q_agent1 = np.max(env.q_table_agent_1[tuple(next_state_agent_1) + (slice(None),)])
        env.q_table_agent_1[tuple(state_agent_1) + (action_agent1,)] += alpha * (
                reward_agent_1 + gamma * best_q_agent1 - env.q_table_agent_1[tuple(state_agent_1) + (action_agent1,)])

        # Determine best actions after update
        best_action_after_update_agent0 = np.argmax(env.q_table_agent_0[tuple(state_agent_0) + (slice(None),)])
        best_action_after_update_agent1 = np.argmax(env.q_table_agent_1[tuple(state_agent_1) + (slice(None),)])
        qvalue_agent0_after = env.q_table_agent_0[tuple(state_agent_0) + (action_agent0,)]
        qvalue_agent1_after = env.q_table_agent_1[tuple(state_agent_1) + (action_agent1,)]

        # Track Q-value changes
        q_value_change_agent0.append(np.abs(qvalue_agent0_after - qvalue_agent0_before))
        q_value_change_agent1.append(np.abs(qvalue_agent1_after - qvalue_agent1_before))

        # Check for policy convergence
        if (best_action_after_update_agent0 == best_action_before_update_agent0) and (
                best_action_after_update_agent1 == best_action_before_update_agent1):
            convergence_step += 1
        else:
            convergence_step = 0
            policy_changed += 1

        # Update state
        state_agent_0 = next_state_agent_0
        state_agent_1 = next_state_agent_1

        # Decay epsilon
        epsilon = np.exp(-beta * iteration)

        # Print progress every 100,000 iterations
        if iteration % 100000 == 0:
            print("Iteration: ", iteration)
            print("Epsilon: ", epsilon)
            print("Action Agent 0: ", env.action_price_space[action_agent0])
            print("Action Agent 1: ", env.action_price_space[action_agent1])

    # Plot results
    plot_tabular_ql(env, actions_agent_0, actions_agent_1, rewards_agent_0, rewards_agent_1, seed)

    return env







