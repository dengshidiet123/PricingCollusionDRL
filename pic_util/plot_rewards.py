import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

########################### Standard Bertrand Reward Visualization

# Define the demand function d1(p1, p2)
def d1(p1, p2):
    d = np.zeros_like(p1)
    d[p1 < p2] = 1 - p1[p1 < p2]  # When p1 < p2, agent 1 captures all the market
    d[p1 == p2] = 0.5 * (1 - p1[p1 == p2])  # When p1 == p2, the market is split equally
    d[p1 > p2] = 0  # When p1 > p2, agent 1 captures no market
    return d

# Define the reward function r1(p1, p2)
def r1(p1, p2):
    return p1 * d1(p1, p2)

# Create a mesh grid for p1 and p2
p1 = np.linspace(0, 1, 100)
p2 = np.linspace(0, 1, 100)
P1, P2 = np.meshgrid(p1, p2)
R1 = r1(P1, P2)

# Create the figure and axes
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Plot the surface
surface = ax.plot_surface(P1, P2, R1, cmap='viridis')

# Add a color bar
fig.colorbar(surface)

# Set axis labels
ax.set_xlabel('$p_i$')
ax.set_ylabel('$p_j$')
ax.set_zlabel('$r_i$')

# Add a title
ax.set_title('Standard Bertrand: Reward of Agent $i$')

# Show the plot
plt.show()

############################## Edgeworth Bertrand

# Define the demand function d1(p1, p2) for Edgeworth competition
# def d1(p1, p2, k=0.6):
#     d = np.zeros_like(p1)
#     cond_less = p1 < p2
#     cond_equal = p1 == p2
#     cond_greater = p1 > p2

#     # When p1 < p2
#     d[cond_less] = np.minimum(k, 1 - p1[cond_less])

#     # When p1 == p2
#     d[cond_equal] = 0.5 * (1 - p1[cond_equal])

#     # When p1 > p2
#     d[cond_greater] = np.maximum(0, 1 - p1[cond_greater] - k)

#     return d

# Define the reward function r1(p1, p2) for Edgeworth competition
# def r1(p1, p2):
#     return p1 * d1(p1, p2)

# Create a mesh grid for p1 and p2
# p1 = np.linspace(0, 1, 100)
# p2 = np.linspace(0, 1, 100)
# P1, P2 = np.meshgrid(p1, p2)
# R1 = r1(P1, P2)

# Create the figure and axes
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')

# Plot the surface
# surface = ax.plot_surface(P1, P2, R1, cmap='viridis')

# Add a color bar
# fig.colorbar(surface)

# Set axis labels
# ax.set_xlabel('$p_i$')
# ax.set_ylabel('$p_j$')
# ax.set_zlabel('$r_i$')

# Add a title
# ax.set_title('Edgeworth Bertrand: Reward of Agent $i$')

# Show the plot
# plt.show()

########################## Logit Bertrand

# Define the demand function for Logit competition
# def demand_calvano(a, p, mu, a_0, agent_idx):
#     # Calculate the demand
#     num = np.exp((a[agent_idx] - p[agent_idx]) / mu)
#     den = np.sum(np.exp((a - p) / mu)) + np.exp(a_0 / mu)
#     return num / den

# Define the reward function
# def reward_function(p, c, d):
#     return (p - c) * d

# Set parameters
# a = np.array([2, 2])  # a is 2 for both agents
# mu = 0.25
# a_0 = 0
# c = 1  # Cost

# Create a mesh grid for p1 and p2
# p1 = np.linspace(1.3, 2.0, 100)
# p2 = np.linspace(1.3, 2.0, 100)
# P1, P2 = np.meshgrid(p1, p2)

# Initialize the reward matrix
# R = np.zeros_like(P1)

# Calculate the reward for each (p1, p2) pair
# for i in range(len(p1)):
#     for j in range(len(p2)):
#         p = np.array([P1[i, j], P2[i, j]])
#         d1 = demand_calvano(a, p, mu, a_0, 0)  # Demand for agent 1
#         R[i, j] = reward_function(P1[i, j], c, d1)

# Create the figure and axes
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')

# Plot the surface
# surface = ax.plot_surface(P1, P2, R, cmap='viridis')

# Add a color bar
# fig.colorbar(surface)

# Set axis labels
# ax.set_xlabel('$p_i$')
# ax.set_ylabel('$p_j$')
# ax.set_zlabel('Reward $r_i$')

# Add a title
# ax.set_title('Logit Bertrand: Reward of Agent $i$')

# Show the plot
# plt.show()
