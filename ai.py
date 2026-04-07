import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal

# 1. The Actor: Decides the smooth steering angle
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim=1):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(state_dim, 256)
        self.mu = nn.Linear(256, action_dim)
        self.log_std = nn.Linear(256, action_dim)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        mu = self.mu(x)
        log_std = torch.clamp(self.log_std(x), -20, 2)
        std = torch.exp(log_std)
        dist = Normal(mu, std)
        z = dist.rsample() # Sample with reparameterization trick
        action = torch.tanh(z)
        return action * 45 # Scale to 45 degree steering range

# 2. The Critic: Evaluates how "good" a state and action are
class Critic(nn.Module):
    def __init__(self, state_dim, action_dim=1):
        super(Critic, self).__init__()
        # Takes both State AND Action as input
        self.fc1 = nn.Linear(state_dim + action_dim, 256)
        self.q_out = nn.Linear(256, 1)

    def forward(self, state, action):
        x = torch.cat([state, action], 1)
        x = F.relu(self.fc1(x))
        return self.q_out(x)

# 3. The SAC Agent: Manages training and memory
class SAC:
    def __init__(self, state_dim):
        self.actor = Actor(state_dim)
        self.critic = Critic(state_dim)
        self.target_critic = Critic(state_dim)
        self.target_critic.load_state_dict(self.critic.state_dict())
        self.optimizer_a = optim.Adam(self.actor.parameters(), lr=0.0003)
        self.optimizer_c = optim.Adam(self.critic.parameters(), lr=0.0003)
        self.gamma = 0.99
        self.tau = 0.005 # Soft update parameter

    def update(self, reward, last_signal):
        state = torch.Tensor(last_signal).float().unsqueeze(0)
        action = self.actor(state)
        # Return the float steering angle to map.py
        return action.item()
       
       
