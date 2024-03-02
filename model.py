import gym
import env.envs
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from collections import namedtuple
import random

class DQN(nn.Module):
  def __init__(self, input_size, hidden_size, output_size):
    super(DQN, self).__init__()
    self.input_size = input_size
    self.fc1 = nn.Linear(input_size**2*3, hidden_size)
    self.relu = nn.ReLU()
    self.fc2 = nn.Linear(hidden_size, output_size)

  def forward(self, x):
    x = self.fc1(x)
    x = self.relu(x)
    x = self.fc2(x)
    return x


class Agent:
  def __init__(self, env, lr, gamma, eps):
    self.env = env
    self.lr = lr
    self.gamma = gamma
    self.eps = eps
    self.state_size = self.env.observation_space.shape[0]
    self.action_size = self.env.action_space.n
    self.policy_net = DQN(self.state_size, 128, self.action_size)
    self.target_net = DQN(self.state_size, 128, self.action_size)
    self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.lr)
    self.update_target_every = 10

  def choose_action(self, state):
    if random.random() < self.eps:
      return self.env.action_space.sample()
    with torch.no_grad():
      state = torch.tensor(state, dtype=torch.float).unsqueeze(0)
      q_values = self.policy_net(state)
      action = torch.argmax(q_values, dim=1).item()
    return action

  def learn(self, state, action, reward, next_state, done):
    state = torch.tensor(state, dtype=torch.float).unsqueeze(0)
    next_state = torch.tensor(next_state, dtype=torch.float).unsqueeze(0)

    q_values = self.policy_net(state)[0]
    next_q_values = self.target_net(next_state)[0]

    q_target = reward
    if not done:
      q_target += self.gamma * torch.max(next_q_values)

    expected_q = q_values[action]
    loss = (expected_q - q_target) ** 2

    self.optimizer.zero_grad()
    loss.backward()
    self.optimizer.step()

    # Update target network periodically
    if self.env.t % self.update_target_every == 0:
      self.target_net.load_state_dict(self.policy_net.state_dict())

# Hyperparameters
env = gym.make('Chess-v1')
lr = 0.001
gamma = 0.99
eps = 0.1

# Create agent
agent = Agent(env, lr, gamma, eps)

# Training loop
episodes = 1000
for episode in range(episodes):
  state = env.reset()
  done = False
  while not done:
    env.render()
    action = agent.choose_action(state)
    next_state, reward, done, info = env.step(action)
    agent.learn(state, action, reward, next_state, done)
    state = next_state

  # Update exploration rate
  agent.eps = max(agent.eps * 0.99, 0.01)

env.close()
