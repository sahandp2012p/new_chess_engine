import gym
import env.envs
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from collections import namedtuple
import random
torch.set_default_device('cuda')

class DQN(nn.Module):
  def __init__(self, input_size, hidden_size, output_size):
    super(DQN, self).__init__()
    self.input_size = input_size
    # Replace F.linear with nn.Linear and adjust input size for flattened state

    self.fc1 = nn.Linear(input_size, hidden_size)
    self.output_size = output_size

  def forward(self, x, output_size=None):
        x = torch.flatten(x)
        x = self.fc1(x)
        
        # If output_size is provided, adjust the output layer dynamically
        if output_size:
            self.fc2 = nn.Linear(self.fc1.out_features, self.output_size)
        
        return self.fc2(x)
class Agent:
  def __init__(self, env, lr, gamma):
    self.env = env
    self.lr = lr
    self.gamma = gamma
    self.state_size = self.env.observation_space.shape
    self.action_size = self.env.action_space.n
    self.policy_net = DQN(self.state_size, 32, self.action_size)
    self.target_net = DQN(self.state_size, 32, self.action_size)
    self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.lr)
    self.update_target_every = 10

  def update_action_size(self, action_size):
        self.action_size = action_size
        self.policy_net.forward(torch.zeros(self.state_size), output_size=action_size)
        self.target_net.forward(torch.zeros(self.state_size), output_size=action_size)

  def choose_action(self, state):
    with torch.no_grad():
      state = torch.tensor(state, dtype=torch.float).unsqueeze(0)
      print(state.shape)
      q_values = self.policy_net(state, )
      action = torch.argmax(q_values, dim=0).item()
    return action

  def learn(self, state, action, reward, next_state, done):
    state = torch.tensor(state, dtype=torch.float).unsqueeze(0)
    next_state = torch.tensor(next_state, dtype=torch.float).unsqueeze(0)

    q_values = self.policy_net(state)
    next_q_values = self.target_net(next_state)

    q_target = reward
    if not done:
      q_target += self.gamma * torch.max(next_q_values)

    expected_q = q_values[action]
    loss = F.mse_loss(expected_q, torch.tensor(q_target, dtype=torch.float))

    self.optimizer.zero_grad()
    loss.backward()
    self.optimizer.step()

    # Update target network periodically
    self.target_net.load_state_dict(self.policy_net.state_dict())

# Hyperparameters
env = gym.make('Chess-v1')
lr = 0.001
gamma = 0.99

# Create agent
agent = Agent(env, lr, gamma)

# Training loop
episodes = 1000
for episode in range(episodes):
  state = env.reset()
  done = False
  total_reward = 0
  while not done:
    legal_moves = list(env.action_space.board.legal_moves)
    agent.update_action_size(len(legal_moves))
    env.render()
    action = agent.choose_action(state)
    if len(legal_moves) == 0:
       print('No legal moves')
       break
    next_state, reward, done, info = env.step(legal_moves[action])
    total_reward += reward
    agent.learn(state, action, reward, next_state, done)
    state = next_state

  # Update exploration rate
  print('Episode', episode+1, 'Reward for both sides', total_reward)

env.close()
