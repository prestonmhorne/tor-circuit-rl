# dqn_agent.py

import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import numpy as np
import random
import config

class LinearQNetwork(nn.Module):
    
    def __init__(self, state_dim, action_dim):
        super(LinearQNetwork, self).__init__()
        self.linear = nn.Linear(state_dim, action_dim)

    def forward(self, state):
        return self.linear(state)

class ReplayBuffer:

    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            np.array(states),
            np.array(actions),
            np.array(rewards, dtype=np.float32),
            np.array(next_states),
            np.array(dones, dtype=np.float32)
        )
    
    def __len__(self):
        return len(self.buffer)

class DQNAgent:

    def __init__(self, action_dim, state_dim):
        self.action_dim = action_dim
        self.state_dim = state_dim

        self.learning_rate = config.DQN_LEARNING_RATE
        self.discount_factor = config.DISCOUNT_FACTOR
        self.epsilon = config.DQN_EPSILON
        self.epsilon_decay = config.DQN_EPSILON_DECAY
        self.min_epsilon = config.DQN_MIN_EPSILON
        self.batch_size = config.DQN_BATCH_SIZE
        self.target_update_freq = config.DQN_TARGET_UPDATE_FREQ

        self.q_network = LinearQNetwork(state_dim, action_dim)
        self.target_network = LinearQNetwork(state_dim, action_dim)
        self.target_network.load_state_dict(self.q_network.state_dict())

        self.optimizer = optim.Adam(self.q_network.parameters(), lr=self.learning_rate)

        self.memory = ReplayBuffer(config.DQN_MEMORY_SIZE)

        self.steps = 0

    def policy(self, state):
        if np.random.random() < self.epsilon:
            return np.random.randint(self.action_dim)
        else:
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0)
                q_values = self.q_network(state_tensor)
                return q_values.argmax().item()

    def update(self, state, action, reward, next_state, terminated):
        self.memory.push(state, action, reward, next_state, terminated)

        if len(self.memory) < self.batch_size:
            return
        
        states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size)

        states = torch.FloatTensor(states)
        actions = torch.LongTensor(actions)
        rewards = torch.FloatTensor(rewards)
        next_states = torch.FloatTensor(next_states)
        dones = torch.FloatTensor(dones)

        current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            next_q_values = self.target_network(next_states).max(1)[0]
            target_q_values = rewards + (1 - dones) * self.discount_factor * next_q_values

        loss = nn.MSELoss()(current_q_values, target_q_values)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.steps += 1
        if self.steps % self.target_update_freq == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())

    def decay_epsilon(self):
        self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)