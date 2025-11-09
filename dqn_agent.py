# dqn_agent.py

import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import numpy as np
import config

class DeepQNetwork(nn.Module):
    
    def __init__(self, state_dim, action_dim, hidden_dims=[512, 256]):
        super(DeepQNetwork, self).__init__()
        layers = []
        in_dim = state_dim

        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.ReLU())
            in_dim = hidden_dim

        layers.append(nn.Linear(in_dim, action_dim))

        self.network = nn.Sequential(*layers)

    def forward(self, state):
        return self.network(state)

class ReplayBuffer:

    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        batch = [self.buffer[idx] for idx in indices]

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

    def __init__(self, action_dim, state_dim, hidden_dims=[512, 256]):
        self.action_dim = action_dim
        self.state_dim = state_dim

        self.learning_rate = config.DQN_LEARNING_RATE
        self.discount_factor = config.DISCOUNT_FACTOR
        self.epsilon = config.DQN_EPSILON
        self.epsilon_decay = config.DQN_EPSILON_DECAY
        self.min_epsilon = config.DQN_MIN_EPSILON
        self.batch_size = config.DQN_BATCH_SIZE
        self.target_update_freq = config.DQN_TARGET_UPDATE_FREQ

        self.q_network = DeepQNetwork(state_dim, action_dim, hidden_dims)
        self.target_network = DeepQNetwork(state_dim, action_dim, hidden_dims)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()

        self.optimizer = optim.Adam(self.q_network.parameters(), lr=self.learning_rate)

        self.memory = ReplayBuffer(config.DQN_MEMORY_SIZE)

        self.steps = 0

    def policy(self, state, action_mask=None):
        if np.random.random() < self.epsilon:
            if action_mask is not None:
                valid_actions = np.where(action_mask)[0]
                return np.random.choice(valid_actions)
            return np.random.randint(self.action_dim)
        else:
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0)
                q_values = self.q_network(state_tensor).numpy()[0]
                
                if action_mask is not None:
                    q_values[~action_mask] = -np.inf
                
                return q_values.argmax()

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
            next_actions = self.q_network(next_states).argmax(1)
            next_q_values = self.target_network(next_states).gather(1, next_actions.unsqueeze(1)).squeeze(1)
            target_q_values = rewards + (1 - dones) * self.discount_factor * next_q_values

        loss = nn.MSELoss()(current_q_values, target_q_values)

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), max_norm=10.0)
        self.optimizer.step()

        self.steps += 1
        if self.steps % self.target_update_freq == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())

    def decay_epsilon(self):
        self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)