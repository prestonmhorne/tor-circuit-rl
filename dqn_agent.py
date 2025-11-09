# dqn_agent.py

import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import numpy as np
import config

class DeepQNetwork(nn.Module):
    
    def __init__(self, state_dim, relay_feature_dim, hidden_dims=[512, 256, 128]):
        super(DeepQNetwork, self).__init__()
        layers = []
        in_dim = state_dim + relay_feature_dim

        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.ReLU())
            in_dim = hidden_dim

        layers.append(nn.Linear(in_dim, 1))

        self.network = nn.Sequential(*layers)

    def forward(self, state, relay_features):
        combined = torch.cat([state, relay_features], dim=-1)
        return self.network(combined)

class ReplayBuffer:

    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, relay_features, reward, next_state, next_relays, next_action_mask, done):
        self.buffer.append((state, action, relay_features, reward, next_state, next_relays, next_action_mask, done))

    def sample(self, batch_size):
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        batch = [self.buffer[idx] for idx in indices]

        states, _, relay_features, rewards, next_states, next_relays_batch, next_action_masks, dones = zip(*batch)
        return (
            np.array(states),
            np.array(relay_features),
            np.array(rewards, dtype=np.float32),
            np.array(next_states),
            list(next_relays_batch),
            list(next_action_masks),
            np.array(dones, dtype=np.float32)
        )

    def __len__(self):
        return len(self.buffer)

class DQNAgent:

    def __init__(self, action_dim, state_dim, relay_feature_dim=4, hidden_dims=[512, 256, 128]):
        self.action_dim = action_dim
        self.state_dim = state_dim
        self.relay_feature_dim = relay_feature_dim

        self.learning_rate = config.DQN_LEARNING_RATE
        self.discount_factor = config.DQN_DISCOUNT_FACTOR
        self.epsilon = config.DQN_EPSILON_START
        self.epsilon_decay = config.DQN_EPSILON_DECAY
        self.min_epsilon = config.DQN_EPSILON_MIN
        self.batch_size = config.DQN_BATCH_SIZE
        self.target_update_freq = config.DQN_TARGET_UPDATE_FREQ

        self.q_network = DeepQNetwork(state_dim, relay_feature_dim, hidden_dims)
        self.target_network = DeepQNetwork(state_dim, relay_feature_dim, hidden_dims)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()

        self.optimizer = optim.Adam(self.q_network.parameters(), lr=self.learning_rate)

        self.memory = ReplayBuffer(config.DQN_MEMORY_SIZE)

        self.steps = 0

    def _extract_relay_features(self, relay):
        return np.array([
            relay['bandwidth'] / config.RELAY_MAX_BANDWIDTH,
            relay['latency'] / config.RELAY_MAX_LATENCY,
            float(relay['guard_flag']),
            float(relay['exit_flag'])
        ], dtype=np.float32)

    def select_action(self, state, action_mask=None, relays=None):
        if np.random.random() < self.epsilon:
            if action_mask is not None:
                valid_actions = np.where(action_mask)[0]

                bandwidths = np.array([relays[i]['bandwidth'] for i in valid_actions])

                probs = bandwidths / bandwidths.sum()
                return np.random.choice(valid_actions, p=probs)
            return np.random.randint(self.action_dim)
        else:
            valid_actions = np.where(action_mask)[0] if action_mask is not None else np.arange(self.action_dim)

            with torch.no_grad():
                relay_features_list = [self._extract_relay_features(relays[action]) for action in valid_actions]
                relay_features_batch = torch.FloatTensor(np.array(relay_features_list))

                state_batch = torch.FloatTensor(state).unsqueeze(0).repeat(len(valid_actions), 1)

                q_values = self.q_network(state_batch, relay_features_batch).squeeze(1)

                best_idx = torch.argmax(q_values).item()
                best_action = valid_actions[best_idx]

            return best_action

    def store_transition(self, state, action, reward, next_state, terminated, relays, next_relays, next_action_mask):
        """Store transition in replay buffer"""
        relay_features = self._extract_relay_features(relays[action])
        self.memory.push(state, action, relay_features, reward, next_state, next_relays, next_action_mask, terminated)

    def train_step(self):
        """Perform one step of training on a batch from replay buffer"""
        if len(self.memory) < self.batch_size:
            return

        states, relay_features_batch, rewards, next_states, next_relays_batch, next_action_masks, dones = self.memory.sample(self.batch_size)

        states = torch.FloatTensor(states)
        relay_features_batch = torch.FloatTensor(relay_features_batch)
        rewards = torch.FloatTensor(rewards)
        next_states = torch.FloatTensor(next_states)
        dones = torch.FloatTensor(dones)

        current_q_values = self.q_network(states, relay_features_batch).squeeze(1)

        with torch.no_grad():
            next_q_values = torch.zeros(self.batch_size)

            batch_states = []
            batch_action_features = []
            batch_indices = []  

            for i in range(self.batch_size):
                if not dones[i]:
                    valid_actions = np.where(next_action_masks[i])[0]

                    for action in valid_actions:
                        batch_states.append(next_states[i])
                        batch_action_features.append(
                            self._extract_relay_features(next_relays_batch[i][action])
                        )
                        batch_indices.append(i)

            if len(batch_states) > 0:
                batch_states_tensor = torch.FloatTensor(np.array(batch_states))
                batch_action_features_tensor = torch.FloatTensor(np.array(batch_action_features))

                all_q_values = self.target_network(
                    batch_states_tensor,
                    batch_action_features_tensor
                ).squeeze(1)

                current_idx = 0
                for i in range(self.batch_size):
                    if not dones[i]:
                        valid_actions = np.where(next_action_masks[i])[0]
                        num_valid = len(valid_actions)

                        sample_q_values = all_q_values[current_idx:current_idx + num_valid]
                        next_q_values[i] = torch.max(sample_q_values).item()

                        current_idx += num_valid

            target_q_values = rewards + self.discount_factor * next_q_values

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