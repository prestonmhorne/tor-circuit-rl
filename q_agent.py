# q_learning.py

import numpy as np
import config

class QLearningAgent:
    
    def __init__(self, action_dim):

        self.action_dim = action_dim

        self.learning_rate = config.Q_LEARNING_RATE
        self.discount_factor = config.DISCOUNT_FACTOR
        self.epsilon = config.Q_EPSILON
        self.epsilon_decay = config.Q_EPSILON_DECAY
        self.min_epsilon = config.Q_MIN_EPSILON

        self.Q = np.zeros((2, action_dim))

    def policy(self, state):
        
        circuit_pos = self._encode_state(state)

        if np.random.random() < self.epsilon:

            action = np.random.randint(self.action_dim)
        else:
            action = np.argmax(self.Q[circuit_pos])

        return action
    
    def update(self, state, action, reward, next_state, terminated):
        state_idx = self._encode_state(state)
        next_state_idx = self._encode_state(next_state)

        current_q = self.Q[state_idx, action]

        if terminated:
            max_next_q = 0
        else:
            max_next_q = np.max(self.Q[next_state_idx])

        target = reward + self.discount_factor * max_next_q
        new_q = current_q + self.learning_rate * (target - current_q)

        self.Q[state_idx, action] = new_q

    def decay_epsilon(self):
        self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)

    def _encode_state(self, state):
        return int(state[0])