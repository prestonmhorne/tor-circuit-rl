# tor_circuit_env.py

import config
from gymnasium import spaces
import gymnasium as gym
import numpy as np

class CircuitEnv(gym.Env): 

    def __init__(self):
        super().__init__()

        self.num_relays = config.NUM_RELAYS

        self.relays = self._generate_relays()

        self.action_space = spaces.Discrete(self.num_relays)

        self.observation_space = spaces.Box(
            low=0, 
            high=1, 
            shape=(self.num_relays * 3 + 1,),
            dtype=np.float32
        )

        self.entry_guard = None
        self.middle_relay = None
        self.exit_relay = None
        self.circuit_pos = 0
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.entry_guard = None
        self.middle_relay = None
        self.exit_relay = None
        self.circuit_pos = 0

        obs = self._get_observation()
        info = {}

        return obs, info

    def step(self, action):
        
        reward = 0
        terminated = False

        if self.circuit_pos == 0:
            if self.relays[action]['guard_flag']:
                self.entry_guard = action
                reward = config.REWARD_VALID
                self.circuit_pos = 1
            else:
                reward = config.REWARD_INVALID
                terminated = True

        elif self.circuit_pos == 1:
            if action != self.entry_guard:
                self.middle_relay = action
                reward = config.REWARD_VALID
                self.circuit_pos = 2
            else: 
                reward = config.REWARD_INVALID
                terminated = True


        elif self.circuit_pos == 2:
            if (action != self.entry_guard and
                action != self.middle_relay and
                self.relays[action]['exit_flag']):
                self.exit_relay = action
                reward = self._calculate_reward()
                terminated = True
            else:
                reward = config.REWARD_INVALID
                terminated = True

        obs = self._get_observation()
        truncated = False
        info = {}

        return obs, reward, terminated, truncated, info

    def _generate_relays(self):
        num_guards = int(self.num_relays * config.GUARD_FRACTION)
        num_exits = int(self.num_relays * config.EXIT_FRACTION)

        guard_indices = set(np.random.choice(
            self.num_relays, 
            num_guards,
            replace=False
        ))
        
        exit_indices = set(np.random.choice(
            self.num_relays,
            num_exits,
            replace=False
        ))

        relays = []

        for i in range(self.num_relays):
            relay = {
                'id': i,
                'bandwidth': np.random.uniform(config.MIN_BANDWIDTH, config.MAX_BANDWIDTH),
                'latency': np.random.uniform(config.MIN_LATENCY, config.MAX_LATENCY),
                'guard_flag': i in guard_indices,
                'exit_flag': i in exit_indices,
            }
            relays.append(relay)

        if config.VERBOSE:
            print(f"Guards: {num_guards}, Exits: {num_exits}")

        return relays
    
    def _get_observation(self):
        obs = np.zeros(self.num_relays * 3 + 1, dtype=np.float32)

        obs[0] = self.circuit_pos / 2.0

        for i in range(self.num_relays):
            obs[i*3 + 1] = self.relays[i]['bandwidth'] / config.MAX_BANDWIDTH
            obs[i*3 + 2] = float(self.relays[i]['guard_flag'])
            obs[i*3 + 3] = float(self.relays[i]['exit_flag'])

        return obs
    
    def _calculate_reward(self):

        entry_guard = self.relays[self.entry_guard]
        middle_relay = self.relays[self.middle_relay]
        exit_relay = self.relays[self.exit_relay]

        circuit_bandwidth = min(entry_guard['bandwidth'], middle_relay['bandwidth'], exit_relay['bandwidth'])
        bandwidth_reward = (circuit_bandwidth / config.MAX_BANDWIDTH) * config.REWARD_BANDWIDTH_WEIGHT

        circuit_latency = entry_guard['latency'] + middle_relay['latency'] + exit_relay['latency']
        latency_penalty = (circuit_latency / (config.MAX_LATENCY * 3)) * config.REWARD_LATENCY_WEIGHT

        return bandwidth_reward + latency_penalty