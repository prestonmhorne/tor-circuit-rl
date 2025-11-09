# circuit_env.py

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
            shape=(self.num_relays * 4 + 1 + 8,),
            dtype=np.float32
        )

        self.persistent_guard = self._select_persistent_guard()
        self.entry_guard = None
        self.middle_relay = None
        self.exit_relay = None
        self.circuit_pos = 0

    def _select_persistent_guard(self):
        guards = [i for i in range(self.num_relays) if self.relays[i]['guard_flag']]
        
        quality_guards = [i for i in guards
                          if self.relays[i]['bandwidth'] > 200 and
                          self.relays[i]['latency'] < 150]
        
        if quality_guards:
            bandwidths = np.array([self.relays[i]['bandwidth'] for i in quality_guards])
            return np.random.choice(quality_guards, p=bandwidths/bandwidths.sum())

        bandwidths = np.array([self.relays[i]['bandwidth'] for i in guards])
        return np.random.choice(guards, p=bandwidths/bandwidths.sum())
    
    def get_action_mask(self):
        mask = np.ones(self.num_relays, dtype=bool)

        if self.circuit_pos == 0:
            mask[self.entry_guard] = False
        
        elif self.circuit_pos == 1:
            mask[self.entry_guard] = False
            mask[self.middle_relay] = False

            for i in range(self.num_relays):
                if not self.relays[i]['exit_flag']:
                    mask[i] = False
        return mask 
    
    def get_relay_info(self):
        return self.relays

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.entry_guard = self.persistent_guard
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
            if action != self.entry_guard:
                self.middle_relay = action
                self.circuit_pos = 1
                reward = 0
                terminated = False
            else: 
                reward = config.REWARD_INVALID
                terminated = True

        elif self.circuit_pos == 1:
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

            bandwidth = np.random.pareto(a=1.2) * 50 + config.MIN_BANDWIDTH
            bandwidth = min(bandwidth, config.MAX_BANDWIDTH)

            latency = np.random.exponential(scale=50) + config.MIN_LATENCY
            latency = min(latency, config.MAX_LATENCY)

            relay = {
                'id': i,
                'bandwidth': bandwidth,
                'latency': latency,
                'guard_flag': i in guard_indices,
                'exit_flag': i in exit_indices,
            }
            relays.append(relay)

        print(f"Guards: {num_guards}, Exits: {num_exits}")

        return relays
    
    def _get_observation(self):
        obs = np.zeros(self.num_relays * 4 + 1 + 8, dtype=np.float32)

        obs[0] = self.circuit_pos / 2.0

        for i in range(self.num_relays):
            obs[i*4 + 1] = self.relays[i]['bandwidth'] / config.MAX_BANDWIDTH
            obs[i*4 + 2] = self.relays[i]['latency'] / config.MAX_LATENCY
            obs[i*4 + 3] = float(self.relays[i]['guard_flag'])
            obs[i*4 + 4] = float(self.relays[i]['exit_flag'])


        base_idx = self.num_relays * 4 + 1

        obs[base_idx] = self.relays[self.entry_guard]['bandwidth'] / config.MAX_BANDWIDTH
        obs[base_idx + 1] = self.relays[self.entry_guard]['latency'] / config.MAX_LATENCY
        obs[base_idx + 2] = float(self.relays[self.entry_guard]['guard_flag'])
        obs[base_idx + 3] = float(self.relays[self.entry_guard]['exit_flag'])

        if self.middle_relay is not None:
            obs[base_idx + 4] = self.relays[self.middle_relay]['bandwidth'] / config.MAX_BANDWIDTH
            obs[base_idx + 5] = self.relays[self.middle_relay]['latency'] / config.MAX_LATENCY
            obs[base_idx + 6] = float(self.relays[self.middle_relay]['guard_flag'])
            obs[base_idx + 7] = float(self.relays[self.middle_relay]['exit_flag'])

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