# circuit_env.py

import config
from gymnasium import spaces
import gymnasium as gym
import numpy as np
from collections import deque

class CircuitEnv(gym.Env): 

    def __init__(self):
        super().__init__()

        self.num_relays = config.ENV_NUM_RELAYS

        self.relays = self._generate_relays()

        self.action_space = spaces.Discrete(self.num_relays)

        self.observation_space = spaces.Box(
            low=0,
            high=1,
            shape=(9,),
            dtype=np.float32
        )

        self.persistent_guard = self._select_persistent_guard()
        self.entry_guard = None
        self.middle_relay = None
        self.exit_relay = None
        self.circuit_pos = 0

        self.circuit_history = deque(maxlen=config.ANONYMITY_HISTORY_SIZE)

        self.relay_status = np.ones(self.num_relays, dtype=bool)
        self.episode_count = 0
        self.guard_rotation_counter = 0

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

        if config.ENABLE_RELAY_FAILURES:
            mask = mask & self.relay_status

        if self.circuit_pos == 0:
            mask[self.entry_guard] = False

        elif self.circuit_pos == 1:
            mask[self.entry_guard] = False
            mask[self.middle_relay] = False

            for i in range(self.num_relays):
                if not self.relays[i]['exit_flag']:
                    mask[i] = False
        return mask 
    
    def get_relays(self):
        return self.relays

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self._update_network_state()
        self.episode_count += 1

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
                reward = config.REWARD_INVALID_ACTION
                terminated = True

        elif self.circuit_pos == 1:
            if (action != self.entry_guard and
                action != self.middle_relay and
                self.relays[action]['exit_flag']):
                self.exit_relay = action
                reward = self._calculate_reward()
                terminated = True
            else:
                reward = config.REWARD_INVALID_ACTION
                terminated = True

        obs = self._get_observation()
        truncated = False
        info = {}

        return obs, reward, terminated, truncated, info

    def _generate_relays(self):
        num_guards = int(self.num_relays * config.ENV_GUARD_FRACTION)
        num_exits = int(self.num_relays * config.ENV_EXIT_FRACTION)

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

            bandwidth = np.random.pareto(a=1.2) * 50 + config.RELAY_MIN_BANDWIDTH
            bandwidth = min(bandwidth, config.RELAY_MAX_BANDWIDTH)

            latency = np.random.exponential(scale=50) + config.RELAY_MIN_LATENCY
            latency = min(latency, config.RELAY_MAX_LATENCY)

            relay = {
                'id': i,
                'bandwidth': bandwidth,
                'base_bandwidth': bandwidth, 
                'latency': latency,
                'guard_flag': i in guard_indices,
                'exit_flag': i in exit_indices,
            }

            if config.ENABLE_GEOGRAPHIC_DIVERSITY:
                relay['region'] = np.random.randint(0, config.NUM_GEOGRAPHIC_REGIONS)
                relay['operator'] = np.random.randint(0, config.NUM_OPERATORS)

            relays.append(relay)

        print(f"Guards: {num_guards}, Exits: {num_exits}")

        return relays

    def _update_network_state(self):
        """Update network state: relay failures, recovery, congestion, guard rotation"""

        if config.ENABLE_RELAY_FAILURES:
            for i in range(self.num_relays):
                if self.relay_status[i]:  
                    if np.random.random() < config.RELAY_FAILURE_RATE:
                        self.relay_status[i] = False  
                else:  
                    if np.random.random() < config.RELAY_RECOVERY_RATE:
                        self.relay_status[i] = True  

        if config.ENABLE_CONGESTION:
            for relay in self.relays:
                base_bw = relay['base_bandwidth']
                variance = config.CONGESTION_VARIANCE * base_bw
                relay['bandwidth'] = max(
                    config.RELAY_MIN_BANDWIDTH,
                    base_bw + np.random.uniform(-variance, variance)
                )

        if config.ENABLE_GUARD_ROTATION:
            self.guard_rotation_counter += 1
            if self.guard_rotation_counter >= config.GUARD_ROTATION_INTERVAL:
                self.guard_rotation_counter = 0
                old_guard = self.persistent_guard
                self.persistent_guard = self._select_persistent_guard()
                if old_guard != self.persistent_guard:
                    print(f"Guard rotation: {old_guard} → {self.persistent_guard}")

    def _get_observation(self):
        """
        Standard state representation for sequential circuit construction:
        - Position in circuit (which relay we're selecting)
        - Features of already-selected relays (context)
        - Action features (relay candidates) passed separately to network
        """
        obs = np.zeros(9, dtype=np.float32)

        obs[0] = self.circuit_pos / 2.0

        obs[1] = self.relays[self.entry_guard]['bandwidth'] / config.RELAY_MAX_BANDWIDTH
        obs[2] = self.relays[self.entry_guard]['latency'] / config.RELAY_MAX_LATENCY
        obs[3] = float(self.relays[self.entry_guard]['guard_flag'])
        obs[4] = float(self.relays[self.entry_guard]['exit_flag'])

        if self.middle_relay is not None:
            obs[5] = self.relays[self.middle_relay]['bandwidth'] / config.RELAY_MAX_BANDWIDTH
            obs[6] = self.relays[self.middle_relay]['latency'] / config.RELAY_MAX_LATENCY
            obs[7] = float(self.relays[self.middle_relay]['guard_flag'])
            obs[8] = float(self.relays[self.middle_relay]['exit_flag'])

        return obs
    
    def _calculate_diversity_bonus(self):
        """
        Calculate diversity bonus based on how unique this circuit is from recent history.
        Uses temporal weighting: recent circuits penalized more than older ones.
        Heavily penalizes exact circuit matches.
        """
        if len(self.circuit_history) == 0:
            return 1.0

        current_circuit = (self.entry_guard, self.middle_relay, self.exit_relay)

        if current_circuit in self.circuit_history:
            return 0.0 

        current_relays = set(current_circuit)

        weighted_overlaps = []
        history_list = list(self.circuit_history)

        for i, past_circuit in enumerate(history_list):
            past_relays = set(past_circuit)
            overlap = len(current_relays & past_relays) / 3.0

            recency = (i + 1) / len(history_list)
            temporal_weight = 0.1 + 0.9 * recency

            weighted_overlaps.append(overlap * temporal_weight)

        avg_weighted_overlap = np.mean(weighted_overlaps)
        diversity_bonus = 1.0 - avg_weighted_overlap

        return diversity_bonus

    def _calculate_reward(self):

        entry_guard = self.relays[self.entry_guard]
        middle_relay = self.relays[self.middle_relay]
        exit_relay = self.relays[self.exit_relay]

        circuit_bandwidth = min(entry_guard['bandwidth'], middle_relay['bandwidth'], exit_relay['bandwidth'])
        bandwidth_reward = (circuit_bandwidth / config.RELAY_MAX_BANDWIDTH) * config.REWARD_BANDWIDTH_WEIGHT

        circuit_latency = entry_guard['latency'] + middle_relay['latency'] + exit_relay['latency']
        latency_penalty = (circuit_latency / (config.RELAY_MAX_LATENCY * 3)) * config.REWARD_LATENCY_WEIGHT

        diversity_bonus = self._calculate_diversity_bonus()
        diversity_reward = diversity_bonus * config.REWARD_DIVERSITY_WEIGHT

        self.circuit_history.append((self.entry_guard, self.middle_relay, self.exit_relay))

        return bandwidth_reward + latency_penalty + diversity_reward