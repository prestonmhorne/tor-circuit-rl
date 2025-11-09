# main.py

import numpy as np
import torch
import argparse
import config
from circuit_env import CircuitEnv
from dqn_agent import DQNAgent
from baseline_agent import BaselineAgent

def evaluate_agent(agent, env, num_episodes=100, agent_name="Agent"):
    """Evaluate an agent and return detailed metrics"""

    if hasattr(agent, 'epsilon'):
        original_epsilon = agent.epsilon
        agent.epsilon = config.EVAL_EPSILON

    eval_rewards = []
    successful_circuits = []
    failed_circuits = 0
    bandwidths = []
    latencies = []

    all_circuits = []
    middle_relays_used = []
    exit_relays_used = []

    for _ in range(num_episodes):
        obs, _ = env.reset()
        terminated = False
        episode_reward = 0

        while not terminated:
            action_mask = env.get_action_mask()
            relays = env.get_relays()

            if isinstance(agent, BaselineAgent):
                action = agent.select_action(action_mask, relays)
            else:
                action = agent.select_action(obs, action_mask, relays)

            next_obs, reward, terminated, _, _ = env.step(action)
            obs = next_obs
            episode_reward += reward

        eval_rewards.append(episode_reward)

        if env.exit_relay is not None and episode_reward != config.REWARD_INVALID_ACTION:
            circuit = {
                'entry': env.entry_guard,
                'middle': env.middle_relay,
                'exit': env.exit_relay,
                'reward': episode_reward
            }
            successful_circuits.append(circuit)
            all_circuits.append((env.entry_guard, env.middle_relay, env.exit_relay))
            middle_relays_used.append(env.middle_relay)
            exit_relays_used.append(env.exit_relay)

            entry_guard = env.relays[env.entry_guard]
            middle_relay = env.relays[env.middle_relay]
            exit_relay = env.relays[env.exit_relay]

            circuit_bandwidth = min(entry_guard['bandwidth'], middle_relay['bandwidth'], exit_relay['bandwidth'])
            circuit_latency = entry_guard['latency'] + middle_relay['latency'] + exit_relay['latency']

            bandwidths.append(circuit_bandwidth)
            latencies.append(circuit_latency)
        else:
            failed_circuits += 1

    if hasattr(agent, 'epsilon'):
        agent.epsilon = original_epsilon

    success_rate = len(successful_circuits) / num_episodes * 100

    unique_circuits = len(set(all_circuits))
    unique_middle = len(set(middle_relays_used))
    unique_exit = len(set(exit_relays_used))
    circuit_diversity = unique_circuits / len(all_circuits) if all_circuits else 0

    return {
        'agent_name': agent_name,
        'rewards': eval_rewards,
        'mean_reward': np.mean(eval_rewards),
        'std_reward': np.std(eval_rewards),
        'success_rate': success_rate,
        'failed_circuits': failed_circuits,
        'mean_bandwidth': np.mean(bandwidths) if bandwidths else 0,
        'std_bandwidth': np.std(bandwidths) if bandwidths else 0,
        'mean_latency': np.mean(latencies) if latencies else 0,
        'std_latency': np.std(latencies) if latencies else 0,
        'best_circuit': successful_circuits[np.argmax([c['reward'] for c in successful_circuits])] if successful_circuits else None,
        'unique_circuits': unique_circuits,
        'unique_middle_relays': unique_middle,
        'unique_exit_relays': unique_exit,
        'circuit_diversity': circuit_diversity
    }
    
def print_comparison(dqn_metrics, baseline_metrics):
    print("\n" + "="*80)
    print("COMPARISON: DQN vs Baseline")
    print("="*80)

    print(f"\n{'Metric':<30} {'DQN':>20} {'Baseline':>20}")
    print("-"*80)

    print(f"{'Success Rate (%)':<30} {dqn_metrics['success_rate']:>19.1f}% {baseline_metrics['success_rate']:>19.1f}%")
    print(f"{'Failed Circuits':<30} {dqn_metrics['failed_circuits']:>20d} {baseline_metrics['failed_circuits']:>20d}")

    print(f"\n{'Mean Reward':<30} {dqn_metrics['mean_reward']:>20.2f} {baseline_metrics['mean_reward']:>20.2f}")
    print(f"{'Std Reward':<30} {dqn_metrics['std_reward']:>20.2f} {baseline_metrics['std_reward']:>20.2f}")

    print(f"\n{'Mean Bandwidth (MB/s)':<30} {dqn_metrics['mean_bandwidth']:>20.2f} {baseline_metrics['mean_bandwidth']:>20.2f}")
    print(f"{'Std Bandwidth (MB/s)':<30} {dqn_metrics['std_bandwidth']:>20.2f} {baseline_metrics['std_bandwidth']:>20.2f}")

    print(f"\n{'Mean Latency (ms)':<30} {dqn_metrics['mean_latency']:>20.2f} {baseline_metrics['mean_latency']:>20.2f}")
    print(f"{'Std Latency (ms)':<30} {dqn_metrics['std_latency']:>20.2f} {baseline_metrics['std_latency']:>20.2f}")

    print(f"\n{'Anonymity Metrics':<30}")
    print(f"{'Unique Circuits Used':<30} {dqn_metrics['unique_circuits']:>20d} {baseline_metrics['unique_circuits']:>20d}")
    print(f"{'Unique Middle Relays':<30} {dqn_metrics['unique_middle_relays']:>20d} {baseline_metrics['unique_middle_relays']:>20d}")
    print(f"{'Unique Exit Relays':<30} {dqn_metrics['unique_exit_relays']:>20d} {baseline_metrics['unique_exit_relays']:>20d}")
    print(f"{'Circuit Diversity (%)':<30} {dqn_metrics['circuit_diversity']*100:>19.1f}% {baseline_metrics['circuit_diversity']*100:>19.1f}%")

    if baseline_metrics['mean_reward'] != 0:
          reward_improvement = ((dqn_metrics['mean_reward'] - baseline_metrics['mean_reward']) / abs(baseline_metrics['mean_reward']) * 100)
    else:
        reward_improvement = 0

    if baseline_metrics['mean_bandwidth'] != 0:
        bandwidth_improvement = ((dqn_metrics['mean_bandwidth'] - baseline_metrics['mean_bandwidth']) / baseline_metrics['mean_bandwidth'] * 100)
    else:
        bandwidth_improvement = 0

    if baseline_metrics['mean_latency'] != 0:
        latency_improvement = ((baseline_metrics['mean_latency'] - dqn_metrics['mean_latency']) / baseline_metrics['mean_latency'] * 100)
    else:
        latency_improvement = 0

    print(f"\n{'DQN Improvement over Baseline:':<30}")
    print(f"{'  Reward':<30} {reward_improvement:>19.1f}%")
    print(f"{'  Bandwidth':<30} {bandwidth_improvement:>19.1f}%")
    print(f"{'  Latency Reduction':<30} {latency_improvement:>19.1f}%")

    print("\n" + "="*80)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--episodes', type=int, default=config.TRAIN_NUM_EPISODES)
    parser.add_argument('--seed', type=int, default=config.RANDOM_SEED)
    parser.add_argument('--eval_episodes', type=int, default=100)
    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    env = CircuitEnv()

    dqn_agent = DQNAgent(
        action_dim=config.ENV_NUM_RELAYS,
        state_dim=env.observation_space.shape[0]
    )

    baseline_agent = BaselineAgent()

    print("="*80)
    print("TRAINING DQN AGENT")
    print("="*80)
    print(f"Episodes: {args.episodes}")
    print(f"Relays: {config.ENV_NUM_RELAYS}")

    for episode in range(args.episodes):
        obs, _ = env.reset()
        terminated = False
        episode_reward = 0

        while not terminated:
            action_mask = env.get_action_mask()
            relays = env.get_relays()
            action = dqn_agent.select_action(obs, action_mask, relays)

            next_obs, reward, terminated, _, _ = env.step(action)

            next_action_mask = env.get_action_mask()
            next_relays = env.get_relays()
            dqn_agent.store_transition(obs, action, reward, next_obs, terminated, relays, next_relays, next_action_mask)
            dqn_agent.train_step()

            obs = next_obs
            episode_reward += reward

        dqn_agent.decay_epsilon()

        if episode % config.TRAIN_LOG_FREQ == 0:
            print(f"Episode {episode:5d}/{args.episodes} | Reward: {episode_reward:7.2f} | Epsilon: {dqn_agent.epsilon:.3f}")

    print("Training Complete!")

    dqn_metrics = evaluate_agent(dqn_agent, env, args.eval_episodes, "DQN")
    baseline_metrics = evaluate_agent(baseline_agent, env, args.eval_episodes, "Baseline")

    print_comparison(dqn_metrics, baseline_metrics)

if __name__ == "__main__":
    main()