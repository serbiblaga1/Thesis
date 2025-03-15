import os
import random
import time
import cv2
import matplotlib.pyplot as plt

import gym
import isaacgym  # noqa
from isaacgym import gymutil
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.normal import Normal
from torch.utils.tensorboard import SummaryWriter
from stable_baselines3 import PPO

from aerial_gym.envs import *
from aerial_gym.utils import task_registry


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class Agent(nn.Module):
    def __init__(self, envs):
        super().__init__()
        self.critic = nn.Sequential(
            layer_init(nn.Linear(np.array(envs.num_obs).prod(), 256)),
            nn.Tanh(),
            layer_init(nn.Linear(256, 256)),
            nn.Tanh(),
            layer_init(nn.Linear(256, 1), std=1.0),
        )
        self.actor_mean = nn.Sequential(
            layer_init(nn.Linear(np.array(envs.num_obs).prod(), 256)),
            nn.Tanh(),
            layer_init(nn.Linear(256, 256)),
            nn.Tanh(),
            layer_init(nn.Linear(256, np.prod(envs.num_actions_rl)), std=0.01),
        )
        self.actor_logstd = nn.Parameter(torch.zeros(1, np.prod(envs.num_actions_rl)))

    def get_value(self, x):
        return self.critic(x)
    

    def get_action_and_value(self, x, action=None):
        action_mean = self.actor_mean(x)
        action_mean = torch.tanh(action_mean)
        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        probs = Normal(action_mean, action_std)

        if action is None:
            action = probs.sample()
        #action = torch.clamp(action, -0.7, 0.7)
        return action, probs.log_prob(action).sum(1), probs.entropy().sum(1), self.critic(x)

def evaluate_model(model, env, num_test_episodes=50, render=False):
    """
    Evaluates the trained PPO model on the drone environment.
    
    Parameters:
        model (PPO): Trained PPO model.
        env (gym.Env): The drone environment.
        num_test_episodes (int): Number of episodes to evaluate.
        render (bool): Whether to render the environment.

    Returns:
        DataFrame containing evaluation results.
    """
    success_count = 0
    collision_count = 0
    episode_rewards = []
    episode_lengths = []

    print("\n📊 Running PPO Evaluation...\n")
    
    for episode in range(num_test_episodes):
        print(f"🎯 Starting Test Episode {episode + 1}/{num_test_episodes}")
        obs = env.reset()
        done = False
        total_reward = 0
        episode_length = 0

        while not done:
            with torch.no_grad():
                action, _ = model.predict(obs, deterministic=True)  # Use deterministic inference

            obs, reward, done, info = env.step(action)
            total_reward += reward
            episode_length += 1

            if info.get("collision", False):  # Track collisions
                collision_count += 1

            if render:
                env.render()

        episode_rewards.append(total_reward)
        episode_lengths.append(episode_length)

        if info.get("success", False):  # Check if the drone reached a goal
            success_count += 1

    env.close()

    # Compute evaluation metrics
    success_rate = success_count / num_test_episodes
    collision_rate = collision_count / num_test_episodes
    avg_return = np.mean(episode_rewards)
    avg_length = np.mean(episode_lengths)

    # Store results in a DataFrame
    results_df = pd.DataFrame({
        "Episode": np.arange(1, num_test_episodes + 1),
        "Reward": episode_rewards,
        "Length": episode_lengths,
        "Success": [1 if i < success_count else 0 for i in range(num_test_episodes)],
        "Collisions": [1 if i < collision_count else 0 for i in range(num_test_episodes)]
    })

    print("\n==== ✅ Evaluation Complete! ====")
    print(f"📈 Average Return: {avg_return:.2f}")
    print(f"📏 Average Episode Length: {avg_length}")
    print(f"✅ Success Rate: {success_rate * 100:.2f}%")
    print(f"💥 Collision Rate: {collision_rate * 100:.2f}%")

    return results_df


def get_args():
    custom_parameters = [
        {"name": "--task", "type": str, "default": "quad", "help": "Task for RL training/testing."},
        {"name": "--experiment_name", "type": str, "default": os.path.basename(__file__).rstrip(".py"), "help": "Experiment name."},
        {"name": "--checkpoint", "type": str, "default": None, "help": "Path to trained model checkpoint."},        
        {"name": "--test", "action": "store_true", "default": False, "help": "Run trained model in evaluation mode."},
        {"name": "--num_test_episodes", "type": int, "default": 50, "help": "Number of episodes to test."},
        {"name": "--render", "action": "store_true", "default": False, "help": "Render the environment during testing."},
        {"name": "--num_envs", "type": int, "default": 100, "help": "Number of environments."},
        {"name": "--seed", "type": int, "default": 1, "help": "Random seed."},
    ]

    args = gymutil.parse_arguments(description="RL Policy", custom_parameters=custom_parameters)
    return args


if __name__ == "__main__":
    args = get_args()
    
    print(f"Using device: {args.rl_device}")

    # Initialize environment
    envs, env_cfg = task_registry.make_env(name="quad_with_obstacles", args=args)

    # TESTING MODE: Run Evaluation Instead of Training
    if args.test:
        if args.checkpoint is None:
            raise ValueError("⚠️ No checkpoint provided for testing. Use --checkpoint <model_path>.")

        print(f"\n📥 Loading trained model from {args.checkpoint}...")
        model = PPO.load(args.checkpoint)

        results_df = evaluate_model(model, envs, num_test_episodes=args.num_test_episodes, render=args.render)

        # Save results to CSV
        results_df.to_csv("ppo_test_results.csv", index=False)
        print("\n✅ Results saved to 'ppo_test_results.csv'")

        exit()  # Exit after testing

    # TRAINING MODE: Proceed as usual if not testing
    agent = Agent(envs).to(args.rl_device)
    optimizer = torch.optim.Adam(agent.parameters(), lr=2e-4, eps=1e-5)

    if args.checkpoint is not None:
        print("Loading checkpoint...")
        checkpoint = torch.load(args.checkpoint)
        agent.load_state_dict(checkpoint)
        print("✅ Loaded checkpoint")

    # Start Training
    print("\n🚀 Starting PPO Training...")
    for update in range(1, 6000 + 1):
        print(f"🔄 Update {update}")
        obs = envs.reset()
        for step in range(16):
            with torch.no_grad():
                action, logprob, _, value = agent.get_action_and_value(obs)
            
            obs, rewards, done, info = envs.step(action)

    print("\n✅ Training Complete!")
