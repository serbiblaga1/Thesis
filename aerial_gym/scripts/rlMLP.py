from aerial_gym import AERIAL_GYM_ROOT_DIR
import os
import math
import isaacgym
import gym
from aerial_gym.envs import *
from aerial_gym.utils import get_args, task_registry, Logger
from aerial_gym.utils.printing import (
    print_depth_maps_to_file, save_all_combined_images,print_depth_map
)

import numpy as np
import torch
import cv2
import time
import sys
from scipy.optimize import minimize
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random
from isaacgym import gymutil
import matplotlib.pyplot as plt
import time

import argparse

from torch.distributions.normal import Normal
from torch.utils.tensorboard import SummaryWriter

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.linear_model import RANSACRegressor
from aerial_gym.envs.controllers.altitude_control import AltitudeStabilizationController

sys.path.append('/home/serbiblaga/workspaces/aerial_gym_simulator/aerial_gym/rl_training/cleanrl')
#from controller_training import Agent
writer = SummaryWriter(log_dir='./runs/AltitudeTrain')

def plot_metrics(critic_losses, actor_losses, episode_rewards):
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 3, 1)
    plt.plot(critic_losses)
    plt.title("Critic Loss over Time")
    plt.xlabel("Training Steps")
    plt.ylabel("Critic Loss")

    plt.subplot(1, 3, 2)
    plt.plot(actor_losses)
    plt.title("Actor Loss over Time")
    plt.xlabel("Training Steps")
    plt.ylabel("Actor Loss")

    plt.subplot(1, 3, 3)
    plt.plot(episode_rewards)
    plt.title("Episode Rewards")
    plt.xlabel("Episodes")
    plt.ylabel("Reward")

    plt.tight_layout()
    plt.show()

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

class ActorCriticMLP(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()

        # Actor Network
        self.actor = nn.Sequential(
            layer_init(nn.Linear(state_dim, 64)),
            nn.ReLU(),
            layer_init(nn.Linear(64, action_dim), std=0.01),
            nn.Tanh()  # Output between -1 and 1
        )

        # Critic Network
        self.critic = nn.Sequential(
            layer_init(nn.Linear(state_dim + action_dim, 64)),
            nn.ReLU(),
            layer_init(nn.Linear(64, 1), std=1.0)
        )

    def get_action(self, state):
        return self.actor(state)

    def get_value(self, state, action):
        state = state.squeeze(1)
        action = action.squeeze(1)

        x = torch.cat([state, action], dim=1)
        return self.critic(x)

class DDPGAgent:
    def __init__(self, state_dim, action_dim):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.actor_critic = ActorCriticMLP(state_dim, action_dim).to(self.device)
        self.target_actor_critic = ActorCriticMLP(state_dim, action_dim).to(self.device)

        # Optimizers with modified learning rates
        self.actor_optimizer = optim.Adam(self.actor_critic.actor.parameters(), lr=0.0001)
        self.critic_optimizer = optim.Adam(self.actor_critic.critic.parameters(), lr=0.0001)

        self.memory = deque(maxlen=10000)
        self.gamma = 0.99
        self.tau = 0.001
        self.train_step = 0

        self.critic_losses = []
        self.actor_losses = []
        self.episode_rewards = []

        # Randomization factor for exploration
        self.noise_scale = 0.1
        self.noise_decay = 0.995

    def choose_action(self, state, noise_scale=None):
        if not isinstance(state, torch.Tensor):
            state = torch.tensor(state, dtype=torch.float32).to(self.device)

        action = self.actor_critic.get_action(state).detach().cpu().numpy()
        
        adjusted_action = np.zeros_like(action)
        
        noise_scale = noise_scale or self.noise_scale
        
        noise = noise_scale * np.random.randn(action.shape[0])
        
        adjusted_action[:, 0] = np.clip(action[:, 0] + noise, -1, 1)
        
        return adjusted_action


    def store_transition(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def train(self):
        if len(self.memory) < 512:
            return

        batch = random.sample(self.memory, 512)
        states, actions, rewards, next_states, dones = zip(*batch)

        # Prepare batch tensors
        states = torch.stack([torch.tensor(state, dtype=torch.float32).to(self.device) for state in states]).squeeze(1)
        actions = torch.stack([torch.tensor(action, dtype=torch.float32).to(self.device) for action in actions]).squeeze(1)
        rewards = torch.tensor(rewards, dtype=torch.float32).to(self.device).unsqueeze(1)
        next_states = torch.stack([torch.tensor(next_state, dtype=torch.float32).to(self.device) for next_state in next_states]).squeeze(1)
        dones = torch.tensor(dones, dtype=torch.float32).to(self.device).unsqueeze(1)

        # Compute target Q-values (using the target networks)
        next_actions = self.target_actor_critic.get_action(next_states)
        target_q = self.target_actor_critic.get_value(next_states, next_actions)
        target_q = rewards + (1 - dones) * self.gamma * target_q

        # Compute critic loss
        current_q = self.actor_critic.get_value(states, actions)
        critic_loss = torch.mean((current_q - target_q.detach()) ** 2)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Actor loss (maximize value function)
        actor_loss = -self.actor_critic.get_value(states, self.actor_critic.get_action(states)).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # Update target networks using soft update
        self.update_target_network(self.target_actor_critic.actor, self.actor_critic.actor)
        self.update_target_network(self.target_actor_critic.critic, self.actor_critic.critic)

        # Store loss for visualization
        self.critic_losses.append(critic_loss.item())
        self.actor_losses.append(actor_loss.item())
        self.train_step += 1
        self.noise_scale *= self.noise_decay
        torch.cuda.empty_cache()

    def update_target_network(self, target, source):
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(self.tau * param.data + (1.0 - self.tau) * target_param.data)

def integrate_depth_data(root_positions, root_quats, root_linvels, root_angvels, depth_values):
    flattened_depths = [depth.flatten() for depth in depth_values]
    
    obs = np.concatenate([
        root_positions,
        root_quats,
        root_linvels,
        root_angvels,
        *flattened_depths  # Flatten depth images
    ])
    return obs


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Set random seeds for reproducibility
    np.random.seed(42)
    torch.manual_seed(42)
    random.seed(42)

    args = get_args()  # Assuming this function is defined elsewhere for argument parsing

    # Configure environment
    env_cfg = task_registry.get_cfgs(name=args.task)
    env_cfg.env.num_envs = min(env_cfg.env.num_envs, 50)
    env_cfg.control.controller = "lee_attitude_control"
    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)

    state, additional_info = env.reset()
    state = state.to(device)

    state_dim = state.shape[1]
    action_dim = env.action_input.shape[1]

    # Initialize agent
    agent = DDPGAgent(state_dim, action_dim)

    total_timesteps = 100000
    episode_reward = 0
    episode_timesteps = 0
    episode_num = 0

    for t in range(total_timesteps):
        episode_timesteps += 1
        start_time = time.time()

        action = agent.choose_action(state)
        next_state, privileged_obs, reward, reset, extras, root_quats = env.step(action)
        next_state = next_state.to(device)
        done = reset > 0

        agent.store_transition(state, action, reward, next_state, done)
        agent.train()
        reset_penalty = -100.0 if done else 0.0  
        reward += reset_penalty
        episode_reward += reward
        state = next_state

        if done:
            end_time = time.time()
            episode_duration = end_time - start_time

            print(f"Episode: {episode_num + 1}, Reward: {episode_reward}, Timesteps: {episode_timesteps}, Duration: {episode_duration:.2f} seconds")
            agent.episode_rewards.append(episode_reward.item())

            episode_reward = 0
            episode_timesteps = 0
            episode_num += 1
            state, additional_info = env.reset()
            state = state.to(device)

    # Plot metrics
    plot_metrics(agent.critic_losses, agent.actor_losses, agent.episode_rewards)

if __name__ == "__main__":
    main()