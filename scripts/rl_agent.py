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

class RecordEpisodeStatisticsTorch(AerialRobotWithObstacles):
    def __init__(self, env, device):
        super().__init__(env)
        self.num_envs = getattr(env, "num_envs", 1)
        self.device = device
        self.episode_returns = None
        self.episode_lengths = None

    def reset(self, **kwargs):
        observations = super().reset(**kwargs)
        self.episode_returns = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
        self.episode_lengths = torch.zeros(self.num_envs, dtype=torch.int32, device=self.device)
        self.returned_episode_returns = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
        self.returned_episode_lengths = torch.zeros(self.num_envs, dtype=torch.int32, device=self.device)
        return observations

    def step(self, action):
        observations, privileged_observations, rewards, dones, infos = super().step(action)
        self.episode_returns += rewards
        self.episode_lengths += 1
        self.returned_episode_returns[:] = self.episode_returns
        self.returned_episode_lengths[:] = self.episode_lengths
        self.episode_returns *= 1 - dones
        self.episode_lengths *= 1 - dones
        infos["r"] = self.returned_episode_returns
        infos["l"] = self.returned_episode_lengths
        return (
            observations,
            rewards,
            dones,
            infos,
        )


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        
        # Critic Network
        self.critic = nn.Sequential(
            layer_init(nn.Linear(state_dim + action_dim, 64)),
            nn.ReLU(),
            layer_init(nn.Linear(64, 1), std=1.0)
        )
        
        # Actor Network
        self.actor_mean = nn.Sequential(
            layer_init(nn.Linear(state_dim, 64)),
            nn.ReLU(),
            layer_init(nn.Linear(64, action_dim), std=0.01),
            nn.Tanh()
        )

    def get_action(self, state):
        return self.actor_mean(state)

    def get_value(self, state, action):
        action = action.to(state.device)
        state = state.squeeze(1)  
        action = action.squeeze(1)

        x = torch.cat([state, action], dim=1)  
        return self.critic(x)



class DDPGAgent:
    def __init__(self, state_dim, action_dim):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.actor_critic = ActorCritic(state_dim, action_dim).to(self.device)
        self.target_actor_critic = ActorCritic(state_dim, action_dim).to(self.device)
        self.actor = self.actor_critic.actor_mean
        self.critic = self.actor_critic.critic
       
        self.actor_optimizer = optim.Adam(self.actor_critic.actor_mean.parameters(), lr=0.001)
        self.critic_optimizer = optim.Adam(self.actor_critic.critic.parameters(), lr=0.001)

        self.memory = deque(maxlen=10000)
        self.gamma = 0.99
        self.tau = 0.001   

        self.train_step = 0 

        self.critic_losses = []
        self.actor_losses = []
        self.episode_rewards = []

        #### LSTM, history of states, give velocity inputs (derivatives), maybe a P + D gain would work.
        # reduce dt for the drag.

    def choose_action(self, state):
        if not isinstance(state, torch.Tensor):
            state = torch.tensor(state, dtype=torch.float32)
        
        state = state.to(self.actor_critic.actor_mean[0].weight.device)
        
        action = self.actor_critic.get_action(state).detach().cpu().numpy()

        adjusted_action = np.zeros_like(action)
        
        adjusted_action[:, 0] = action[:, 0] 
        return adjusted_action


    def store_transition(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def train(self):
        if len(self.memory) < 64:
            return
        batch = random.sample(self.memory, 64)
        states, actions, rewards, next_states, dones = zip(*batch)
        
        states = torch.stack([torch.tensor(state, dtype=torch.float32, device=self.device) for state in states])
        actions = torch.stack([torch.tensor(action, dtype=torch.float32, device=self.device) for action in actions])
        rewards = torch.stack([torch.tensor(r, dtype=torch.float32, device=self.device) for r in rewards]).unsqueeze(1)
        next_states = torch.stack([torch.tensor(next_state, dtype=torch.float32, device=self.device) for next_state in next_states])
        dones = torch.tensor(dones, dtype=torch.float32, device=self.device).unsqueeze(1) 
        self.target_actor_critic.to(self.device)
        dones = dones.squeeze(1)  
        next_actions = self.target_actor_critic.get_action(next_states)
        target_q = self.target_actor_critic.get_value(next_states, next_actions)

        target_q = rewards + (1 - dones) * self.gamma * target_q
        current_q = self.actor_critic.get_value(states, actions)
        critic_loss = torch.mean((current_q - target_q.detach()) ** 2)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        actor_loss = -self.actor_critic.get_value(states, self.actor_critic.get_action(states)).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        self.update_target_network(self.target_actor_critic.actor_mean, self.actor_critic.actor_mean)
        self.update_target_network(self.target_actor_critic.critic, self.actor_critic.critic)

        self.critic_losses.append(critic_loss.item())
        self.actor_losses.append(actor_loss.item())

            # Log losses to TensorBoard
        writer.add_scalar('Loss/Critic', critic_loss.item(), self.train_step)
        writer.add_scalar('Loss/Actor', actor_loss.item(), self.train_step)

        # Optional: Log weight and activation distributions
        for name, param in self.actor_critic.named_parameters():
            writer.add_histogram(f'Weights/{name}', param, self.train_step)
            if param.grad is not None:
                writer.add_histogram(f'Gradients/{name}', param.grad, self.train_step)

        # Increase train step counter
        self.train_step += 1

    def update_target_network(self, target, source):
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(self.tau * param.data + (1.0 - self.tau) * target_param.data)


def integrate_depth_data(root_positions, root_quats, root_linvels, root_angvels, depth_values):
    """
    Combine positional, orientation, velocity, and depth sensor data into a single observation array.
    """
    flattened_depths = [depth.flatten() for depth in depth_values]
    
    obs = np.concatenate([
        root_positions,
        root_quats,
        root_linvels,
        root_angvels,
        *flattened_depths  
    ])
    return obs


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    args = get_args()

    env_cfg = task_registry.get_cfgs(name=args.task)
    env_cfg.env.num_envs = min(env_cfg.env.num_envs, 50)
    env_cfg.control.controller = "lee_attitude_control"
    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
   
    state, additional_info = env.reset()
    state = state.to(device)
    
    state_dim = state.shape[1] 
    action_dim = env.action_input.shape[1]
  
    agent = DDPGAgent(state_dim, action_dim)
   
    env.seed(42)
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)

    total_timesteps = 1000
    batch_size = 64 
    episode_reward = 0
    episode_timesteps = 0
    episode_num = 0

    for t in range(total_timesteps):
        episode_timesteps += 1
        
        action = agent.choose_action(state)
       
        next_state, privileged_obs, reward, reset, extras, root_quats = env.step(action) 
        next_state = next_state.to(device)
        done = reset > 0
        agent.store_transition(state, action, reward, next_state, done)
        
        state = next_state
        episode_reward += reward
        
        agent.train()

        if done:
            print(f"Episode: {episode_num + 1}, Reward: {episode_reward}, Timesteps: {episode_timesteps}")
            agent.episode_rewards.append(episode_reward.item())

            state, additional_info = env.reset()
            state = state.to(device)

            episode_reward = 0
            episode_timesteps = 0
            episode_num += 1

            if (episode_num + 1) % 100 == 0:
                torch.save(agent.actor.state_dict(), f"actor_{episode_num + 1}.pth")
                torch.save(agent.critic.state_dict(), f"critic_{episode_num + 1}.pth")
    plot_metrics(agent.critic_losses, agent.actor_losses, agent.episode_rewards)

if __name__ == "__main__":
    main()

