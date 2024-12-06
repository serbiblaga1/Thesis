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

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
import random


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim, lstm_hidden_dim=64, seq_length=5):
        super().__init__()
        
        # LSTM Layer
        self.lstm = nn.LSTM(input_size=state_dim, hidden_size=lstm_hidden_dim, batch_first=True)

        # Critic Network
        self.critic = nn.Sequential(
            layer_init(nn.Linear(lstm_hidden_dim + action_dim, 64)),
            nn.ReLU(),
            layer_init(nn.Linear(64, 1), std=1.0)
        )
        
        # Actor Network
        self.actor_mean = nn.Sequential(
            layer_init(nn.Linear(lstm_hidden_dim, 64)),
            nn.ReLU(),
            layer_init(nn.Linear(64, action_dim), std=0.01),
            nn.Tanh()
        )

    def forward(self, state_history):
      #  print(f"Input to LSTM shape: {state_history.shape}") 
        lstm_output, _ = self.lstm(state_history) 
        lstm_output = lstm_output[:, -1, :] 
        return lstm_output

    def get_action(self, state_history):
        lstm_output = self(state_history)
        return self.actor_mean(lstm_output)

    def get_value(self, state_history, action):
        lstm_output = self(state_history) 
        action = action.to(lstm_output.device)
        lstm_output = lstm_output.squeeze(1) 
        action = action.squeeze(1)
       # print("Shpae lstm_output ", lstm_output.shape)
       # print("Action shape ", action.shape)
        x = torch.cat([lstm_output, action], dim=1) 
        return self.critic(x)


class DDPGAgent:
    def __init__(self, state_dim, action_dim, lstm_seq_length=5):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.actor_critic = ActorCritic(state_dim, action_dim, seq_length=lstm_seq_length).to(self.device)
        self.target_actor_critic = ActorCritic(state_dim, action_dim, seq_length=lstm_seq_length).to(self.device)
        self.actor = self.actor_critic.actor_mean
        self.critic = self.actor_critic.critic

        self.actor_optimizer = optim.Adam(self.actor_critic.actor_mean.parameters(), lr=0.0001)
        self.critic_optimizer = optim.Adam(self.actor_critic.critic.parameters(), lr=0.0001)

        self.memory = deque(maxlen=10000)
        self.gamma = 0.95
        self.tau = 0.001   
        self.train_step = 0 

        self.critic_losses = []
        self.actor_losses = []
        self.episode_rewards = []
        self.state_history = [] 

    def choose_action(self, state, noise_scale=0.1):
        if not isinstance(state, torch.Tensor):
            state = torch.tensor(state, dtype=torch.float32)

        state = state.to(self.actor_critic.actor_mean[0].weight.device)
        self.state_history.append(state)

        if len(self.state_history) > 5:
            self.state_history.pop(0)

        state_history_tensor = torch.stack(self.state_history, dim=0).to(self.device)
        state_history_tensor = state_history_tensor.transpose(0, 1)

        action = self.actor_critic.get_action(state_history_tensor).detach().cpu().numpy()

        noise = noise_scale * np.random.randn() 
        adjusted_action = np.zeros_like(action)  

        adjusted_action[:, 0] = np.clip(action[:, 0] + noise, -1, 1) 
        
        return adjusted_action


    def store_transition(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def train_old(self):
        if len(self.memory) < 512:
            return
        batch = random.sample(self.memory, 512)
        states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.stack([state.clone().detach().to(self.device) for state in states])
        actions = torch.stack([torch.tensor(action, dtype=torch.float32, device=self.device).clone().detach() for action in actions])
        rewards = torch.stack([r.clone().detach().to(self.device) for r in rewards]).unsqueeze(1)
        next_states = torch.stack([next_state.clone().detach().to(self.device) for next_state in next_states])

        dones = torch.tensor(dones, dtype=torch.float32, device=self.device).unsqueeze(1) 
        self.target_actor_critic.to(self.device)
        dones = dones.squeeze(1)  

        states_history = torch.stack([states[i-4:i+1] for i in range(4, len(states))]).to(self.device)
        next_states_history = torch.stack([next_states[i-4:i+1] for i in range(4, len(next_states))]).to(self.device) 

        states_history = states_history.view(-1, 5, 8) 
        next_states_history = next_states_history.view(-1, 5, 8)
        actions = actions[4:]
        next_actions = self.target_actor_critic.get_action(next_states_history)
        target_q = self.target_actor_critic.get_value(next_states_history, next_actions)

        target_q = rewards + (1 - dones) * self.gamma * target_q
        current_q = self.actor_critic.get_value(states_history, actions)
        critic_loss = torch.mean((current_q - target_q.detach()) ** 2)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        actor_loss = -self.actor_critic.get_value(states_history, self.actor_critic.get_action(states_history)).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        self.update_target_network(self.target_actor_critic.actor_mean, self.actor_critic.actor_mean)
        self.update_target_network(self.target_actor_critic.critic, self.actor_critic.critic)

        self.critic_losses.append(critic_loss.item())
        self.actor_losses.append(actor_loss.item())

        # Log losses to TensorBoard (assuming writer is defined elsewhere)
        # writer.add_scalar('Loss/Critic', critic_loss.item(), self.train_step)
        # writer.add_scalar('Loss/Actor', actor_loss.item(), self.train_step)

        # Optional: Log weight and activation distributions
        # for name, param in self.actor_critic.named_parameters():
        #     writer.add_histogram(f'Weights/{name}', param, self.train_step)
        #     if param.grad is not None:
        #         writer.add_histogram(f'Gradients/{name}', param.grad, self.train_step)
        self.train_step += 1
        torch.cuda.empty_cache()

    def train(self):
        if len(self.memory) < 512:
            return

        sample_size = min(len(self.memory) - 4, 512)
        indices = np.random.choice(range(4, len(self.memory)), size=sample_size, replace=False)
        
        states, actions, rewards, next_states, dones = zip(*[self.memory[i] for i in indices])
        
        states = torch.stack([state.clone().detach().to(self.device) for state in states])
        actions = torch.stack([torch.tensor(action, dtype=torch.float32, device=self.device).clone().detach() for action in actions])
        rewards = torch.tensor(rewards, dtype=torch.float32, device=self.device).unsqueeze(1)
        next_states = torch.stack([next_state.clone().detach().to(self.device) for next_state in next_states])
        dones = torch.tensor(dones, dtype=torch.float32, device=self.device).unsqueeze(1)

        states_history = []
        next_states_history = []

        # Use only indices that allow for history collection
        for i in indices:
            if i >= 4:  # Ensure we can access the history
                state_sequence = states[i - 4:i + 1]  
                next_state_sequence = next_states[i - 4:i + 1]  
                # Log shapes for debugging
                print(f"Index: {i}, State sequence shape: {state_sequence.shape}, Next state sequence shape: {next_state_sequence.shape}")

                if state_sequence.shape == (5, states.shape[1]) and next_state_sequence.shape == (5, next_states.shape[1]):
                    states_history.append(state_sequence)
                    next_states_history.append(next_state_sequence)

        # Check if we have collected valid histories
        if states_history:
            states_history = torch.stack(states_history).to(self.device)  # Convert to tensor if not empty
        else:
            print("No valid state histories collected for states.")
            states_history = None

        if next_states_history:
            next_states_history = torch.stack(next_states_history).to(self.device)  # Convert to tensor if not empty
        else:
            print("No valid state histories collected for next states.")
            next_states_history = None

        if states_history is None or next_states_history is None:
            print("No valid state histories collected. Check state storage consistency.")
            return 

        next_actions = self.target_actor_critic.get_action(next_states_history)
        target_q = self.target_actor_critic.get_value(next_states_history, next_actions)
        target_q = rewards + (1 - dones) * self.gamma * target_q
        current_q = self.actor_critic.get_value(states_history, actions)
        critic_loss = torch.mean((current_q - target_q.detach()) ** 2)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        actor_loss = -self.actor_critic.get_value(states_history, self.actor_critic.get_action(states_history)).mean()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        self.update_target_network(self.target_actor_critic.actor_mean, self.actor_critic.actor_mean)
        self.update_target_network(self.target_actor_critic.critic, self.actor_critic.critic)

        self.critic_losses.append(critic_loss.item())
        self.actor_losses.append(actor_loss.item())
        print(critic_loss)
        torch.cuda.empty_cache()

        self.train_step += 1


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

    total_timesteps = 3000
    batch_size = 64 
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
    plot_metrics(agent.critic_losses, agent.actor_losses, agent.episode_rewards)


if __name__ == "__main__":
    main()
