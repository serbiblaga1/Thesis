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

def plot_metrics_original(critic_losses, actor_losses, episode_rewards):
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
            nn.Tanh() 
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
    def __init__(self, state_dim, action_dim, imitation_data=None):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.actor_critic = ActorCriticMLP(state_dim, action_dim).to(self.device)
        self.target_actor_critic = ActorCriticMLP(state_dim, action_dim).to(self.device)
        self.critic_losses = []
        self.actor_losses = []
        self.episode_rewards = []
        self.imitation_data = imitation_data
        self.imitation_weight = 0.5  

        self.actor_optimizer = optim.Adam(self.actor_critic.actor.parameters(), lr=1e-5)
        self.critic_optimizer = optim.Adam(self.actor_critic.critic.parameters(), lr=1e-5)

        self.memory = deque(maxlen=10000)
        self.gamma = 0.99
        self.tau = 0.005
        self.train_step = 0
        self.noise_scale = 0.1
        self.noise_decay = 0.995

        self.target_q_values = []
        self.current_q_values = []
        self.rewards = []
        self.actor_grad_norms = []
        self.critic_grad_norms = []
        self.episode_rewards = []

        self.patience = 1000 
        self.best_critic_loss = float('inf')
        self.epochs_without_improvement = 0
        self.stop_training = False  

    def plot_metrics(self):
        critic_losses_cpu = [loss.item() if isinstance(loss, torch.Tensor) else loss for loss in self.critic_losses]
        actor_losses_cpu = [loss.item() if isinstance(loss, torch.Tensor) else loss for loss in self.actor_losses]
        target_q_values_cpu = [q.item() if isinstance(q, torch.Tensor) else q for q in self.target_q_values]
        current_q_values_cpu = [q.item() if isinstance(q, torch.Tensor) else q for q in self.current_q_values]
        actor_grad_norms_cpu = [norm.item() if isinstance(norm, torch.Tensor) else norm for norm in self.actor_grad_norms]
        critic_grad_norms_cpu = [norm.item() if isinstance(norm, torch.Tensor) else norm for norm in self.critic_grad_norms]

        episode_rewards_cpu = [reward.item() if isinstance(reward, torch.Tensor) else reward for reward in self.episode_rewards]

        plt.figure(figsize=(12, 8))

        # Plot Critic Loss
        plt.subplot(2, 3, 1)
        plt.plot(critic_losses_cpu, label='Critic Loss')
        plt.xlabel('Training Steps')
        plt.ylabel('Loss')
        plt.title('Critic Loss')
        plt.legend()

        # Plot Actor Loss
        plt.subplot(2, 3, 2)
        plt.plot(actor_losses_cpu, label='Actor Loss', color='orange')
        plt.xlabel('Training Steps')
        plt.ylabel('Loss')
        plt.title('Actor Loss')
        plt.legend()

        # Plot Target vs Current Q-value
        plt.subplot(2, 3, 3)
        plt.plot(target_q_values_cpu, label='Target Q-Value', color='green')
        plt.plot(current_q_values_cpu, label='Current Q-Value', color='red')
        plt.xlabel('Training Steps')
        plt.ylabel('Q-Value')
        plt.title('Target vs Current Q-Value')
        plt.legend()

        # Plot Episode Rewards (Total rewards per episode)
        plt.subplot(2, 3, 4)
        plt.plot(episode_rewards_cpu, label='Epitsode Rewards', color='blue')
        plt.xlabel('Episode')
        plt.ylabel('Reward')
        plt.title('Rewards per Episode')
        plt.legend()

        # Plot Actor Gradient Norm
        plt.subplot(2, 3, 5)
        plt.plot(actor_grad_norms_cpu, label='Actor Gradient Norm', color='purple')
        plt.xlabel('Training Steps')
        plt.ylabel('Gradient Norm')
        plt.title('Actor Gradient Norm')
        plt.legend()

        # Plot Critic Gradient Norm
        plt.subplot(2, 3, 6)
        plt.plot(critic_grad_norms_cpu, label='Critic Gradient Norm', color='red')
        plt.xlabel('Training Steps')
        plt.ylabel('Gradient Norm')
        plt.title('Critic Gradient Norm')
        plt.legend()

        plt.tight_layout()
        plt.show()


    def pretrain_imitation(self, num_epochs=500, batch_size=64, scaling_factor=10000, initial_lr=1e-5, lr_decay_epoch=100):        
        criterion = nn.MSELoss()
        imitation_data = torch.utils.data.DataLoader(self.imitation_data, batch_size=batch_size, shuffle=True)
        
        self.actor_optimizer = torch.optim.Adam(self.actor_critic.parameters(), lr=initial_lr)
        scheduler = torch.optim.lr_scheduler.StepLR(self.actor_optimizer, step_size=lr_decay_epoch, gamma=0.2) 
        
        for epoch in range(num_epochs):
            epoch_loss = 0  
            
            for states, actions in imitation_data:
                states = states.to(self.device).float()
                actions = actions.to(self.device).float()
                target_thrust = actions[:, :1]  

                try:
                    predicted_actions = self.actor_critic.get_action(states)
                    predicted_thrust = predicted_actions[:, :1]
                except RuntimeError as e:
                    print(f"Error during forward pass: {e}")
                    continue  

                imitation_loss = criterion(predicted_thrust, target_thrust)
                imitation_loss *= scaling_factor 

                epoch_loss += imitation_loss.item()  

                self.actor_optimizer.zero_grad()
                imitation_loss.backward()

                actor_grads_before = [p.grad.norm().item() for p in self.actor_critic.parameters() if p.grad is not None]
                print(f"Epoch {epoch+1}, Grads Before Clipping: {actor_grads_before[:5]}") 

                torch.nn.utils.clip_grad_norm_(self.actor_critic.parameters(), max_norm=5.0)

                actor_grads_after = [p.grad.norm().item() for p in self.actor_critic.parameters() if p.grad is not None]
                print(f"Epoch {epoch+1}, Grads After Clipping: {actor_grads_after[:5]}") 

                self.actor_optimizer.step()

                if epoch % 10 == 0 and len(imitation_data) % 10 == 0:
                    print(f"Epoch {epoch+1}, Predicted Thrust (First 5): {predicted_thrust[:5]}")

            scheduler.step()

            print(f"Epoch {epoch+1}/{num_epochs}, Average Imitation Loss: {epoch_loss / len(imitation_data)}")
            print(f"Epoch {epoch+1}, Learning Rate: {self.actor_optimizer.param_groups[0]['lr']}")

        print("Pre-training finished!")

    def choose_action(self, state, noise_scale=None):
        if not isinstance(state, torch.Tensor):
            state = torch.tensor(state, dtype=torch.float32).to(self.device)

        action = self.actor_critic.get_action(state).detach().cpu().numpy()
        
        adjusted_action = np.copy(action)

        if len(adjusted_action.shape) == 2 and adjusted_action.shape[0] == 1: 
            adjusted_action = np.squeeze(adjusted_action) 

        noise_scale = noise_scale or self.noise_scale
        noise = noise_scale * np.random.randn(adjusted_action.shape[0])  
        adjusted_action[0] = np.clip(adjusted_action[0] + noise[0], -1, 1)  

        adjusted_action[1:] = 0

        final_action = adjusted_action.reshape(1, -1)  

        #print(f"final_action: {final_action}")
        return final_action



    def store_transition(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def train(self):
        if len(self.memory) < 128:
            return

        batch = random.sample(self.memory, 128)
        states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.stack([
            torch.tensor(s, dtype=torch.float32, device=self.device).unsqueeze(0) 
            if isinstance(s, np.ndarray) and s.ndim == 1 else 
            torch.tensor(s, dtype=torch.float32, device=self.device)
            for s in states])
        
        actions = torch.stack([
            torch.tensor(a, dtype=torch.float32, device=self.device).unsqueeze(0) 
            if isinstance(a, np.ndarray) and a.ndim == 1 else 
            torch.tensor(a, dtype=torch.float32, device=self.device)
            for a in actions]).squeeze(1)

        rewards = torch.tensor(rewards, dtype=torch.float32, device=self.device).unsqueeze(1)

        next_states = torch.stack([
            torch.tensor(ns, dtype=torch.float32, device=self.device).unsqueeze(0) 
            if isinstance(ns, np.ndarray) and ns.ndim == 1 else 
            torch.tensor(ns, dtype=torch.float32, device=self.device)
            for ns in next_states]).squeeze(1)

        dones = torch.tensor(dones, dtype=torch.float32, device=self.device).unsqueeze(1)

        next_actions = self.target_actor_critic.get_action(next_states)
        target_q = rewards + (1 - dones) * self.gamma * self.target_actor_critic.get_value(next_states, next_actions)
        
        current_q = self.actor_critic.get_value(states, actions)
        critic_loss = torch.mean((current_q - target_q.detach()) ** 2)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor_critic.critic.parameters(), max_norm=1.0)
        self.critic_optimizer.step()

        predicted_actions = self.actor_critic.get_action(states)
        actor_loss = -self.actor_critic.get_value(states, predicted_actions).mean()

        imitation_loss = torch.tensor(0.0, device=self.device)
        if self.imitation_data:
            imitation_loss_fn = nn.MSELoss()
            imitation_batch = random.sample(self.imitation_data, min(64, len(self.imitation_data)))
            imitation_states, target_actions = zip(*imitation_batch)
            imitation_states = torch.stack(imitation_states).to(self.device)
            target_actions = torch.stack(target_actions).to(self.device)
            predicted_actions = self.actor_critic.get_action(imitation_states)
            imitation_loss = imitation_loss_fn(predicted_actions, target_actions)
            actor_loss += self.imitation_weight * imitation_loss

        total_reward = torch.sum(rewards).item()
        self.episode_rewards.append(total_reward)

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor_critic.actor.parameters(), max_norm=1.0)
        self.actor_optimizer.step()

        self.update_target_network(self.target_actor_critic.actor, self.actor_critic.actor, tau=0.005)
        self.update_target_network(self.target_actor_critic.critic, self.actor_critic.critic, tau=0.005)

        self.critic_losses.append(critic_loss.item())
        self.actor_losses.append(actor_loss.item())
        
        self.target_q_values.append(target_q.mean().item())
        self.current_q_values.append(current_q.mean().item())
        
        actor_grad_norm = torch.nn.utils.clip_grad_norm_(self.actor_critic.actor.parameters(), max_norm=1.0)
        critic_grad_norm = torch.nn.utils.clip_grad_norm_(self.actor_critic.critic.parameters(), max_norm=5.0)
        self.actor_grad_norms.append(actor_grad_norm)
        self.critic_grad_norms.append(critic_grad_norm)

        if critic_loss.item() < self.best_critic_loss:
            self.best_critic_loss = critic_loss.item()
            self.epochs_without_improvement = 0 
        else:
            self.epochs_without_improvement += 1
            if self.epochs_without_improvement >= self.patience:
                print("Early stopping triggered!")
                self.stop_training = True

        if len(self.critic_losses) % 1000 == 0:
            self.plot_metrics()

    def update_target_network(self, target, source, tau=0.001):
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(tau * param.data + (1.0 - tau) * target_param.data)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    np.random.seed(42)
    torch.manual_seed(42)
    random.seed(42)

    args = get_args()
    env_cfg = task_registry.get_cfgs(name=args.task)
    env_cfg.env.num_envs = min(env_cfg.env.num_envs, 50)
    env_cfg.control.controller = "lee_attitude_control"
    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)

    state, _ = env.reset()
    state = state.to(device)
    state_dim = state.shape[1]
    action_dim = env.action_input.shape[1]

    imitation_data = np.load('imitation_data.npy', allow_pickle=True)
    
    imitation_data_tensors = []
    for state_action_pair in imitation_data:
         state = state_action_pair[0][0]  
         action = state_action_pair[1][0]  

         state_tensor = torch.tensor(state, dtype=torch.float32).to(device)
         action_tensor = torch.tensor(action, dtype=torch.float32).to(device)

         imitation_data_tensors.append((state_tensor, action_tensor))

    agent = DDPGAgent(state_dim, action_dim, imitation_data=imitation_data_tensors)

    agent.pretrain_imitation()

    total_timesteps = 100000
    episode_reward = 0
    episode_timesteps = 0
    episode_num = 0
    reset_penalty = 0

    for t in range(total_timesteps):
        if t % 1000 == 0:
            print("TIMESTEP ", t)
        if agent.stop_training:
            print("Training stopped due to early stopping.")
            break
        episode_timesteps += 1
        reset_penalty = 0
        start_time = time.time()

        action = agent.choose_action(state)
        #agent.noise_scale *= agent.noise_decay
        next_state, privileged_obs, reward, reset, extras, root_quats = env.step(action)
        next_state = next_state.to(device)

        agent.store_transition(state, action, reward, next_state, reset)
        agent.train()
        episode_reward += reward
        # if reset:
        #      training_progress = min(t / 100000, 1.0)
        #      crash_penalty = -100 * training_progress
        #      reward += crash_penalty
        #      episode_reward += reward

        state = next_state
        if reset:
            end_time = time.time()
            print(f"Episode: {episode_num + 1}, Reward: {episode_reward}, Timesteps: {episode_timesteps}")
            agent.episode_rewards.append(episode_reward)
            episode_timesteps = 0
            episode_reward = 0
            episode_num += 1
            state, _ = env.reset()
            state = state.to(device)
    #plot_metrics(agent.critic_losses, agent.actor_losses, agent.episode_rewards)


if __name__ == "__main__":
    main()
