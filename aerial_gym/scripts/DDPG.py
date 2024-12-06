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

import setuptools
import distutils.version
setuptools.version = distutils.version

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
from torch.utils.tensorboard import SummaryWriter
from torch.distributions.normal import Normal
import torch.nn.functional as F
import argparse
import matplotlib.pyplot as plt

def plot_metrics(actor_losses, critic_losses, episodic_rewards):
    plt.figure(figsize=(15, 5))

    # Plot Actor Loss
    plt.subplot(1, 3, 1)
    plt.plot(actor_losses, label="Actor Loss", color='b')
    plt.xlabel("Timesteps")
    plt.ylabel("Actor Loss")
    plt.legend()

    # Plot Critic Loss
    plt.subplot(1, 3, 2)
    plt.plot(critic_losses, label="Critic Loss", color='r')
    plt.xlabel("Timesteps")
    plt.ylabel("Critic Loss")
    plt.legend()

    # Plot Rewards per Episode
    plt.subplot(1, 3, 3)
    plt.plot(episodic_rewards, label="Rewards per Episode", color='g')
    plt.xlabel("Episodes")
    plt.ylabel("Rewards")
    plt.legend()

    plt.tight_layout()
    plt.show()

def get_args():
    custom_parameters = [
        {"name": "--task", "type": str, "default": "quad", "help": "Resume training or start testing from a checkpoint. Overrides config file if provided."},
        {"name": "--experiment_name", "type": str, "default": os.path.basename(__file__).rstrip(".py"), "help": "Name of the experiment to run or load. Overrides config file if provided."},
        {"name": "--checkpoint", "type": str, "default": None, "help": "Saved model checkpoint number."},
        {"name": "--headless", "action": "store_true", "default": False, "help": "Force display off at all times"},
        {"name": "--horovod", "action": "store_true", "default": False, "help": "Use horovod for multi-gpu training"},
        {"name": "--rl_device", "type": str, "default": "cuda:0", "help": 'Device used by the RL algorithm, (cpu, gpu, cuda:0, cuda:1 etc..)'},
        {"name": "--num_envs", "type": int, "default": 1, "help": "Number of environments to create. Overrides config file if provided."},
        {"name": "--seed", "type": int, "default": 1, "help": "Random seed. Overrides config file if provided."},
        {"name": "--play", "required": False, "help": "only run network", "action": 'store_true'},
        {"name": "--torch-deterministic-off", "action": "store_true", "default": False, "help": "if toggled, `torch.backends.cudnn.deterministic=False`"},
        {"name": "--track", "action": "store_true", "default": False,"help": "if toggled, this experiment will be tracked with Weights and Biases"},
        {"name": "--wandb-project-name", "type":str, "default": "cleanRL", "help": "the wandb's project name"},
        {"name": "--wandb-entity", "type":str, "default": None, "help": "the entity (team) of wandb's project"},
        {"name": "--total-timesteps", "type":int, "default": 3000000, "help": "total timesteps of the experiments"},
        {"name": "--learning-rate", "type":float, "default": 1e-2, "help": "the learning rate of the optimizer"},
        {"name": "--num-steps", "type":int, "default": 64, "help": "the number of steps to run in each environment per policy rollout"},
        {"name": "--anneal-lr", "action": "store_true", "default": True, "help": "Toggle learning rate annealing for policy and value networks"},
        {"name": "--gamma", "type":float, "default": 0.95, "help": "the discount factor gamma"},
        {"name": "--gae-lambda", "type":float, "default": 0.99, "help": "the lambda for the general advantage estimation"},
        {"name": "--num-minibatches", "type":int, "default": 5, "help": "the number of mini-batches"},
        {"name": "--update-epochs", "type":int, "default": 4, "help": "the K epochs to update the policy"},
        {"name": "--norm-adv-off", "action": "store_true", "default": False, "help": "Toggles advantages normalization"},
        {"name": "--clip-coef", "type":float, "default": 0.2, "help": "the surrogate clipping coefficient"},
        {"name": "--clip-vloss", "action": "store_true", "default": False, "help": "Toggles whether or not to use a clipped loss for the value function, as per the paper."},
        {"name": "--ent-coef", "type":float, "default": 0.01, "help": "coefficient of the entropy"},
        {"name": "--vf-coef", "type":float, "default": 2, "help": "coefficient of the value function"},
        {"name": "--max-grad-norm", "type":float, "default": 1, "help": "the maximum norm for the gradient clipping"},
        {"name": "--target-kl", "type":float, "default": 0.1, "help": "the target KL divergence threshold"},
        {"name": "--save_model_interval", "type": int, "default": 10000, "help": "Interval (in updates) to save the model."}, 
        {"name": "--imitation_epochs", "type": int, "default": 243, "help": "Number of imitation learning epochs."},
        {"name": "--expert_data_path", "type": str, "default": "imitation_data.npy", "help": "Path to the expert demonstrations."},

    ]

    args = gymutil.parse_arguments(description="RL Policy", custom_parameters=custom_parameters)
    
    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    args.torch_deterministic = not args.torch_deterministic_off
    args.norm_adv = not args.norm_adv_off

    args.sim_device_id = args.rl_device.split(":")[-1] if "cuda" in args.rl_device else "0"
    args.sim_device = args.rl_device

    return args



class RecordEpisodeStatisticsTorch(gym.Wrapper):
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
        observations, _, rewards, dones, infos, _ = super().step(action)
        self.episode_returns += rewards
        self.episode_lengths += 1
        self.returned_episode_returns[:] = self.episode_returns
        self.returned_episode_lengths[:] = self.episode_lengths
        self.episode_returns *= 1 - dones
        self.episode_lengths *= 1 - dones
        infos["r"] = self.returned_episode_returns
        infos["l"] = self.returned_episode_lengths
        return observations, rewards, dones, infos


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

# Replay Buffer
class ReplayBuffer:
    def __init__(self, capacity, state_dim, action_dim):
        self.capacity = capacity
        self.ptr = 0
        self.size = 0
        self.states = torch.zeros((capacity, state_dim))
        self.actions = torch.zeros((capacity, action_dim))
        self.rewards = torch.zeros(capacity)
        self.next_states = torch.zeros((capacity, state_dim))
        self.dones = torch.zeros(capacity)

    def add(self, state, action, reward, next_state, done):
        self.states[self.ptr] = torch.tensor(state)
        self.actions[self.ptr] = torch.tensor(action)
        self.rewards[self.ptr] = torch.tensor(reward)
        self.next_states[self.ptr] = torch.tensor(next_state)
        self.dones[self.ptr] = torch.tensor(done)
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size):
        idxs = np.random.randint(0, self.size, size=batch_size)
        return (self.states[idxs], self.actions[idxs],
                self.rewards[idxs], self.next_states[idxs], self.dones[idxs])


class DDPGAgent(nn.Module):
    def __init__(self, state_dim, action_dim, max_action):
        super(DDPGAgent, self).__init__()
        
        # Actor network
        self.actor = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, action_dim),
            nn.Tanh(),  
        )

        # Critic network
        self.critic = nn.Sequential(
            nn.Linear(state_dim + action_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1), 
        )

        # Target actor and critic (for stable training)
        self.target_actor = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, action_dim),
            nn.Tanh(),
        )

        self.target_critic = nn.Sequential(
            nn.Linear(state_dim + action_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
        )

        self.max_action = max_action

        self.target_actor.load_state_dict(self.actor.state_dict())
        self.target_critic.load_state_dict(self.critic.state_dict())

    def forward(self, state):
        return self.actor(state)

    def get_action(self, state):
        action = self.actor(state) 

        thrust = action[..., 0]  
        fixed_action = torch.zeros_like(action) 
        fixed_action[..., 0] = thrust 
        
        fixed_action[:, 1] = 0
        return fixed_action

    def get_value(self, state, action):
        return self.critic(torch.cat([state, action], dim=-1))  

def ddpg_update(agent, target_agent, replay_buffer, optimizer_actor, optimizer_critic, gamma, tau, batch_size):
    states, actions, rewards, next_states, dones = replay_buffer.sample(batch_size)
    states = states.to(device)
    actions = actions.to(device)
    rewards = rewards.to(device)
    next_states = next_states.to(device)
    dones = dones.to(device)
    # Update Critic
    with torch.no_grad():
        target_actions = target_agent.target_actor(next_states)
        target_q = target_agent.target_critic(torch.cat([next_states, target_actions], dim=-1))
        target = rewards + gamma * (1 - dones) * target_q
    q_values = agent.critic(torch.cat([states, actions], dim=-1))
    critic_loss = F.mse_loss(q_values, target)

    optimizer_critic.zero_grad()
    critic_loss.backward()
    optimizer_critic.step()

    # Update Actor
    predicted_actions = agent.actor(states)
    actor_loss = -agent.critic(torch.cat([states, predicted_actions], dim=-1)).mean()

    optimizer_actor.zero_grad()
    actor_loss.backward()
    optimizer_actor.step()

    # Update Target Networks
    for target_param, param in zip(target_agent.target_actor.parameters(), agent.actor.parameters()):
        target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)
    for target_param, param in zip(target_agent.target_critic.parameters(), agent.critic.parameters()):
        target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

# PID Controller
class PIDController:
    def __init__(self):
        self.kP, self.kI, self.kD = 1.0, 0.0, 0.0  # Default gains
        self.prev_error = 0.0
        self.integral = 0.0

    def update(self, error, dt):
        self.integral += error * dt
        derivative = (error - self.prev_error) / dt
        self.prev_error = error
        return self.kP * error + self.kI * self.integral + self.kD * derivative

    def set_gains(self, kP, kI, kD):
        self.kP = kP
        self.kI = kI
        self.kD = kD


# Training Loop
def train_ddpg(env, agent, replay_buffer, pid_controller, num_episodes, gamma=0.99, tau=0.005, batch_size=256):
    optimizer_actor = torch.optim.Adam(agent.actor.parameters(), lr=0.0003)
    optimizer_critic = torch.optim.Adam(agent.critic.parameters(), lr=0.001)

    total_rewards = []
    for episode in range(num_episodes):
        reset_output = env.reset()
        if isinstance(reset_output, tuple):
            state = reset_output[0] 
        else:
            state = reset_output

        state = state.float().to(device)
        episode_reward = 0

        for step in range(500):  # Max steps per episode
            action = agent.get_action(state.unsqueeze(0))  # Get action
            #thrust, kP, kI, kD = action[0]  # Assume 4 actions

            # Set PID gains dynamically
            #pid_controller.set_gains(kP.item(), kI.item(), kD.item())

            # Use PID for control
            target_altitude = 0.3
            #error = target_altitude - state[0].item()  # Altitude error
            #pid_output = pid_controller.update(error, dt=0.02)

            # Apply actions to environment
            #combined_action = [thrust.item(), pid_output]
            next_state, reward, done, _ = env.step(action)

            replay_buffer.add(state, action, reward, next_state, done)
            episode_reward += reward

            state = torch.tensor(next_state, dtype=torch.float32)

            # Train the agent
            if replay_buffer.size >= batch_size:
                ddpg_update(agent, agent, replay_buffer, optimizer_actor, optimizer_critic, gamma, tau, batch_size)

            if done:
                break

        total_rewards.append(episode_reward)
        print(f"Episode {episode + 1}/{num_episodes}, Reward: {episode_reward.item():.2f}")


    return total_rewards


# Main Function
if __name__ == "__main__":
    args = get_args()
    run_name = f"{args.task}__{args.experiment_name}__{args.seed}__{int(time.time())}"
    if args.track:
        import wandb
        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            sync_tensorboard=True,
            config=vars(args),
            name=run_name,
            monitor_gym=True,
            save_code=True,
        )
    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = args.rl_device
    print("Using device:", device)
    
    env_cfg = task_registry.get_cfgs(name=args.task)

    env_cfg.env.num_envs = min(env_cfg.env.num_envs, args.num_envs)  
    env_cfg.control.controller = "lee_attitude_control"

    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)

    env = RecordEpisodeStatisticsTorch(env, device)

    print("Action space:", env.action_space)
    print("Observation space:", env.observation_space)
    state_dim = env.observation_space.shape[0] 
    action_dim = env.action_space.shape[0] 
    agent = DDPGAgent(state_dim, action_dim, 0.5).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)

    replay_buffer = ReplayBuffer(capacity=1000, state_dim=state_dim, action_dim=action_dim)
    pid_controller = PIDController()

    train_ddpg(env, agent, replay_buffer, pid_controller, num_episodes=1000)
