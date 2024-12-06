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
        {"name": "--num_envs", "type": int, "default": 100, "help": "Number of environments to create. Overrides config file if provided."},
        {"name": "--seed", "type": int, "default": 1, "help": "Random seed. Overrides config file if provided."},
        {"name": "--play", "required": False, "help": "only run network", "action": 'store_true'},
        {"name": "--torch-deterministic-off", "action": "store_true", "default": False, "help": "if toggled, `torch.backends.cudnn.deterministic=False`"},
        {"name": "--track", "action": "store_true", "default": False,"help": "if toggled, this experiment will be tracked with Weights and Biases"},
        {"name": "--wandb-project-name", "type":str, "default": "cleanRL", "help": "the wandb's project name"},
        {"name": "--wandb-entity", "type":str, "default": None, "help": "the entity (team) of wandb's project"},
        {"name": "--total-timesteps", "type":int, "default": 3000000, "help": "total timesteps of the experiments"},
        {"name": "--learning-rate", "type":float, "default": 1e-3, "help": "the learning rate of the optimizer"},
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
        {"name": "--actor_learning_rate", "type":float, "default": 0.0003, "help": "Learning rate of the actor network."},
        {"name": "--critic_learning_rate", "type":float, "default": 0.001, "help": "Learning rate of the critic network."},

    ]

    args = gymutil.parse_arguments(description="RL Policy", custom_parameters=custom_parameters)
    
    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    args.torch_deterministic = not args.torch_deterministic_off
    args.norm_adv = not args.norm_adv_off

    args.sim_device_id = args.rl_device.split(":")[-1] if "cuda" in args.rl_device else "0"
    args.sim_device = args.rl_device

    return args

# ============================
# Environment Setup
# ============================
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


# ============================
# Agent Setup
# ============================
def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class Agent(nn.Module):
    def __init__(self, envs):
        super().__init__()
        self.critic = nn.Sequential(
            layer_init(nn.Linear(np.prod(envs.observation_space.shape), 256)),
            nn.Tanh(),
            layer_init(nn.Linear(256, 256)),
            nn.Tanh(),
            layer_init(nn.Linear(256, 1), std=1.0),
        )
        self.actor_mean = nn.Sequential(
            layer_init(nn.Linear(np.prod(envs.observation_space.shape), 256)),
            nn.ReLU(),
            layer_init(nn.Linear(256, 256)),
            nn.Tanh(),
            layer_init(nn.Linear(256, 1), std=0.01),
            nn.Tanh(),
        )
        self.actor_logstd = nn.Parameter(torch.zeros(1, np.prod(envs.num_actions))) 

    def get_value(self, x):
        return self.critic(x)
    
    def get_action_and_value(self, x, action=None):
        action_mean = self.actor_mean(x) 
        action_logstd = self.actor_logstd[:, :1].expand_as(action_mean)  
        action_std = torch.exp(action_logstd)  
        probs = Normal(action_mean, action_std) 

        if action is None:
            action = probs.sample()  

        if not isinstance(action, torch.Tensor):
            action = torch.tensor(action, dtype=torch.float32, device=x.device) 

        full_action = torch.zeros((action.shape[0], 4), device=x.device)  
        full_action[:, 0] = action.squeeze(-1) * 0.1

        log_prob = probs.log_prob(full_action).sum(-1)  
        entropy = probs.entropy().sum(-1)  
        value = self.critic(x) 

        return full_action, log_prob, entropy, value


# ============================
# Advantage Calculation
# ============================
def compute_advantages(rewards, dones, values, next_value, gamma, gae_lambda):
    advantages = torch.zeros_like(rewards)
    lastgaelam = 0
    for t in reversed(range(len(rewards))):
        if t == len(rewards) - 1:
            next_non_terminal = 1.0 - dones[t]
            next_values = next_value
        else:
            next_non_terminal = 1.0 - dones[t + 1]
            next_values = values[t + 1]
        delta = rewards[t] + gamma * next_values * next_non_terminal - values[t]
        advantages[t] = lastgaelam = delta + gamma * gae_lambda * next_non_terminal * lastgaelam
    return advantages 


# ============================
# Training Step
# ============================
def train_step(agent, optimizer_actor, optimizer_critic, obs, actions, advantages, logprobs, values, rewards, args):
    b_obs = obs.reshape((-1, np.prod(obs.shape[2:])))
    b_actions = actions.reshape((-1, 1))
    b_logprobs = logprobs.reshape(-1)
    b_advantages = advantages.reshape(-1)
    b_returns = (advantages + values).reshape(-1)
    b_values = values.reshape(-1)

    clipfracs = []
    
    for epoch in range(args.update_epochs):
        b_inds = torch.randperm(b_obs.size(0), device=obs.device)
        for start in range(0, b_obs.size(0), args.minibatch_size):
            end = start + args.minibatch_size
            mb_inds = b_inds[start:end]

            new_actions, new_logprobs, entropy, new_value = agent.get_action_and_value(b_obs[mb_inds])

            logratio = new_logprobs - b_logprobs[mb_inds]
            ratio = logratio.exp()

            with torch.no_grad():
                old_approx_kl = (-logratio).mean()
                approx_kl = ((ratio - 1) - logratio).mean()
                clipfracs += [((ratio - 1.0).abs() > args.clip_coef).float().mean().item()]

            mb_advantages = b_advantages[mb_inds]
            if args.norm_adv:
                mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

            pg_loss1 = -mb_advantages * ratio
            pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
            pg_loss = torch.max(pg_loss1, pg_loss2).mean()

            v_loss = 0.5 * ((new_value.view(-1) - b_returns[mb_inds]) ** 2).mean()

            entropy_loss = entropy.mean()
            loss = pg_loss + v_loss * args.vf_coef + entropy_loss * args.ent_coef

            optimizer_actor.zero_grad()
            optimizer_critic.zero_grad()
            loss.backward()

            nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)

            optimizer_actor.step()
            optimizer_critic.step()

        if args.target_kl is not None:
            if approx_kl > args.target_kl:
                break

    return {
        "loss/actor": pg_loss.item(),
        "loss/critic": v_loss.item(),
        "loss/entropy": entropy_loss.item(),
        "clip_frac": np.mean(clipfracs)
    }


# ============================
# Main Training Loop
# ============================
def train_altitude_control(env, agent, args):
    writer = SummaryWriter(f"runs/{args.experiment_name}")
    optimizer_actor = optim.Adam(agent.actor_mean.parameters(), lr=args.actor_learning_rate, eps=1e-5)
    optimizer_critic = optim.Adam(agent.critic.parameters(), lr=args.critic_learning_rate, eps=1e-5)

    episodic_rewards = deque(maxlen=100)
    actor_losses = deque(maxlen=100)
    critic_losses = deque(maxlen=100)

    obs = torch.zeros((args.num_steps, args.num_envs, np.prod(env.observation_space.shape)), dtype=torch.float).to(args.rl_device)
    actions = torch.zeros((args.num_steps, args.num_envs, 4), dtype=torch.float).to(args.rl_device)  # Only thrust
    rewards = torch.zeros((args.num_steps, args.num_envs), dtype=torch.float).to(args.rl_device)
    dones = torch.zeros((args.num_steps, args.num_envs), dtype=torch.float).to(args.rl_device)
    values = torch.zeros((args.num_steps, args.num_envs), dtype=torch.float).to(args.rl_device)
    logprobs = torch.zeros((args.num_steps, args.num_envs), dtype=torch.float).to(args.rl_device)
    advantages = torch.zeros_like(rewards, dtype=torch.float).to(args.rl_device)

    global_step = 0
    episode_number = 0
    num_updates = args.total_timesteps // args.batch_size
    next_obs, _info = env.reset()
    next_done = torch.zeros(args.num_envs, dtype=torch.float).to(args.rl_device)

    for update in range(1, num_updates + 1):
        for step in range(0, args.num_steps):
            global_step += 1
            next_obs_tensor = next_obs[0].unsqueeze(0) if next_obs[0].dim() == 1 else next_obs[0]
            obs[step] = next_obs_tensor
            dones[step] = next_done

            with torch.no_grad():
                action, logprob, _, value = agent.get_action_and_value(next_obs_tensor)
                values[step] = value.flatten()
           
            actions[step] = action
            logprobs[step] = logprob

            next_obs, rewards[step], next_done, info = env.step(action)

            skip_reward = False
            if 0 <= step <= 2:
                    for idx, d in enumerate(next_done):
                        if d:
                            episode_number += 1
                            episodic_return = info["r"][idx].item()  
                            episodic_length = info["l"][idx]  

                            print(f"Episode {episode_number}, global_step={global_step}, episodic_return={episodic_return}")
                            writer.add_scalar("charts/episodic_return", episodic_return, global_step)
                            writer.add_scalar("charts/episodic_length", episodic_length, global_step)
                            
                            if "consecutive_successes" in info:
                                writer.add_scalar("charts/consecutive_successes", info["consecutive_successes"].item(), global_step)
                            break



        with torch.no_grad():
            next_value = agent.get_value(next_obs).reshape(1, -1)
            advantages = compute_advantages(rewards, dones, values, next_value, args.gamma, args.gae_lambda)
        
        train_step(agent, optimizer_actor, optimizer_critic, obs, actions, advantages, logprobs, values, rewards, args)

    writer.close()


# ============================
# Main Entry Point
# ============================
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

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    env_cfg = task_registry.get_cfgs(name=args.task)
    env_cfg.env.num_envs = min(env_cfg.env.num_envs, args.num_envs)
    env_cfg.control.controller = "lee_attitude_control"
    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)

    env = RecordEpisodeStatisticsTorch(env, args.rl_device)

    agent = Agent(env).to(args.rl_device)
    train_altitude_control(env, agent, args)