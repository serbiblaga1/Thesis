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
        {"name": "--num_envs", "type": int, "default": 1, "help": "Number of environments to create. Overrides config file if provided."},
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


class Agent(nn.Module):
    def __init__(self, envs):
        super().__init__()
        self.critic = nn.Sequential(
            layer_init(nn.Linear(np.array(envs.observation_space.shape).prod(), 256)),
            nn.ReLU(),
            layer_init(nn.Linear(256, 256)),
            nn.Tanh(),
            layer_init(nn.Linear(256, 1), std=1.0),
        )
        self.actor_mean = nn.Sequential(
            layer_init(nn.Linear(np.array(envs.observation_space.shape).prod(), 256)),
            nn.ReLU(),
            layer_init(nn.Linear(256, 256)),
            nn.Tanh(),
            layer_init(nn.Linear(256, np.prod(envs.action_space.shape)), std=0.01),
            nn.Tanh(),
        )
        self.actor_logstd = nn.Parameter(torch.zeros(1, np.prod(envs.action_space.shape)))

        for name, param in self.named_parameters():
            print(f"{name} requires_grad: {param.requires_grad}")

    def get_value(self, x):
        return self.critic(x)
    
    def get_action_and_value(self, x, action=None):
        action_mean = self.actor_mean(x) * 0.01
        action_logstd = self.actor_logstd.expand_as(action_mean) 
        action_std = torch.exp(action_logstd) 

        probs = Normal(action_mean, action_std)
        
        log_prob = probs.log_prob(action).sum(1)
        entropy = probs.entropy().sum(1)
        
        return action, log_prob, entropy, self.critic(x)


    # def get_action_and_value_original(self, x, action=None):
    #     action_mean = self.actor_mean(x)
    #     action_logstd = self.actor_logstd.expand_as(action_mean)
    #     action_std = torch.exp(action_logstd)
    #     probs = Normal(action_mean, action_std)
    #     if action is None:
    #         action = probs.sample()
    #     print("ACTIONS ", action)
    #     return action, probs.log_prob(action).sum(1), probs.entropy().sum(1), self.critic(x)
    

def imitation_learning(agent, optimizer, expert_data, num_epochs, device):
    print("Starting imitation learning...")

    obs = [item[0][0] for item in expert_data] 
    actions = [item[1][0] for item in expert_data]  
    obs = np.array(obs, dtype=np.float32)
    actions = np.array(actions, dtype=np.float32)
    obs = torch.tensor(obs, dtype=torch.float32, requires_grad=True).to(device) 
    actions = torch.tensor(actions, dtype=torch.float32).to(device) 
    criterion = nn.MSELoss()
    agent.train()  

    for epoch in range(num_epochs):
        optimizer.zero_grad()  
        pred_actions, logprob, entropy, value = agent.get_action_and_value(obs)
        print(f"pred_actions requires_grad: {pred_actions.requires_grad}")  

        if not pred_actions.requires_grad:
            raise RuntimeError("Gradient flow is broken. Check model parameters and forward pass.")
        
        loss = criterion(pred_actions, actions)
        print(f"Loss: {loss.item()}")  
        loss.backward()
        optimizer.step()
        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {loss.item():.4f}")

    print("Imitation learning completed.")



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

    agent = Agent(env).to(device)
    optimizer_actor = optim.Adam(agent.actor_mean.parameters(), lr=args.actor_learning_rate, eps=1e-5)
    optimizer_critic = optim.Adam(agent.critic.parameters(), lr=args.critic_learning_rate, eps=1e-5)


    if args.play and args.checkpoint is None:
        raise ValueError("No checkpoint provided for testing.")

    if args.checkpoint is not None:
        print("Loading checkpoint...")
        checkpoint = torch.load(args.checkpoint)
        agent.load_state_dict(checkpoint)
        print("Loaded checkpoint")

    episodic_rewards = deque(maxlen=100)
    actor_losses = deque(maxlen=100)
    critic_losses = deque(maxlen=100)

    # if os.path.exists(args.expert_data_path):
    #     expert_data = np.load(args.expert_data_path, allow_pickle=True)
    #     imitation_learning(agent, optimizer, expert_data, args.imitation_epochs, device)
    # else:
    #     print(f"Expert data not found at {args.expert_data_path}. Skipping imitation learning.")


    obs = torch.zeros((args.num_steps, args.num_envs, np.prod(env.observation_space.shape)), dtype=torch.float).to(device)
    actions = torch.zeros((args.num_steps, args.num_envs, np.prod(env.action_space.shape)), dtype=torch.float).to(device)
    logprobs = torch.zeros((args.num_steps, args.num_envs), dtype=torch.float).to(device)
    rewards = torch.zeros((args.num_steps, args.num_envs), dtype=torch.float).to(device)
    dones = torch.zeros((args.num_steps, args.num_envs), dtype=torch.float).to(device)
    values = torch.zeros((args.num_steps, args.num_envs), dtype=torch.float).to(device)
    advantages = torch.zeros_like(rewards, dtype=torch.float).to(device)

    global_step = 0
    start_time = time.time()
    next_obs, _info = env.reset()
    next_done = torch.zeros(args.num_envs, dtype=torch.float).to(device)
    num_updates = args.total_timesteps // args.batch_size
    episode_number = 0
    #previous_done = torch.zeros_like(next_done)

    if not args.play:
        for update in range(1, num_updates + 1):
            if args.anneal_lr:
                frac = 1.0 - (update - 1.0) / num_updates
                lrnow_actor = frac * args.actor_learning_rate
                lrnow_critic = frac * args.critic_learning_rate
                optimizer_actor.param_groups[0]["lr"] = lrnow_actor
                optimizer_critic.param_groups[0]["lr"] = lrnow_critic
            
            for step in range(0, args.num_steps):
                global_step += 1 * args.num_envs
                next_obs_tensor = next_obs[0] 
                if next_obs_tensor.dim() == 1:
                    next_obs_tensor = next_obs_tensor.unsqueeze(0) 
                obs[step] = next_obs_tensor 
                dones[step] = next_done

                with torch.no_grad():
                    action, logprob, _, value = agent.get_action_and_value(next_obs_tensor)
                    values[step] = value.flatten()
                actions[step] = action
                logprobs[step] = logprob

                # Now, proceed with the normal flow
                next_obs, rewards[step], next_done, info = env.step(action)
                #if any(previous_done == 1):  # Check if any of the previous done flags were 1
                #    next_done = torch.zeros_like(next_done)  # Reset next_done to 0 for all environments
                skip_reward = False
                if next_done:
                    for idx, d in enumerate(next_done):
                        #print(f"Environment {idx}, Done: {d}")
                        # If an environment was done in the previous step, we ensure we reset it now
                        if d:
                            episode_number += 1
                            episodic_return = info["r"][idx].item()  
                            episodic_length = info["l"][idx]  

                            if episodic_length < 2:
                                print(f"Spurious reset detected for env {idx}. Skipping reward.")
                                skip_reward = True
                                episode_number -= 1  
                                continue

                            print(f"Episode {episode_number}, global_step={global_step}, episodic_return={episodic_return}")
                            writer.add_scalar("charts/episodic_return", episodic_return, episode_number)
                            writer.add_scalar("charts/episodic_length", episodic_length, episode_number)
                            
                            if "consecutive_successes" in info:
                                writer.add_scalar("charts/consecutive_successes", info["consecutive_successes"].item(), global_step)
                            break

                #if skip_reward:
                #    rewards[step] = 0.0  
                #    skip_reward = False  
           # previous_done = next_done.clone()

            with torch.no_grad():
                next_value = agent.get_value(next_obs).reshape(1, -1)
                advantages = torch.zeros_like(rewards).to(device)
                lastgaelam = 0
                for t in reversed(range(args.num_steps)):
                    if t == args.num_steps - 1:
                        nextnonterminal = 1.0 - next_done
                        nextvalues = next_value
                    else:
                        nextnonterminal = 1.0 - dones[t + 1]
                        nextvalues = values[t + 1]
                    delta = rewards[t] + args.gamma * nextvalues * nextnonterminal - values[t]
                    advantages[t] = lastgaelam = delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam
                returns = advantages + values

            b_obs = obs.reshape((-1, np.prod(env.observation_space.shape)))
            b_actions = actions.reshape((-1, np.prod(env.action_space.shape)))
            b_logprobs = logprobs.reshape(-1)
            b_advantages = advantages.reshape(-1)
            b_returns = returns.reshape(-1)
            b_values = values.reshape(-1)
            
            clipfracs = []
            for epoch in range(args.update_epochs):
                b_inds = torch.randperm(args.batch_size, device=device)
                for start in range(0, args.batch_size, args.minibatch_size):
                    end = start + args.minibatch_size
                    mb_inds = b_inds[start:end]

                    _, newlogprob, entropy, newvalue = agent.get_action_and_value(b_obs[mb_inds], b_actions[mb_inds])
                    logratio = newlogprob - b_logprobs[mb_inds]
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

                    newvalue = newvalue.view(-1)
                    if args.clip_vloss:
                        v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                        v_clipped = b_values[mb_inds] + torch.clamp(
                            newvalue - b_values[mb_inds],
                            -args.clip_coef,
                            args.clip_coef,
                        )
                        v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                        v_loss = 0.5 * torch.max(v_loss_unclipped, v_loss_clipped).mean()
                    else:
                        v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                    entropy_loss = entropy.mean()
                    loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef

                    actor_loss = -(ratio * b_advantages[mb_inds]).mean()

                    critic_loss = (newvalue - b_returns[mb_inds]).pow(2).mean()

                    total_loss = actor_loss + 0.5 * critic_loss - 0.01 * entropy.mean()

                    # Actor update
                    optimizer_actor.zero_grad()
                    actor_loss.backward()
                    optimizer_actor.step()

                    # Critic update
                    optimizer_critic.zero_grad()
                    critic_loss.backward()
                    optimizer_critic.step()

                    #for name, param in agent.named_parameters():
                    #    if param.grad is not None:
                    #        print(f"{name} grad norm: {param.grad.norm()}")

                    nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)

                if args.target_kl is not None:
                    if approx_kl > args.target_kl:
                        break

            actor_losses.append(actor_loss.item())
            critic_losses.append(critic_loss.item()) 
                        
            if update % 50 == 0:
                print("Saving model.")
                torch.save(agent.state_dict(), f"runs/{run_name}/latest_model.pth")

            #if update % 50 == 0:
            #    plot_metrics(list(actor_losses), list(critic_losses), list(episodic_rewards))

            if update % args.save_model_interval == 0:
                torch.save(agent.state_dict(), f"checkpoints/{run_name}_{update}.pth")

        writer.add_scalar("charts/learning_rate", optimizer_actor.param_groups[0]["lr"], global_step)
        writer.add_scalar("charts/learning_rate", optimizer_critic.param_groups[0]["lr"], global_step)
        writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
        writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
        writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
        writer.add_scalar("losses/old_approx_kl", old_approx_kl.item(), global_step)
        writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
        writer.add_scalar("losses/clipfrac", np.mean(clipfracs), global_step)
        print("SPS:", int(global_step / (time.time() - start_time)))
        writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)
            
    else:
        for step in range(0, 5000000):
            # ALGO LOGIC: action logic
            with torch.no_grad():
                action, logprob, _, value = agent.get_action_and_value(next_obs)
            next_obs, rewards, next_done, info = env.step(action)

   # env.close()
    writer.close()
