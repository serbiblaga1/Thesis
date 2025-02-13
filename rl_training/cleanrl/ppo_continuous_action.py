# Copyright (c) 2018-2022, NVIDIA Corporation
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/ppo/#ppo_continuous_action_isaacgympy
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


from aerial_gym.envs import *
from aerial_gym.utils import task_registry

def process_depth_images(env):
    depth_images = []
    depth_values = []
    
    for i in range(1, 6):
        depth = torch.clamp(getattr(env, f'full_camera_array{i}'), 0.0, 4.0)
        
        depth_cm = depth * 100  
        
        depth_normalized = depth_cm / 400 
        
        depth_np = depth_normalized.cpu().numpy()  
        depth_img = np.uint8(depth_np * 255)  
        
        depth_images.append(depth_img)
        depth_values.append(depth_np)
    
    return depth_images, depth_values

def display_depth_images(depth_images):
    titles = ['Front', 'Left', 'Right', 'Back', 'Down']
    
    for img, title in zip(depth_images, titles):
        grayscale_img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)  
        
        close_mask = img < int(0.5 * 255)  
        grayscale_img[close_mask] = [0, 0, 255]  
        
        cv2.imshow(title, grayscale_img)

    cv2.waitKey(1)
    


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

        # Algorithm specific arguments
        {"name": "--total-timesteps", "type":int, "default": 30000000,
            "help": "total timesteps of the experiments"},
        {"name": "--learning-rate", "type":float, "default": 0.0026,
            "help": "the learning rate of the optimizer"},
        {"name": "--num-steps", "type":int, "default": 16,
            "help": "the number of steps to run in each environment per policy rollout"},
        {"name": "--anneal-lr", "action": "store_true", "default": False,
            "help": "Toggle learning rate annealing for policy and value networks"},
        {"name": "--gamma", "type":float, "default": 0.99,
            "help": "the discount factor gamma"},
        {"name": "--gae-lambda", "type":float, "default": 0.95,
            "help": "the lambda for the general advantage estimation"},
        {"name": "--num-minibatches", "type":int, "default": 2,
            "help": "the number of mini-batches"},
        {"name": "--update-epochs", "type":int, "default": 4,
            "help": "the K epochs to update the policy"},
        {"name": "--norm-adv-off", "action": "store_true", "default": False,
            "help": "Toggles advantages normalization"},
        {"name": "--clip-coef", "type":float, "default": 0.2,
            "help": "the surrogate clipping coefficient"},
        {"name": "--clip-vloss", "action": "store_true", "default": False,
            "help": "Toggles whether or not to use a clipped loss for the value function, as per the paper."},
        {"name": "--ent-coef", "type":float, "default": 0.0,
            "help": "coefficient of the entropy"},
        {"name": "--vf-coef", "type":float, "default": 2,
            "help": "coefficient of the value function"},
        {"name": "--max-grad-norm", "type":float, "default": 1,
            "help": "the maximum norm for the gradient clipping"},
        {"name": "--target-kl", "type":float, "default": None,
            "help": "the target KL divergence threshold"},
        ]

    # parse arguments
    args = gymutil.parse_arguments(
        description="RL Policy",
        custom_parameters=custom_parameters)
    
    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)

    args.torch_deterministic = not args.torch_deterministic_off
    args.norm_adv = not args.norm_adv_off

    # name allignment
    args.sim_device_id = args.compute_device_id
    args.sim_device = args.sim_device_type
    if args.sim_device=='cuda':
        args.sim_device += f":{args.sim_device_id}"
    return args

class PIDController:
    def __init__(self, p, i, d, target):
        self.p = p  
        self.i = i  
        self.d = d 
        self.target = target 
        
        self.integral = torch.zeros(100, device='cuda')  
        self.prev_error = torch.zeros(100, device='cuda')  

    def update(self, current_value):
        error = self.target - current_value  
        
        p_term = self.p * error 
        self.integral += error  
        i_term = self.i * self.integral  
        d_term = self.d * (error - self.prev_error) 

        self.prev_error = error

        pid_output = p_term + i_term + d_term
        return pid_output


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
            layer_init(nn.Linear(256, 3), std=0.01),
        )
        self.actor_logstd = nn.Parameter(torch.zeros(1, 3))

    def get_value(self, x):
        return self.critic(x)
    

    def get_action_and_value(self, x, action=None):
        action_mean = self.actor_mean(x)
        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        probs = Normal(action_mean, action_std)

        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action).sum(1), probs.entropy().sum(1), self.critic(x)

# After collecting some data, you can plot it:
def plot_observations(corrected_altitude_log,altitude_rate_of_change_log,roll_log,pitch_log,yaw_log,distance_front_log,distance_back_log,collision_log,too_high_log):
    plt.figure(figsize=(15, 10))
    
    # Plot Corrected Altitude
    plt.subplot(3, 3, 1)
    plt.plot(corrected_altitude_log, label="Corrected Altitude")
    plt.xlabel("Timestep")
    plt.ylabel("Altitude (Normalized)")
    plt.legend()

    # Plot Altitude Rate of Change
    plt.subplot(3, 3, 2)
    plt.plot(altitude_rate_of_change_log, label="Altitude Rate of Change")
    plt.xlabel("Timestep")
    plt.ylabel("Altitude Rate of Change (Normalized)")
    plt.legend()

    # Plot Roll
    plt.subplot(3, 3, 3)
    plt.plot(roll_log, label="Roll")
    plt.xlabel("Timestep")
    plt.ylabel("Roll (Normalized)")
    plt.legend()

    # Plot Pitch
    plt.subplot(3, 3, 4)
    plt.plot(pitch_log, label="Pitch")
    plt.xlabel("Timestep")
    plt.ylabel("Pitch (Normalized)")
    plt.legend()

    # Plot Yaw
    plt.subplot(3, 3, 5)
    plt.plot(yaw_log, label="Yaw")
    plt.xlabel("Timestep")
    plt.ylabel("Yaw (Normalized)")
    plt.legend()

    # Plot Distance Front
    plt.subplot(3, 3, 6)
    plt.plot(distance_front_log, label="Distance Front")
    plt.xlabel("Timestep")
    plt.ylabel("Distance (Normalized)")
    plt.legend()

    # Plot Distance Back
    plt.subplot(3, 3, 7)
    plt.plot(distance_back_log, label="Distance Back")
    plt.xlabel("Timestep")
    plt.ylabel("Distance (Normalized)")
    plt.legend()

    # Plot Collision
    plt.subplot(3, 3, 8)
    plt.plot(collision_log, label="Collision", color='red')
    plt.xlabel("Timestep")
    plt.ylabel("Collision (0 or 1)")
    plt.legend()

    # Plot Too High
    plt.subplot(3, 3, 9)
    plt.plot(too_high_log, label="Too High", color='orange')
    plt.xlabel("Timestep")
    plt.ylabel("Too High (0 or 1)")
    plt.legend()

    # Show the plots
    plt.tight_layout()
    plt.show()


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

    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = args.sim_device
    print("using device:", device)

    # env setup
    envs, env_cfg = task_registry.make_env(name="quad_with_obstacles", args=args)

    envs = RecordEpisodeStatisticsTorch(envs, device)


    print("num actions: ",envs.num_actions)
    print("num obs: ", envs.num_obs)
    
    agent = Agent(envs).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)

    if args.play and args.checkpoint is None:
        raise ValueError("No checkpoint provided for testing.")

    if args.checkpoint is not None:
        print("Loading checkpoint...")
        checkpoint = torch.load(args.checkpoint)
        logstd = checkpoint["actor_logstd"]

        print("Log Standard Deviation Values (logstd):")
        print(logstd)

        std = torch.exp(logstd)
        print("Standard Deviation Values (std):")
        print(std)
        agent.load_state_dict(checkpoint)
        print("Loaded checkpoint")
        

    # ALGO Logic: Storage setup
    obs = torch.zeros((args.num_steps, args.num_envs, envs.num_obs), dtype=torch.float).to(device)
    actions = torch.zeros((args.num_steps, args.num_envs, envs.num_actions), dtype=torch.float).to(device)
    logprobs = torch.zeros((args.num_steps, args.num_envs), dtype=torch.float).to(device)
    rewards = torch.zeros((args.num_steps, args.num_envs), dtype=torch.float).to(device)
    dones = torch.zeros((args.num_steps, args.num_envs), dtype=torch.float).to(device)
    values = torch.zeros((args.num_steps, args.num_envs), dtype=torch.float).to(device)
    advantages = torch.zeros_like(rewards, dtype=torch.float).to(device)

    corrected_altitude_log = []
    altitude_rate_of_change_log = []
    roll_log = []
    pitch_log = []
    yaw_log = []
    distance_front_log = []
    distance_back_log = []
    collision_log = []
    too_high_log = []


    # TRY NOT TO MODIFY: start the game
    global_step = 0
    start_time = time.time()
    next_obs,_info = envs.reset()
    next_done = torch.zeros(args.num_envs, dtype=torch.float).to(device)
    num_updates = args.total_timesteps // args.batch_size
    num_updates = 6000
    if not args.play:
        for update in range(1, num_updates + 1):
            # Annealing the rate if instructed to do so.
            if args.anneal_lr:
                frac = 1.0 - (update - 1.0) / num_updates
                lrnow = frac * args.learning_rate
                optimizer.param_groups[0]["lr"] = lrnow

            for step in range(0, args.num_steps):
                depth_images, depth_values = process_depth_images(envs)
                #display_depth_images(depth_images)
                min_depths = [torch.amin(torch.tensor(depth, device=device), dim=1) for depth in depth_values]
                raw_altitude = min_depths[4].min(dim=-1).values.squeeze()
                corrected_altitude = next_obs[..., 0]
 
                if cv2.waitKey(1) & 0xFF == ord('q'):
                     break  
                global_step += 1 * args.num_envs
                obs[step] = next_obs
                dones[step] = next_done

                # ALGO LOGIC: action logic
                with torch.no_grad():
                    action, logprob, _, value = agent.get_action_and_value(next_obs)
                    values[step] = value.flatten()
                actions[step] = action
                logprobs[step] = logprob

                # TRY NOT TO MODIFY: execute the game and log data.
                next_obs, rewards[step], next_done, info = envs.step(action)

                ### SAVE FOR PLOTTING
                corrected_altitude_log.append(next_obs[0, 0].cpu().numpy())  # Log normalized corrected altitude
                altitude_rate_of_change_log.append(next_obs[0, 1].cpu().numpy())  # Log normalized altitude rate of change
                roll_log.append(next_obs[0, 2].cpu().numpy())  # Log normalized roll
                pitch_log.append(next_obs[0, 3].cpu().numpy())  # Log normalized pitch
                yaw_log.append(next_obs[0, 4].cpu().numpy())  # Log normalized yaw
                distance_front_log.append(next_obs[0, 9].cpu().numpy())  # Log normalized distance front
                distance_back_log.append(next_obs[0, 10].cpu().numpy())  # Log normalized distance back
                collision_log.append(next_obs[0, 11].cpu().numpy())  # Log collision data (binary)
                too_high_log.append(next_obs[0, 12].cpu().numpy())  # Log too high data (binary)
                #if update == 100:
                #    plot_observations(corrected_altitude_log,altitude_rate_of_change_log,roll_log,pitch_log,yaw_log,distance_front_log,distance_back_log,collision_log,too_high_log)
                
                if 0 <= step <= 2:
                    for idx, d in enumerate(next_done):
                        if d:
                            episodic_return = info["r"][idx].item()
                            episodic_length = info["l"][idx].item()
                            
                            # Skip logging for episodes with a length of 1
                            if episodic_length > 1:
                                print(f"global_step={global_step}, episodic_return={episodic_return}, episodic_length={episodic_length}")
                                writer.add_scalar("charts/episodic_return", episodic_return, global_step)
                                writer.add_scalar("charts/episodic_length", episodic_length, global_step)
                                
                                if "consecutive_successes" in info:  # ShadowHand and AllegroHand metric
                                    writer.add_scalar(
                                        "charts/consecutive_successes", info["consecutive_successes"].item(), global_step
                                    )

                            break


            # bootstrap value if not done
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

            # flatten the batch
            b_obs = obs.reshape((-1, envs.num_obs))
            b_logprobs = logprobs.reshape(-1)
            b_actions = actions.reshape((-1, envs.num_actions))
            b_advantages = advantages.reshape(-1)
            b_returns = returns.reshape(-1)
            b_values = values.reshape(-1)

            # Optimizing the policy and value network
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
                        # calculate approx_kl http://joschu.net/blog/kl-approx.html
                        old_approx_kl = (-logratio).mean()
                        approx_kl = ((ratio - 1) - logratio).mean()
                        clipfracs += [((ratio - 1.0).abs() > args.clip_coef).float().mean().item()]

                    mb_advantages = b_advantages[mb_inds]
                    if args.norm_adv:
                        mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                    # Policy loss
                    pg_loss1 = -mb_advantages * ratio
                    pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
                    pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                    # Value loss
                    newvalue = newvalue.view(-1)
                    if args.clip_vloss:
                        v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                        v_clipped = b_values[mb_inds] + torch.clamp(
                            newvalue - b_values[mb_inds],
                            -args.clip_coef,
                            args.clip_coef,
                        )
                        v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                        v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                        v_loss = 0.5 * v_loss_max.mean()
                    else:
                        v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                    entropy_loss = entropy.mean()
                    loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef

                    optimizer.zero_grad()
                    loss.backward()

                    nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)

                    optimizer.step()

                if args.target_kl is not None:
                    if approx_kl > args.target_kl:
                        break

            # TRY NOT TO MODIFY: record rewards for plotting purposes
            writer.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], global_step)
            writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
            writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
            writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
            writer.add_scalar("losses/old_approx_kl", old_approx_kl.item(), global_step)
            writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
            writer.add_scalar("losses/clipfrac", np.mean(clipfracs), global_step)
            print("SPS:", int(global_step / (time.time() - start_time)))
            writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)
            
            # save the model levery 50 updates
            if update % 50 == 0:
                print("Saving model.")
                torch.save(agent.state_dict(), f"runs/{run_name}/latest_model.pth")

    else:
        for step in range(0, 5000000):
            # ALGO LOGIC: action logic
            with torch.no_grad():
                action, logprob, _, value = agent.get_action_and_value(next_obs)
            next_obs, rewards, next_done, info = envs.step(action)


    # envs.close()
    writer.close()