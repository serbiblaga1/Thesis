from aerial_gym import AERIAL_GYM_ROOT_DIR
import os

import isaacgym
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

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.linear_model import RANSACRegressor
from aerial_gym.envs.controllers.altitude_control import AltitudeStabilizationController

sys.path.append('/home/serbiblaga/workspaces/aerial_gym_simulator/aerial_gym/rl_training/cleanrl')

def initialize_pid(target_altitude, pid_gains):
    """
    Initialize PID control parameters.
    
    :param target_altitude: Desired altitude to reach and maintain.
    :param pid_gains: Gains for the PID controller (a tuple: (Kp, Ki, Kd)).
    :return: A dictionary containing initial PID parameters.
    """
    return {
        'target_altitude': target_altitude,
        'Kp': pid_gains[0],
        'Ki': pid_gains[1],
        'Kd': pid_gains[2],
        'altitude_error_sum': 0, 
        'prev_altitude_error': 0,  
        'prev_altitude': 0,  
        'cumulative_time': 0  
    }

def compute_thrust(pid_params, current_altitude, dt, max_rate=0.1, log_file='thrust_log.csv'):
    """
    Compute the thrust needed to reach the target altitude using PID control.
    Added an optional max_rate to limit ascent speed.
    
    :param pid_params: Dictionary containing PID parameters and state.
    :param current_altitude: Current altitude of the drone.
    :param dt: Time delta since the last computation.
    :param max_rate: Maximum allowed ascent rate (m/s).
    :return: Thrust to apply to stabilize altitude.
    """
    target_altitude = pid_params['target_altitude']
    Kp = pid_params['Kp']
    Ki = pid_params['Ki']
    Kd = pid_params['Kd']
    altitude_error_sum = pid_params['altitude_error_sum']
    prev_altitude_error = pid_params['prev_altitude_error']

    altitude_error = target_altitude - current_altitude
    altitude_error_sum += altitude_error * dt  
    altitude_error_derivative = (altitude_error - prev_altitude_error) / dt 

    thrust = (Kp * altitude_error + 
              Ki * altitude_error_sum + 
              Kd * altitude_error_derivative)

    if 'prev_altitude' in pid_params:
        current_rate = (current_altitude - pid_params['prev_altitude']) / dt
        if abs(current_rate) > max_rate:
            thrust *= max_rate / abs(current_rate)  

    thrust = np.clip(thrust, -1.0, 1.0)

    pid_params['prev_altitude'] = current_altitude
    pid_params['prev_altitude_error'] = altitude_error
    pid_params['altitude_error_sum'] = altitude_error_sum

    pid_params['cumulative_time'] += dt
    t = pid_params['cumulative_time']

    return thrust

def collect_imitation_data(env, num_timesteps=1000, target_altitude=0.3, pid_gains=(0.5, 0.1, 0.1), stabilization_timesteps=200):
    """
    Collect imitation data from the environment using a PID controller for takeoff and stabilization.

    :param env: The environment object (assumed to have `reset` and `step` methods).
    :param num_timesteps: Total number of timesteps for the entire data collection.
    :param target_altitude: The target altitude to reach and stabilize at.
    :param pid_gains: The PID gains (Kp, Ki, Kd) for controlling the drone's altitude.
    :param stabilization_timesteps: Number of timesteps to maintain the altitude after takeoff.
    :return: A list of tuples containing state-action pairs.
    """
    pid_params = initialize_pid(target_altitude, pid_gains)
    
    imitation_data = []
    states_list = []
    actions_list = []

    state, _ = env.reset() 
    takeoff_complete = False
    stabilized_timesteps = 0  
    for step in range(num_timesteps):
        actions = torch.zeros(env.num_envs, 4, device=env.device)

        if not takeoff_complete:
            current_altitude = state[:, 3].item() 
            dt = env_cfg.sim.dt  
            thrust = compute_thrust(pid_params, current_altitude, dt)  
            
            actions[:, 0] = thrust
            actions[:, 1] = 0
            actions[:, 2] = 0
            actions[:, 3] = 0

            obs, privileged_obs, rewards, resets, extras, quats = env.step(actions)

            state = obs 

            if isinstance(state, torch.Tensor):
                state = state.cpu().numpy()  
            if isinstance(actions, torch.Tensor):
                actions = actions.cpu().numpy() 

            imitation_data.append((tuple(state), tuple(actions)))
            states_list.append(tuple(state))
            actions_list.append(tuple(actions))
            current_altitude = state[:, 3].item() 
            if current_altitude >= target_altitude:
                takeoff_complete = True
                print(f"Takeoff complete! Current altitude: {current_altitude}")

        if takeoff_complete and stabilized_timesteps < stabilization_timesteps:
            current_altitude = state[:, 3].item()

            dt = env_cfg.sim.dt 
            thrust = compute_thrust(pid_params, current_altitude, dt) 
            #print(f"Thrust during stabilization: {thrust}")
            
            actions[:, 0] = thrust
            actions[:, 1] = 0
            actions[:, 2] = 0
            actions[:, 3] = 0

            obs, privileged_obs, rewards, resets, extras, quats = env.step(actions)

            state = obs 

            if isinstance(state, torch.Tensor):
                state = state.cpu().numpy() 
            if isinstance(actions, torch.Tensor):
                actions = actions.cpu().numpy() 
 
            imitation_data.append((tuple(state), tuple(actions)))
            states_list.append(tuple(state))
            actions_list.append(tuple(actions))
            stabilized_timesteps += 1

        if stabilized_timesteps >= stabilization_timesteps:
            print(f"Data collection complete. Stabilized for {stabilized_timesteps} timesteps.")
            break

    return imitation_data, states_list, actions_list

if __name__ == "__main__":
    args = get_args() 

    env_cfg = task_registry.get_cfgs(name=args.task)
    env_cfg.env.num_envs = min(env_cfg.env.num_envs, 50)
    env_cfg.control.controller = "lee_attitude_control"
    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)

    imitation_data, states_list, actions_list = collect_imitation_data(env, num_timesteps=1000, target_altitude=0.3)
    #imitation_data = [(np.array(state, dtype=np.float32), np.array(action, dtype=np.float32)) for state, action in imitation_data]

    print(imitation_data)
    with open('imitation_data.npy', 'wb') as f:
        np.save(f, imitation_data, allow_pickle=True)
    np.save('states.npy', states_list, allow_pickle=True)
    np.save('actions.npy', actions_list, allow_pickle=True)
    print(f"Collected {len(imitation_data)} state-action pairs.")
