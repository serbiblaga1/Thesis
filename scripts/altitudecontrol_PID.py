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
from controller_training import Agent

altitude_error_sum = 0.0
prev_altitude_error = 0.0

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
        'altitude_error_sum': altitude_error_sum,
        'prev_altitude_error': prev_altitude_error
    }

def compute_thrust(pid_params, current_altitude, dt):
    """
    Compute the thrust needed to reach the target altitude using PID control.

    :param pid_params: Dictionary containing PID parameters and state.
    :param current_altitude: Current altitude of the drone.
    :param dt: Time delta since the last computation.
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

    pid_params['prev_altitude_error'] = altitude_error
    pid_params['altitude_error_sum'] = altitude_error_sum

    thrust = max(-1.0, min(1.0, thrust))


    return thrust


def process_depth_images(env):
    depth_images = []
    depth_values = []
    
    for i in range(1, 6):
        depth = torch.clamp(getattr(env, f'full_camera_array{i}')[0], 0.0, 10.0)
        depth_np = depth.cpu().numpy() / 10.0
        depth_img = np.uint8(depth_np * 255)
        
        depth_images.append(depth_img)
        depth_values.append(depth_np)
    
    return depth_images, depth_values

def display_depth_images(depth_images):
    titles = ['depth_camera1', 'depth_camera2', 'depth_camera3', 'depth_camera4', 'depth_camera_down']
    for img, title in zip(depth_images, titles):
        cv2.imshow(title, img)

def simple_pid_takeoff_and_move(args):
    env_cfg = task_registry.get_cfgs(name=args.task)
    env_cfg.env.num_envs = min(env_cfg.env.num_envs, 50)
    env_cfg.control.controller = "lee_attitude_control" 
    
    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    
    env.reset()
    print("EPISODE LENGTH ", env.max_episode_length)

    target_altitude = 0.3
    initial_thrust = 0.05
    pid_gains = (0.2, 0.1, 0.05)
    pid_params = initialize_pid(target_altitude, pid_gains)

    all_images = []

    with open('depth_maps.txt', 'w') as file:
        pass

    takeoff_complete = False
    while not takeoff_complete:
        actions = torch.zeros(env.num_envs, 4, device=env.device)
        
        actions[:, 0] = initial_thrust

        obs, privileged_obs, rewards, resets, extras = env.step(actions.detach())
        
        depth_images, depth_values = process_depth_images(env)
        display_depth_images(depth_images)
        print_depth_maps_to_file('depth_maps.txt', *depth_values)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        
        current_altitude = depth_values[4].mean().item()
        print("Current Altitude during Takeoff:", current_altitude)

        if current_altitude >= target_altitude:
            takeoff_complete = True
            print("Takeoff complete. Switching to PID control.")

        
        all_images.append(depth_images)
    for step in range(1000 * int(env.max_episode_length)):
        depth_images, depth_values = process_depth_images(env)
        display_depth_images(depth_images)
        print_depth_maps_to_file('depth_maps.txt', *depth_values)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        dt = env_cfg.sim.dt 
        current_altitude = depth_values[4].mean().item()
        print("Current Altitude during Control:", current_altitude)
        thrust = compute_thrust(pid_params, current_altitude, dt)
        
        actions[:, 0] = thrust
        obs, privileged_obs, rewards, resets, extras = env.step(actions.detach())

        all_images.append(depth_images)

    save_all_combined_images(all_images, 'combined_depth_images_large.png')


if __name__ == '__main__':
    args = get_args()
    args.test = True
    args.checkpoint = None
    simple_pid_takeoff_and_move(args)

    