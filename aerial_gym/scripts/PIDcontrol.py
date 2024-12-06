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

class PIDController:
    def __init__(self, kp, ki, kd, target):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.target = target
        self.integral = 0.0
        self.previous_error = 0.0

    def compute(self, current_value, dt=0.01):
        error = self.target - current_value
        self.integral += error * dt
        derivative = (error - self.previous_error) / dt
        self.previous_error = error
        output = self.kp * error + self.ki * self.integral + self.kd * derivative
        output = max(-0.5, min(0.5, output))
        return output

def obstacle_avoidance(env):
    pid_controller_altitude = PIDController(kp=0.4, ki=0.1, kd=0.1, target=0.2)
    pid_controller_pitch = PIDController(kp=0.5, ki=0.2, kd=0.2, target=0.2)
    pid_controller_roll = PIDController(kp=0.3, ki=0.1, kd=0.1, target=0.2) 
    actions = torch.zeros(env.num_envs, 4, device=env.device)
    actions_yaw = torch.zeros(env.num_envs, 4, device=env.device)
    actions_pitch = torch.zeros(env.num_envs, 4, device=env.device)
    actions_roll = torch.zeros(env.num_envs, 4, device=env.device)
    actions_thrust = torch.zeros(env.num_envs, 4, device=env.device)
    start_actions = 0
    target_altitude = 0.2
    resets = 0
    detected = 0
    yaw_counter = 0 
    yawing = False
    stable_distance_counter = 0  
    previous_distance = None
    no_detection_sides = 0
    detected_front = 0
    detected_left = 0
    detected_right = 0
    reset_before = 0  

    for step in range(1000 * int(env.max_episode_length)):
        actions = torch.zeros(env.num_envs, 4, device=env.device)  
       
        depth_images, depth_values = process_depth_images(env)

        actions, adjusted_altitude = take_off(env, pid_controller_altitude, depth_values)
        obstacle_distance_right = depth_values[2].min().item()  
        obstacle_distance_left  = depth_values[3].min().item() 
        middle_4x4 = depth_values[0][2:6, 2:6]
        obstacle_distance = depth_values[0].min().item()

        detected_front = 0
        detected_left = 0
        detected_right = 0  

        if resets == 1 or reset_before == 1:
            print("resets")  
            start_actions = 0
            resets = 0
            yawing = False
            stable_distance_counter = 0
            previous_distance = None  
                # if no_detection_sides == 1:
                #        turning = 0
                #        while turning <= 40:
                #            actions_yaw = yaw_control(env, actions)
                #            actions_pitch, _ = pitch_command(env, actions, pid_controller_pitch, depth_values)
                #            actions_roll = roll_command(env, actions, pid_controller_roll, depth_values)
                #            actions_thrust, _ = take_off(env, pid_controller_altitude, depth_values)
                #            actions[:, 0] = actions_thrust[:, 0]
                #            actions[:, 1] = actions_roll[:, 1]
                #            actions[:, 2] = actions_pitch[:, 2]
                #            actions[:, 3] = actions_yaw[:, 3]
                #            turning += 1
                #            obs, privileged_obs, rewards, resets, extras = env.step(actions)
                #            if resets == 1:
                #               turning = 41
        stabilized = torch.allclose(torch.tensor(adjusted_altitude, device=env.device), 
                                    torch.tensor(target_altitude, device=env.device), atol=0.05)
        if stabilized:
            start_actions += 1

        if start_actions >= 50:
            turning = 0
            actions[:, 1] = 0
            actions[:, 2] = 0.01
            actions[:, 3] = 0

            if obstacle_distance < 0.4:
                actions, detected = pitch_command(env, actions, pid_controller_pitch, depth_values)
            if obstacle_distance_left <= 0.3 or obstacle_distance_right <= 0.3:
                actions = roll_command(env, actions, pid_controller_roll, depth_values)
                if obstacle_distance_left <= 0.3:
                    detected_left = 1
                    actions[:, 1] = -actions[:, 1]
                elif obstacle_distance_right <= 0.3:
                    detected_right = 1
            if detected_left == 0 and detected_right == 0:
                no_detection_sides = 1
            if obstacle_distance <= 0.2:
                detected_front = 1
            if detected_front == 1:
                if detected_right == 1:
                    while turning <= 40:
                        actions_yaw = yaw_control(env, actions)
                        actions_pitch, _ = pitch_command(env, actions, pid_controller_pitch, depth_values)
                        actions_roll = roll_command(env, actions, pid_controller_roll, depth_values)
                        actions_thrust, _ = take_off(env, pid_controller_altitude, depth_values)
                        actions[:, 0] = actions_thrust[:, 0]
                        actions[:, 1] = actions_roll[:, 1]
                        actions[:, 2] = actions_pitch[:, 2]
                        actions[:, 3] = actions_yaw[:, 3]
                        turning += 1
                        obs, privileged_obs, rewards, resets, extras = env.step(actions)
                        if resets == 1:
                            turning = 41
                if detected_left == 1 and detected_right == 0:
                    while turning <= 40:
                        actions_yaw = yaw_control(env, actions)
                        actions_pitch, _ = pitch_command(env, actions, pid_controller_pitch, depth_values)
                        actions_roll = roll_command(env, actions, pid_controller_roll, depth_values)
                        actions_thrust, _ = take_off(env, pid_controller_altitude, depth_values)
                        actions[:, 0] = actions_thrust[:, 0]
                        actions[:, 1] = actions_roll[:, 1]
                        actions[:, 2] = actions_pitch[:, 2]
                        actions[:, 3] = -actions_yaw[:, 3]
                        turning += 1
                        obs, privileged_obs, rewards, resets, extras = env.step(actions)
                        if resets == 1:
                            turning = 41
                # if no_detection_sides == 1:
                #        turning = 0
                #        while turning <= 40:
                #            actions_yaw = yaw_control(env, actions)
                #            actions_pitch, _ = pitch_command(env, actions, pid_controller_pitch, depth_values)
                #            actions_roll = roll_command(env, actions, pid_controller_roll, depth_values)
                #            actions_thrust, _ = take_off(env, pid_controller_altitude, depth_values)
                #            actions[:, 0] = actions_thrust[:, 0]
                #            actions[:, 1] = actions_roll[:, 1]
                #            actions[:, 2] = actions_pitch[:, 2]
                #            actions[:, 3] = actions_yaw[:, 3]
                #            turning += 1
                #            obs, privileged_obs, rewards, resets, extras = env.step(actions)
                #            if resets == 1:
                #               turning = 41
        if resets == 1:
            reset_before = 1
        else:
            reset_before = 0
        obs, privileged_obs, rewards, resets, extras, quats = env.step(actions)


def yaw_control(env, actions):
    yaw_rate = 0.2 

    actions[:, 3] = yaw_rate        
    
    return actions

def pitch_command(env, actions, pid_controller_pitch, depth_values):
    #middle_4x4 = depth_values[0][2:6, 2:6]
    obstacle_distance = depth_values[0].min().item() 
    detected = 0
    target_distance = pid_controller_pitch.target

    pitch_correction = pid_controller_pitch.compute(obstacle_distance)

    if obstacle_distance >= (target_distance - 0.05) and obstacle_distance <= (target_distance + 0.05):
        detected = 1
        actions[:, 2] = 0.0 
    else:
        actions[:, 2] = -pitch_correction  

    return actions, detected

def roll_command(env, actions, pid_controller_roll, depth_values):
    obstacle_distance_right = depth_values[2].min().item()  
    obstacle_distance_left  = depth_values[3].min().item() 
    roll_correction = 0
    if obstacle_distance_right <= 0.3:
        roll_correction = pid_controller_roll.compute(obstacle_distance_right)
    if obstacle_distance_left <= 0.3:
        roll_correction = pid_controller_roll.compute(obstacle_distance_right)
    actions[:, 1] = roll_correction
    
    return actions


def take_off(env, pid_controller, depth_values):
    actions = torch.zeros(env.num_envs, 4, device=env.device)
    safety_threshold = 0.2  
    ground_altitude_target = 0.2  
    obstacle_clearance = 0.1  
    yaw_direction = random.choice([-1, 1]) 

    ground_distance = depth_values[4].mean().item()  
    obstacle_distance = depth_values[2].mean().item()

    if obstacle_distance < safety_threshold: 
        target_altitude = max(ground_altitude_target, obstacle_distance + obstacle_clearance)
        target_altitude = ground_altitude_target
    
    pitch_angle = env.obs_buf[:, 0].mean().item() 
    roll_angle = env.obs_buf[:, 1].mean().item() 
    adjusted_altitude = ground_distance * math.cos(math.radians(pitch_angle)) * math.cos(math.radians(roll_angle))
    
    pid_thrust = pid_controller.compute(adjusted_altitude)
    
    actions[:, 0]  = pid_thrust
            
    return actions, adjusted_altitude


# ============================
# Main Entry Point
# ============================
if __name__ == "__main__":
    args = get_args()

    env_cfg = task_registry.get_cfgs(name=args.task)
    env_cfg.env.num_envs = min(env_cfg.env.num_envs, args.num_envs)
    env_cfg.control.controller = "lee_attitude_control"
    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)

    obstacle_avoidance(env)


