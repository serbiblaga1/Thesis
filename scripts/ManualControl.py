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
        depth = torch.clamp(getattr(env, f'full_camera_array{i}')[0], 0.0, 4.0)
        
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
        
        close_mask = img < int(0.3 * 255)  
        grayscale_img[close_mask] = [0, 0, 255]  
        
        cv2.imshow(title, grayscale_img)

    cv2.waitKey(1)



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
    pid_controller_altitude = PIDController(kp=0.3, ki=0.0, kd=0.1, target=1.5)
    pid_controller_pitch = PIDController(kp=0.1, ki=0.0, kd=0.1, target=1)
    actions = torch.zeros(env.num_envs, 4, device=env.device)
    reached_altitude = False
    counter = 0
    stuck_threshold = 150  
    path = [0, 0]

    altitude_log = [] 
    step_log = []  
    for step in range(1000 * int(env.max_episode_length)):
        depth_images, depth_values = process_depth_images(env)
        display_depth_images(depth_images)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break  

        actions = torch.zeros(env.num_envs, 4, device=env.device)
        actions, reached_altitude = take_off(env, pid_controller_altitude, depth_values)

        current_altitude = depth_values[4].min().item()
        altitude_log.append(current_altitude)
        step_log.append(step)

        # Stop and plot after 200 steps
        # if step == 400:
        #     plt.figure(figsize=(10, 5))
        #     plt.plot(step_log, altitude_log, label="Altitude")
        #     plt.axhline(y=0.5, color='r', linestyle='--', label="Target Altitude")
        #     plt.xlabel("Step")
        #     plt.ylabel("Altitude (cm)")
        #     plt.title("Drone Altitude Over 200 Steps")
        #     plt.legend()
        #     plt.grid()
        #     plt.show()
        #     break

        if reached_altitude:
            pitch_value, detected = pitch_command(env, pid_controller_pitch, depth_values)
            actions[:, 2] = pitch_value  

            if pitch_value == 0:
                counter += 1
            else:
                counter = 0

            if counter >= stuck_threshold:
                check_right = False
                back_to_position = False
                for yaw_step in range(50):
                    depth_images, depth_values = process_depth_images(env)
                    display_depth_images(depth_images)
                    actions, reached_altitude = take_off(env, pid_controller_altitude, depth_values)
                    actions[:, 3] = 0.2 
                    env.step(actions)

                if depth_values[0].min().item() <= 0.3:
                    path = [0, 0]
                    check_right = True
                    print("Check right")
                else:
                    path = [1, 0]

                if check_right:
                    for yaw_step in range(100):
                        depth_images, depth_values = process_depth_images(env)
                        display_depth_images(depth_images)
                        actions, reached_altitude = take_off(env, pid_controller_altitude, depth_values)
                        actions[:, 3] = -0.2 
                        env.step(actions)

                    if depth_values[0].min().item() <= 0.3:
                        path = [0, 0]
                        back_to_position = True
                        print("Back to position")
                    else:
                        path = [0, 1]

                if back_to_position:
                    for yaw_step in range(50):
                        depth_images, depth_values = process_depth_images(env)
                        display_depth_images(depth_images)
                        actions, reached_altitude = take_off(env, pid_controller_altitude, depth_values)
                        actions[:, 3] = 0.2  
                        env.step(actions)
                    for yaw_step in range(200):
                        depth_images, depth_values = process_depth_images(env)
                        display_depth_images(depth_images)
                        actions, reached_altitude = take_off(env, pid_controller_altitude, depth_values)
                        actions[:, 3] = 0.2
                        env.step(actions)
                elif path == [1, 0]:
                    for yaw_step in range(50):
                        depth_images, depth_values = process_depth_images(env)
                        display_depth_images(depth_images)
                        actions, reached_altitude = take_off(env, pid_controller_altitude, depth_values)
                        actions[:, 3] = 0.2
                        env.step(actions)

        actions[:, 3] = 0 
        obs, privileged_obs, rewards, resets, extras = env.step(actions)




def bang_bang_control(front_sensor_reading, target_distance=0.2, pitch_step=0.05, dead_zone=0.2):
    error = abs(front_sensor_reading - target_distance)
    if error > dead_zone:  
        return pitch_step
    elif error < dead_zone:  
        return -pitch_step
    else:  
        return 0
    
def pitch_command(env, pid_controller_pitch, depth_values):
    obstacle_distance = depth_values[0].min().item() 
    detected = 0
    target_distance = pid_controller_pitch.target
    pitch_correction = pid_controller_pitch.compute(obstacle_distance)
    if obstacle_distance >= (target_distance - 0.05) and obstacle_distance <= (target_distance + 0.05):
        detected = 1
        pitch = 0.0 
    else:
        pitch = -pitch_correction  
    return pitch, detected


def take_off(env, pid_controller, depth_values):
    actions = torch.zeros(env.num_envs, 4, device=env.device)
    safety_threshold = 0.5  
    ground_altitude_target = 1.5  
    obstacle_clearance = 0.05  
    yaw_direction = random.choice([-1, 1])
    reached_altitude = False  

    ground_distance = depth_values[4].mean().item()
     
   # obstacle_distance = depth_values[2].mean().item()

    #if obstacle_distance < safety_threshold: 
    #    target_altitude = max(ground_altitude_target, obstacle_distance + obstacle_clearance)
    #    target_altitude = ground_altitude_target
    
    #pitch_angle = env.obs_buf[:, 0].mean().item() 
    #roll_angle = env.obs_buf[:, 1].mean().item() 
    #adjusted_altitude = ground_distance * math.cos(math.radians(pitch_angle)) * math.cos(math.radians(roll_angle))
    
    pid_thrust = pid_controller.compute(ground_distance)
    
    actions[:, 0]  = pid_thrust

    if ground_altitude_target - 0.1 <= ground_distance <= ground_altitude_target + 0.1:
        reached_altitude = True
            
    return actions, reached_altitude


if __name__ == "__main__":
    args = get_args()

    env_cfg = task_registry.get_cfgs(name=args.task)
    env_cfg.env.num_envs = min(env_cfg.env.num_envs, args.num_envs)
    env_cfg.control.controller = "lee_attitude_control"
    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)

    obstacle_avoidance(env)


