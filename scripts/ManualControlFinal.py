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

import numpy as np
import cv2
import torch

def add_gaussian_noise(array, noise_level=0.05):

    noise = np.random.normal(0, noise_level, array.shape)
    noisy_array = array + noise
    return np.clip(noisy_array, 0.0, 4.0)  

def process_depth_images(env, save_dir="tof_readings", step=0, noise_level=0.05):
    depth_images = []
    depth_values = []
    camera_name = ["front", "back", "right", "left", "down"]
    save_cameras = {"front": 0, "right": 2, "left": 3}

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    for i in range(1, 6):
        depth = torch.clamp(getattr(env, f'full_camera_array{i}')[0], 0.0, 4.0)
        depth_np = depth.cpu().numpy()

        noisy_depth_np = add_gaussian_noise(depth_np, noise_level)

        depth_img = np.uint8((noisy_depth_np / 4.0) * 255)
        depth_images.append(depth_img)

        depth_values.append(noisy_depth_np)

        if i - 1 in save_cameras.values():
            normalized_depth = noisy_depth_np / 4.0
            file_path = os.path.join(save_dir, f"{camera_name[i - 1]}_step_{step:05d}.npy")
            np.save(file_path, normalized_depth)

    left_matrix = depth_values[3]  
    right_matrix = depth_values[2]
    front_matrix = depth_values[0]

    # print("\nFront (F), Left (L) and Right (R) Depth Matrices (Side by Side):")
    # for front_row, left_row, right_row in zip(front_matrix, left_matrix, right_matrix):
    #     print(
    #         "F: " + ", ".join(f"{val:4.2f}" for val in front_row) + "   ||   " +
    #         "L: " + ", ".join(f"{val:4.2f}" for val in left_row) + "   ||   " +
    #         "R: " + ", ".join(f"{val:4.2f}" for val in right_row)
    #     )

    return depth_images, depth_values

def display_depth_images(depth_images):
    titles = ['Front', 'Left', 'Right', 'Back', 'Down']
    
    for img, title in zip(depth_images, titles):
        grayscale_img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        half_meter_threshold = int((0.5 / 4.0) * 255)
        close_mask = img <= half_meter_threshold
        grayscale_img[close_mask] = [0, 0, 255]
        cv2.imshow(title, grayscale_img)

    cv2.waitKey(1)

def calculate_performance_metrics(altitudes, final_value, tolerance=0.02):
    # 1. Calculate **Rise Time**
    # Find the time when altitude reaches 10% and 90% of the final value
    y_10 = 0.1 * final_value
    y_90 = 0.9 * final_value

    rise_time_start = next(i for i, v in enumerate(altitudes) if v >= y_10)
    rise_time_end = next(i for i, v in enumerate(altitudes) if v >= y_90)
    rise_time = rise_time_end - rise_time_start

    # 2. Calculate **Settling Time**
    # Settling time is when the altitude stays within tolerance (2%) of the final value
    settling_time = next(i for i, v in enumerate(altitudes) if abs(v - final_value) <= tolerance * final_value)

    # 3. Calculate **Overshoot**
    overshoot = (max(altitudes) - final_value) / final_value * 100  # percentage overshoot

    # 4. Calculate **Peak Time**
    peak_time = altitudes.index(max(altitudes))  # Time (step) of the first peak

    # Return all the metrics
    return rise_time, settling_time, overshoot, peak_time

def add_roll_drift(actions, step, drift_magnitude=0.0001, drift_update_steps=1000):
    if not hasattr(add_roll_drift, "roll_drift"):
        add_roll_drift.roll_drift = np.random.uniform(-drift_magnitude, drift_magnitude)

    if step % drift_update_steps == 0:
        add_roll_drift.roll_drift = np.random.uniform(-drift_magnitude, drift_magnitude)

    actions[:, 1] += add_roll_drift.roll_drift
    return actions


class PIDController:
    def __init__(self, kp, ki, kd, target):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.target = target
        self.integral = 0.0
        self.previous_error = 0.0

    def compute(self, current_value, dt=0.01, deadband = 0.05):
        error = self.target - current_value
        if abs(error) < deadband:  
            error = 0.0
        self.integral += error * dt
        derivative = (error - self.previous_error) / dt
        self.previous_error = error
        output = self.kp * error + self.ki * self.integral + self.kd * derivative
        output = max(-0.5, min(0.5, output))
        return output
    
    def reset_integral(self):
        """Resets the integral term to prevent accumulation of error."""
        self.integral = 0.0


def initialize_controllers():
    pid_controller_altitude = PIDController(kp=0.007, ki=0.0, kd=0.005, target=1.5)
    pid_controller_pitch = PIDController(kp=0.015, ki=0.0, kd=0.006, target=0.6)
    pid_controller_roll  = PIDController(kp=0.01, ki=0.0, kd=0.006, target=0.6)
    return pid_controller_altitude, pid_controller_pitch, pid_controller_roll

def process_step(env, actions, pid_controller_altitude, depth_values, observations):
    actions = torch.zeros(env.num_envs, 4, device=env.device)
    actions, reached_altitude = take_off(env, pid_controller_altitude, depth_values, observations)
    return actions, reached_altitude

def check_obstacle_direction(env, actions, pid_controller_altitude, pid_controller_roll, pid_controller_pitch, depth_values, observations, yaw_step_range, yaw_direction, step):
    for yaw_step in range(yaw_step_range):
        depth_images, depth_values = process_depth_images(env, save_dir="vae_tof_data", step=step)
        display_depth_images(depth_images)
        actions, _ = take_off(env, pid_controller_altitude, depth_values, observations)
        actions[:, 3] = yaw_direction
        actions = add_roll_drift(actions, yaw_step)
        if 0.5 <= depth_values[3].min().item() <= 0.7:
            actions[:, 1] += pid_controller_roll.compute(depth_values[3].min().item())
        elif 0.5 <= depth_values[2].min().item() <= 0.7:
            actions[:, 1] += -pid_controller_roll.compute(depth_values[2].min().item())
        print(actions)
        observations, privileged_obs, reward_buf, reset_buf, extras = env.step(actions)
        step += 1

    return depth_values[0].min().item() <= 0.6, step

def handle_obstacle_checks(env, actions, pid_controller_altitude, pid_controller_roll, pid_controller_pitch, depth_values, observations, step):
    print("Position hold activated. Staying in place.")
    obstacle_detected_left, step = check_obstacle_direction(env, actions, pid_controller_altitude, pid_controller_roll, pid_controller_pitch, depth_values, observations, 50, 0.2, step)
    go_back_to_position, step = check_obstacle_direction(env, actions, pid_controller_altitude, pid_controller_roll, pid_controller_pitch, depth_values, observations, 50, -0.2, step)
    obstacle_detected_right, step = check_obstacle_direction(env, actions, pid_controller_altitude, pid_controller_roll, pid_controller_pitch, depth_values, observations, 50, -0.2, step)
    go_back_to_position, step = check_obstacle_direction(env, actions, pid_controller_altitude, pid_controller_roll, pid_controller_pitch, depth_values, observations, 50, 0.2, step)
    
    if obstacle_detected_left and obstacle_detected_right:
        return [1, 1], step
    elif obstacle_detected_left:
        return [1, 0], step
    elif obstacle_detected_right:
        return [0, 1], step
    else:
        return [0, 0], step

def execute_path(env, actions, pid_controller_altitude, pid_controller_roll, pid_controller_pitch, depth_values, observations, path, checked, step):
    if path == [0, 0] and checked:
        print("No obstacles left or right, try left.")
        _, step = check_obstacle_direction(env, actions, pid_controller_altitude, pid_controller_roll, pid_controller_pitch, depth_values, observations, 100, 0.2, step)
    elif path == [0, 1] and checked:
        print("Obstacle in the right, trying left.")
        _, step = check_obstacle_direction(env, actions, pid_controller_altitude, pid_controller_roll, pid_controller_pitch, depth_values, observations, 100, 0.2, step)
    elif path == [1, 0] and checked:
        print("Obstacle in the left, trying right.")
        _, step = check_obstacle_direction(env, actions, pid_controller_altitude, pid_controller_roll, pid_controller_pitch, depth_values, observations, 100, -0.2, step)
    elif path == [1, 1] and checked:
        print("Obstacle in both directions, turn around.")
        _, step = check_obstacle_direction(env, actions, pid_controller_altitude, pid_controller_roll, pid_controller_pitch, depth_values, observations, 200, 0.2, step)
    return step


def preprocess_calculate_drift(env, actions, pid_pitch, pid_roll, depth_values, target_distance_front, dt=0.1):
    print("Starting pre-processing to calculate drift...")
    drift_samples = []
    while True:
        front_distance = depth_values[0].min().item()
        left_distance = depth_values[3].min().item()
        right_distance = depth_values[2].min().item()

        pitch_error = target_distance_front - front_distance
        pitch_correction = pid_pitch.compute(pitch_error, dt)
        actions[:, 2] = pitch_correction

        roll_error = right_distance - left_distance
        drift_samples.append(roll_error)

        roll_correction = pid_roll.compute(roll_error, dt)
        actions[:, 1] += roll_correction

        observations, privileged_obs, reward_buf, reset_buf, extras = env.step(actions)

        if abs(pitch_error) < 0.05:
            break

    calculated_drift = np.mean(drift_samples)
    print(f"Calculated drift: {calculated_drift:.4f}")
    return calculated_drift


def obstacle_avoidance(env):
    pid_controller_altitude, pid_controller_pitch, pid_controller_roll = initialize_controllers()
    actions = torch.zeros(env.num_envs, 4, device=env.device)
    reached_altitude = False
    stay_counter = 0
    initial_target_distance_front = 0.6
    target_distance_front = 1.0
    target_distance_left  = 0.6
    target_distance_right = 0.6

    save_dir = "vae_tof_data"
    num_steps_to_save = 1000

    stay_steps_required = 50
    observations = torch.zeros(1, 8, device=env.device)
    path = [0, 0]
    checked = False
    altitude_log, step_log = [], []

    # Store altitude data for performance metrics
    altitudes = []

    for step in range(500):
        depth_images, depth_values = process_depth_images(env, save_dir=save_dir, step=step)
        display_depth_images(depth_images)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break  

        actions, reached_altitude = process_step(env, actions, pid_controller_altitude, depth_values, observations)
        actions = add_roll_drift(actions, step)

        current_altitude = depth_values[4].min().item()
        if step > 1:
            altitudes.append(current_altitude)  # Log the altitude for performance metrics

        current_distance = depth_values[0].min().item()
        current_distance_left  = depth_values[3].min().item()
        current_distance_right = depth_values[2].min().item()

        if target_distance_left - 0.1 <= current_distance_left <= target_distance_left + 0.1:
            actions[:, 1] += pid_controller_roll.compute(current_distance_left)
        elif target_distance_right - 0.1 <= current_distance_right <= target_distance_right + 0.1:
            actions[:, 1] += -pid_controller_roll.compute(current_distance_right)
    

        altitude_log.append(current_altitude)
        step_log.append(step)

        if reached_altitude:
            pitch_value = -pid_controller_pitch.compute(depth_values[0].min().item())
            actions[:, 2] = pitch_value
            actions = add_roll_drift(actions, step)

            if depth_values[0].min().item() <= initial_target_distance_front:
                pid_controller_pitch.target = target_distance_front

            if target_distance_front - 0.1 <= current_distance <= target_distance_front + 0.1:
                stay_counter += 1
                actions[:, 2] = 0
                actions = add_roll_drift(actions, step)
                if target_distance_left - 0.1 <= current_distance_left <= target_distance_left + 0.1:
                    actions[:, 1] += pid_controller_roll.compute(current_distance_left)
                elif target_distance_right - 0.1 <= current_distance_right <= target_distance_right + 0.1:
                    actions[:, 1] += -pid_controller_roll.compute(current_distance_right)
                if stay_counter >= stay_steps_required:
                    stay_counter = 0
                    path, step = handle_obstacle_checks(env, actions, pid_controller_altitude, pid_controller_roll, pid_controller_pitch, depth_values, observations, step)
                    checked = True
                    pid_controller_pitch.target = initial_target_distance_front

            else:
                stay_counter = 0

            step =  execute_path(env, actions, pid_controller_altitude, pid_controller_roll, pid_controller_pitch, depth_values, observations, path, checked, step)
            checked = False
        print(step)
        actions[:, 3] = 0
        observations, privileged_obs, reward_buf, reset_buf, extras = env.step(actions)

    # Now, calculate the performance metrics based on the logged altitudes
    final_value = 1.5  # The target altitude (or the final desired altitude)
    rise_time, settling_time, overshoot, peak_time = calculate_performance_metrics(altitudes, final_value)

    # Output the calculated performance metrics
    print(f"Rise Time: {rise_time} steps")
    print(f"Settling Time: {settling_time} steps")
    print(f"Overshoot: {overshoot:.2f}%")
    print(f"Peak Time: {peak_time} steps")

    # Optionally, you can also plot the altitude data
    plt.plot(altitudes)
    plt.title("Altitude Over Time")
    plt.xlabel("Time Step")
    plt.ylabel("Altitude")
    plt.show()

def obstacle_avoidance_good(env):
    pid_controller_altitude, pid_controller_pitch, pid_controller_roll = initialize_controllers()
    actions = torch.zeros(env.num_envs, 4, device=env.device)
    reached_altitude = False
    stay_counter = 0
    initial_target_distance_front = 0.6
    target_distance_front = 1.0
    target_distance_left  = 0.6
    target_distance_right = 0.6

    save_dir = "vae_tof_data"
    num_steps_to_save = 1000

    stay_steps_required = 50
    observations = torch.zeros(1, 8, device=env.device)
    path = [0, 0]
    checked = False
    altitude_log, step_log = [], []

    for step in range(100000 * int(env.max_episode_length)):
        depth_images, depth_values = process_depth_images(env, save_dir=save_dir, step=step)
        display_depth_images(depth_images)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break  

        actions, reached_altitude = process_step(env, actions, pid_controller_altitude, depth_values, observations)
        actions = add_roll_drift(actions, step)

        current_altitude = depth_values[4].min().item()
        current_distance = depth_values[0].min().item()
        current_distance_left  = depth_values[3].min().item()
        current_distance_right = depth_values[2].min().item()

        if target_distance_left - 0.1 <= current_distance_left <= target_distance_left + 0.1:
            actions[:, 1] += pid_controller_roll.compute(current_distance_left)
        elif target_distance_right - 0.1 <= current_distance_right <= target_distance_right + 0.1:
            actions[:, 1] += -pid_controller_roll.compute(current_distance_right)
    

        altitude_log.append(current_altitude)
        step_log.append(step)

        if reached_altitude:
            pitch_value = -pid_controller_pitch.compute(depth_values[0].min().item())
            #pitch_value, detected = pitch_command(env, pid_controller_pitch, depth_values)

            if target_distance_left - 0.1 <= current_distance_left <= target_distance_left + 0.1:
                actions[:, 1] += pid_controller_roll.compute(current_distance_left)
            elif target_distance_right - 0.1 <= current_distance_right <= target_distance_right + 0.1:
                actions[:, 1] += -pid_controller_roll.compute(current_distance_right)

            actions[:, 2] = pitch_value
            actions = add_roll_drift(actions, step)

            if depth_values[0].min().item() <= initial_target_distance_front:
                pid_controller_pitch.target = target_distance_front

            if target_distance_front - 0.1 <= current_distance <= target_distance_front + 0.1:
                stay_counter += 1
                actions[:, 2] = 0
                actions = add_roll_drift(actions, step)
                if target_distance_left - 0.1 <= current_distance_left <= target_distance_left + 0.1:
                    actions[:, 1] += pid_controller_roll.compute(current_distance_left)
                elif target_distance_right - 0.1 <= current_distance_right <= target_distance_right + 0.1:
                    actions[:, 1] += -pid_controller_roll.compute(current_distance_right)
                if stay_counter >= stay_steps_required:
                    stay_counter = 0
                    path, step = handle_obstacle_checks(env, actions, pid_controller_altitude, pid_controller_roll, pid_controller_pitch, depth_values, observations, step)
                    checked = True
                    pid_controller_pitch.target = initial_target_distance_front

            else:
                stay_counter = 0

            step =  execute_path(env, actions, pid_controller_altitude, pid_controller_roll, pid_controller_pitch, depth_values, observations, path, checked, step)
            checked = False
        print(step)
        actions[:, 3] = 0
        observations, privileged_obs, reward_buf, reset_buf, extras = env.step(actions)



    
def pitch_command(env, pid_controller_pitch, depth_values):
    obstacle_distance = depth_values[0].min().item()
    detected = 0
    target_distance = pid_controller_pitch.target
    pitch_correction = pid_controller_pitch.compute(obstacle_distance)

    pitch = -pitch_correction  
    return pitch, detected


def take_off(env, pid_controller, depth_values, observations):
    actions = torch.zeros(env.num_envs, 4, device=env.device)
    safety_threshold = 0.5  
    ground_altitude_target = 1.5  
    obstacle_clearance = 0.05  
    yaw_direction = random.choice([-1, 1])
    reached_altitude = False  

    ground_distance = depth_values[4].mean().item()
    horizontal_distance = torch.cos(observations[0, 2]) * ground_distance
    
    pid_thrust = pid_controller.compute(horizontal_distance)
    
    actions[:, 0]  = pid_thrust

    if ground_altitude_target - 0.1 <= horizontal_distance <= ground_altitude_target + 0.1:
        reached_altitude = True
            
    return actions, reached_altitude


if __name__ == "__main__":
    args = get_args()

    env_cfg = task_registry.get_cfgs(name=args.task)
    env_cfg.env.num_envs = min(env_cfg.env.num_envs, args.num_envs)
    env_cfg.control.controller = "lee_attitude_control"
    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)

    obstacle_avoidance(env)


