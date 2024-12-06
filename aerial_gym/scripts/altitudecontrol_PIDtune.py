from aerial_gym import AERIAL_GYM_ROOT_DIR
import os
import math
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
from scipy.optimize import minimize

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
    pid_params['prev_altitude'] = current_altitude

    thrust = max(-1.0, min(1.0, thrust))

    pid_params['prev_altitude_error'] = altitude_error
    pid_params['altitude_error_sum'] = altitude_error_sum

    if 'cumulative_time' not in pid_params:
        pid_params['cumulative_time'] = 0
    pid_params['cumulative_time'] += dt
    t = pid_params['cumulative_time']

    with open(log_file, 'a') as file:
        file.write(f"{t},{thrust},{altitude_error},{current_altitude}\n")

    return thrust

def initialize_pitch_pid(target_distance, pid_gains):
    return {
        'target_distance': target_distance,
        'Kp': pid_gains[0],
        'Ki': pid_gains[1],
        'Kd': pid_gains[2],
        'distance_error_sum': 0.0,
        'prev_distance_error': 0.0
    }

def compute_pitch_correction(pid_params, current_distance, dt, log_file='pitch_log.csv'):
    target_distance = pid_params['target_distance']
    Kp = pid_params['Kp']
    Ki = pid_params['Ki']
    Kd = pid_params['Kd']
    distance_error_sum = pid_params['distance_error_sum']
    prev_distance_error = pid_params['prev_distance_error']

    distance_error = target_distance - current_distance
    distance_error_sum += distance_error * dt
    distance_error_derivative = (distance_error - prev_distance_error) / dt

    pitch_correction = (Kp * distance_error +
                        Ki * distance_error_sum +
                        Kd * distance_error_derivative)

    pid_params['prev_distance_error'] = distance_error
    pid_params['distance_error_sum'] = distance_error_sum

    if 'cumulative_time' not in pid_params:
        pid_params['cumulative_time'] = 0
    pid_params['cumulative_time'] += dt
    t = pid_params['cumulative_time']

    with open(log_file, 'a') as file:
        file.write(f"{t},{max(-1.0, min(1.0, pitch_correction))},{distance_error},{current_distance}\n")

    return max(-1.0, min(1.0, pitch_correction)), distance_error

def initialize_roll_pid(target_distance, pid_gains):
    return {
        'target_distance': target_distance,
        'Kp': pid_gains[0],
        'Ki': pid_gains[1],
        'Kd': pid_gains[2],
        'distance_error_sum': 0.0,
        'prev_distance_error': 0.0
    }

def compute_roll_correction(pid_params, current_distance, dt, log_file='roll_log.csv'):
    target_distance = pid_params['target_distance']
    Kp = pid_params['Kp']
    Ki = pid_params['Ki']
    Kd = pid_params['Kd']
    distance_error_sum = pid_params['distance_error_sum']
    prev_distance_error = pid_params['prev_distance_error']

    distance_error = target_distance - current_distance
    distance_error_sum += distance_error * dt
    distance_error_derivative = (distance_error - prev_distance_error) / dt

    roll_correction = (Kp * distance_error +
                       Ki * distance_error_sum +
                       Kd * distance_error_derivative)

    pid_params['prev_distance_error'] = distance_error
    pid_params['distance_error_sum'] = distance_error_sum

    if 'cumulative_time' not in pid_params:
        pid_params['cumulative_time'] = 0
    pid_params['cumulative_time'] += dt
    t = pid_params['cumulative_time']

    with open(log_file, 'a') as file:
        file.write(f"{t},{max(-1.0, min(1.0, roll_correction))},{distance_error},{current_distance}\n")

    return max(-1.0, min(1.0, roll_correction)), distance_error


def compute_roll_correction_stabilization(pid_params, left_distance, right_distance, dt):
    current_distance = (left_distance - right_distance) / 2 
    target_distance = pid_params['target_distance']
    Kp = pid_params['Kp']
    Ki = pid_params['Ki']
    Kd = pid_params['Kd']
    distance_error_sum = pid_params['distance_error_sum']
    prev_distance_error = pid_params['prev_distance_error']

    distance_error = target_distance - abs(current_distance)
    distance_error_sum += distance_error * dt
    distance_error_derivative = (distance_error - prev_distance_error) / dt

    roll_correction = (Kp * distance_error +
                       Ki * distance_error_sum +
                       Kd * distance_error_derivative)

    pid_params['prev_distance_error'] = distance_error
    pid_params['distance_error_sum'] = distance_error_sum

    return max(-1.0, min(1.0, roll_correction)), distance_error


def initialize_yaw_pid(yaw_gains):
    return {
        'Kp': yaw_gains[0],
        'Ki': yaw_gains[1],
        'Kd': yaw_gains[2],
        'yaw_error_sum': 0.0,
        'prev_yaw_error': 0.0
    }

def compute_yaw_correction(yaw_pid_params, target_yaw, current_yaw, dt):
    Kp = yaw_pid_params['Kp']
    Ki = yaw_pid_params['Ki']
    Kd = yaw_pid_params['Kd']
    yaw_error_sum = yaw_pid_params['yaw_error_sum']
    prev_yaw_error = yaw_pid_params['prev_yaw_error']

    yaw_error = target_yaw - current_yaw
    yaw_error_sum += yaw_error * dt
    yaw_error_derivative = (yaw_error - prev_yaw_error) / dt

    yaw_correction = (Kp * yaw_error +
                      Ki * yaw_error_sum +
                      Kd * yaw_error_derivative)

    yaw_pid_params['prev_yaw_error'] = yaw_error
    yaw_pid_params['yaw_error_sum'] = yaw_error_sum
    
    return max(-1.0, min(1.0, yaw_correction)), yaw_error

def normalize_angle(angle):
    return (angle + np.pi) % (2 * np.pi) - np.pi


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

def objective_function(pid_params, target_altitude, env, sim_steps=100, threshold=1e-6):
    Kp, Ki, Kd = pid_params
    print(f"Testing PID: Kp={Kp}, Ki={Ki}, Kd={Kd}")

    pid_params_dict = initialize_pid(target_altitude, (Kp, Ki, Kd))
    total_error = 0.0
    current_altitude = 0.0
    dt = 0.01
    prev_error = None

    for step in range(sim_steps):
        thrust = compute_thrust(pid_params_dict, current_altitude, dt)
        actions = torch.zeros(env.num_envs, 4, device=env.device)
        actions[:, 0] = thrust

        obs, privileged_obs, rewards, resets, extras = env.step(actions.detach())
        depth_images, depth_values = process_depth_images(env)
        current_altitude = depth_values[4].mean().item()

        altitude_error = target_altitude - current_altitude
        total_error += altitude_error ** 2

        if prev_error is not None and abs(total_error - prev_error) < threshold:
            print("Convergence threshold reached. Exiting optimization early.")
            break

        prev_error = total_error

    print(f"Total error for Kp={Kp}, Ki={Ki}, Kd={Kd}: {total_error}")
    return total_error

def quaternion_to_yaw(quaternion):
    x, y, z, w = quaternion[:, 0], quaternion[:, 1], quaternion[:, 2], quaternion[:, 3]
    
    yaw = torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y**2 + z**2))
    
    return yaw

def optimize_pid(target_altitude, env):
    bounds = [(0, 2), (0, 1), (0, 1)]

    initial_guess = [0.5, 0.1, 0.1]

    result = minimize(objective_function, initial_guess,
                      args=(target_altitude, env),
                      bounds=bounds,
                      method='Nelder-Mead',
                      options={'disp': True})
    print("RESULT ", result)
    Kp_opt, Ki_opt, Kd_opt = result.x
    print(f"Optimized PID parameters: Kp={Kp_opt}, Ki={Ki_opt}, Kd={Kd_opt}")

    return Kp_opt, Ki_opt, Kd_opt


def simple_pid_takeoff_and_move(args):

    ############################### INITIALIZE ########################################

    env_cfg = task_registry.get_cfgs(name=args.task)
    env_cfg.env.num_envs = min(env_cfg.env.num_envs, 50)
    env_cfg.control.controller = "lee_attitude_control"
    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    env.reset()
    all_images = []
    target_altitude = 0.3
    Kp_opt = 0.5
    Ki_opt = 0.1
    Kd_opt = 0.1
    pid_params = initialize_pid(target_altitude, (Kp_opt, Ki_opt, Kd_opt))
    
    pid_params['prev_altitude'] = 0.0

    cruise_thrust = 0

    target_distance_from_wall = 0.2
    pitch_pid_params = initialize_pitch_pid(target_distance_from_wall, (0.5, 0.05, 0.05))

    yaw_target_angle = math.radians(90) 
    current_yaw_angle = 0.0  
    yaw_pid_params = initialize_yaw_pid([0.1, 0.01, 0.05])  

    target_distance_from_wall_side = 0.2  
    roll_pid_params = initialize_roll_pid(target_distance_from_wall_side, (0.2, 0.05, 0.05))

    takeoff_complete = False
    has_stopped_near_wall = False
    currently_yawing = False
    yaw_complete = False
    reached_obstacle = False
    yaw_target_angle = np.radians(90) 
    dt = env_cfg.sim.dt
    alternating_pitch = False 
    last_pitch_correction = 0
    yaw_torque = 0.5

    for step in range(1000 * int(env.max_episode_length)):
    ############################### TAKE-OFF #########################################        
        while not takeoff_complete:
            depth_images, depth_values = process_depth_images(env)
            actions = torch.zeros(env.num_envs, 4, device=env.device)
            current_altitude = depth_values[4].mean().item()
            thrust = compute_thrust(pid_params, current_altitude, dt, max_rate=0.1) 
            actions[:, 0] = thrust

            obs, privileged_obs, rewards, resets, extras, quats = env.step(actions.detach())
            depth_images, depth_values = process_depth_images(env)
            display_depth_images(depth_images)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break        
            current_altitude = depth_values[4].mean().item()
            if current_altitude >= target_altitude:
                takeoff_complete = True
                cruise_thrust = thrust
                
    ############################### MOVE #############################################
    
        depth_images, depth_values = process_depth_images(env)
        display_depth_images(depth_images)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break        

        if takeoff_complete:
            current_altitude = depth_values[4].mean().item()
            thrust = compute_thrust(pid_params, current_altitude, dt, max_rate=0.1) 
            actions[:, 0] = thrust
            ##### PITCH #####
            current_distance_to_wall = depth_values[0].mean().item()
            pitch_correction, distance_error_pitch = compute_pitch_correction(pitch_pid_params, current_distance_to_wall, dt)
            actions[:, 2] = -pitch_correction 

            ##### ROLL #####
            current_distance_side = depth_values[2].mean().item()
            roll_correction, _ = compute_roll_correction(roll_pid_params, current_distance_side, dt)
            if current_distance_side <= target_distance_from_wall_side:
                actions[:, 1] = -roll_correction

            if current_distance_to_wall <= target_distance_from_wall:
                reached_obstacle = True

            ###### YAW ######
            if reached_obstacle:
                while not stop_yawing and resets == 0:
                    depth_images, depth_values = process_depth_images(env)
                    current_distance_to_wall = depth_values[0].mean().item()
                    pitch_correction, distance_error_pitch = compute_pitch_correction(pitch_pid_params, current_distance_to_wall, dt)

                    actions[:, 3] = yaw_torque
                    actions[:, 2] = -pitch_correction 

                    current_altitude = depth_values[4].mean().item()
                    thrust = compute_thrust(pid_params, current_altitude, dt, max_rate=0.1) 
                    current_distance_side = depth_values[2].mean().item()
                    roll_correction, _ = compute_roll_correction(roll_pid_params, current_distance_side, dt)

                   # print(current_distance_to_wall)
                    
                    if current_distance_side <= target_distance_from_wall_side:
                        actions[:, 1] = -roll_correction
                    if current_distance_side == target_distance_from_wall:
                        actions[:, 3] = 0
                        stop_yawing = True
                        reached_obstacle = False
                        print(" yaw stop ")
                    actions[:, 0] = thrust
                    obs, privileged_obs, rewards, resets, extras, quats = env.step(actions.detach())
            
            reached_obstacle = False
            stop_yawing = False
            
            obs, privileged_obs, rewards, resets, extras, quats = env.step(actions.detach())

            
            if resets == 1:
                    takeoff_complete = False
                    stop_yawing = False
                    reached_obstacle = False
                    has_stopped_near_wall = False
                    yaw_complete = False
                    currently_yawing = False


            all_images.append(depth_images)            
            
            # if not has_stopped_near_wall:
            #     actions[:, 2] = -pitch_correction 
            #     last_pitch_correction = pitch_correction 
            #     if abs(distance_error_pitch) < 0.005:
            #         has_stopped_near_wall = True 

            # if takeoff_complete and has_stopped_near_wall and not yaw_complete:
            #         #print(current_yaw_angle)
            #         currently_yawing = True
            #         actions[:, 0] = thrust  
            #         actions[:, 2] = -pitch_correction
            #         actions[:, 1] = -roll_correction
            #         #actions[:, 1] = -pitch_correction
            #         yaw_correction, _ = compute_yaw_correction(yaw_pid_params, yaw_target_angle, current_yaw_angle, dt)
            #         actions[:, 3] = yaw_correction
            #         current_yaw_angle += yaw_correction * dt
                    
                   
            #         yaw_error = current_yaw_angle - yaw_target_angle
            #         #print(f"Yaw error: {yaw_error}")

            #         if abs(yaw_error) < 0.01:
            #             yaw_complete = True
            #             currently_yawing = False
            #             actions[:, 3] = 0  
            #             current_yaw_angle = 0
            #             print("Yaw Completed")         
                
            #actions[:, 0] = thrust 

    save_all_combined_images(all_images, 'combined_depth_images_large.png')

if __name__ == '__main__':
    args = get_args()
    args.test = True
    args.checkpoint = None
    simple_pid_takeoff_and_move(args)
