# Copyright (c) 2023, Autonomous Robots Lab, Norwegian University of Science and Technology
# All rights reserved.

# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import math
import numpy as np
import os
import torch
import sys
from gym import spaces
import csv
import matplotlib.pyplot as plt
from aerial_gym import AERIAL_GYM_ROOT_DIR, AERIAL_GYM_ROOT_DIR
import torch.nn.functional as F

from isaacgym import gymutil, gymtorch, gymapi
from isaacgym.torch_utils import *

from aerial_gym.envs.base.base_task import BaseTask
from aerial_gym.envs.base.aerial_robot_with_obstacles_config import AerialRobotWithObstaclesCfg

from aerial_gym.envs.controllers.controller import Controller
#from aerial_gym.envs.controllers.altitude_control import AltitudeStabilizationController
import pandas as pd

from prettytable import PrettyTable

from aerial_gym.utils.asset_manager import AssetManager

from aerial_gym.utils.helpers import asset_class_to_AssetOptions
import time



def plot_observation_reward_correlation(obs_buf, rew_buf, feature_names=None):
    num_features = obs_buf.shape[1]

    obs_numpy = obs_buf.cpu().numpy()
    rewards_numpy = rew_buf.cpu().numpy()

    fig, axes = plt.subplots(nrows=num_features, ncols=1, figsize=(8, num_features * 3))
    fig.suptitle("Observation-Reward Correlation", fontsize=14)

    if num_features == 1:
        axes = [axes]  

    for i in range(num_features):
        ax = axes[i]
        ax.scatter(obs_numpy[:, i], rewards_numpy, alpha=0.5)
        ax.set_xlabel(f"Feature {i}" if feature_names is None else feature_names[i])
        ax.set_ylabel("Reward")
        ax.grid(True)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()


class RewardNormalizer:
    def __init__(self, epsilon=1e-8):
        self.mean = 0.0
        self.var = 1.0
        self.count = 1e-4  # Small initial count to avoid division by zero
        self.epsilon = epsilon  # Small value for numerical stability

    def update(self, rewards):
        batch_mean = rewards.mean()
        batch_var = rewards.var(unbiased=False)

        self.count += 1
        self.mean = 0.999 * self.mean + 0.001 * batch_mean
        self.var = 0.999 * self.var + 0.001 * batch_var

    def normalize(self, rewards):
        return (rewards - self.mean) / (self.var + self.epsilon) ** 0.5



class AerialRobotWithObstacles(BaseTask):

    def __init__(self, cfg: AerialRobotWithObstaclesCfg, sim_params, physics_engine, sim_device, headless):
        self.cfg = cfg

        self.max_episode_length = int(self.cfg.env.episode_length_s / self.cfg.sim.dt)
        self.debug_viz = False
        self.max_saved_steps = 5000
        self.trajectory_buffer = [] 
        self.sim_params = sim_params
        self.physics_engine = physics_engine
        self.sim_device_id = sim_device
        self.headless = headless

        self.reward_normalizer = RewardNormalizer()

        self.action_space = spaces.Box(low=-1, high=1, shape=(4,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(24,), dtype=np.float32)

        self.enable_onboard_cameras = self.cfg.env.enable_onboard_cameras

        self.env_asset_manager = AssetManager(self.cfg, sim_device)
        self.cam_resolution = (8, 8)

        super().__init__(self.cfg, sim_params, physics_engine, sim_device, headless)
        self.root_tensor = self.gym.acquire_actor_root_state_tensor(self.sim)
        self.contact_force_tensor = self.gym.acquire_net_contact_force_tensor(self.sim)

        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)
        num_actors = self.env_asset_manager.get_env_actor_count() + 1 # Number of obstacles in the environment + one robot
        bodies_per_env = self.env_asset_manager.get_env_link_count() + self.robot_num_bodies # Number of links in the environment + robot

        self.vec_root_tensor = gymtorch.wrap_tensor(
            self.root_tensor).view(self.num_envs, num_actors, 13)
        
        self.logged_observations = []
        self.logged_rewards = []
        self.global_step_counter = 0  # Track PPO steps
        self.max_logged_steps = 500  # Limit stored data
        
        self.yaw_triggered_sum = 0
        self.root_states = self.vec_root_tensor[:, 0, :]
        self.root_positions = self.root_states[..., 0:3]
        self.root_quats = self.root_states[..., 3:7]
        self.root_linvels = self.root_states[..., 7:10]
        self.root_angvels = self.root_states[..., 10:13]

        self.reward_mean = 0.0  # Running mean of rewards
        self.reward_variance = 1.0  # Running variance of rewards
        self.reward_count = 0  # Number of rewards observed

        self.env_asset_root_states = self.vec_root_tensor[:, 1:, :]
        self.steps_list = []
        self.last_altitude = 0.0  # Initialize with target altitude or some reasonable default value
        self.previous_altitude = 0.0
        self.num_steps = 0
        history_length = 0
        self.history_length = history_length
        self.previous_altitude = torch.full((self.num_envs,), 0.1565, dtype=torch.float32, device=self.device)
        self.previous_min_front = torch.full((self.num_envs,), fill_value=4.0, dtype=torch.float32, device=self.device)
        self.attitude_history = torch.zeros((history_length, 3), device=self.device) 

        self.step_global = 0

        self.altitude_history = []  
        self.altitude_rate_history = []
        self.distance_front_history = []
        self.distance_rate_front_history = []
        self.pitch_history = []
        self.roll_history = []
        self.yaw_history = []
        self.altitude_differences = [] 
        self.current_timestep = 0
        self.num_actions = 4
        self.num_actions_rl = 2
        self.num_obs = 24
        self.frame_stack = []
        self.obs_buf_size = 10
        self.prev_altitude = 0
        self.privileged_obs_buf = None
        if self.vec_root_tensor.shape[1] > 1:
            if self.get_privileged_obs:
                self.privileged_obs_buf = self.env_asset_root_states.clone()


        self.previous_compressed_tof_front = torch.zeros((self.num_envs, 3, 1), device=self.device)  # (Min, Max, Mean)
        self.previous_compressed_tof_left = torch.zeros((self.num_envs, 3, 1), device=self.device)
        self.previous_compressed_tof_right = torch.zeros((self.num_envs, 3, 1), device=self.device)

        self.all_pitch_rewards = []
        self.all_yaw_rewards = []
        self.all_yaw_corrections = []
        self.all_collision_penalties = []
        self.all_distance_rewards = []
        self.time_alive = torch.zeros(self.num_envs, device=self.device)

        self.contact_forces = gymtorch.wrap_tensor(self.contact_force_tensor).view(self.num_envs, bodies_per_env, 3)[:, 0]

        self.collisions = torch.zeros(self.num_envs, device=self.device)
        self.too_high = torch.zeros(self.num_envs, device=self.device)

        self.initial_root_states = self.root_states.clone()
        self.counter = 0

        self.action_upper_limits = torch.tensor(
            [10, 10, 10, 10], device=self.device, dtype=torch.float32)
        self.action_lower_limits = torch.tensor(
            [-10, -10, -10, -10], device=self.device, dtype=torch.float32)
        
        # control tensors
        self.action_input = torch.zeros(
            (self.num_envs, 4), dtype=torch.float32, device=self.device, requires_grad=False)
        self.forces = torch.zeros((self.num_envs, bodies_per_env, 3),
                                  dtype=torch.float32, device=self.device, requires_grad=False)
        self.torques = torch.zeros((self.num_envs, bodies_per_env, 3),
                                   dtype=torch.float32, device=self.device, requires_grad=False)
        
        self.controller = Controller(self.cfg.control, self.device)

        # Getting environment bounds
        self.env_lower_bound = torch.zeros(
            (self.num_envs, 3), dtype=torch.float32, device=self.device)
        self.env_upper_bound = torch.zeros(
            (self.num_envs, 3), dtype=torch.float32, device=self.device)

        if self.cfg.env.enable_onboard_cameras:
            self.full_camera_array1     = torch.zeros((self.num_envs, 8, 8), device=self.device)
            self.full_camera_array2     = torch.zeros((self.num_envs, 8, 8), device=self.device)
            self.full_camera_array3     = torch.zeros((self.num_envs, 8, 8), device=self.device)
            self.full_camera_array4     = torch.zeros((self.num_envs, 8, 8), device=self.device)
            self.full_camera_array5     = torch.zeros((self.num_envs, 8, 8), device=self.device)



        if self.viewer:
            cam_pos_x, cam_pos_y, cam_pos_z = self.cfg.viewer.pos[0], self.cfg.viewer.pos[1], self.cfg.viewer.pos[2]
            cam_target_x, cam_target_y, cam_target_z = self.cfg.viewer.lookat[0], self.cfg.viewer.lookat[1], self.cfg.viewer.lookat[2]
            cam_pos = gymapi.Vec3(cam_pos_x, cam_pos_y, cam_pos_z)
            cam_target = gymapi.Vec3(cam_target_x, cam_target_y, cam_target_z)
            cam_ref_env = self.cfg.viewer.ref_env
            
            self.gym.viewer_camera_look_at(self.viewer, None, cam_pos, cam_target)

        self.log_file = "simulation_data.csv"
        if not os.path.exists(self.log_file):
            with open(self.log_file, mode="w", newline="") as file:
                writer = csv.writer(file)
                writer.writerow(["Step", "v_body_x", "v_body_y", "v_body_z", 
                                 "force_x", "force_y", "force_z", 
                                 "drag_force_x", "drag_force_y", "drag_force_z"])

    def seed(self, seed=None):
        """Sets the random seed for reproducibility."""
        if seed is not None:
            # Set numpy seed
            np.random.seed(seed)
            # Set PyTorch seed
            torch.manual_seed(seed)
            # If using CUDA
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)

    def create_sim(self):
        self.sim = self.gym.create_sim(
            self.sim_device_id, self.graphics_device_id, self.physics_engine, self.sim_params)
        if self.cfg.env.create_ground_plane:
            self._create_ground_plane()
        self._create_envs()
        self.progress_buf = torch.zeros(
            self.cfg.env.num_envs, device=self.sim_device, dtype=torch.long)

    def _create_ground_plane(self):
        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
        self.gym.add_ground(self.sim, plane_params)
        return

    def _create_envs(self):
        print("\n\n\n\n\n CREATING ENVIRONMENT \n\n\n\n\n\n")
        asset_path = self.cfg.robot_asset.file.format(
            AERIAL_GYM_ROOT_DIR=AERIAL_GYM_ROOT_DIR)
        asset_root = os.path.dirname(asset_path)
        asset_file = os.path.basename(asset_path)

        asset_options = asset_class_to_AssetOptions(self.cfg.robot_asset)

        robot_asset = self.gym.load_asset(
            self.sim, asset_root, asset_file, asset_options)

        self.robot_num_bodies = self.gym.get_asset_rigid_body_count(robot_asset)

        start_pose = gymapi.Transform()
        # create env instance
        pos = torch.tensor([0, 0, 0], device=self.device)
        start_pose.p = gymapi.Vec3(*pos)
        self.env_spacing = self.cfg.env.env_spacing
        env_lower = gymapi.Vec3(-self.env_spacing, -
                                self.env_spacing, -self.env_spacing)
        env_upper = gymapi.Vec3(
            self.env_spacing, self.env_spacing, self.env_spacing)
        self.actor_handles = []
        self.env_asset_handles = []
        self.envs = []
        self.camera_handles = []
        self.camera_tensors = []
        self.secondary_camera_handles = []
        self.secondary_camera_tensors = []
        self.third_camera_handles = []
        self.third_camera_tensors = []
        self.fourth_camera_handles = []
        self.fourth_camera_tensors = []
        self.fifth_camera_handles = []
        self.fifth_camera_tensors = []


        # Set Camera Properties
        camera_props = gymapi.CameraProperties()
        camera_props.enable_tensors = True
        camera_props.width = self.cam_resolution[0]
        camera_props.height = self.cam_resolution[1]
        camera_props.far_plane = 4.0
        camera_props.horizontal_fov = 45.0
        
        # local transform for the first camera
        local_transform = gymapi.Transform()
        local_transform.p = gymapi.Vec3(0.15, 0.00, 0.05)
        #local_transform.r = gymapi.Quat(0.0, -0.0436, 0.0, 0.999)        
        local_transform.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)

        # # local transform for the second camera
        local_transform_fourth = gymapi.Transform()
        local_transform_fourth.p = gymapi.Vec3(-0.15, 0.00, 0.05)
        local_transform_fourth.r = gymapi.Quat(0.0, 0.0, 1.0, 0.0) 

        # local transform for the third camera
        local_transform_third = gymapi.Transform()
        local_transform_third.p = gymapi.Vec3(0.0, -0.15, 0.05)
        local_transform_third.r = gymapi.Quat(0.0, 0.0, 0.7071, -0.7071)

        # local transform for the fourth camera
        local_transform_second = gymapi.Transform()
        local_transform_second.p = gymapi.Vec3(0.0, 0.15, 0.05)
        local_transform_second.r = gymapi.Quat(0.0, 0.0, 0.7071, 0.7071)

        # local transform for the down camera
        local_transform_fifth = gymapi.Transform()
        local_transform_fifth.p = gymapi.Vec3(0.0, 0.00, -0.05)
        local_transform_fifth.r = gymapi.Quat(0.7071, 0.0, -0.7071, 0.0)

        self.segmentation_counter = 0


        for i in range(self.num_envs):
            # create environment
            env_handle = self.gym.create_env(self.sim, env_lower, env_upper, int(np.sqrt(self.num_envs)))
            # insert robot asset
            actor_handle = self.gym.create_actor(env_handle, robot_asset, start_pose, "robot", i, self.cfg.robot_asset.collision_mask, 0)
            # append to lists
            self.envs.append(env_handle)
            self.actor_handles.append(actor_handle)

            if self.enable_onboard_cameras:
                # Create the first camera
                cam_handle = self.gym.create_camera_sensor(env_handle, camera_props)
                self.gym.attach_camera_to_body(cam_handle, env_handle, actor_handle, local_transform, gymapi.FOLLOW_TRANSFORM)
                self.camera_handles.append(cam_handle)
                camera_tensor = self.gym.get_camera_image_gpu_tensor(self.sim, env_handle, cam_handle, gymapi.IMAGE_DEPTH)
                torch_cam_tensor = gymtorch.wrap_tensor(camera_tensor)
                self.camera_tensors.append(torch_cam_tensor)

                # Create the second camera
                cam_handle_second = self.gym.create_camera_sensor(env_handle, camera_props)
                self.gym.attach_camera_to_body(cam_handle_second, env_handle, actor_handle, local_transform_second, gymapi.FOLLOW_TRANSFORM)
                self.secondary_camera_handles.append(cam_handle_second)
                camera_tensor_second = self.gym.get_camera_image_gpu_tensor(self.sim, env_handle, cam_handle_second, gymapi.IMAGE_DEPTH)
                torch_cam_tensor_second = gymtorch.wrap_tensor(camera_tensor_second)
                self.secondary_camera_tensors.append(torch_cam_tensor_second)

                # Create the third camera
                cam_handle_third = self.gym.create_camera_sensor(env_handle, camera_props)
                self.gym.attach_camera_to_body(cam_handle_third, env_handle, actor_handle, local_transform_third, gymapi.FOLLOW_TRANSFORM)
                self.third_camera_handles.append(cam_handle_third)
                camera_tensor_third = self.gym.get_camera_image_gpu_tensor(self.sim, env_handle, cam_handle_third, gymapi.IMAGE_DEPTH)
                torch_cam_tensor_third = gymtorch.wrap_tensor(camera_tensor_third)
                self.third_camera_tensors.append(torch_cam_tensor_third)

                # Create the fourth camera
                cam_handle_fourth = self.gym.create_camera_sensor(env_handle, camera_props)
                self.gym.attach_camera_to_body(cam_handle_fourth, env_handle, actor_handle, local_transform_fourth, gymapi.FOLLOW_TRANSFORM)
                self.fourth_camera_handles.append(cam_handle_fourth)
                camera_tensor_fourth = self.gym.get_camera_image_gpu_tensor(self.sim, env_handle, cam_handle_fourth, gymapi.IMAGE_DEPTH)
                torch_cam_tensor_fourth = gymtorch.wrap_tensor(camera_tensor_fourth)
                self.fourth_camera_tensors.append(torch_cam_tensor_fourth)

                #Create the down camera
                cam_handle_fifth = self.gym.create_camera_sensor(env_handle, camera_props)
                self.gym.attach_camera_to_body(cam_handle_fifth, env_handle, actor_handle, local_transform_fifth, gymapi.FOLLOW_TRANSFORM)
                self.fifth_camera_handles.append(cam_handle_fifth)
                camera_tensor_fifth = self.gym.get_camera_image_gpu_tensor(self.sim, env_handle, cam_handle_fifth, gymapi.IMAGE_DEPTH)
                torch_cam_tensor_fifth = gymtorch.wrap_tensor(camera_tensor_fifth)
                self.fifth_camera_tensors.append(torch_cam_tensor_fifth)

            env_asset_list = self.env_asset_manager.prepare_assets_for_simulation(self.gym, self.sim)
            asset_counter = 0

            # have the segmentation counter be the max defined semantic id + 1. Use this to set the semantic mask of objects that are
            # do not have a defined semantic id in the config file, but still requre one. Increment for every instance in the next snippet
            for dict_item in env_asset_list:
                self.segmentation_counter = max(self.segmentation_counter, int(dict_item["semantic_id"])+1)

            for dict_item in env_asset_list:
                folder_path = dict_item["asset_folder_path"]
                filename = dict_item["asset_file_name"]
                asset_options = dict_item["asset_options"]
                whole_body_semantic = dict_item["body_semantic_label"]
                per_link_semantic = dict_item["link_semantic_label"]
                semantic_masked_links = dict_item["semantic_masked_links"]
                semantic_id = dict_item["semantic_id"]
                color = dict_item["color"]
                collision_mask = dict_item["collision_mask"]

                loaded_asset = self.gym.load_asset(self.sim, folder_path, filename, asset_options)


                assert not (whole_body_semantic and per_link_semantic)
                if semantic_id < 0:
                    object_segmentation_id = self.segmentation_counter
                    self.segmentation_counter += 1
                else:
                    object_segmentation_id = semantic_id

                asset_counter += 1

                env_asset_handle = self.gym.create_actor(env_handle, loaded_asset, start_pose, "env_asset_"+str(asset_counter), i, collision_mask, object_segmentation_id)
                self.env_asset_handles.append(env_asset_handle)
                if len(self.gym.get_actor_rigid_body_names(env_handle, env_asset_handle)) > 1:
                    print("Env asset has rigid body with more than 1 link: ", len(self.gym.get_actor_rigid_body_names(env_handle, env_asset_handle)))
                    sys.exit(0)

                if per_link_semantic:
                    rigid_body_names = None
                    if len(semantic_masked_links) == 0:
                        rigid_body_names = self.gym.get_actor_rigid_body_names(env_handle, env_asset_handle)
                    else:
                        rigid_body_names = semantic_masked_links
                    for rb_index in range(len(rigid_body_names)):
                        self.segmentation_counter += 1
                        self.gym.set_rigid_body_segmentation_id(env_handle, env_asset_handle, rb_index, self.segmentation_counter)
            
                if color is None:
                    color = np.random.randint(low=50,high=200,size=3)

                self.gym.set_rigid_body_color(env_handle, env_asset_handle, 0, gymapi.MESH_VISUAL,
                        gymapi.Vec3(color[0]/255,color[1]/255,color[2]/255))


            self.envs.append(env_handle)
            self.actor_handles.append(actor_handle)


        self.robot_body_props = self.gym.get_actor_rigid_body_properties(self.envs[0],self.actor_handles[0])
        self.robot_mass = 0
        for prop in self.robot_body_props:
            self.robot_mass += prop.mass
            
        print("Total robot mass: ", self.robot_mass)
        
        print("\n\n\n\n\n ENVIRONMENT CREATED \n\n\n\n\n\n")

    def analyze_observation_reward_correlation(self):
        print(f"Analyzing {len(self.logged_observations)} samples of observations and rewards...")

        obs_data = torch.cat(self.logged_observations, dim=0)
        rew_data = torch.cat(self.logged_rewards, dim=0)

        self.logged_observations.clear()
        self.logged_rewards.clear()
        
        plot_observation_reward_correlation(obs_data, rew_data, feature_names=[
            "Pitch", "Prev Yaw Action", "Yaw Rate", "Collision",
            "Min Front",
            "Min Left", 
            "Min Right",
            "Min Back",
        ])

    def save_trajectory(self):
        """
        Saves the trajectory of Drone [0] (x, y, z) positions to a text file.
        - Resets the file when Drone [0] resets.
        """
        drone_id = 0  # Only track the first drone

        # Convert root_positions tensor to numpy and extract (x, y, z) for Drone [0]
        position = self.root_positions[drone_id, :3].cpu().numpy()

        # If Drone [0] resets, clear the trajectory file
        if self.reset_buf[drone_id] > 0:
            self.trajectory_buffer = []  # Clear the buffer
            with open("drone_trajectory.txt", "w") as f:
                f.write("x,y,z\n")  # Write header to reset file
            print("Drone [0] reset detected. Clearing trajectory file.")

        # Append new position to the trajectory buffer
        self.trajectory_buffer.append(position.tolist())
        #print(len(self.trajectory_buffer))
        # Save every `max_saved_steps` steps
        if len(self.trajectory_buffer) >= self.max_saved_steps:
            np.savetxt("drone_trajectory.txt", self.trajectory_buffer, fmt="%.3f", delimiter=",", header="x,y,z", comments="")
            print(f"Saved {len(self.trajectory_buffer)} steps for Drone [0] to drone_trajectory.txt")

            # Clear buffer after saving
            self.trajectory_buffer = []


    def step(self, actions):
        self.reset_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
        self.old_obs = self.obs_buf.clone()
        max_episode_timesteps = 15000
        for i in range(self.cfg.env.num_control_steps_per_env_step):
            self.pre_physics_step(actions)
            self.gym.simulate(self.sim)
            self.post_physics_step()

        self.render(sync_frame_time=False)
        if self.enable_onboard_cameras:
            self.render_cameras()
        
        self.progress_buf += 1
        self.num_steps += 1
        
        self.collisions = torch.zeros(self.num_envs, device=self.device)
        self.too_high = torch.zeros(self.num_envs, device=self.device)

        self.check_collisions()
        self.compute_observations_two(actions)
        #self.check_altitude()
        self.compute_reward_two()

        self.time_alive += 1

        self.time_alive = torch.where(self.collisions > 0, torch.zeros_like(self.time_alive), self.time_alive)
        self.time_alive = torch.where(self.progress_buf >= max_episode_timesteps, torch.zeros_like(self.time_alive), self.time_alive)


        if self.cfg.env.reset_on_collision:
            ones = torch.ones_like(self.reset_buf)
            self.reset_buf = torch.where(self.collisions > 0, ones, self.reset_buf)
            self.reset_buf = torch.where(self.too_high > 0, ones, self.reset_buf)

        self.reset_buf = torch.where(self.progress_buf >= max_episode_timesteps, torch.ones_like(self.reset_buf), self.reset_buf)

        reset_env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)

        if len(reset_env_ids) > 0:
            self.reset_idx(reset_env_ids)

        self.time_out_buf = self.progress_buf > self.max_episode_length
        self.extras["time_outs"] = self.time_out_buf
        self.extras["time_alive"] = self.time_alive.clone()
        
        self.save_trajectory()

        return self.obs_buf, self.privileged_obs_buf, self.rew_buf, self.reset_buf, self.extras
   

    def reset_idx(self, env_ids):
        num_resets = len(env_ids)
        if 0 in env_ids:
            print("\n\n\n RESETTING ENV 0 \n\n\n")

        self.env_asset_manager.randomize_pose()
        
        self.env_asset_root_states[env_ids, :, 0:3] = self.env_asset_manager.asset_pose_tensor[env_ids, :, 0:3]

        euler_angles = self.env_asset_manager.asset_pose_tensor[env_ids, :, 3:6]
        self.env_asset_root_states[env_ids, :, 3:7] = quat_from_euler_xyz(euler_angles[..., 0], euler_angles[..., 1], euler_angles[..., 2])
        self.env_asset_root_states[env_ids, :, 7:13] = 0.0


        # get environment lower and upper bounds
        self.env_lower_bound[env_ids] = self.env_asset_manager.env_lower_bound.diagonal(dim1=-2, dim2=-1)
        self.env_upper_bound[env_ids] = self.env_asset_manager.env_upper_bound.diagonal(dim1=-2, dim2=-1)
        drone_pos_rand_sample = torch.rand((num_resets, 3), device=self.device)
        
        """
        drone_positions = (self.env_upper_bound[env_ids] - self.env_lower_bound[env_ids] -
                           0.50)*drone_pos_rand_sample + (self.env_lower_bound[env_ids]+ 0.25)
        """
        drone_positions = torch.tensor([0.0, 0.0, 0.2], device=self.device)

        # set drone positions that are sampled within environment bounds

        self.root_states[env_ids,
                         0:3] = drone_positions
        self.root_states[env_ids,
                         7:10] = 0.2*torch_rand_float(-1.0, 1.0, (num_resets, 3), self.device)
        self.root_states[env_ids,
                         10:13] = 0.2*torch_rand_float(-1.0, 1.0, (num_resets, 3), self.device)

        self.root_states[env_ids, 3:6] = 0 
        self.root_states[env_ids, 6] = 1

        self.gym.set_actor_root_state_tensor(self.sim, self.root_tensor)
        self.progress_buf[env_ids] = 0
        self.reset_buf[env_ids] = 1


    def quat_rotate(quat, vec):
        q_xyz = quat[..., :3]  
        q_w = quat[..., 3:4]   

        t = 2.0 * torch.cross(q_xyz, vec, dim=-1)
        rotated_vec = vec + q_w * t + torch.cross(q_xyz, t, dim=-1)

        return rotated_vec


    def quat_rotate_inverse(quat, vec):
        quat_conjugate = torch.cat([-quat[..., :3], quat[..., 3:4]], dim=-1)
        return quat_rotate(quat_conjugate, vec)

    def pre_physics_step(self, _actions):
        #if self.counter % 250 == 0:
            #print("self.counter:", self.counter)
        self.counter += 1

        actions = torch.tensor(_actions, dtype=torch.float32).to(self.device)
        actions = tensor_clamp(actions, self.action_lower_limits, self.action_upper_limits)
        self.action_input[:] = actions

        self.forces[:] = 0.0
        self.torques[:, :] = 0.0

        output_thrusts_mass_normalized, output_torques_inertia_normalized = self.controller(self.root_states, self.action_input)
        self.forces[:, 0, 2] = self.robot_mass * (-self.sim_params.gravity.z) * output_thrusts_mass_normalized
        self.torques[:, 0] = output_torques_inertia_normalized

        body_velocity = quat_rotate_inverse(self.root_quats, self.root_linvels)

        v_body = body_velocity[:, :3]

        k_v_linear = torch.tensor([4.2, 1.8, 0.0], device=self.device) 
        #v_body[:, 0] = torch.clamp(v_body[:, 0], min=-0.5, max=0.5)
        #v_body[:, 1] = torch.clamp(v_body[:, 1], min=-0.5, max=0.5)
        #v_body[:, 2] = torch.clamp(v_body[:, 2], min=-0.5, max=0.5)

        drag_force_x = -k_v_linear[0] * v_body[:, 0] * torch.abs(v_body[:, 0])
        drag_force_y = -k_v_linear[1] * v_body[:, 1] * torch.abs(v_body[:, 1])

        

        drag_forces_body = torch.stack((drag_force_x, drag_force_y, torch.zeros_like(drag_force_x)), dim=-1)

        drag_forces_world = drag_forces_body
        #drag_forces_world = torch.clamp(drag_forces_world, min=-0.2, max=0.2)
        self.forces[:, 0, :3] += drag_forces_world

        body_angular_velocity = quat_rotate_inverse(self.root_quats, self.root_angvels)  
        k_w_angular = torch.tensor([0.0, 0.0, 0.0], device=self.device)  
        angular_drag_torques_body = -k_w_angular * body_angular_velocity * torch.abs(body_angular_velocity)

        angular_drag_torques_world = quat_rotate(self.root_quats, angular_drag_torques_body)
        self.torques[:, 0] += angular_drag_torques_world
    
        self.gym.apply_rigid_body_force_tensors(
            self.sim,
            gymtorch.unwrap_tensor(self.forces),
            gymtorch.unwrap_tensor(self.torques),
            gymapi.LOCAL_SPACE
        )
        

    def get_drone_position(self):
        return self.root_positions[:, 0].cpu().numpy()

    def render_cameras(self):        
        self.gym.render_all_camera_sensors(self.sim)
        self.gym.start_access_image_tensors(self.sim)
        self.dump_images()
        self.gym.end_access_image_tensors(self.sim)
        return

    def post_physics_step(self):
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)
    
    def check_collisions(self):
        ones = torch.ones((self.num_envs), device=self.device)
        zeros = torch.zeros((self.num_envs), device=self.device)
        self.collisions[:] = 0
        self.collisions = torch.where(torch.norm(self.contact_forces, dim=1) > 0.1, ones, zeros)


    def dump_images(self):
        for env_id in range(self.num_envs):
            self.full_camera_array1[env_id]     = -self.camera_tensors[env_id]
            self.full_camera_array2[env_id]     = -self.secondary_camera_tensors[env_id]
            self.full_camera_array3[env_id]     = -self.third_camera_tensors[env_id]
            self.full_camera_array4[env_id]     = -self.fourth_camera_tensors[env_id]
            self.full_camera_array5[env_id]     = -self.fifth_camera_tensors[env_id]


    def estimate_yaw_from_tof(self, left_distance, right_distance):
        max_diff = 0.4 

        distance_diff = left_distance - right_distance

        estimated_yaw = (distance_diff / max_diff) * 90  

        return estimated_yaw

    def process_depth_images(self):
        depth_images = []
        depth_values = []
        
        max_depth = 4.0 

        for camera_array in [self.full_camera_array1, self.full_camera_array2, self.full_camera_array3,
                            self.full_camera_array4, self.full_camera_array5]:
            depth_np = camera_array.cpu().numpy() 
            #print(depth_np)
            depth_np = np.clip(depth_np, 0, max_depth)
            depth_img = np.uint8(depth_np * 255 / max_depth)
            depth_images.append(depth_img)
            depth_values.append(depth_np)

        return depth_images, depth_values  
    

    def check_altitude(self):
        self.too_high = torch.zeros_like(self.obs_raw[:, 7], dtype=torch.int)       
        self.too_high[self.obs_raw[:, 7] >= 4] = 1
        
    def downsample_tof(self, matrix, grid_size=3):
        block_size = 8 // grid_size
        num_envs = matrix.shape[0]  # Extract batch size
        
        # Ensure min_matrix and mean_matrix have the correct batch dimension
        min_matrix = torch.zeros((num_envs, grid_size, grid_size), device=self.device)
        mean_matrix = torch.zeros((num_envs, grid_size, grid_size), device=self.device)

        for i in range(grid_size):
            for j in range(grid_size):
                # Extract the corresponding 3x3 or 4x4 region for all environments
                region = matrix[:, i * block_size:(i + 1) * block_size, j * block_size:(j + 1) * block_size]
                
                # Compute min and mean across the last two spatial dimensions
                min_matrix[:, i, j] = torch.amin(region, dim=(-1, -2))
                mean_matrix[:, i, j] = torch.mean(region, dim=(-1, -2))

        return min_matrix, mean_matrix

    
    def quaternion_to_euler(self, quaternion):
        x, y, z, w = quaternion[..., 0], quaternion[..., 1], quaternion[..., 2], quaternion[..., 3]
        t0 = +2.0 * (w * x + y * z)
        t1 = +1.0 - 2.0 * (x * x + y * y)
        roll = torch.atan2(t0, t1)

        t2 = +2.0 * (w * y - z * x)
        #t2 = torch.clamp(t2, -1.0, 1.0)
        pitch = torch.asin(t2)

        t3 = +2.0 * (w * z + x * y)
        t4 = +1.0 - 2.0 * (y * y + z * z)
        yaw = torch.atan2(t3, t4)

        return roll, pitch, yaw
    

    def compress_tof_to_3x1_minmaxmean(self, tof_matrix):
        tof_matrix = tof_matrix.float()  # Ensure dtype is float32

        # Compute min, max, and mean values along the columns
        min_depth = torch.min(tof_matrix, dim=2, keepdim=True)[0]  # (num_envs, 8, 1) - Closest obstacle
        max_depth = torch.max(tof_matrix, dim=2, keepdim=True)[0]  # (num_envs, 8, 1) - Farthest obstacle
        mean_depth = torch.mean(tof_matrix, dim=2, keepdim=True)   # (num_envs, 8, 1) - Average distance

        # Concatenate (min, max, mean) into a single feature vector
        compressed_matrix = torch.cat((min_depth, max_depth, mean_depth), dim=2)  # (num_envs, 8, 3)

        # Further compress from (8,3) to (3,1) by averaging across rows
        compressed_matrix = torch.mean(compressed_matrix, dim=1, keepdim=True).permute(0, 2, 1)  # (num_envs, 3, 1)

        return compressed_matrix

    
    def compress_tof_to_3x1(self, tof_matrix):
        tof_matrix = tof_matrix.float()  # Ensure dtype is float32

        # Compute column-wise averages for each environment
        avg_col_01 = torch.mean(tof_matrix[:, :, 0:2], dim=2, keepdim=True)  # (num_envs, 8, 1)
        avg_col_2345 = torch.mean(tof_matrix[:, :, 2:6], dim=2, keepdim=True)  # (num_envs, 8, 1)
        avg_col_67 = torch.mean(tof_matrix[:, :, 6:8], dim=2, keepdim=True)  # (num_envs, 8, 1)

        # Compute row-wise averages to compress (8,1) → (3,1)
        compressed_matrix = torch.cat((avg_col_01, avg_col_2345, avg_col_67), dim=2)  # (num_envs, 8, 3)
        compressed_matrix = torch.mean(compressed_matrix, dim=1, keepdim=True).permute(0, 2, 1)  # (num_envs, 3, 1)

        return compressed_matrix
    
    def print_observations(self):
        """ Prints real-time observations for drone [0] in a table format. """
        
        obs_names = [
            "Pitch", "Yaw Action", "Yaw",
            "Left Min", "Center Min", "Right Min"
        ]
        
        # Extract observations for drone [0]
        obs_values = self.obs_buf[0].cpu().numpy()
        
        # Convert to table format
        table = PrettyTable()
        table.field_names = ["Feature", "Value"]
        
        for name, value in zip(obs_names, obs_values):
            table.add_row([name, f"{value:.3f}"])
        
        print("\n🚀 **Real-Time Observations for Drone [0]**")
        print(table)



    def compute_observations_two(self, actions):
        history_length = 4  
        num_features = 6  
        num_raw_features = 11  
        self.obs_raw = torch.zeros((self.num_envs, num_raw_features), device=self.device)  

        total_features = num_features * history_length  

        # Initialize the observation buffer
        if not hasattr(self, 'obs_history'):
            self.obs_history = torch.zeros((self.num_envs, history_length, num_features), device=self.device)

        # 📡 **Process ToF sensor data**
        depth_images, depth_values = self.process_depth_images()
        depth_values = [torch.tensor(depth, device=self.device) for depth in depth_values]

        tof_front = depth_values[0].view(self.num_envs, 8, 8)
        tof_back = depth_values[1].view(self.num_envs, 8, 8)
        tof_left = depth_values[2].view(self.num_envs, 8, 8)
        tof_right = depth_values[3].view(self.num_envs, 8, 8)

        # Extract min & mean for better smoothing
        left_min = tof_front[:, :, :2].min(dim=-1).values.min(dim=-1).values  
        left_mean = tof_front[:, :, :2].mean(dim=-1).mean(dim=-1)  
        left_min_mean = (left_min + left_mean) / 2  # Smoothed perception

        center_min = tof_front[:, :, 2:6].min(dim=-1).values.min(dim=-1).values  
        center_mean = tof_front[:, :, 2:6].mean(dim=-1).mean(dim=-1)  
        center_min_mean = (center_min + center_mean) / 2  

        right_min = tof_front[:, :, 6:].min(dim=-1).values.min(dim=-1).values  
        right_mean = tof_front[:, :, 6:].mean(dim=-1).mean(dim=-1)  
        right_min_mean = (right_min + right_mean) / 2  

        back_min = tof_back.min(dim=-1).values.min(dim=-1).values  
        back_mean = tof_back.mean(dim=-1).mean(dim=-1)  
        back_min_mean = (back_min + back_mean) / 2  

        roll, pitch, yaw = self.quaternion_to_euler(self.root_quats)
        yaw_rate = (yaw - self.previous_yaw) if hasattr(self, 'previous_yaw') else torch.tensor(0.0, device=self.device)

        collision = self.collisions

        prev_pitch_action = getattr(self, 'previous_pitch_action', torch.zeros(self.num_envs, device=self.device))
        prev_yaw_action = getattr(self, 'previous_yaw_action', torch.zeros(self.num_envs, device=self.device))

        # 📏 **Normalize distances**
        safe_min_distance = 0.02  
        pitch_norm = pitch / np.pi  
        yaw_action_norm = prev_yaw_action / 4  

        left_min_norm = torch.log1p(left_min_mean + safe_min_distance) / np.log(4.0 + 1)
        center_min_norm = torch.log1p(center_min_mean + safe_min_distance) / np.log(4.0 + 1)  
        right_min_norm = torch.log1p(right_min_mean + safe_min_distance) / np.log(4.0 + 1)
        back_min_norm = torch.log1p(back_min_mean + safe_min_distance) / np.log(4.0 + 1)

        # Distance traveled based on center_min variations
        distance_traveled = torch.abs(self.previous_min_front - center_min_norm)

        # **Save Raw Observations (for debugging, optional)**
        self.obs_raw[..., 0] = prev_pitch_action
        self.obs_raw[..., 1] = pitch
        self.obs_raw[..., 2] = prev_yaw_action
        self.obs_raw[..., 3] = yaw_rate
        self.obs_raw[..., 4] = collision
        self.obs_raw[..., 5] = left_min_norm
        self.obs_raw[..., 6] = center_min_norm
        self.obs_raw[..., 7] = right_min_norm
        self.obs_raw[..., 8] = back_min_norm
        self.obs_raw[..., 9] = torch.where(self.reset_buf > 0, torch.tensor(0.9, device=self.device), self.obs_raw[..., 9] * 0.9)
        self.obs_raw[..., 10] = distance_traveled

        # 🚀 **Update the observation history buffer**
        self.obs_history = torch.roll(self.obs_history, shifts=-1, dims=1)  # Shift old data
        self.obs_history[:, -1, 0] = pitch_norm
        self.obs_history[:, -1, 1] = yaw_action_norm
        self.obs_history[:, -1, 2] = yaw / np.pi 
        self.obs_history[:, -1, 3] = left_min_norm
        self.obs_history[:, -1, 4] = center_min_norm
        self.obs_history[:, -1, 5] = right_min_norm

        # 🚀 **Flatten before returning (concatenate past frames)**
        self.obs_buf = self.obs_history.view(self.num_envs, -1)  

        # Save previous states
        self.previous_pitch_action = actions[:, 2] if actions is not None else torch.zeros(self.num_envs, device=self.device)
        self.previous_yaw_action = actions[:, 3] if actions is not None else torch.zeros(self.num_envs, device=self.device)
        self.previous_pitch = pitch.clone()
        self.previous_yaw = yaw.clone()
        self.previous_min_front = center_min_norm.clone()
        #self.print_observations()
        return self.obs_buf


    def compute_observations_five(self, actions):
        num_features = 8  # Reduced to store only min ToF values
        num_raw_features = 11  
        self.obs_buf = torch.zeros((self.num_envs, num_features), device=self.device)
        self.obs_raw = torch.zeros((self.num_envs, num_raw_features), device=self.device)  

        # Process depth images
        depth_images, depth_values = self.process_depth_images()
        depth_values = [torch.tensor(depth, device=self.device) for depth in depth_values]

        # Convert depth images to 8x8 matrices
        tof_front = depth_values[0].view(self.num_envs, 8, 8)
        tof_back = depth_values[1].view(self.num_envs, 8, 8)
        tof_left = depth_values[2].view(self.num_envs, 8, 8)
        tof_right = depth_values[3].view(self.num_envs, 8, 8)

        # Compress ToF data
        compressed_tof_front = self.compress_tof_to_3x1_minmaxmean(tof_front)
        compressed_tof_left = self.compress_tof_to_3x1_minmaxmean(tof_left)
        compressed_tof_right = self.compress_tof_to_3x1_minmaxmean(tof_right)

        # Downsample and extract min values
        min_front_value, _ = self.downsample_tof(tof_front, grid_size=3)
        min_back_value, _ = self.downsample_tof(tof_back, grid_size=3)
        min_left_value, _ = self.downsample_tof(tof_left, grid_size=3)
        min_right_value, _ = self.downsample_tof(tof_right, grid_size=3)

        min_front_value = min_front_value.min(dim=-1).values
        min_back_value = min_back_value.min(dim=-1).values
        min_left_value = min_left_value.min(dim=-1).values
        min_right_value = min_right_value.min(dim=-1).values

        # Compute pitch, yaw
        roll, pitch, yaw = self.quaternion_to_euler(self.root_quats)
        yaw_rate = (yaw - self.previous_yaw) if hasattr(self, 'previous_yaw') else torch.tensor(0.0, device=self.device)
        
        collision = self.collisions
        
        prev_pitch_action = getattr(self, 'previous_pitch_action', torch.zeros(self.num_envs, device=self.device))
        prev_yaw_action = getattr(self, 'previous_yaw_action', torch.zeros(self.num_envs, device=self.device))
        compressed_tof_front[:, 0, 0] = compressed_tof_front[:, 0, 0] * torch.cos(pitch)
        min_back_value = min_back_value * torch.cos(pitch).unsqueeze(-1)


        distance_traveled = torch.abs(self.previous_min_front - compressed_tof_front[:, 0, 0])
        # Store raw observations
        self.obs_raw[..., 0] = prev_pitch_action
        self.obs_raw[..., 1] = pitch
        self.obs_raw[..., 2] = prev_yaw_action
        self.obs_raw[..., 3] = yaw_rate
        self.obs_raw[..., 4] = collision
        self.obs_raw[..., 5] = min_front_value.min(dim=-1).values
        self.obs_raw[..., 6] = min_back_value.min(dim=-1).values
        self.obs_raw[..., 7] = min_left_value.min(dim=-1).values
        self.obs_raw[..., 8] = min_right_value.min(dim=-1).values
        self.obs_raw[..., 9] = torch.where(self.reset_buf > 0, torch.tensor(0.9, device=self.device), self.obs_raw[..., 9] * 0.9)
        self.obs_raw[..., 10] = distance_traveled
        # Normalize and store in obs_buf
        safe_min_distance = 0.02  # Avoid log(0)
        self.obs_buf[..., 0] = pitch / np.pi  
        self.obs_buf[..., 1] = prev_yaw_action / 4
        self.obs_buf[..., 2] = yaw_rate / (2 * np.pi)
        self.obs_buf[..., 3] = 1.0 - torch.exp(-collision * 5.0)
        self.obs_buf[..., 4] = torch.log1p(compressed_tof_front[:, 0, 0] + safe_min_distance) / np.log(4.0 + 1)
        self.obs_buf[..., 5] = torch.log1p(compressed_tof_left[:, 0, 0] + safe_min_distance) / np.log(4.0 + 1)
        self.obs_buf[..., 6] = torch.log1p(compressed_tof_right[:, 0, 0] + safe_min_distance) / np.log(4.0 + 1)
        self.obs_buf[..., 7] = torch.log1p(min_back_value.min(dim=-1).values + safe_min_distance) / np.log(4.0 + 1)

        # Save previous states
        self.previous_pitch_action = actions[:, 2] if actions is not None else torch.zeros(self.num_envs, device=self.device)
        self.previous_yaw_action = actions[:, 3] if actions is not None else torch.zeros(self.num_envs, device=self.device)
        self.previous_pitch = pitch.clone()
        self.previous_yaw = yaw.clone()

        self.previous_min_front = min_front_value.clone()
        self.previous_min_back = min_back_value.clone()
        self.previous_min_left = min_left_value.clone()
        self.previous_min_right = min_right_value.clone()
        #self.print_observations(drone_id=0)
        self.previous_min_front = compressed_tof_front[:, 0, 0].clone()

        return self.obs_buf
    

    def compute_reward_two(self):
        self.rew_buf[:], self.reset_buf[:]= compute_rewards_flapper_two(
            self.obs_buf[..., 0],
            self.obs_buf[..., 2],
            self.obs_buf[..., 1],
            self.obs_buf[..., 4],
            self.collisions,
            self.obs_raw[..., 9],
            self.time_alive,
            self.obs_raw[..., 10]
        )

    def compute_reward_five(self):
        self.rew_buf[:], self.reset_buf[:] = compute_rewards_flapper_five(
            self.obs_buf[..., 0],
            self.obs_buf[..., 2],
            self.obs_buf[..., 1],
            self.obs_buf[..., 4],
            self.obs_buf[..., 3],
            self.obs_raw[..., 9],
            self.time_alive,
            self.obs_raw[..., 10]
        )

        self.global_step_counter += 1

        # if self.global_step_counter % 10 == 0 and len(self.logged_observations) < self.max_logged_steps:
        #     self.logged_observations.append(self.obs_buf.clone().cpu())
        #     self.logged_rewards.append(self.rew_buf.clone().cpu())

        # if len(self.logged_observations) >= self.max_logged_steps:
        #     self.analyze_observation_reward_correlation()





###=========================jit functions=========================###
#####################################################################

@torch.jit.script
def quat_rotate(q, v):
    shape = q.shape
    q_w = q[:, -1]
    q_vec = q[:, :3]
    a = v * (2.0 * q_w ** 2 - 1.0).unsqueeze(-1)
    b = torch.cross(q_vec, v, dim=-1) * q_w.unsqueeze(-1) * 2.0
    c = q_vec * \
        torch.bmm(q_vec.view(shape[0], 1, 3), v.view(
            shape[0], 3, 1)).squeeze(-1) * 2.0
    return a + b + c

@torch.jit.script
def quat_axis(q, axis=0):
    # type: (Tensor, int) -> Tensor
    basis_vec = torch.zeros(q.shape[0], 3, device=q.device)
    basis_vec[:, axis] = 1
    return quat_rotate(q, basis_vec)


@torch.jit.script
def compute_rewards_flapper_two(
    pitch: torch.Tensor,  
    yaw: torch.Tensor,
    yaw_rate: torch.Tensor,  
    front_tof_min: torch.Tensor,  
    collision: torch.Tensor,
    reset: torch.Tensor,  
    time_alive: torch.Tensor,
    distance_traveled: torch.Tensor 
):
    dtype = pitch.dtype  
    device = pitch.device  

    # 🚨 **1. Collision Penalty (Strongest)**
    collision_penalty = torch.where(
        collision > 0, 
        torch.tensor(-50.0, dtype=dtype, device=device),  
        torch.tensor(0.0, dtype=dtype, device=device)
    )

    # 📐 **2. Reward for Slow Forward Pitch (~0.1)**
    desired_pitch = torch.tensor(0.07, dtype=dtype, device=device)
    pitch_reward = torch.where(
        pitch >= 0.0,  # Only reward non-negative pitch (forward tilt)
        torch.exp(-torch.abs(pitch - desired_pitch) * 6.0),  
        torch.tensor(0.0, dtype=dtype, device=device)  # No reward if pitch < 0
    )


    # 🔄 **3. Yaw Randomly to Explore**
    yaw_exploration_reward = 0.2 * torch.abs(yaw_rate)  

    # 🔥 **Final Reward Calculation**
    reward = (
        collision_penalty  
        + pitch_reward  
        + yaw_exploration_reward  
    )

    return reward, reset




@torch.jit.script
def compute_rewards_flapper_five(
    pitch: torch.Tensor,  
    yaw: torch.Tensor,  
    prev_yaw_action: torch.Tensor,  
    front_tof_min: torch.Tensor,  
    collision: torch.Tensor,
    reset: torch.Tensor,  
    time_alive: torch.Tensor,
    distance_traveled: torch.Tensor):

    dtype = pitch.dtype  
    device = pitch.device  

    collision_penalty = torch.where(
        collision > 0, 
        torch.tensor(-10.0, dtype=dtype, device=device),  
        torch.tensor(0.0, dtype=dtype, device=device)
    )

    desired_pitch = torch.tensor(0.1, dtype=dtype, device=device)
    pitch_reward = torch.exp(-torch.abs(pitch - desired_pitch) * 5.0)  

    obstacle_detected = front_tof_min < 0.2 
    yaw_reward = torch.where(
        obstacle_detected, 
        torch.abs(yaw) * 0.2,  
        torch.tensor(0.0, dtype=dtype, device=device)  
    )

    time_alive_reward = 0.02 * time_alive 
    direct_path_reward = torch.where(
        distance_traveled > 0.02,  
        torch.tensor(0.05, dtype=dtype, device=device),  
        torch.tensor(0.0, dtype=dtype, device=device)
    )

    reward = (
        collision_penalty  
        + 0.3 * pitch_reward  
        + yaw_reward 
        + 0.5 * direct_path_reward

    )

    return reward, reset



def compute_rewards_flapper_idk(
    pitch: torch.Tensor,  
    yaw: torch.Tensor,  
    prev_yaw_action: torch.Tensor,  
    front_tof_min: torch.Tensor,  
    collision: torch.Tensor,
    reset: torch.Tensor,  
    time_alive: torch.Tensor
     ):

    dtype = pitch.dtype  
    device = pitch.device  

    collision_penalty = torch.where(
        collision > 0, 
        torch.tensor(-10.0, dtype=dtype, device=device),  
        torch.tensor(0.0, dtype=dtype, device=device)
    )

    desired_pitch = torch.tensor(0.1, dtype=dtype, device=device)
    pitch_reward = torch.exp(-torch.abs(pitch - desired_pitch) * 5.0)  

    # 🔹 Reward for yawing when an obstacle is detected (to encourage turning)
    obstacle_detected = front_tof_min < 0.3  # If obstacle is within 30% of range
    yaw_reward = torch.where(
        obstacle_detected, 
        torch.abs(yaw) * 0.2,  # Reward for turning when an obstacle is near
        torch.tensor(0.0, dtype=dtype, device=device)  
    )

    # 🔹 Combine all rewards
    reward = (
        collision_penalty  # Only penalty in the function
        + 0.3 * pitch_reward  # Encourage stable pitch
        + yaw_reward  # Reward yawing near obstacles
    )

    return reward, reset

