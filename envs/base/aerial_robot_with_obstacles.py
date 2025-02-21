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


from aerial_gym.utils.asset_manager import AssetManager

from aerial_gym.utils.helpers import asset_class_to_AssetOptions
import time


class AerialRobotWithObstacles(BaseTask):

    def __init__(self, cfg: AerialRobotWithObstaclesCfg, sim_params, physics_engine, sim_device, headless):
        self.cfg = cfg

        self.max_episode_length = int(self.cfg.env.episode_length_s / self.cfg.sim.dt)
        self.debug_viz = False

        self.sim_params = sim_params
        self.physics_engine = physics_engine
        self.sim_device_id = sim_device
        self.headless = headless

        self.action_space = spaces.Box(low=-1, high=1, shape=(4,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(78,), dtype=np.float32)

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

        self.attitude_history = torch.zeros((history_length, 3), device=self.device) 

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
        self.num_obs = 78
        self.frame_stack = []
        self.obs_buf_size = 10
        self.prev_altitude = 0
        self.privileged_obs_buf = None
        if self.vec_root_tensor.shape[1] > 1:
            if self.get_privileged_obs:
                self.privileged_obs_buf = self.env_asset_root_states.clone()


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
        local_transform_second = gymapi.Transform()
        local_transform_second.p = gymapi.Vec3(-0.15, 0.00, 0.05)
        local_transform_second.r = gymapi.Quat(0.0, 0.0, 1.0, 0.0) 

        # local transform for the third camera
        local_transform_third = gymapi.Transform()
        local_transform_third.p = gymapi.Vec3(0.0, -0.15, 0.05)
        local_transform_third.r = gymapi.Quat(0.0, 0.0, 0.7071, -0.7071)

        # local transform for the fourth camera
        local_transform_fourth = gymapi.Transform()
        local_transform_fourth.p = gymapi.Vec3(0.0, 0.15, 0.05)
        local_transform_fourth.r = gymapi.Quat(0.0, 0.0, 0.7071, 0.7071)

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

    def plot_altitude_difference(self, current_altitudes, target_altitude=1.5, tolerance=0.02, step_threshold=6000):
        altitude_diff = target_altitude - current_altitudes
        self.steps_list.append(self.num_steps)
        self.altitude_differences.append(altitude_diff)
        
        if len(self.altitude_differences) < step_threshold:
            return  
        
        altitude_differences_tensor = torch.stack(self.altitude_differences)
        
    
        plt.figure(figsize=(10, 6))
        
        plt.plot(self.steps_list, altitude_differences_tensor.cpu().numpy(), label="Altitude Difference (Drone 1)", color="blue")

        plt.axhline(y=tolerance, color="green", linestyle="--", label="Upper Tolerance Bound")
        plt.axhline(y=-tolerance, color="red", linestyle="--", label="Lower Tolerance Bound")
        
        plt.xlabel("Steps")
        plt.ylabel("Altitude Difference (m)")
        plt.title("Altitude Difference Between Target and Current Altitude (Drone 1)")
        plt.legend()
        
        plt.show()

    def step(self, actions):
        self.reset_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
        self.old_obs = self.obs_buf.clone()
        max_episode_timesteps = 500
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
        self.compute_observations(actions)
        #self.check_altitude()
        altitude_reward = self.compute_reward_pitch()
        altitude_reward = torch.tensor(altitude_reward, device=self.device)
        
        self.rew_buf[:] = altitude_reward  
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
        """
        Rotate a vector using a quaternion.

        :param quat: Tensor of quaternions (shape: [N, 4]), with (x, y, z, w).
        :param vec: Tensor of vectors to rotate (shape: [N, 3]).
        :return: Rotated vectors (shape: [N, 3]).
        """
        q_xyz = quat[..., :3]  
        q_w = quat[..., 3:4]   

        t = 2.0 * torch.cross(q_xyz, vec, dim=-1)
        rotated_vec = vec + q_w * t + torch.cross(q_xyz, t, dim=-1)

        return rotated_vec


    def quat_rotate_inverse(quat, vec):
        """
        Rotate a vector using the inverse of a quaternion.

        :param quat: Tensor of quaternions (shape: [N, 4]), with (x, y, z, w).
        :param vec: Tensor of vectors to rotate (shape: [N, 3]).
        :return: Rotated vectors (shape: [N, 3]).
        """
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
        v_body[:, 0] = torch.clamp(v_body[:, 0], min=-0.5, max=0.5)
        v_body[:, 1] = torch.clamp(v_body[:, 1], min=-0.5, max=0.5)
        v_body[:, 2] = torch.clamp(v_body[:, 2], min=-0.5, max=0.5)

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
        """
        Downsamples an 8x8 ToF matrix into a smaller grid (3x3 or 4x4).
        
        Args:
            matrix (torch.Tensor): (num_envs, 8, 8) ToF depth matrix.
            grid_size (int): Target downsampled grid size (3 or 4).
        
        Returns:
            tuple: (min_matrix, mean_matrix) - Both are (num_envs, grid_size, grid_size).
        """
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

    
    def compute_observations_original(self):        
        obs_buf_size = 13 + 5 
        self.obs_buf = torch.zeros((self.num_envs, obs_buf_size), device=self.device)
        self.obs_buf[..., :3] = self.root_positions
        self.obs_buf[..., 3:7] = self.root_quats
        self.obs_buf[..., 7:10] = self.root_linvels
        self.obs_buf[..., 10:13] = self.root_angvels
        print(self.root_quats)

        depth_images, depth_values = self.process_depth_images()
        
        depth_values = [torch.tensor(depth, device=self.device) for depth in depth_values]

        min_depths = [depth.min().unsqueeze(0) for depth in depth_values] 

        self.obs_buf[..., 13:13 + len(min_depths)] = torch.cat(min_depths, dim=0)

        detection_threshold = 0.3

        left_distance  = depth_values[3].mean().item() 
        right_distance = depth_values[2].mean().item() 
        yaw = self.estimate_yaw_from_tof(left_distance, right_distance)
            
        front_detected = torch.any(depth_values[0] < detection_threshold)
        left_detected = torch.any(depth_values[1] < detection_threshold)
        right_detected = torch.any(depth_values[2] < detection_threshold)
        back_detected = torch.any(depth_values[3] < detection_threshold)
        down_detected = torch.any(depth_values[4] < detection_threshold)
        
        self.extras["front_detected"] = front_detected.float()
        self.extras["left_detected"] = left_detected.float()
        self.extras["right_detected"] = right_detected.float()
        self.extras["back_detected"] = back_detected.float()
        self.extras["down_detected"] = down_detected.float()
        return self.obs_buf
    
    def quaternion_to_euler(self, quaternion):
        """Convert quaternion to Euler angles (pitch, roll, yaw)."""
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


    def compute_observations_OLD(self):
        num_features = 13
        self.obs_buf = torch.zeros((self.num_envs, num_features), device=self.device)
        #self.old_obs = torch.zeros((self.num_envs, num_features), device=self.device)
        self.obs_raw = torch.zeros((self.num_envs, num_features+1), device=self.device) 

        depth_images, depth_values = self.process_depth_images()
        depth_values = [torch.tensor(depth, device=self.device) for depth in depth_values]

        min_depths = [torch.amin(torch.tensor(depth, device=self.device), dim=1) for depth in depth_values]
        raw_altitude = min_depths[4].min(dim=-1).values.squeeze()
        distance_front = min_depths[0].min(dim=-1).values.squeeze()
        distance_back = min_depths[1].min(dim=-1).values.squeeze()
        roll, pitch, yaw = self.quaternion_to_euler(self.root_quats)
        distance_front[distance_front >= 4.0] = 4.0
        distance_back[distance_back >= 4.0]   = 4.0
        corrected_altitude = torch.clamp(raw_altitude * torch.cos(pitch) * torch.cos(roll), max = 4.0)
        corrected_altitude = torch.clamp(corrected_altitude, min=0.0, max=4.0)

        if hasattr(self, 'previous_altitude') and self.previous_altitude is not None:
            delta_t = self.cfg.sim.dt
            altitude_rate_of_change = (corrected_altitude - self.previous_altitude) / delta_t
        else:
            altitude_rate_of_change = torch.zeros_like(corrected_altitude)  

        desired_altitude = 1.5
        desired_roll     = 0.0
        desired_pitch    = 0.0
        desired_yaw      = 0.0

        collision = self.collisions
        depth_values[4] = torch.tensor(depth_values[4], device=self.device)
        depth_values[0] = torch.tensor(depth_values[0], device=self.device)
        downsampled_downward_depth = self.downsample_depth_matrix(depth_values[4], target_size=(4, 4))
        downsampled_forward_depth = self.downsample_depth_matrix(depth_values[0], target_size=(4, 4))
        too_high  = corrected_altitude >= 4.0

        self.obs_raw[..., 0]  = corrected_altitude                                  # raw altitude
        self.obs_raw[..., 1]  = altitude_rate_of_change                             # raw altitude rate of change
        self.obs_raw[..., 2]  = roll                                                # raw roll
        self.obs_raw[..., 3]  = pitch                                               # raw pitch
        self.obs_raw[..., 4]  = yaw                                                 # raw yaw
        self.obs_raw[..., 5]  = desired_altitude                                    # raw desired altitude
        self.obs_raw[..., 6]  = desired_roll                                        # raw desired roll
        self.obs_raw[..., 7]  = desired_pitch                                       # raw desired pitch
        self.obs_raw[..., 8]  = desired_yaw                                         # raw desired yaw
        self.obs_raw[..., 9]  = distance_front                                      # raw distance front
        self.obs_raw[..., 10] = distance_back                                       # raw distance back
        self.obs_raw[..., 11]  = collision                                          # raw collision
        self.obs_raw[..., 12]  = too_high                                           # raw too high flag
        self.obs_raw[..., 13]  = raw_altitude

        self.obs_buf[..., 0]     = (corrected_altitude - 0) / (4 - 0)                               # normalized altitude
        self.obs_buf[..., 1]     = (altitude_rate_of_change / 500)                                  # normalized altitude rate of change
        self.obs_buf[..., 2]     = roll / np.pi                                                     # normalized roll
        self.obs_buf[..., 3]     = pitch / np.pi                                                    # normalized pitch
        self.obs_buf[..., 4]     = yaw / (2 * np.pi)                                                # normalized yaw
        self.obs_buf[..., 5]     = (desired_altitude - 0) / (4 - 0)                                 # normalized desired altitude
        self.obs_buf[..., 6]     = desired_roll / np.pi                                             # normalized desired roll
        self.obs_buf[..., 7]     = desired_pitch / np.pi                                            # normalized desired pitch
        self.obs_buf[..., 8]     = desired_yaw / (2 * np.pi)                                        # normalized desired yaw
        self.obs_buf[..., 9]     = (distance_front / 4)                                             # normalized distance front
        self.obs_buf[..., 10]    = (distance_back / 4)                                              # normalized distance back
        self.obs_buf[..., 11]    = collision                                                        # collision (no normalization needed)
        self.obs_buf[..., 12]    = too_high                                                         # too high (no normalization needed)

        self.previous_altitude = corrected_altitude.clone()
        self.previous_pitch = pitch.clone()
        
        return self.obs_buf


    def compute_observations(self, actions):
        num_features = 78  # Updated to include back, left, right cameras
        num_raw_features = 78  
        self.obs_buf = torch.zeros((self.num_envs, num_features), device=self.device)
        self.obs_raw = torch.zeros((self.num_envs, num_raw_features), device=self.device)  

        depth_images, depth_values = self.process_depth_images()
        depth_values = [torch.tensor(depth, device=self.device) for depth in depth_values]

        min_depths = [torch.amin(torch.tensor(depth, device=self.device), dim=1) for depth in depth_values]
        
        # Process front, back, left, and right cameras (8x8)
        tof_front = depth_values[0].view(self.num_envs, 8, 8)
        tof_back = depth_values[1].view(self.num_envs, 8, 8)
        tof_left = depth_values[2].view(self.num_envs, 8, 8)
        tof_right = depth_values[3].view(self.num_envs, 8, 8)
        
        # Downsample each camera view
        min_front, mean_front = self.downsample_tof(tof_front, grid_size=3)
        min_back, mean_back = self.downsample_tof(tof_back, grid_size=3)
        min_left, mean_left = self.downsample_tof(tof_left, grid_size=3)
        min_right, mean_right = self.downsample_tof(tof_right, grid_size=3)
        
        depth_change_front = min_front - self.previous_min_front if hasattr(self, 'previous_min_front') else torch.zeros_like(min_front)
        depth_change_back = min_back - self.previous_min_back if hasattr(self, 'previous_min_back') else torch.zeros_like(min_back)
        depth_change_left = min_left - self.previous_min_left if hasattr(self, 'previous_min_left') else torch.zeros_like(min_left)
        depth_change_right = min_right - self.previous_min_right if hasattr(self, 'previous_min_right') else torch.zeros_like(min_right)
        
        roll, pitch, yaw = self.quaternion_to_euler(self.root_quats)
        pitch_change = pitch - self.previous_pitch if hasattr(self, 'previous_pitch') else torch.tensor(0.0, device=self.device)
        yaw_rate = (yaw - self.previous_yaw) if hasattr(self, 'previous_yaw') else torch.tensor(0.0, device=self.device)

        collision = self.collisions

        prev_pitch_action = self.previous_pitch_action if hasattr(self, 'previous_pitch_action') else torch.tensor(0.0, device=self.device)
        prev_yaw_action = self.previous_yaw_action if hasattr(self, 'previous_yaw_action') else torch.tensor(0.0, device=self.device)
        prev_thrust_action = self.previous_thrust_action if hasattr(self, 'previous_thrust_action') else torch.tensor(0.0, device=self.device)

        self.obs_raw[..., 0] = prev_pitch_action
        self.obs_raw[..., 1] = pitch
        self.obs_raw[..., 2] = prev_yaw_action
        self.obs_raw[..., 3] = yaw_rate
        self.obs_raw[..., 4] = collision
        
        # Store all downsampled ToF matrices
        self.obs_raw[..., 5:14] = min_front.view(self.num_envs, -1)
        self.obs_raw[..., 14:23] = depth_change_front.view(self.num_envs, -1)
        self.obs_raw[..., 23:32] = min_back.view(self.num_envs, -1)
        self.obs_raw[..., 32:41] = depth_change_back.view(self.num_envs, -1)
        self.obs_raw[..., 41:50] = min_left.view(self.num_envs, -1)
        self.obs_raw[..., 50:59] = depth_change_left.view(self.num_envs, -1)
        self.obs_raw[..., 59:68] = min_right.view(self.num_envs, -1)
        self.obs_raw[..., 68:77] = depth_change_right.view(self.num_envs, -1)

        self.obs_buf[..., 0] = prev_pitch_action / 4
        self.obs_buf[..., 1] = pitch / np.pi  
        self.obs_buf[..., 2] = prev_yaw_action / 4
        self.obs_buf[..., 3] = yaw_rate / (2 * np.pi) 
        self.obs_buf[..., 4] = collision  
        self.obs_buf[..., 5:14]  = min_front.view(self.num_envs, -1) / 4.0  
        self.obs_buf[..., 14:23] = depth_change_front.view(self.num_envs, -1) / 4.0  
        self.obs_buf[..., 23:32] = min_back.view(self.num_envs, -1) / 4.0  
        self.obs_buf[..., 32:41] = depth_change_back.view(self.num_envs, -1) / 4.0  
        self.obs_buf[..., 41:50] = min_left.view(self.num_envs, -1) / 4.0  
        self.obs_buf[..., 50:59] = depth_change_left.view(self.num_envs, -1) / 4.0  
        self.obs_buf[..., 59:68] = min_right.view(self.num_envs, -1) / 4.0  
        self.obs_buf[..., 68:77] = depth_change_right.view(self.num_envs, -1) / 4.0  

        self.previous_pitch_action = actions[:, 2] if actions is not None else torch.zeros(self.num_envs, device=self.device)
        self.previous_yaw_action = actions[:, 3] if actions is not None else torch.zeros(self.num_envs, device=self.device)
        self.previous_thrust_action = actions[:, 0] if actions is not None else torch.zeros(self.num_envs, device=self.device)

        self.previous_pitch = pitch.clone()
        self.previous_yaw = yaw.clone()
        self.previous_min_front = min_front.clone()
        self.previous_min_back = min_back.clone()
        self.previous_min_left = min_left.clone()
        self.previous_min_right = min_right.clone()

        return self.obs_buf



    def compute_reward(self):
        self.rew_buf[:], self.reset_buf[:] = compute_quadcopter_reward(
            self.root_positions,
            self.root_quats,
            self.root_linvels,
            self.root_angvels,
            self.reset_buf, self.progress_buf, self.max_episode_length
        )
 

    def compute_reward_pitch(self):
        pitch = self.obs_buf[..., 1]
        collision = self.obs_buf[..., 4]
        min_front_distance = self.obs_buf[..., 5:14].min(dim=-1).values

        reward_pitch = torch.where(
                pitch <= 0.1,
                torch.exp(-20*torch.abs(pitch-0.1)),
                torch.exp(-3*torch.abs(pitch-0.1))
            )

        reward_pitch = torch.where(pitch < 0.0, torch.tensor(0.0, dtype=torch.float32, device=self.device), reward_pitch)

        penalty_negative_pitch = torch.where(
            pitch < 0.0,
            torch.abs(pitch),
            torch.tensor(0.0, dtype=torch.float32, device=self.device)
        )

        penalty_collision = -1 * collision

        print(f"{'Pitch':<10} {'Reward Pitch':<15} {'Negative Pitch Penalty':<25}")
        print(f"{pitch[0].item():<10.3f} {reward_pitch[0].item():<15.3f} {penalty_negative_pitch[0].item():<25.3f}")
    

        return 0.1 * reward_pitch + penalty_collision + penalty_negative_pitch



    def compute_reward_altitude(self):
        target_altitude = 1.5
        altitude_tolerance = 0.02  
        stability_bonus = 0.5  
        obstacle_distance_target = 0.5  
        obstacle_distance_tolerance = 0.02  

        current_distance = self.obs_buf[..., 0]  
        distance_to_obstacle_front = self.obs_buf[..., 2]
        distance_to_obstacle_back = self.obs_buf[..., 3]    
        pitch_angle = self.obs_buf[..., 4]  

        corrected_altitude = current_distance * torch.cos(pitch_angle)

        distance_penalty = -0.01 * torch.abs(target_altitude - corrected_altitude)
        reward = distance_penalty.clone()
        within_altitude_tolerance = torch.abs(target_altitude - corrected_altitude) < altitude_tolerance
        reward[within_altitude_tolerance] += stability_bonus

        obstacle_detected_front = distance_to_obstacle_front <= 1.0
        obstacle_detected_back = distance_to_obstacle_back <= 1.0

        if obstacle_detected_front.any():
            prev_distance_front = self.prev_distance_to_obstacle_front  
            distance_penalty_front = -0.01 * (distance_to_obstacle_front - prev_distance_front)
            reward += distance_penalty_front
            if (distance_to_obstacle_front > prev_distance_front).any():
                reward += 0.1 

        if obstacle_detected_back.any():
            prev_distance_back = self.prev_distance_to_obstacle_back 
            distance_penalty_back = -0.01 * (distance_to_obstacle_back - prev_distance_back)
            reward += distance_penalty_back
            if (distance_to_obstacle_back > prev_distance_back).any():
                reward += 0.1  

        self.prev_distance_to_obstacle_front = distance_to_obstacle_front
        self.prev_distance_to_obstacle_back = distance_to_obstacle_back

        collision_mask = self.collisions > 0
        too_high_mask = self.too_high > 0
        reward[collision_mask] += -1
        reward[too_high_mask] += -1

        return reward



    def compute_reward_random(self, tolerance_altitude=0.05, bonus=0.5):
        target_altitude = self.obs_buf[..., 5]
        altitude = self.obs_buf[..., 0]  
        roll = self.obs_buf[..., 2]      
        pitch = self.obs_buf[..., 3]
        yaw = self.obs_buf[..., 4]     
        collision = self.collisions      
        too_high = self.too_high        

        lambda1 = -0.1  # Penalize altitude deviation 
        lambda2 = -0.1   # Penalize roll
        lambda3 = 0.1   # Penalize pitch
        lambda4 = -5  # Penalize collision
        lambda5 = -5  # Penalize flying too high
        bonus_altitude = bonus  # Reward for staying within tolerance
        
        min_pitch_threshold = 0.05   
        max_pitch_threshold = 0.1   

        altitude_deviation = torch.abs(altitude - target_altitude)

        altitude_penalty = lambda1 * altitude_deviation
        print(altitude_penalty[0]) 

        altitude_reward = torch.zeros_like(altitude_penalty)  

        within_tolerance = altitude_deviation <= tolerance_altitude
        altitude_reward[within_tolerance] += bonus_altitude  

        altitude_reward[~within_tolerance] = altitude_penalty[~within_tolerance]

        pitch_reward = lambda3 * torch.clamp(pitch - min_pitch_threshold, min=0)  

        pitch_penalty = torch.where(pitch > max_pitch_threshold, -lambda3 * (pitch - max_pitch_threshold), torch.zeros_like(pitch))

        roll_penalty = lambda2 * torch.abs(roll)

        collision_penalty = lambda4 * collision  
        too_high_penalty = lambda5 * too_high  

        total_reward = altitude_reward + pitch_reward + pitch_penalty + roll_penalty + collision_penalty + too_high_penalty

        return total_reward


    
    def compute_reward_no_attitude(self):
        target_altitude = self.obs_buf[..., 5]
        current_altitude = self.obs_buf[..., 0]  
        current_roll = self.obs_buf[..., 2]      
        current_pitch = self.obs_buf[..., 3]
        yaw = self.obs_buf[..., 4]     
        collision = self.collisions      
        too_high = self.too_high

        tolerance = 0.01
        bonus = 1.0
        lambda_altitude_penalty = 1
        lambda_roll_penalty = 0.2
        lambda_pitch_penalty = -0.1
        zero_roll_pitch_bonus = 0.5
        lambda_collision_penalty = 5.0  
        lambda_too_high_penalty = 2.0 
        altitude_deviation = torch.abs(current_altitude - target_altitude)  
        is_at_target = torch.isclose(altitude_deviation, torch.zeros_like(altitude_deviation, device=self.device), atol=1e-4)


        if is_at_target.any():
            altitude_reward = bonus  
        else:
            altitude_reward = -lambda_altitude_penalty * altitude_deviation  

        altitude_reward += torch.where(altitude_deviation <= tolerance, torch.tensor(0.5, device=self.device), torch.tensor(0.0, device=self.device))

        
        roll_penalty = -lambda_roll_penalty * torch.abs(current_roll)  
        
        desired_pitch = 0.05
        pitch_deviation = torch.abs(current_pitch - desired_pitch)
        pitch_reward = torch.exp(-10 * torch.abs(current_pitch - desired_pitch)) * lambda_pitch_penalty


        zero_roll_pitch_bonus = torch.tensor(0.5, dtype=torch.float32, device=self.device)  

        zero_roll_reward = torch.where(
            torch.abs(current_roll) < 1e-3,  
            zero_roll_pitch_bonus,  
            zero_roll_pitch_bonus * torch.exp(-10 * torch.abs(current_roll).to(torch.float32))  
        )

        collision_penalty = -lambda_collision_penalty * collision  

        too_high_penalty = -lambda_too_high_penalty * too_high  

        total_reward = altitude_reward + roll_penalty + pitch_reward + zero_roll_reward + collision_penalty + too_high_penalty

        print(f"Altitude: {current_altitude[0]:.3f}, Target: {target_altitude[0]:.3f}, Reward: {altitude_reward[0]:.3f}")
        print(f"Roll: {current_roll[0]:.3f}, Penalty: {roll_penalty[0]:.3f}")
        print(f"Pitch: {current_pitch[0]:.3f}, Desired: {desired_pitch:.3f}, Reward: {pitch_reward[0]:.3f}")
        print(f"Zero Roll Bonus: {zero_roll_reward[0]:.3f}")
        print(f"Collision: {collision[0]:.0f}, Penalty: {collision_penalty[0]:.3f}")
        print(f"Too High: {too_high[0]:.0f}, Penalty: {too_high_penalty[0]:.3f}")        
        print(f"Total Reward: {total_reward[0]:.3f}")
        print("\n")

        return total_reward




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
def compute_quadcopter_reward(root_positions, root_quats, root_linvels, root_angvels, reset_buf, progress_buf, max_episode_length):
    # type: (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, float) -> Tuple[Tensor, Tensor]

    ## The reward function set here is arbitrary and the user is encouraged to modify this as per their need to achieve collision avoidance.

    # distance to target
    target_dist = torch.sqrt(root_positions[..., 0] * root_positions[..., 0] +
                             root_positions[..., 1] * root_positions[..., 1] +
                             (root_positions[..., 2]) * (root_positions[..., 2]))
    pos_reward = 2.0 / (1.0 + target_dist * target_dist)

    # uprightness
    ups = quat_axis(root_quats, 2)
    tiltage = torch.abs(1 - ups[..., 2])
    up_reward = 1.0 / (1.0 + tiltage * tiltage)

    # spinning
    spinnage = torch.abs(root_angvels[..., 2])
    spinnage_reward = 1.0 / (1.0 + spinnage * spinnage)

    # combined reward
    # uprigness and spinning only matter when close to the target
    reward = pos_reward + pos_reward * (up_reward + spinnage_reward)

    # resets due to misbehavior
    ones = torch.ones_like(reset_buf)
    die = torch.zeros_like(reset_buf)
    # die = torch.where(target_dist > 10.0, ones, die)

    # resets due to episode length
    reset = torch.where(progress_buf >= max_episode_length - 1, ones, die)
    reset = torch.where(torch.norm(root_positions, dim=1) > 20, ones, reset)


    return reward, reset

@torch.jit.script
def compute_rewards(current_altitude, current_distance_to_obstacle, current_yaw_angle, left_distance_to_obstacle, right_distance_to_obstacle):
    # Target values as tensors
    target_altitude = torch.tensor(0.3, device=current_altitude.device)
    target_distance_to_obstacle = torch.tensor(0.3, device=current_distance_to_obstacle.device)
    yaw_target = torch.tensor(90.0, device=current_yaw_angle.device)  # Target yaw angle in degrees

    # Altitude stability reward
    altitude_error = torch.abs(current_altitude - target_altitude)  # Use torch.abs for tensor operations
    altitude_reward = torch.clamp(1.0 - altitude_error, min=0.0)  # Clamp ensures reward is non-negative

    # Front obstacle avoidance reward
    front_distance_error = torch.abs(current_distance_to_obstacle - target_distance_to_obstacle)
    front_distance_reward = torch.clamp(1.0 - front_distance_error, min=0.0)

    # Yaw stability reward (normalized between 0 and 1)
    yaw_error = torch.abs(current_yaw_angle - yaw_target)
    yaw_reward = torch.clamp(1.0 - yaw_error / 180.0, min=0.0)  # Normalize by 180 degrees

    # Side obstacle penalty
    side_penalty = torch.tensor(0.0, device=current_altitude.device)
    if left_distance_to_obstacle < target_distance_to_obstacle:
        side_penalty += 0.5
    if right_distance_to_obstacle < target_distance_to_obstacle:
        side_penalty += 0.5

    # Total reward (higher reward for staying stable and avoiding obstacles)
    reward = altitude_reward + front_distance_reward + yaw_reward - side_penalty

    # Reset condition (e.g., drone crashes or deviates too much)
    reset = torch.tensor(0.0, device=current_altitude.device)
    if current_distance_to_obstacle < 0.05 or current_altitude > 0.4 or altitude_error > 0.5:
         reset = torch.tensor(1.0, device=current_altitude.device)
    return reward, reset


