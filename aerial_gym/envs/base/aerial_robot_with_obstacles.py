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

from aerial_gym import AERIAL_GYM_ROOT_DIR, AERIAL_GYM_ROOT_DIR

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
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(28,), dtype=np.float32)


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

        self.env_asset_root_states = self.vec_root_tensor[:, 1:, :]
        
        self.last_altitude = 0.0  # Initialize with target altitude or some reasonable default value
        self.previous_altitude = 0.0

        history_length = 4
        self.history_length = history_length
        self.attitude_history = torch.zeros((history_length, 3), device=self.device) 
        #self.altitude_history = torch.zeros((history_length, 1), device=self.device)
        #self.altitude_rate_history = torch.zeros((history_length, 1), device=self.device)
        self.altitude_history = []  # Python list for storing altitude history
        self.altitude_rate_history = []
        self.distance_front_history = []
        self.distance_rate_front_history = []
        self.pitch_history = []
        self.roll_history = []
        self.yaw_history = [] 
        self.current_timestep = 0

        self.frame_stack = []
        self.obs_buf_size = 10
        
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
        local_transform.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)

        # local transform for the second camera
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

    def step(self, actions):
        self.reset_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
        
        max_episode_timesteps = 5000
        for i in range(self.cfg.env.num_control_steps_per_env_step):
            self.pre_physics_step(actions)
            self.gym.simulate(self.sim)
            self.post_physics_step()

        self.render(sync_frame_time=False)
        if self.enable_onboard_cameras:
            self.render_cameras()
        
        self.progress_buf += 1
        self.collisions = torch.zeros(self.num_envs, device=self.device)
        self.too_high = torch.zeros(self.num_envs, device=self.device)

        self.check_collisions()

        self.compute_observations()
        self.check_altitude()
        altitude_reward = self.compute_reward_altitude()
        altitude_reward = torch.tensor(altitude_reward, device=self.device)

        #if altitude_reward.dim() == 1:
        #    altitude_reward = altitude_reward.unsqueeze(1) 
        
        

        self.rew_buf[:] = altitude_reward  
        if self.cfg.env.reset_on_collision:
            ones = torch.ones_like(self.reset_buf)
            self.reset_buf = torch.where(self.collisions > 0, ones, self.reset_buf)
            self.reset_buf = torch.where(self.too_high > 0, ones, self.reset_buf)

        #if self.collisions > 0:
        #    print("COLLISION")

        # Reset environments that exceed 300 timesteps
        self.reset_buf = torch.where(self.progress_buf >= max_episode_timesteps, torch.ones_like(self.reset_buf), self.reset_buf)

        reset_env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)

        if len(reset_env_ids) > 0:
            self.reset_idx(reset_env_ids)

        self.time_out_buf = self.progress_buf > self.max_episode_length
        self.extras["time_outs"] = self.time_out_buf

        return self.obs_buf, self.privileged_obs_buf, self.rew_buf, self.reset_buf, self.extras, self.root_quats

    def steptwo(self, actions):
        # step physics and render each frame
        for i in range(self.cfg.env.num_control_steps_per_env_step):
            self.pre_physics_step(actions)
            self.gym.simulate(self.sim)
            # NOTE: as per the isaacgym docs, self.gym.fetch_results must be called after self.gym.simulate, but not having it here seems to work fine
            # it is called in the render function.
            self.post_physics_step()

        self.render(sync_frame_time=False)
        if self.enable_onboard_cameras:
            self.render_cameras()
        
        self.progress_buf += 1

        self.check_collisions()
        self.compute_observations()
        self.compute_reward_altitude()
        if self.cfg.env.reset_on_collision:
            ones = torch.ones_like(self.reset_buf)
            self.reset_buf = torch.where(self.collisions > 0, ones, self.reset_buf)
            self.reset_buf = torch.where(self.too_high > 0, ones, self.reset_buf)

        reset_env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        if len(reset_env_ids) > 0:
            self.reset_idx(reset_env_ids)

        self.time_out_buf = self.progress_buf > self.max_episode_length
        self.extras["time_outs"] = self.time_out_buf

        return self.obs_buf, self.privileged_obs_buf, self.rew_buf, self.reset_buf, self.extras, self.root_quats

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
        drone_positions = torch.tensor([0.0, 0.0, 0.5], device=self.device)

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

    # def pre_physics_step(self, _actions):
    #     # resets
    #     if self.counter % 250 == 0:
    #         print("self.counter:", self.counter)
    #     self.counter += 1

        
    #     actions = _actions.to(self.device)
    #     actions = tensor_clamp(
    #         actions, self.action_lower_limits, self.action_upper_limits)
    #     self.action_input[:] = actions

    #     # clear actions for reset envs
    #     self.forces[:] = 0.0
    #     self.torques[:, :] = 0.0

    #     output_thrusts_mass_normalized, output_torques_inertia_normalized = self.controller(self.root_states, self.action_input)
    #     self.forces[:, 0, 2] = self.robot_mass * (-self.sim_params.gravity.z) * output_thrusts_mass_normalized
    #     self.torques[:, 0] = output_torques_inertia_normalized
    #     self.forces = torch.where(self.forces < 0, torch.zeros_like(self.forces), self.forces)

    #     # apply actions
    #     self.gym.apply_rigid_body_force_tensors(self.sim, gymtorch.unwrap_tensor(
    #         self.forces), gymtorch.unwrap_tensor(self.torques), gymapi.LOCAL_SPACE)

    def pre_physics_step(self, _actions):
         if self.counter % 250 == 0:
             print("self.counter:", self.counter)
         self.counter += 1

         actions = torch.tensor(_actions, dtype=torch.float32).to(self.device)
         actions = tensor_clamp(actions, self.action_lower_limits, self.action_upper_limits)
         self.action_input[:] = actions

         self.forces[:] = 0.0
         self.torques[:, :] = 0.0

         output_thrusts_mass_normalized, output_torques_inertia_normalized = self.controller(self.root_states, self.action_input)

         self.forces[:, 0, 2] = self.robot_mass * (-self.sim_params.gravity.z) * output_thrusts_mass_normalized
         self.torques[:, 0] = output_torques_inertia_normalized

         v_x_k = self.root_linvels[:, 0]  
         v_y_k = self.root_linvels[:, 1] 

         g = 9.81  
         dt = self.sim_params.dt  
         b_x = 0.42  
         b_y = 0.18  

         drag_force_x = -b_x * v_x_k
         drag_force_y = -b_y * v_y_k 

         self.forces[:, 0, 0] += drag_force_x  
         self.forces[:, 0, 1] += drag_force_y 
         
         
         self.gym.apply_rigid_body_force_tensors(self.sim, gymtorch.unwrap_tensor(self.forces), gymtorch.unwrap_tensor(self.torques), gymapi.LOCAL_SPACE)



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
        
        max_depth = 1.0 

        for camera_array in [self.full_camera_array1, self.full_camera_array2, self.full_camera_array3,
                            self.full_camera_array4, self.full_camera_array5]:
            
            depth_np = camera_array.cpu().numpy() / 10.0
            
            depth_np = np.clip(depth_np, 0, max_depth)
            
            depth_img = np.uint8(depth_np * 255 / max_depth)
            
            depth_images.append(depth_img)
            depth_values.append(depth_np)

        return depth_images, depth_values  
    
    # def pre_physics_step(self, _actions):
    #     # resets
    #     if self.counter % 250 == 0:
    #         print("self.counter:", self.counter)
    #     self.counter += 1

        
    #     actions = _actions.to(self.device)
    #     actions = tensor_clamp(
    #         actions, self.action_lower_limits, self.action_upper_limits)
    #     self.action_input[:] = actions

    #     # clear actions for reset envs
    #     self.forces[:] = 0.0
    #     self.torques[:, :] = 0.0

    #     output_thrusts_mass_normalized, output_torques_inertia_normalized = self.controller(self.root_states, self.action_input)
    #     self.forces[:, 0, 2] = self.robot_mass * (-self.sim_params.gravity.z) * output_thrusts_mass_normalized
    #     self.torques[:, 0] = output_torques_inertia_normalized
    #     self.forces = torch.where(self.forces < 0, torch.zeros_like(self.forces), self.forces)

    #     # apply actions
    #     self.gym.apply_rigid_body_force_tensors(self.sim, gymtorch.unwrap_tensor(
    #         self.forces), gymtorch.unwrap_tensor(self.torques), gymapi.LOCAL_SPACE)


    ######## USUALLY ALTITUDE IS 7 BUT FOR NOW IT IS 3 SO CHANGE IT
    def check_altitude(self):
        self.too_high = torch.zeros_like(self.obs_buf[:, 0], dtype=torch.int)  # Create a tensor of zeros        
        self.too_high[self.obs_buf[:, 0] >= 1] = 1

    
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
        t2 = torch.clamp(t2, -1.0, 1.0)
        pitch = torch.asin(t2)

        t3 = +2.0 * (w * z + x * y)
        t4 = +1.0 - 2.0 * (y * y + z * z)
        yaw = torch.atan2(t3, t4)

        return roll, pitch, yaw

    # def compute_observations(self):
    #     obs_buf_size = 8
    #     self.obs_buf = torch.zeros((self.num_envs, obs_buf_size), device=self.device)

    #     roll, pitch, yaw = self.quaternion_to_euler(self.root_quats)
        
    #     self.obs_buf[..., 0] = pitch
    #     self.obs_buf[..., 1] = roll
    #     self.obs_buf[..., 2] = yaw
    #     depth_images, depth_values = self.process_depth_images()

    #     depth_values = [torch.tensor(depth, device=self.device) for depth in depth_values]

    #     min_depths = [depth.min().unsqueeze(0) for depth in depth_values] 
    #     self.obs_buf[..., 3:8] = torch.cat(min_depths, dim=0)

    #     detection_threshold = 0.3
    #     self.extras["front_detected"] = (min_depths[0] < detection_threshold).float()
    #     self.extras["left_detected"] = (min_depths[1] < detection_threshold).float()
    #     self.extras["right_detected"] = (min_depths[2] < detection_threshold).float()
    #     self.extras["back_detected"] = (min_depths[3] < detection_threshold).float()
    #     self.extras["down_detected"] = (min_depths[4] < detection_threshold).float()

    #     return self.obs_buf

    def compute_observations_v1(self):
        """
        Compute the current observation buffer, including pitch, roll, yaw,
        distance to ground, and rate of change in altitude (vertical velocity).
        """
        obs_buf_size = 5 
        self.obs_buf = torch.zeros((self.num_envs, obs_buf_size), device=self.device)

        roll, pitch, yaw = self.quaternion_to_euler(self.root_quats)
        self.obs_buf[..., 0] = pitch
        self.obs_buf[..., 1] = roll
        self.obs_buf[..., 2] = yaw

        depth_images, depth_values = self.process_depth_images()
        depth_values = [torch.tensor(depth, device=self.device) for depth in depth_values]
        min_depths = [depth.min().unsqueeze(0) for depth in depth_values]
        current_altitude = min_depths[4]  
        self.obs_buf[..., 3] = current_altitude

        if hasattr(self, 'previous_altitude') and self.previous_altitude is not None:
            delta_t = self.cfg.sim.dt  
            altitude_rate_of_change = (current_altitude - self.previous_altitude) / delta_t
        else:
            altitude_rate_of_change = torch.zeros_like(current_altitude) 

        self.obs_buf[..., 4] = altitude_rate_of_change

        self.previous_altitude = current_altitude.clone()

        current_attitudes = torch.tensor([pitch, roll, yaw], device=self.device)
        current_distances = torch.cat(min_depths)

        detection_threshold = 0.3
        self.extras["down_detected"] = (current_altitude < detection_threshold).float()
        #print("Altitude change ",self.obs_buf[..., 4])

        return self.obs_buf
    
    def compute_observations_v2(self):
        obs_buf_size = 2 + 2 * self.history_length  
        self.obs_buf = torch.zeros((self.num_envs, obs_buf_size), device=self.device)

        depth_images, depth_values = self.process_depth_images()
        depth_values = [torch.tensor(depth, device=self.device) for depth in depth_values]
        min_depths = [depth.min().unsqueeze(0) for depth in depth_values]
        current_altitude = min_depths[4] 
        
        if hasattr(self, 'previous_altitude') and self.previous_altitude is not None:
            delta_t = self.cfg.sim.dt 
            altitude_rate_of_change = (current_altitude - self.previous_altitude) / delta_t
        else:
            altitude_rate_of_change = torch.zeros_like(current_altitude)  

       # print("current altitude: ", current_altitude)
       # print("previous altitude: ", self.previous_altitude)
       # print("Difference: ", current_altitude - self.previous_altitude)

        self.altitude_history.append(current_altitude.item()) 
        self.altitude_rate_history.append(altitude_rate_of_change.item())  
        
        if len(self.altitude_history) > self.history_length:
            self.altitude_history.pop(0)
        if len(self.altitude_rate_history) > self.history_length:
            self.altitude_rate_history.pop(0)

        self.obs_buf[..., 0] = current_altitude
        self.obs_buf[..., 1] = altitude_rate_of_change

        for i in range(self.history_length):
            if i < len(self.altitude_history):
                self.obs_buf[..., 2 + i] = torch.tensor(self.altitude_history[-(i + 1)], device=self.device)
            else:
                self.obs_buf[..., 2 + i] = torch.tensor(0.0, device=self.device) 

            if i < len(self.altitude_rate_history):
                self.obs_buf[..., 2 + self.history_length + i] = torch.tensor(self.altitude_rate_history[-(i + 1)], device=self.device)
            else:
                self.obs_buf[..., 2 + self.history_length + i] = torch.tensor(0.0, device=self.device) 

        # Save the previous observation buffer
        if not hasattr(self, 'previous_obs_buf'):
            self.previous_obs_buf = torch.zeros_like(self.obs_buf, device=self.device)  # Initialize if not present
        self.previous_obs_buf.copy_(self.obs_buf)

        # Save the current altitude for future computations
        self.previous_altitude = current_altitude.clone()

        # Optional: Add a detection for when the drone is below a threshold
        detection_threshold = 0.3
        self.extras["down_detected"] = (current_altitude < detection_threshold).float()

        return self.obs_bufs

    def compute_observations(self):
        num_features = 7  # altitude, rate of change, pitch, roll, yaw, distance front, rate front
        stacked_features = num_features * self.history_length  # Total stacked features
        self.obs_buf = torch.zeros((self.num_envs, stacked_features), device=self.device)

        depth_images, depth_values = self.process_depth_images()
        depth_values = [torch.tensor(depth, device=self.device) for depth in depth_values]
        min_depths = [depth.min().unsqueeze(0) for depth in depth_values]
        current_altitude = min_depths[4]
        distance_front = min_depths[0]

        if hasattr(self, 'previous_altitude') and self.previous_altitude is not None:
            delta_t = self.cfg.sim.dt
            altitude_rate_of_change = (current_altitude - self.previous_altitude) / delta_t
        else:
            altitude_rate_of_change = torch.zeros_like(current_altitude)

        if hasattr(self, 'previous_distance_front') and self.previous_distance_front is not None:
            distance_rate_front = (distance_front - self.previous_distance_front) / delta_t
        else:
            distance_rate_front = torch.zeros_like(distance_front)

        roll, pitch, yaw = self.quaternion_to_euler(self.root_quats)

        self.altitude_history.append(current_altitude.item())
        self.altitude_rate_history.append(altitude_rate_of_change.item())
        self.distance_front_history.append(distance_front.item())
        self.distance_rate_front_history.append(distance_rate_front.item())
        self.pitch_history.append(pitch)
        self.roll_history.append(roll)
        self.yaw_history.append(yaw)

        for history in [self.altitude_history, self.altitude_rate_history,
                        self.distance_front_history, self.distance_rate_front_history,
                        self.pitch_history, self.roll_history, self.yaw_history]:
            if len(history) > self.history_length:
                history.pop(0)

        self.obs_buf[..., 0] = current_altitude
        self.obs_buf[..., 1] = altitude_rate_of_change
        self.obs_buf[..., 2] = distance_front
        self.obs_buf[..., 3] = distance_rate_front
        self.obs_buf[..., 4] = pitch
        self.obs_buf[..., 5] = roll
        self.obs_buf[..., 6] = yaw

        stacked_obs = torch.zeros((self.num_envs, num_features * self.history_length), device=self.device)
        
        for i in range(self.history_length):
            idx = i * num_features  
            stacked_obs[:, idx + 0] = self.altitude_history[-(i + 1)] if i < len(self.altitude_history) else 0.0
            stacked_obs[:, idx + 1] = self.altitude_rate_history[-(i + 1)] if i < len(self.altitude_rate_history) else 0.0
            stacked_obs[:, idx + 2] = self.distance_front_history[-(i + 1)] if i < len(self.distance_front_history) else 0.0
            stacked_obs[:, idx + 3] = self.distance_rate_front_history[-(i + 1)] if i < len(self.distance_rate_front_history) else 0.0
            stacked_obs[:, idx + 4] = self.pitch_history[-(i + 1)] if i < len(self.pitch_history) else 0.0
            stacked_obs[:, idx + 5] = self.roll_history[-(i + 1)] if i < len(self.roll_history) else 0.0
            stacked_obs[:, idx + 6] = self.yaw_history[-(i + 1)] if i < len(self.yaw_history) else 0.0

        self.obs_buf[:, :] = stacked_obs

        self.previous_altitude = current_altitude.clone()
        self.previous_distance_front = distance_front.clone()

        return self.obs_buf
   
    def compute_observations_nohistory(self):
        obs_buf_size = 8 #+ (self.history_length * 3) + (self.history_length * 5)
        self.obs_buf = torch.zeros((self.num_envs, obs_buf_size), device=self.device)

        roll, pitch, yaw = self.quaternion_to_euler(self.root_quats)
        
        self.obs_buf[..., 0] = pitch
        self.obs_buf[..., 1] = roll
        self.obs_buf[..., 2] = yaw

        depth_images, depth_values = self.process_depth_images()
        depth_values = [torch.tensor(depth, device=self.device) for depth in depth_values]
        
        min_depths = [depth.min().unsqueeze(0) for depth in depth_values]
        self.obs_buf[..., 3:8] = torch.cat(min_depths, dim=0)

        current_attitudes = torch.tensor([pitch, roll, yaw], device=self.device)
        current_distances = torch.cat(min_depths) 

      #  self.update_history(current_attitudes, current_distances)

        #history_attitudes = self.attitude_history.flatten()
        #history_distances = self.distance_history.flatten()
      
        detection_threshold = 0.3
        self.extras["front_detected"] = (min_depths[0] < detection_threshold).float()
        self.extras["left_detected"] = (min_depths[1] < detection_threshold).float()
        self.extras["right_detected"] = (min_depths[2] < detection_threshold).float()
        self.extras["back_detected"] = (min_depths[3] < detection_threshold).float()
        self.extras["down_detected"] = (min_depths[4] < detection_threshold).float()

        return self.obs_buf

    def update_history(self, current_attitudes, current_distances):
        self.attitude_history[self.current_timestep % self.history_length] = current_attitudes
        self.distance_history[self.current_timestep % self.history_length] = current_distances
        self.current_timestep += 1

    def compute_reward(self):
        self.rew_buf[:], self.reset_buf[:] = compute_quadcopter_reward(
            self.root_positions,
            self.root_quats,
            self.root_linvels,
            self.root_angvels,
            self.reset_buf, self.progress_buf, self.max_episode_length
        )
 

    def compute_reward_altitude_old(self):
        target_altitude = 0.3
        tolerance = 0.01
        a = 10  # Scaling factor for altitude rewards
        b = 5   # Scaling factor for movement rewards
        c = 100  # Collision penalty
        d = 100  # Too-high penalty
        
        current_distance = self.obs_buf[..., 0]
        reward = 0

        # Positive reward for moving closer to target altitude
        distance_to_target = torch.abs(target_altitude - current_distance)
        reward -= a * distance_to_target  # Penalize distance

        # Extra reward for being within the tolerance range
        if torch.abs(current_distance - target_altitude) <= tolerance:
            reward += b

        # Strong penalties for dangerous states
        if self.collisions == 1:
            reward -= c  # Penalty for collision
        elif self.too_high == 1:
            reward -= d  # Penalty for flying too high

        print("REWARD: ", reward)
        return reward


    def compute_reward_altitude_cases(self):
        target_altitude = 0.3
        tolerance = 0.2
        median_low = target_altitude - tolerance / 2
        median_high = target_altitude + tolerance / 2
        a = 10
        reward = 0
        current_distance = self.obs_buf[..., 0]

        # Case 1: current altitude <= target - tolerance:
        if torch.where(current_distance <= target_altitude - tolerance):
            reward = -a**2 * torch.abs((target_altitude - tolerance) -  current_distance)
        # Case 2: current altitude > target - tolerance & current altitude < target & current altitude <= median_low:
        if current_distance > target_altitude - tolerance and current_distance < target_altitude and current_distance <= median_low:
            reward = -a * torch.abs((target_altitude - tolerance / 2) - current_distance)
        # Case 3: current altitude > target - tolerance & current altitude <= target & current altitude > median_low:
        if current_distance > target_altitude - tolerance and current_distance <= target_altitude and current_distance > median_low:
            reward = -a * torch.abs(target_altitude - current_distance)
        # Case 4: current altitude > target & current altitude < target + tolerance & current altitude <= median_high:
        if current_distance > target_altitude and current_distance < target_altitude + tolerance and current_distance <= median_high:
            reward = -a * torch.abs(target_altitude - current_distance)
        # Case 5: current altitude > target & current altitude <= target + tolerance & current altitude > median_high:
        if current_distance > target_altitude and current_distance <= target_altitude + tolerance and current_distance > median_high:
            reward = -a * torch.abs((target_altitude + tolerance / 2) - current_distance)
        # Case 6: current altitude > target + tolerance
        if current_distance > target_altitude + tolerance:
            reward = -a**2 * torch.abs((target_altitude + tolerance) - current_distance)

        if self.collisions == 1:
             reward += -100
        if self.too_high == 1:
             reward += -100

      #  print("REWARD: ", reward)

        return reward

    def compute_reward_altitude(self):
        target_altitude = 0.3
        tolerance = 0.05
        current_distance = self.obs_buf[..., 0]

        reward = -0.003 * torch.abs(target_altitude - current_distance)

        return reward

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


