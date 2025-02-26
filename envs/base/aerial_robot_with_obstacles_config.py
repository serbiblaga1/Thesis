# Copyright (c) 2023, Autonomous Robots Lab, Norwegian University of Science and Technology
# All rights reserved.

# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from aerial_gym.envs.base.base_config import BaseConfig

import numpy as np
from aerial_gym import AERIAL_GYM_ROOT_DIR

THIN_SEMANTIC_ID = 1
TREE_SEMANTIC_ID = 2
OBJECT_SEMANTIC_ID = 3
WALL_SEMANTIC_ID = 8
TOMATO_SEMANTIC_ID = 4

class AerialRobotWithObstaclesCfg(BaseConfig):
    seed = 1

    class env:
        num_envs = 1
        num_observations = 13
        get_privileged_obs = True # if True the states of all entitites in the environment will be returned as privileged observations, otherwise None will be returned
        num_actions = 4
        env_spacing = 5.0  # not used with heightfields/trimeshes
        episode_length_s = 100 # episode length in seconds
        num_control_steps_per_env_step = 10 # number of control & physics steps between camera renders
        enable_onboard_cameras = True # enable onboard cameras
        reset_on_collision = True # reset environment when contact force on quadrotor is above a threshold
        create_ground_plane = True # create a ground plane

    class viewer:
        ref_env = 0
        pos = [-5, -5, 4]  # [m]
        lookat = [0, 0, 0]  # [m]

    class sim:
        dt =  0.01
        substeps = 1
        gravity = [0., 0. , -9.81]  # [m/s^2]
        up_axis = 1  # 0 is y, 1 is z

        class physx:
            num_threads = 10
            solver_type = 1  # 0: pgs, 1: tgs
            num_position_iterations = 4
            num_velocity_iterations = 0
            contact_offset = 0.01  # [m]
            rest_offset = 0.0   # [m]
            bounce_threshold_velocity = 0.5 # [m/s]
            max_depenetration_velocity = 1.0
            max_gpu_contact_pairs = 2**23 #2**24 -> needed for 8000 envs and more
            default_buffer_size_multiplier = 5
            contact_collection = 1 # 0: never, 1: last sub-step, 2: all sub-steps (default=2)

    class control:
        """
        Control parameters
        controller:
            lee_position_control: command_actions = [x, y, z, yaw] in environment frame scaled between -1 and 1
            lee_velocity_control: command_actions = [vx, vy, vz, yaw_rate] in vehicle frame scaled between -1 and 1
            lee_attitude_control: command_actions = [thrust, roll, pitch, yaw_rate] in vehicle frame scaled between -1 and 1
        kP: gains for position
        kV: gains for velocity
        kR: gains for attitude
        kOmega: gains for angular velocity
        """
        controller = "lee_attitude_control" # or "lee_velocity_control" or "lee_attitude_control"
        kP = [0.8, 0.8, 1.0] # used for lee_position_control only
        kV = [0.5, 0.5, 0.4] # used for lee_position_control, lee_velocity_control only
        kR = [3.0, 3.0, 1.0] # used for lee_position_control, lee_velocity_control and lee_attitude_control
        kOmega = [0.5, 0.5, 1.20] # used for lee_position_control, lee_velocity_control and lee_attitude_control
        scale_input =[2.0, 1.0, 1.0, np.pi/4.0] # scale the input to the controller from -1 to 1 for each dimension

    class robot_asset:
        file = "{AERIAL_GYM_ROOT_DIR}/resources/robots/quad/model.urdf"
        name = "aerial_robot"  # actor name
        base_link_name = "base_link"
        disable_gravity = False
        collapse_fixed_joints = True # merge bodies connected by fixed joints.
        fix_base_link = False # fix the base of the robot
        collision_mask = 0 # 1 to disable, 0 to enable...bitwise filter
        replace_cylinder_with_capsule = False # replace collision cylinders with capsules, leads to faster/more stable simulation
        flip_visual_attachments = True # Some .obj meshes must be flipped from y-up to z-up
        density = 0.001
        angular_damping = 0.
        linear_damping = 0.
        max_angular_velocity = 100.
        max_linear_velocity = 100.
        armature = 0.001

    class asset_state_params(robot_asset):
        num_assets = 2                  # number of assets to include

        min_position_ratio = [0.5, 0.5, 0.5] # min position as a ratio of the bounds
        max_position_ratio = [0.5, 0.5, 0.5] # max position as a ratio of the bounds

        collision_mask = 1

        collapse_fixed_joints = True
        fix_base_link = True
        links_per_asset = 1
        set_whole_body_semantic_mask = False
        set_semantic_mask_per_link = False
        semantic_mask_link_list = [] # For empty list, all links are labeled
        specific_filepath = None # if not None, use this folder instead randomizing
        color = None


    class thin_asset_params(asset_state_params):
        num_assets = 1  # Increase the number of assets to match the positions

        collision_mask = 1  # objects with the same collision mask will not collide

        # Setting both min and max position ratio to the same values for specified positions
        max_position_ratio = [0.5, 0.5, 0.5]  
        min_position_ratio = [0.5, 0.5, 0.5]  

        # Manually specified positions for each asset
        specified_positions = [
            [0.0, 3.0, 0.2] 
        ] 

        min_euler_angles = [0.0, 0.0, 0.0]  # min euler angles
        max_euler_angles = [0.0, 0.0, 0.0]  # max euler angles

        # Orientation for each specified position
        specified_euler_angles = [
            [0.0, 0.0, 0.0]  # Orientation for Asset 1
        ]
        
        collapse_fixed_joints = True
        links_per_asset = 1
        set_whole_body_semantic_mask = True
        semantic_id = THIN_SEMANTIC_ID
        set_semantic_mask_per_link = False
        semantic_mask_link_list = []  # If nothing is specified, all links are labeled
        color = [170, 66, 66]  # Color of the asset

    class tree_asset_params(asset_state_params):
        num_assets = 4

        collision_mask = 1  # objects with the same collision mask will not collide

        max_position_ratio = [0.95, 0.95, 0.1]  # min position as a ratio of the bounds
        min_position_ratio = [0.05, 0.05, 0.0]  # max position as a ratio of the bounds

        specified_positions = [
            [0.0, 3.0, 0.2],  # Position for the asset in front
            [0.0, -7.0, 0.7], # Position for the asset in back
            [5.0, 0.0, 0.5],  # Position for the asset to the right
            [-4.0, 0.0, 0.4]  # Position for the asset to the left
        ]  # if > -900, use this value instead of randomizing the ratios

        min_euler_angles = [0, -np.pi / 6.0, -np.pi]  # min euler angles
        max_euler_angles = [0, np.pi / 6.0, np.pi]  # max euler angles

        specified_euler_angles = [
            [0.0, 0.0, 0.0],  # Orientation for the asset in front
            [0.0, 0.0, 0.0],  # Orientation for the asset in back
            [0.0, 0.0, 0.0],  # Orientation for the asset to the right
            [0.0, 0.0, 0.0]   # Orientation for the asset to the left
        ]
        # if > -900, use this value instead of randomizing

        collapse_fixed_joints = True
        links_per_asset = 1
        set_whole_body_semantic_mask = False
        set_semantic_mask_per_link = True
        semantic_mask_link_list = []  # If nothing is specified, all links are labeled
        semantic_id = TREE_SEMANTIC_ID
        color = [70, 200, 100]

    class object_asset_params(asset_state_params):
        num_assets = 27  # Increased to 10 assets

        # Setting both min and max position ratio to the same values for specified positions
        max_position_ratio = [0.5, 0.5, 0.5]  
        min_position_ratio = [0.5, 0.5, 0.5]  

        # Explicitly specified positions for each asset (increased number of assets)
        # Updated specified positions for assets
        specified_positions = [
            [5.0, 10.0, 1.0],    # Asset 1
            [10.0, -15.0, 1.2],  # Asset 2
            [-7.0, 3.0, 0.8],    # Asset 3
            [12.0, 5.0, 1.5],    # Asset 4
            [-15.0, -10.0, 0.6], # Asset 5
            [8.0, -12.0, 1.0],   # Asset 6
            [13.0, 7.0, 0.9],    # Asset 7
            [-10.0, 12.0, 1.3],  # Asset 8
            [-5.0, -18.0, 1.0],  # Asset 9
            [17.0, 0.0, 1.2],    # Asset 10
            [-12.0, 8.0, 1.1],   # Asset 11
            [0.0, -20.0, 1.0],   # Asset 12
            [20.0, -5.0, 1.5],   # Asset 13
            [4.0, 15.0, 1.2],    # Asset 14
            [-15.0, -5.0, 1.4],  # Asset 15
            [-20.0, 18.0, 0.7],  # Asset 16
            [2.0, -10.0, 1.0],   # Asset 17
            [10.0, 12.0, 0.8],   # Asset 18
            [-8.0, -10.0, 1.1],  # Asset 19
            [15.0, 5.0, 1.3],    # Asset 20
            [-18.0, 7.0, 0.9],   # Asset 21
            [-5.0, 18.0, 1.0],   # Asset 22
            [6.0, 8.0, 1.1],     # Asset 23 (new, close)
            [8.0, 10.0, 1.2],    # Asset 24 (new, close)
            [-6.0, -8.0, 0.9],   # Asset 25 (new, close)
            [-7.0, -6.0, 1.0],   # Asset 26 (new, close)
            [7.5, -9.0, 1.0],    # Asset 27 (new, close)
        ]

        specified_euler_angles = [
            [0.0, 0.0, 0.0],    # Asset 1
            [0.0, 0.0, 0.0],    # Asset 2
            [0.0, 0.0, 0.0],    # Asset 3
            [0.0, 0.0, 0.0],    # Asset 4
            [0.0, 0.0, 0.0],    # Asset 5
            [0.0, 0.0, 0.0],    # Asset 6
            [0.0, 0.0, 0.0],    # Asset 7
            [0.0, 0.0, 0.0],    # Asset 8
            [0.0, 0.0, 0.0],    # Asset 9
            [0.0, 0.0, 0.0],    # Asset 10
            [10.0, 0.0, 0.0],   # Asset 11
            [0.0, 15.0, 0.0],   # Asset 12
            [0.0, 0.0, 30.0],   # Asset 13
            [5.0, 10.0, 0.0],   # Asset 14
            [0.0, 0.0, 45.0],   # Asset 15
            [20.0, 0.0, 0.0],   # Asset 16
            [0.0, 25.0, 0.0],   # Asset 17
            [0.0, 0.0, 90.0],   # Asset 18
            [0.0, 10.0, 20.0],  # Asset 19
            [15.0, 0.0, 30.0],  # Asset 20
            [5.0, 5.0, 5.0],    # Asset 21
            [10.0, 0.0, 15.0],  # Asset 22
            [0.0, 0.0, 0.0],    # Asset 23 (new, close)
            [5.0, 15.0, 10.0],  # Asset 24 (new, close)
            [0.0, 0.0, 30.0],   # Asset 25 (new, close)
            [10.0, 0.0, 10.0],  # Asset 26 (new, close)
            [15.0, 5.0, 5.0],   # Asset 27 (new, close)
        ]


        min_euler_angles = [0.0, 0.0, 0.0]  # Min Euler angles
        max_euler_angles = [0.0, 0.0, 0.0]  # Max Euler angles

        links_per_asset = 1
        set_whole_body_semantic_mask = False
        set_semantic_mask_per_link = False
        semantic_id = OBJECT_SEMANTIC_ID

        # Optional color parameter
        # color = [80, 255, 100]


    class left_wall(asset_state_params):
        num_assets = 1

        collision_mask = 1  # objects with the same collision mask will not collide
        
        min_position_ratio = [0.5, -1.5, 0.5]  # min position as a ratio of the bounds
        max_position_ratio = [0.5, -1.5, 0.5]  # max position as a ratio of the bounds

        specified_positions = [[-1000.0, -1000.0, -1000.0]]  # if > -900, use this value instead of randomizing the ratios

        min_euler_angles = [0.0, 0.0, 0.0]  # min euler angles
        max_euler_angles = [0.0, 0.0, 0.0]  # max euler angles

        specified_euler_angles = [[-1000.0, -1000.0, -1000.0]]  # if > -900, use this value instead of randomizing
                
        collapse_fixed_joints = False
        links_per_asset = 1
        specific_filepath = "cube.urdf"
        semantic_id = WALL_SEMANTIC_ID
        color = [100, 200, 210]

    class right_wall(asset_state_params):
        num_assets = 1

        collision_mask = 1  # objects with the same collision mask will not collide

        min_position_ratio = [0.5, 2.5, 0.5]  # min position as a ratio of the bounds
        max_position_ratio = [0.5, 2.5, 0.5]  # max position as a ratio of the bounds

        specified_positions = [[-1000.0, -1000.0, -1000.0]]  # if > -900, use this value instead of randomizing the ratios

        min_euler_angles = [0.0, 0.0, 0.0]  # min euler angles
        max_euler_angles = [0.0, 0.0, 0.0]  # max euler angles

        specified_euler_angles = [[-1000.0, -1000.0, -1000.0]]  # if > -900, use this value instead of randomizing
        
        collapse_fixed_joints = False
        links_per_asset = 1
        specific_filepath = "cube.urdf"
        semantic_id = WALL_SEMANTIC_ID
        color = [100, 200, 210]

    class top_wall(asset_state_params):
        num_assets = 1

        collision_mask = 1  # objects with the same collision mask will not collide

        min_position_ratio = [0.5, 0.5, 1.0]  # min position as a ratio of the bounds
        max_position_ratio = [0.5, 0.5, 1.0]  # max position as a ratio of the bounds

        specified_position = [-1000.0, -1000.0, -1000.0]  # if > -900, use this value instead of randomizing the ratios

        min_euler_angles = [0.0, 0.0, 0.0]  # min euler angles
        max_euler_angles = [0.0, 0.0, 0.0]  # max euler angles

        specified_euler_angle = [-1000.0, -1000.0, -1000.0]  # if > -900, use this value instead of randomizing

        collapse_fixed_joints = False
        links_per_asset = 1
        specific_filepath = "cube.urdf"
        semantic_id = WALL_SEMANTIC_ID
        color = [100, 200, 210]

    class bottom_wall(asset_state_params):
        num_assets = 1

        collision_mask = 1  # objects with the same collision mask will not collide

        min_position_ratio = [0.5, 0.5, 0.0]  # min position as a ratio of the bounds
        max_position_ratio = [0.5, 0.5, 0.0]  # max position as a ratio of the bounds

        specified_position = [-1000.0, -1000.0, -1000.0]  # if > -900, use this value instead of randomizing the ratios

        min_euler_angles = [0.0, 0.0, 0.0]  # min euler angles
        max_euler_angles = [0.0, 0.0, 0.0]  # max euler angles

        specified_euler_angle = [-1000.0, -1000.0, -1000.0]  # if > -900, use this value instead of randomizing

        collapse_fixed_joints = False
        links_per_asset = 1
        specific_filepath = "cube.urdf"
        semantic_id = WALL_SEMANTIC_ID
        color = [100, 150, 150]

    class front_wall(asset_state_params):
        num_assets = 1

        collision_mask = 1  # objects with the same collision mask will not collide

        min_position_ratio = [2, 0.5, 0.5]  # min position as a ratio of the bounds
        max_position_ratio = [2, 0.5, 0.5]  # max position as a ratio of the bounds

        specified_positions = [[-1000.0, -1000.0, -1000.0]]  # if > -900, use this value instead of randomizing the ratios

        min_euler_angles = [0.0, 0.0, 0.0]  # min euler angles
        max_euler_angles = [0.0, 0.0, 0.0]  # max euler angles

        specified_euler_angles = [[-1000.0, -1000.0, -1000.0]]  # if > -900, use this value instead of randomizing

        collapse_fixed_joints = False
        links_per_asset = 1
        specific_filepath = "cube.urdf"
        semantic_id = WALL_SEMANTIC_ID
        color = [180, 150, 150]

    class back_wall(asset_state_params):
        num_assets = 1

        collision_mask = 1  # objects with the same collision mask will not collide

        min_position_ratio = [-1.5, 0.5, 0.5]  # min position as a ratio of the bounds
        max_position_ratio = [-1.5, 0.5, 0.5]  # max position as a ratio of the bounds

        specified_positions = [[-1000.0, -1000.0, -1000.0]]  # if > -900, use this value instead of randomizing the ratios

        min_euler_angles = [0.0, 0.0, 0.0]  # min euler angles
        max_euler_angles = [0.0, 0.0, 0.0]  # max euler angles

        specified_euler_angles = [[-1000.0, -1000.0, -1000.0]]  # if > -900, use this value instead of randomizing

        collapse_fixed_joints = False
        links_per_asset = 1
        specific_filepath = "cube.urdf"
        semantic_id = WALL_SEMANTIC_ID
        color = [180, 100, 150]


    class asset_config:
        folder_path = f"{AERIAL_GYM_ROOT_DIR}/resources/models/environment_assets"

        include_asset_type = {
            "thin": False,
            "trees": False,
            "objects": True
        }

        include_env_bound_type = {
            "front_wall": True,
            "left_wall": True,
            "top_wall": False,
            "back_wall": True,
            "right_wall": True,
            "bottom_wall": False
        }

        env_lower_bound_min = [-5.0, -5.0, 0.0]  # lower bound for the environment space
        env_lower_bound_max = [-5.0, -5.0, 0.0]  # lower bound for the environment space
        env_upper_bound_min = [5.0, 5.0, 5.0]  # upper bound for the environment space
        env_upper_bound_max = [5.0, 5.0, 5.0]  # upper bound for the environment space
    