# utils.py

import numpy as np

def cast_rays(env, robot_index, grid_size=8, max_distance=5.0):
    """
    Simulates Time-of-Flight (ToF) sensor by casting rays in multiple directions
    from the drone's position and checking for collisions with obstacles.

    Args:
    - env (isaacgym.VecTaskPython): The Isaac Gym environment instance.
    - robot_index (int): Index of the drone in the environment.
    - grid_size (int): Size of the grid (e.g., 8x8) for ray directions.
    - max_distance (float): Maximum distance to check for obstacles.

    Returns:
    - numpy.ndarray: Array of distances for each ray direction. 
      If an obstacle is detected within max_distance, it returns the distance; 
      otherwise, it returns max_distance.
    """
    if isinstance(grid_size, list) or isinstance(grid_size, tuple):
        num_rays = grid_size[0] * grid_size[1]  # Example for 2D grid
    else:
        num_rays = grid_size * grid_size
    ray_directions = generate_ray_directions(grid_size)

    # Get the position of the drone
    drone_position = env.root_states[robot_index, :3]

    distances = np.full(num_rays, max_distance)

    for i in range(num_rays):
        direction = ray_directions[i]
        endpoint = drone_position + direction * max_distance

        # Perform collision check using the physics engine
        hit_info = env.cast_ray(start=drone_position, end=endpoint)

        if hit_info.has_hit:
            distances[i] = hit_info.distance
    
    return distances

def generate_ray_directions(grid_size):
    directions = []
    for i in range(grid_size):
        # Generate ray direction logic here
        direction = [0, 0, 1]  # Example direction, replace with actual logic
        directions.append(direction)
    return directions

