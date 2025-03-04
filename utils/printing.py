import numpy as np
import cv2
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.linear_model import RANSACRegressor

def print_depth_maps_to_file(file_path, *depth_maps):
    max_val = max(np.max(depth_map) for depth_map in depth_maps)
    max_len = len(f"{max_val:.3f}")

    border = '+' + ('-' * (max_len + 2) + '+') * depth_maps[0].shape[1]
    
    combined_border = (' ' * 5).join([border] * len(depth_maps))

    with open(file_path, 'a') as f:
        f.write(combined_border + '\n')
        
        for rows in zip(*depth_maps):
            row_strs = []
            for row in rows:
                row_str = '|' + '|'.join(f" {value:{max_len}.3f} " for value in row) + '|'
                row_strs.append(row_str)
            f.write((' ' * 5).join(row_strs) + '\n')
            f.write(combined_border + '\n')
        f.write('------------------------------------------------------------------------------------------------------------------------' + '\n')

def print_depth_map(depth_map):
    max_val = np.max(depth_map)
    max_len = len(f"{max_val:.3f}")

    border = '+' + ('-' * (max_len + 2) + '+') * depth_map.shape[1]
    print(border)
    
    for row in depth_map:
        row_str = '|' + '|'.join(f" {value:{max_len}.3f} " for value in row) + '|'
        print(row_str)
        print(border)

def plot_point_cloud(points, title='3D Point Cloud'):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    ax.scatter(points[:, 0], points[:, 1], points[:, 2], c=points[:, 2], cmap='viridis', marker='o')
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('3D Point Cloud')
    
    plt.show()

def remove_ground_ransac(points, threshold=0.01):
    X = points[:, :2]  
    y = points[:, 2]  
    ransac = RANSACRegressor(residual_threshold=threshold)
    ransac.fit(X, y)
    inlier_mask = ransac.inlier_mask_

    filtered_points = points[~inlier_mask]
    return filtered_points

def filter_floor(points, floor_height=0.04, threshold=0.01):
    filtered_points = points[(points[:, 2] > floor_height + threshold) | (points[:, 2] < floor_height - threshold)]
    return filtered_points

def apply_custom_colormap(depth, max_depth=1.0):
    depth_normalized = depth / max_depth  
    depth_normalized = np.clip(depth_normalized, 0, 1)

    color_map = [
        (0.00, 0.04, (139, 0, 0)),       # Dark red
        (0.04, 0.08, (255, 0, 0)),       # Red
        (0.08, 0.12, (255, 69, 0)),      # Orange red
        (0.12, 0.16, (255, 140, 0)),     # Dark orange
        (0.16, 0.20, (255, 165, 0)),     # Orange
        (0.20, 0.24, (255, 215, 0)),     # Gold
        (0.24, 0.28, (255, 255, 0)),     # Yellow
        (0.28, 0.32, (173, 255, 47)),    # Green yellow
        (0.32, 0.36, (127, 255, 0)),     # Chartreuse
        (0.36, 0.40, (0, 255, 0)),       # Green
        (0.40, 0.44, (0, 255, 127)),     # Spring green
        (0.44, 0.48, (0, 255, 255)),     # Cyan
        (0.48, 0.52, (0, 191, 255)),     # Deep sky blue
        (0.52, 0.56, (0, 0, 255)),       # Blue
        (0.56, 0.60, (75, 0, 130)),      # Indigo
        (0.60, 0.64, (139, 0, 139)),     # Dark violet
        (0.64, 0.68, (255, 0, 255)),     # Magenta
        (0.68, 0.72, (255, 0, 127)),     # Deep pink
        (0.72, 0.76, (255, 0, 255)),     # Fuchsia
        (0.76, 0.80, (255, 20, 147)),    # Deep pink
        (0.80, 0.84, (255, 105, 180)),   # Hot pink
        (0.84, 0.88, (255, 182, 193)),   # Light pink
        (0.88, 0.92, (255, 192, 203)),   # Pink
        (0.92, 0.96, (255, 222, 173)),   # Navajo white
        (0.96, 1.00, (255, 239, 213))    # Papaya whip
    ]

    colored_image = np.zeros((depth.shape[0], depth.shape[1], 3), dtype=np.uint8)
    for lower, upper, color in color_map:
        mask = (depth_normalized >= lower) & (depth_normalized < upper)
        colored_image[mask] = color

    return colored_image


def project_points_to_depth_image(points, width, height):
    """
    Convert 3D points to a 2D depth image.
    """
    x = points[:, 0]
    y = points[:, 1]
    z = points[:, 2]

    u = (x / z).astype(int)
    v = (y / z).astype(int)

    u = np.clip(u, 0, width - 1)
    v = np.clip(v, 0, height - 1)

    depth_image = np.zeros((height, width), dtype=np.float32)

    depth_image[v, u] = z

    return depth_image

def save_combined_depth_images(depth_images, file_path):
    padding = 10 
    combined_image = np.hstack([
        np.pad(img, ((0, 0), (0, padding)), mode='constant', constant_values=255)
        for img in depth_images
    ])
    cv2.imwrite(file_path, combined_image)

def save_all_combined_images(all_images, file_path):
    image_height, image_width = all_images[0][0].shape
    padding = 10  
    num_images = len(all_images)
    
    combined_image_width = image_width * len(all_images[0]) + padding * (len(all_images[0]) - 1)
    combined_image_height = image_height * num_images + padding * (num_images - 1)
    
    combined_image = np.ones((combined_image_height, combined_image_width), dtype=np.uint8) * 255
    
    for i, images in enumerate(all_images):
        y_offset = i * (image_height + padding)
        for j, img in enumerate(images):
            x_offset = j * (image_width + padding)
            combined_image[y_offset:y_offset + image_height, x_offset:x_offset + image_width] = img
    
    cv2.imwrite(file_path, combined_image)
