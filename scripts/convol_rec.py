import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import os
import numpy as np
from convol import DepthToDecisionCNN

# Load trained CNN model
model = DepthToDecisionCNN()
model.load_state_dict(torch.load("cnn_decision_model.pth", map_location=torch.device("cpu")))
model.to("cpu")
model.eval()

# Function to get a decision matrix from a depth image
def get_decision_matrix(model, depth_image, threshold=0.5):
    with torch.no_grad():
        depth_image = depth_image.unsqueeze(0).unsqueeze(0)  # Add batch and channel dimensions
        output = model(depth_image)
        decision_matrix = (output.squeeze(0) > threshold).int()  # Convert probabilities to binary 0/1
        return decision_matrix

# Example depth image (Simulated)
depth_image = torch.rand((8, 8)) * 4  # Random depth values (0 to 4 meters)

# Get the 3x3 decision matrix
decision_matrix = get_decision_matrix(model, depth_image)
print("3x3 Decision Matrix:")
print(decision_matrix.numpy())
