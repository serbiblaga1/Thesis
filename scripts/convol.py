import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import Dataset, DataLoader
import cv2

# Define Dataset for Depth Images
class DepthImageDataset(Dataset):
    def __init__(self, directory):
        self.image_paths = [os.path.join(directory, fname) for fname in os.listdir(directory) if fname.endswith(".npy")]

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = np.load(self.image_paths[idx])
        image = torch.tensor(image, dtype=torch.float32).unsqueeze(0)  # Add channel dimension
        return image

# CNN Model for Converting Depth Map to 3x3 Decision Matrix
class DepthToDecisionCNN(nn.Module):
    def __init__(self):
        super(DepthToDecisionCNN, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),  # Keep (8x8)
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),  # Reduce to (4x4)
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),  # Keep (4x4)
            nn.ReLU(),
            nn.Conv2d(128, 1, kernel_size=3, stride=1, padding=1)  # Output single-channel (4x4)
        )
        self.fc = nn.Linear(4 * 4, 3 * 3)  # Fully connected layer to map (4x4) → (3x3)

    def forward(self, x):
        x = self.conv_layers(x)  # Convolutional feature extraction
        x = x.view(x.size(0), -1)  # Flatten (4x4) feature map
        x = self.fc(x)  # Fully connected layer for 3x3 output
        x = x.view(-1, 3, 3)  # Reshape to (batch_size, 3, 3)
        return torch.sigmoid(x)  # Sigmoid activation for probabilities

# Function to Convert Depth Image to 3x3 Decision Matrix
def generate_labels(depth_image, threshold=1.0):
    """
    Converts an 8x8 depth image into a 3x3 navigation decision matrix.

    Parameters:
    depth_image (torch.Tensor): (8x8) depth matrix from the camera
    threshold (float): Depth value below which an obstacle is detected

    Returns:
    torch.Tensor: A 3x3 matrix where 1 = obstacle, 0 = safe
    """
    left_region = depth_image[:, :3]  # Left side
    center_region = depth_image[:, 3:5]  # Center
    right_region = depth_image[:, 5:]  # Right side
    front_region = depth_image[:4, :]  # Front half

    # Obstacle detection based on depth threshold
    obstacle_left = (left_region < threshold).any()
    obstacle_center = (center_region < threshold).any()
    obstacle_right = (right_region < threshold).any()
    obstacle_front = (front_region < threshold).any()

    # Create 3x3 movement decision matrix
    decision_matrix = torch.zeros((3, 3))
    decision_matrix[0, 0] = 1 if obstacle_left else 0
    decision_matrix[0, 1] = 1 if obstacle_front else 0
    decision_matrix[0, 2] = 1 if obstacle_right else 0

    return decision_matrix

# Loss Function (Binary Cross Entropy for obstacle classification)
def loss_function(predictions, labels):
    return nn.BCELoss()(predictions, labels)

# Training Function
def train_model(model, dataloader, num_epochs=50, learning_rate=0.001):
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0

        for images in dataloader:
            images = images.to(device)

            # Generate labels dynamically from depth data
            labels = torch.stack([generate_labels(img) for img in images]).to(device)

            optimizer.zero_grad()
            predictions = model(images)
            loss = loss_function(predictions, labels)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {epoch_loss / len(dataloader):.4f}")

    return model

# Load dataset
dataset = DepthImageDataset("vae_tof_data")
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# Initialize and train the CNN
model = DepthToDecisionCNN()
#trained_model = train_model(model, dataloader)

# Save the trained model
#torch.save(trained_model.state_dict(), "cnn_decision_model.pth")
#print("CNN Model saved as cnn_decision_model.pth")

# Function to Run Inference on a Depth Image
def get_decision_matrix(model, depth_image, threshold=0.5):
    model.eval()
    with torch.no_grad():
        depth_image = depth_image.unsqueeze(0).unsqueeze(0)  # Add batch and channel dimensions
        output = model(depth_image)
        decision_matrix = (output.squeeze(0) > threshold).int()  # Convert probabilities to binary 0/1
        return decision_matrix

def test_image(model, image_path=None, threshold=0.5):
    """
    Tests a depth image (either .npy or .png) using the trained CNN.

    Parameters:
    model (DepthToDecisionCNN): Trained model
    image_path (str, optional): Path to a .npy or .png depth image
    threshold (float): Decision threshold for obstacles

    Returns:
    torch.Tensor: Predicted 3x3 decision matrix
    """
    # Load PNG or NumPy file correctly
    if image_path.endswith(".npy"):
        depth_image = torch.tensor(np.load(image_path), dtype=torch.float32)
    elif image_path.endswith(".png"):
        depth_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)  # Load as grayscale
        depth_image = cv2.resize(depth_image, (8, 8))  # Resize to 8x8
        depth_image = torch.tensor(depth_image, dtype=torch.float32) / 255.0  # Normalize to 0-1
    else:
        raise ValueError("Unsupported file format. Use .npy or .png")

    # Ensure image is 8x8
    if depth_image.shape != (8, 8):
        raise ValueError("Expected an 8x8 depth image")

    # Convert to tensor and add required dimensions
    depth_image = depth_image.unsqueeze(0).unsqueeze(0)  # Add batch & channel dimensions

    # Run inference
    with torch.no_grad():
        output = model(depth_image)  # Forward pass
        decision_matrix = (output.squeeze(0) > threshold).int()  # Convert to binary 0/1

    # Print results
    print("\n **Predicted 3x3 Decision Matrix**:")
    print(decision_matrix.numpy())  # Convert tensor to NumPy for printing

    return decision_matrix

# Load trained CNN model for inference
model = DepthToDecisionCNN()
model.load_state_dict(torch.load("cnn_decision_model.pth", map_location="cpu"))
model.to("cpu")
model.eval()

# Example Inference with a Simulated Depth Image
depth_image = torch.rand((8, 8)) * 4  # Simulate a random 8x8 depth image (0 to 4 meters)
decision_matrix = test_image(model, image_path="/home/serbiblaga/aerial_gym_simulator/testimage.png")

print("Predicted 3x3 Decision Matrix:")
print(decision_matrix.numpy())
