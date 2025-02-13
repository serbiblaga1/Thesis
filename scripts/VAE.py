import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import Dataset, DataLoader

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

# Define the Encoder
class Encoder(nn.Module):
    def __init__(self, latent_dim):
        super(Encoder, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1),  # Output: (32, 4, 4)
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),  # Output: (64, 2, 2)
            nn.ReLU(),
            nn.Flatten(),  # Output: 64 * 2 * 2 = 256
        )
        self.flattened_size = 64 * 2 * 2
        self.fc_mu = nn.Linear(self.flattened_size, latent_dim)
        self.fc_logvar = nn.Linear(self.flattened_size, latent_dim)

    def forward(self, x):
        x = self.conv(x)
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        return mu, logvar

# Reparameterization Trick
def reparameterize(mu, logvar):
    std = torch.exp(0.5 * logvar)
    eps = torch.randn_like(std)
    return mu + eps * std

# Define the Decoder
class Decoder(nn.Module):
    def __init__(self, latent_dim):
        super(Decoder, self).__init__()
        self.fc = nn.Linear(latent_dim, 64 * 2 * 2)
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),  # Output: (32, 4, 4)
            nn.ReLU(),
            nn.ConvTranspose2d(32, 1, kernel_size=3, stride=2, padding=1, output_padding=1),  # Output: (1, 8, 8)
            nn.Sigmoid()
        )

    def forward(self, z):
        z = self.fc(z).view(-1, 64, 2, 2)
        x_reconstructed = self.deconv(z)
        return x_reconstructed

# Define the VAE
class VAE(nn.Module):
    def __init__(self, latent_dim):
        super(VAE, self).__init__()
        self.encoder = Encoder(latent_dim)
        self.decoder = Decoder(latent_dim)

    def forward(self, x):
        mu, logvar = self.encoder(x)
        z = reparameterize(mu, logvar)
        x_reconstructed = self.decoder(z)
        return x_reconstructed, mu, logvar, z  # Return latent vector as well

# Define the Loss Function
def loss_function(recon_x, x, mu, logvar):
    recon_loss = nn.MSELoss()(recon_x, x)
    kl_divergence = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + kl_divergence

# Function to Summarize Obstacles from Reconstructed Image
def summarize_obstacles(reconstructed, threshold=1.0):
    """
    Analyzes the reconstructed depth image and outputs obstacle locations.

    Parameters:
    reconstructed (torch.Tensor): Reconstructed 8x8 depth image
    threshold (float): Depth value below which an obstacle is considered present

    Returns:
    str: Summary of detected obstacles
    """
    reconstructed = reconstructed.squeeze(0)  # Remove batch dimension if necessary

    # Define obstacle regions
    left_region = reconstructed[:, :3]  # Left side
    center_region = reconstructed[:, 3:5]  # Center
    right_region = reconstructed[:, 5:]  # Right side
    front_region = reconstructed[:4, :]  # Front half

    # Check for obstacles
    obstacle_left = (left_region < threshold).any()
    obstacle_center = (center_region < threshold).any()
    obstacle_right = (right_region < threshold).any()
    obstacle_front = (front_region < threshold).any()

    # Generate text output
    messages = []
    if obstacle_front:
        messages.append("Obstacle in front")
    if obstacle_left:
        messages.append("Obstacle to the left")
    if obstacle_right:
        messages.append("Obstacle to the right")

    return ", ".join(messages) if messages else "Clear path"

# Main Training Script
if __name__ == "__main__":
    # Hyperparameters
    latent_dim = 16
    batch_size = 32
    num_epochs = 50
    learning_rate = 0.001

    # Prepare dataset and dataloader
    dataset = DepthImageDataset("vae_tof_data")
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Initialize model, optimizer, and device
    vae = VAE(latent_dim=latent_dim)
    optimizer = optim.Adam(vae.parameters(), lr=learning_rate)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vae.to(device)

    # Training Loop
    for epoch in range(num_epochs):
        vae.train()
        epoch_loss = 0

        for images in dataloader:
            images = images.to(device)

            optimizer.zero_grad()
            reconstructed, mu, logvar, _ = vae(images)
            loss = loss_function(reconstructed, images, mu, logvar)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {epoch_loss / len(dataloader):.4f}")

    # Save the Trained Model
    torch.save(vae.state_dict(), "vae_model.pth")
    print("Model saved as vae_model.pth")

    # Example Inference: Summarizing Obstacles from a Depth Image
    vae.eval()
    with torch.no_grad():
        test_image = torch.rand((1, 8, 8)).to(device) * 4  # Simulated depth image
        reconstructed, _, _, _ = vae(test_image.unsqueeze(0))  # Add batch dimension
        summary = summarize_obstacles(reconstructed, threshold=1.0)

    print(f"Obstacle Summary: {summary}")
