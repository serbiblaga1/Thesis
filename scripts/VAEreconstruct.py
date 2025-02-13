import torch
import matplotlib.pyplot as plt
import numpy as np
from VAE import VAE, DepthImageDataset
from torch.utils.data import Dataset, DataLoader

vae = VAE(latent_dim=16)
vae.load_state_dict(torch.load("vae_model.pth"))
vae.eval()  # Set the model to evaluation mode

# Function to reconstruct images
def reconstruct_images(dataloader, vae, device):
    vae.to(device)
    
    with torch.no_grad():
        for images in dataloader:
            images = images.to(device)

            # Forward pass through the VAE to get reconstructed images
            reconstructed, mu, logvar = vae(images)
            
            # Convert the reconstructed images to numpy for visualization
            reconstructed = reconstructed.cpu().numpy()
            images = images.cpu().numpy()

            # Plot original vs reconstructed for the first batch
            for i in range(min(5, images.shape[0])):  # Show up to 5 images
                plt.subplot(2, 5, i + 1)
                plt.imshow(images[i, 0], cmap='gray')  # Original image
                plt.title("Original")
                plt.axis('off')

                plt.subplot(2, 5, i + 6)
                plt.imshow(reconstructed[i, 0], cmap='gray')  # Reconstructed image
                plt.title("Reconstructed")
                plt.axis('off')

            plt.show()
            break  # Just show the first batch of images

# Example of how to reconstruct and visualize images
# Load the dataset again for reconstruction
dataset = DepthImageDataset("vae_tof_data")
dataloader = DataLoader(dataset, batch_size=32, shuffle=False)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
reconstruct_images(dataloader, vae, device)
