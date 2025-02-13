import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

class DepthImageDataset(Dataset):
    def __init__(self, directory):
        self.image_paths = [os.path.join(directory, fname) for fname in os.listdir(directory)]

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = np.load(self.image_paths[idx])
        image = torch.tensor(image, dtype=torch.float32).unsqueeze(0)
        return image

class Encoder(nn.Module):
    def __init__(self, latent_dim):
        super(Encoder, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1), 
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1), 
            nn.ReLU(),
            nn.Flatten(),  
        )
        self.flattened_size = 64 * 2 * 2
        self.fc_mu = nn.Linear(self.flattened_size, latent_dim)
        self.fc_logvar = nn.Linear(self.flattened_size, latent_dim)

    def forward(self, x):
        x = self.conv(x)
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        return mu, logvar

class Decoder(nn.Module):
    def __init__(self, latent_dim):
        super(Decoder, self).__init__()
        self.fc = nn.Linear(latent_dim, 64 * 2 * 2)
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1), 
            nn.ReLU(),
            nn.ConvTranspose2d(32, 1, kernel_size=3, stride=2, padding=1, output_padding=1), 
            nn.Sigmoid()
        )

    def forward(self, z):
        z = self.fc(z).view(-1, 64, 2, 2)
        x_reconstructed = self.deconv(z)
        return x_reconstructed

class VAE(nn.Module):
    def __init__(self, latent_dim):
        super(VAE, self).__init__()
        self.encoder = Encoder(latent_dim)
        self.decoder = Decoder(latent_dim)

    def forward(self, x):
        mu, logvar = self.encoder(x)
        z = reparameterize(mu, logvar)
        x_reconstructed = self.decoder(z)
        return x_reconstructed, mu, logvar

def reparameterize(mu, logvar):
    std = torch.exp(0.5 * logvar)
    eps = torch.randn_like(std)
    return mu + eps * std

def load_trained_vae(latent_dim, model_path):
    vae = VAE(latent_dim)
    vae.load_state_dict(torch.load(model_path, map_location=torch.device("cpu")))
    vae.eval()
    return vae

def visualize_latent_space(vae, dataloader, latent_dim):
    latent_points = []
    for images in dataloader:
        images = images.to(device)
        _, mu, _ = vae(images)
        latent_points.append(mu.cpu().detach().numpy())

    latent_points = np.concatenate(latent_points, axis=0)

    tsne = TSNE(n_components=2, random_state=42)
    latent_2d = tsne.fit_transform(latent_points)

    plt.figure(figsize=(8, 8))
    plt.scatter(latent_2d[:, 0], latent_2d[:, 1], alpha=0.7, s=10)
    plt.title("Latent Space Visualization")
    plt.xlabel("Dimension 1")
    plt.ylabel("Dimension 2")
    plt.show()

if __name__ == "__main__":
    latent_dim = 16
    batch_size = 32
    model_path = "vae_model.pth"
    data_dir = "vae_tof_data"

    dataset = DepthImageDataset(data_dir)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vae = load_trained_vae(latent_dim, model_path).to(device)

    visualize_latent_space(vae, dataloader, latent_dim)
