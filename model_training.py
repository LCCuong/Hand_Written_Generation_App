# train_mnist_gan
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
import os

# Check GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using:", device)

# Hyperparameters
latent_dim = 100
batch_size = 128
epochs = 30
lr = 0.0002
image_size = 28

# Dataset
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])  # Normalize to [-1, 1]
])

train_dataset = datasets.MNIST(root="./data", train=True, transform=transform, download=True)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)


# Generator
class Generator(nn.Module):
    def __init__(self, latent_dim, num_classes=10):
        super().__init__()
        self.label_emb = nn.Embedding(num_classes, num_classes)
        self.model = nn.Sequential(
            nn.Linear(latent_dim + num_classes, 128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, image_size * image_size),
            nn.Tanh()
        )

    def forward(self, noise, labels):
        # Concatenate noise and label embedding
        x = torch.cat((noise, self.label_emb(labels)), dim=1)
        img = self.model(x)
        img = img.view(img.size(0), 1, image_size, image_size)
        return img


# Discriminator
class Discriminator(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.label_emb = nn.Embedding(num_classes, num_classes)
        self.model = nn.Sequential(
            nn.Linear(image_size * image_size + num_classes, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, img, labels):
        x = img.view(img.size(0), -1)
        x = torch.cat((x, self.label_emb(labels)), dim=1)
        return self.model(x)


generator = Generator(latent_dim).to(device)
discriminator = Discriminator().to(device)

adversarial_loss = nn.BCELoss()

optimizer_G = optim.Adam(generator.parameters(), lr=lr, betas=(0.5, 0.999))
optimizer_D = optim.Adam(discriminator.parameters(), lr=lr, betas=(0.5, 0.999))

# Training Loop
for epoch in range(epochs):
    for i, (imgs, labels) in enumerate(train_loader):
        batch_size = imgs.size(0)
        real = torch.ones(batch_size, 1).to(device)
        fake = torch.zeros(batch_size, 1).to(device)

        # Real images
        real_imgs = imgs.to(device)
        labels = labels.to(device)

        # Train Generator
        optimizer_G.zero_grad()
        z = torch.randn(batch_size, latent_dim).to(device)
        gen_labels = torch.randint(0, 10, (batch_size,)).to(device)
        gen_imgs = generator(z, gen_labels)
        validity = discriminator(gen_imgs, gen_labels)
        g_loss = adversarial_loss(validity, real)
        g_loss.backward()
        optimizer_G.step()

        # Train Discriminator
        optimizer_D.zero_grad()
        real_pred = discriminator(real_imgs, labels)
        d_real_loss = adversarial_loss(real_pred, real)

        fake_pred = discriminator(gen_imgs.detach(), gen_labels)
        d_fake_loss = adversarial_loss(fake_pred, fake)

        d_loss = (d_real_loss + d_fake_loss) / 2
        d_loss.backward()
        optimizer_D.step()

    print(f"[Epoch {epoch+1}/{epochs}] D loss: {d_loss.item():.4f} | G loss: {g_loss.item():.4f}")

    if epoch % 5 == 0:
        generator.eval()
        with torch.no_grad():
            z = torch.randn(10, latent_dim).to(device)
            labels = torch.arange(0, 10).to(device)
            samples = generator(z, labels)
            samples = samples.cpu().numpy()

            fig, axs = plt.subplots(1, 10, figsize=(15, 2))
            for i in range(10):
                axs[i].imshow(samples[i][0], cmap="gray")
                axs[i].axis("off")
            plt.show()
        generator.train()

# Save generator
os.makedirs("models", exist_ok=True)
torch.save(generator.state_dict(), "mnist_gan_generator.pth")