# app.py
import streamlit as st
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

# === MODEL CONFIG ===
LATENT_DIM = 100
IMAGE_SIZE = 28
MODEL_PATH = "mnist_gan_generator.pth"

# === GENERATOR MODEL ===
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
            nn.Linear(512, IMAGE_SIZE * IMAGE_SIZE),
            nn.Tanh()
        )

    def forward(self, noise, labels):
        x = torch.cat((noise, self.label_emb(labels)), dim=1)
        img = self.model(x)
        img = img.view(img.size(0), 1, IMAGE_SIZE, IMAGE_SIZE)
        return img

# === LOAD MODEL ===
@st.cache_resource
def load_model():
    model = Generator(LATENT_DIM)
    model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
    model.eval()
    return model

generator = load_model()

# === APP UI ===
st.title("🧠 Handwritten Digit Generator (0–9)")
st.markdown("This app generates **5 handwritten digit images** based on the digit you select.")

digit = st.selectbox("Select a digit to generate:", list(range(10)), index=0)
generate_button = st.button("Generate Images")

if generate_button:
    with st.spinner("Generating digits..."):
        z = torch.randn(5, LATENT_DIM)
        labels = torch.tensor([digit] * 5)
        with torch.no_grad():
            gen_imgs = generator(z, labels).detach().cpu().numpy()

        # === DISPLAY IMAGES ===
        st.subheader(f"Generated Images for Digit: {digit}")
        fig, axes = plt.subplots(1, 5, figsize=(10, 2))
        for i in range(5):
            axes[i].imshow(gen_imgs[i][0], cmap="gray")
            axes[i].axis("off")
        st.pyplot(fig)