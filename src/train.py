"""
Train pipeline for Autoencoders & GANs.
Trains VAE on sensor anomaly data, DCGAN on wafer defect data.
Saves checkpoints to models/.
Tracks experiments with MLflow.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import mlflow
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
MODEL_DIR = ROOT / "models"
MODEL_DIR.mkdir(exist_ok=True)

DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
LATENT_DIM = 32


# ═══════════════════════════════════════════════
# Variational Autoencoder (VAE)
# ═══════════════════════════════════════════════

class Encoder(nn.Module):
    def __init__(self, latent_dim: int = LATENT_DIM):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1), nn.ReLU(),
        )
        self.fc_mu = nn.Linear(128 * 8 * 8, latent_dim)
        self.fc_logvar = nn.Linear(128 * 8 * 8, latent_dim)

    def forward(self, x):
        h = self.conv(x).view(x.size(0), -1)
        return self.fc_mu(h), self.fc_logvar(h)


class Decoder(nn.Module):
    def __init__(self, latent_dim: int = LATENT_DIM):
        super().__init__()
        self.fc = nn.Linear(latent_dim, 128 * 8 * 8)
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1), nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1), nn.ReLU(),
            nn.ConvTranspose2d(32, 1, 4, stride=2, padding=1), nn.Sigmoid(),
        )

    def forward(self, z):
        h = self.fc(z).view(-1, 128, 8, 8)
        return self.deconv(h)


class VAE(nn.Module):
    def __init__(self, latent_dim: int = LATENT_DIM):
        super().__init__()
        self.encoder = Encoder(latent_dim)
        self.decoder = Decoder(latent_dim)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decoder(z)
        return recon, mu, logvar

    def encode(self, x):
        mu, logvar = self.encoder(x)
        return self.reparameterize(mu, logvar)

    def decode(self, z):
        return self.decoder(z)


def vae_loss(recon_x, x, mu, logvar):
    recon_loss = F.binary_cross_entropy(recon_x, x, reduction="sum")
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + kl_loss


# ═══════════════════════════════════════════════
# DCGAN (Deep Convolutional GAN)
# ═══════════════════════════════════════════════

class Generator(nn.Module):
    def __init__(self, latent_dim: int = LATENT_DIM, n_classes: int = 5):
        super().__init__()
        self.label_emb = nn.Embedding(n_classes, latent_dim)
        self.fc = nn.Linear(latent_dim * 2, 256 * 4 * 4)
        self.main = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, 2, 1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 4, 2, 1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 4, 2, 1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.ConvTranspose2d(32, 1, 4, 2, 1), nn.Tanh(),
        )

    def forward(self, z, labels):
        c = self.label_emb(labels)
        x = torch.cat([z, c], dim=1)
        h = self.fc(x).view(-1, 256, 4, 4)
        return self.main(h)


class Discriminator(nn.Module):
    def __init__(self, n_classes: int = 5):
        super().__init__()
        self.label_emb = nn.Embedding(n_classes, 64 * 64)
        self.main = nn.Sequential(
            nn.Conv2d(2, 64, 4, 2, 1), nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, 4, 2, 1), nn.BatchNorm2d(128), nn.LeakyReLU(0.2),
            nn.Conv2d(128, 256, 4, 2, 1), nn.BatchNorm2d(256), nn.LeakyReLU(0.2),
            nn.Conv2d(256, 512, 4, 2, 1), nn.BatchNorm2d(512), nn.LeakyReLU(0.2),
            nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(512, 1),
        )

    def forward(self, img, labels):
        c = self.label_emb(labels).view(-1, 1, 64, 64)
        x = torch.cat([img, c], dim=1)
        return self.main(x)


def train_vae(epochs: int = 30, batch_size: int = 64, lr: float = 1e-3):
    """Train VAE on sensor anomaly images."""
    print("=" * 60)
    print("Training VAE on Sensor Anomaly Images")
    print("=" * 60)
    data = np.load(DATA_DIR / "sensor_anomaly_images.npz")
    X = data["images"].astype(np.float32)
    if X.ndim == 3:
        X = X[:, np.newaxis, :, :]

    X_t = torch.tensor(X).to(DEVICE)
    model = VAE(LATENT_DIM).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    mlflow.set_experiment("014_autoencoders_and_gans")
    with mlflow.start_run(run_name="VAE_sensor_anomaly"):
        mlflow.log_params({"model": "VAE", "latent_dim": LATENT_DIM, "epochs": epochs,
                           "batch_size": batch_size, "lr": lr, "dataset": "sensor_anomaly_images",
                           "n_samples": len(X_t), "device": str(DEVICE)})

        model.train()
        for epoch in range(epochs):
            perm = torch.randperm(len(X_t))
            total_loss = 0
            for i in range(0, len(X_t), batch_size):
                idx = perm[i:i + batch_size]
                batch = X_t[idx]
                recon, mu, logvar = model(batch)
                loss = vae_loss(recon, batch, mu, logvar) / len(batch)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            n_batches = (len(X_t) + batch_size - 1) // batch_size
            avg_loss = total_loss / n_batches
            mlflow.log_metric("vae_loss", avg_loss, step=epoch)
            print(f"  Epoch {epoch + 1}/{epochs} — Loss: {avg_loss:.4f}")

        # Compute final anomaly scores on normal vs anomaly
        model.eval()
        labels = data["labels"]
        with torch.no_grad():
            recon, _, _ = model(X_t)
            scores = ((recon - X_t) ** 2).mean(dim=(1, 2, 3)).cpu().numpy()
        normal_score = float(scores[labels == 0].mean())
        anomaly_score = float(scores[labels == 1].mean())
        mlflow.log_metrics({"normal_recon_error": normal_score, "anomaly_recon_error": anomaly_score,
                            "separation_ratio": anomaly_score / max(normal_score, 1e-8),
                            "final_loss": avg_loss})
        print(f"  Normal recon error: {normal_score:.6f}")
        print(f"  Anomaly recon error: {anomaly_score:.6f}")
        print(f"  Separation ratio: {anomaly_score / max(normal_score, 1e-8):.2f}x")

        path = MODEL_DIR / "vae_model.pt"
        torch.save(model.state_dict(), path)
        mlflow.log_artifact(str(path))
        print(f"VAE saved → {path}")
    return model


def train_dcgan(epochs: int = 50, batch_size: int = 64, lr: float = 2e-4):
    """Train conditional DCGAN on wafer defect patterns."""
    print("=" * 60)
    print("Training Conditional DCGAN on Wafer Defect Patterns")
    print("=" * 60)
    data = np.load(DATA_DIR / "wafer_defect_patterns.npz")
    X = data["images"].astype(np.float32)
    y = data["labels"]
    n_classes = len(np.unique(y))

    if X.ndim == 3:
        X = X[:, np.newaxis, :, :]
    # Normalize to [-1, 1] for Tanh
    X = X * 2 - 1

    X_t = torch.tensor(X).to(DEVICE)
    y_t = torch.tensor(y, dtype=torch.long).to(DEVICE)

    gen = Generator(LATENT_DIM, n_classes).to(DEVICE)
    disc = Discriminator(n_classes).to(DEVICE)
    opt_g = torch.optim.Adam(gen.parameters(), lr=lr, betas=(0.5, 0.999))
    opt_d = torch.optim.Adam(disc.parameters(), lr=lr, betas=(0.5, 0.999))
    criterion = nn.BCEWithLogitsLoss()

    mlflow.set_experiment("014_autoencoders_and_gans")
    with mlflow.start_run(run_name="DCGAN_wafer_defects"):
        mlflow.log_params({"model": "DCGAN", "latent_dim": LATENT_DIM, "epochs": epochs,
                           "batch_size": batch_size, "lr": lr, "n_classes": n_classes,
                           "dataset": "wafer_defect_patterns", "n_samples": len(X_t),
                           "device": str(DEVICE)})

        gen.train()
        disc.train()
        for epoch in range(epochs):
            perm = torch.randperm(len(X_t))
            g_loss_sum, d_loss_sum, n_batches = 0, 0, 0
            for i in range(0, len(X_t), batch_size):
                idx = perm[i:i + batch_size]
                real = X_t[idx]
                lbl = y_t[idx]
                bs = len(idx)

                # ── Discriminator ──
                z = torch.randn(bs, LATENT_DIM, device=DEVICE)
                fake = gen(z, lbl).detach()
                d_real = disc(real, lbl)
                d_fake = disc(fake, lbl)
                d_loss = (criterion(d_real, torch.ones_like(d_real)) +
                          criterion(d_fake, torch.zeros_like(d_fake))) / 2
                opt_d.zero_grad()
                d_loss.backward()
                opt_d.step()

                # ── Generator ──
                z = torch.randn(bs, LATENT_DIM, device=DEVICE)
                fake = gen(z, lbl)
                g_out = disc(fake, lbl)
                g_loss = criterion(g_out, torch.ones_like(g_out))
                opt_g.zero_grad()
                g_loss.backward()
                opt_g.step()

                g_loss_sum += g_loss.item()
                d_loss_sum += d_loss.item()
                n_batches += 1

            avg_g = g_loss_sum / n_batches
            avg_d = d_loss_sum / n_batches
            mlflow.log_metrics({"g_loss": avg_g, "d_loss": avg_d}, step=epoch)
            if (epoch + 1) % 5 == 0 or epoch == 0:
                print(f"  Epoch {epoch + 1}/{epochs} — D_loss: {avg_d:.4f}, G_loss: {avg_g:.4f}")

        mlflow.log_metrics({"final_g_loss": avg_g, "final_d_loss": avg_d})
        torch.save(gen.state_dict(), MODEL_DIR / "dcgan_generator.pt")
        torch.save(disc.state_dict(), MODEL_DIR / "dcgan_discriminator.pt")
        mlflow.log_artifact(str(MODEL_DIR / "dcgan_generator.pt"))
        mlflow.log_artifact(str(MODEL_DIR / "dcgan_discriminator.pt"))
        print(f"DCGAN saved → {MODEL_DIR / 'dcgan_generator.pt'}")
    return gen, disc


if __name__ == "__main__":
    train_vae()
    print()
    train_dcgan()
    print("\nAll models trained and saved to models/")
