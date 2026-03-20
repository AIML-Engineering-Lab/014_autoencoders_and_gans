"""
Inference for Autoencoders & GANs.
Load trained VAE/GAN, generate new samples, detect anomalies.
"""
import numpy as np
import torch
from pathlib import Path
from train import VAE, Generator, LATENT_DIM

ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = ROOT / "models"
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")


def anomaly_score(images: np.ndarray, model_path: str = None) -> np.ndarray:
    """Compute anomaly score via VAE reconstruction error."""
    if model_path is None:
        model_path = str(MODEL_DIR / "vae_model.pt")

    model = VAE(LATENT_DIM).to(DEVICE)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE, weights_only=True))
    model.eval()

    X = images.astype(np.float32)
    if X.ndim == 3:
        X = X[:, np.newaxis, :, :]
    X_t = torch.tensor(X).to(DEVICE)

    with torch.no_grad():
        recon, _, _ = model(X_t)
        scores = ((recon - X_t) ** 2).mean(dim=(1, 2, 3)).cpu().numpy()
    return scores


def generate_samples(n_samples: int = 16, class_label: int = 0, n_classes: int = 5) -> np.ndarray:
    """Generate new wafer defect images using trained DCGAN."""
    gen = Generator(LATENT_DIM, n_classes).to(DEVICE)
    gen.load_state_dict(torch.load(MODEL_DIR / "dcgan_generator.pt", map_location=DEVICE, weights_only=True))
    gen.eval()

    z = torch.randn(n_samples, LATENT_DIM, device=DEVICE)
    labels = torch.full((n_samples,), class_label, dtype=torch.long, device=DEVICE)

    with torch.no_grad():
        fake = gen(z, labels)
        # Convert from [-1,1] to [0,1]
        fake = (fake + 1) / 2
    return fake.squeeze(1).cpu().numpy()


def latent_interpolation(model_path: str = None, n_steps: int = 10) -> np.ndarray:
    """Interpolate between two points in VAE latent space."""
    if model_path is None:
        model_path = str(MODEL_DIR / "vae_model.pt")

    model = VAE(LATENT_DIM).to(DEVICE)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE, weights_only=True))
    model.eval()

    z1 = torch.randn(1, LATENT_DIM, device=DEVICE)
    z2 = torch.randn(1, LATENT_DIM, device=DEVICE)

    frames = []
    with torch.no_grad():
        for alpha in np.linspace(0, 1, n_steps):
            z = (1 - alpha) * z1 + alpha * z2
            img = model.decode(z).squeeze().cpu().numpy()
            frames.append(img)
    return np.array(frames)


if __name__ == "__main__":
    # Anomaly detection demo
    data = np.load(ROOT / "data" / "sensor_anomaly_images.npz")
    sample = data["images"][:10]
    scores = anomaly_score(sample)
    print(f"Anomaly scores: {scores}")

    # Generation demo
    samples = generate_samples(n_samples=4, class_label=2)
    print(f"Generated {len(samples)} samples, shape: {samples.shape}")
