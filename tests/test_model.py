"""Tests for Autoencoders & GANs models."""
import numpy as np
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))


def test_vae_model_exists():
    assert (ROOT / "models" / "vae_model.pt").exists(), "VAE model not found. Run src/train.py first."


def test_dcgan_model_exists():
    assert (ROOT / "models" / "dcgan_generator.pt").exists(), "DCGAN generator not found. Run src/train.py first."


def test_anomaly_score():
    from predict import anomaly_score
    if not (ROOT / "models" / "vae_model.pt").exists():
        return
    dummy = np.random.rand(5, 64, 64).astype(np.float32)
    scores = anomaly_score(dummy)
    assert len(scores) == 5
    assert all(s >= 0 for s in scores)


def test_generate_samples():
    from predict import generate_samples
    if not (ROOT / "models" / "dcgan_generator.pt").exists():
        return
    samples = generate_samples(n_samples=4, class_label=0)
    assert samples.shape == (4, 64, 64)


if __name__ == "__main__":
    test_vae_model_exists()
    test_dcgan_model_exists()
    test_anomaly_score()
    test_generate_samples()
    print("All tests passed.")
