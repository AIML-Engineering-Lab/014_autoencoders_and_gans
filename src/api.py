"""
FastAPI serving endpoint for Autoencoders & GANs.
POST image data -> anomaly score + reconstruction.
POST class label -> generated wafer defect image.
"""
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from pathlib import Path

app = FastAPI(title="Autoencoders & GANs API", version="1.0.0")

ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = ROOT / "models"
_vae = None
_generator = None

LATENT_DIM = 32
N_CLASSES = 5


class ImageInput(BaseModel):
    pixels: list[list[float]]  # 2D array (H x W) normalized 0-1


class AnomalyResponse(BaseModel):
    anomaly_score: float
    is_anomaly: bool
    threshold: float = 0.01
    model: str = "VAE"


class GenerateRequest(BaseModel):
    class_label: int = Field(ge=0, lt=N_CLASSES, description="Defect class: 0=none, 1=center, 2=edge_ring, 3=scratch, 4=random")
    n_samples: int = Field(default=1, ge=1, le=16)


class GenerateResponse(BaseModel):
    samples: list[list[list[float]]]
    class_label: int
    model: str = "DCGAN"


def get_vae():
    global _vae
    if _vae is None:
        import torch
        from train import VAE
        device = torch.device("cpu")
        _vae = VAE(LATENT_DIM).to(device)
        _vae.load_state_dict(torch.load(MODEL_DIR / "vae_model.pt", map_location=device, weights_only=True))
        _vae.eval()
    return _vae


def get_generator():
    global _generator
    if _generator is None:
        import torch
        from train import Generator
        device = torch.device("cpu")
        _generator = Generator(LATENT_DIM, N_CLASSES).to(device)
        _generator.load_state_dict(torch.load(MODEL_DIR / "dcgan_generator.pt", map_location=device, weights_only=True))
        _generator.eval()
    return _generator


@app.get("/health")
def health():
    return {"status": "healthy", "models": ["VAE", "DCGAN"]}


@app.get("/info")
def info():
    return {
        "project": "014_autoencoders_and_gans",
        "description": "Autoencoders & GANs for anomaly detection and defect synthesis",
        "endpoints": ["/predict", "/generate"],
    }


@app.post("/predict", response_model=AnomalyResponse)
def predict(input_data: ImageInput):
    """Compute anomaly score for an input image using VAE reconstruction error."""
    import torch
    try:
        model = get_vae()
        arr = np.array(input_data.pixels, dtype=np.float32)
        X = torch.tensor(arr).unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)
        with torch.no_grad():
            recon, _, _ = model(X)
            score = float(((recon - X) ** 2).mean().item())
        threshold = 0.01
        return AnomalyResponse(anomaly_score=score, is_anomaly=score > threshold)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/generate", response_model=GenerateResponse)
def generate(request: GenerateRequest):
    """Generate new wafer defect images using conditional DCGAN."""
    import torch
    try:
        gen = get_generator()
        z = torch.randn(request.n_samples, LATENT_DIM)
        labels = torch.full((request.n_samples,), request.class_label, dtype=torch.long)
        with torch.no_grad():
            fake = gen(z, labels)
            fake = ((fake + 1) / 2).squeeze(1).numpy()
        samples = fake.tolist()
        return GenerateResponse(samples=samples, class_label=request.class_label)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
