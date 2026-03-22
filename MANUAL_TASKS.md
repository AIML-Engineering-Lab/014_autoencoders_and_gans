# MANUAL_TASKS — P14 Autoencoders & GANs

## Summary of Infrastructure Limitations

This project was developed on **CPU (Apple M-series, no GPU)**.
GAN training was critically reduced — 10 epochs is far below the minimum viable
for any GAN variant. VAE training was also reduced but less severely.

---

## What Was Reduced & Why

### NB01: Sensor Anomaly Detection (VAE)

| Component | Current (CPU) | Target (GPU) | Impact |
|-----------|--------------|-------------|--------|
| VAE epochs | 40 | 100–150 | ELBO may not converge; reconstruction quality limited |
| Latent dimension | 32 | 64–128 | Compressed representation too narrow |
| Learning rate | 1e-3 | 1e-4 | More stable convergence with more epochs |

### NB02: Wafer Defect Generation (DCGAN)

| Component | Current (CPU) | Target (GPU) | Impact |
|-----------|--------------|-------------|--------|
| DCGAN epochs | **10** | **50–100** | **CRITICAL: 10 epochs is ~10–20% of minimum viable GAN training** |
| Batch size | 64 | 128 | Larger batches stabilize discriminator gradients |
| Learning rates | 2e-4 / 2e-4 | 1e-4 / 1e-4 | Lower LR prevents mode collapse |

### Why GAN Training at 10 Epochs Is Insufficient
- GANs require adversarial equilibrium which takes 50+ epochs to begin stabilizing
- At 10 epochs, the generator likely produces noise or very blurry outputs
- Discriminator hasn't learned meaningful features yet
- FID/IS scores at 10 epochs are not representative of GAN capability

---

## GPU Execution Instructions

### Prerequisites
- GPU with 8GB+ VRAM (GAN training benefits greatly from GPU batch throughput)

### NB01 Changes
1. VAE training cell: change epoch count from 40 → 100
2. Change `latent_dim = 32` → `latent_dim = 64`

### NB02 Changes (CRITICAL)
1. Training cell: change `EPOCHS = 10` → `EPOCHS = 80`
2. Change batch_size from 64 → 128
3. Lower learning rates: `lr_g = 1e-4, lr_d = 1e-4`
4. Add gradient penalty or spectral norm for training stability

Expected time: ~30 min per notebook on T4 GPU (GAN training is compute-heavy).

---

## Checklist After GPU Run
- [ ] Verify GAN-generated images are visually recognizable (not noise/blurry)
- [ ] Check discriminator loss stabilizes (not collapsing to 0)
- [ ] FID score should be under 50 for synthetic wafer images
- [ ] VAE reconstruction loss shows clear convergence plateau
