# Flow-Matching Speech Enhancement via Dynamic Multi-Layer Tokenizer Alignment

> **Course Project** — CSC_52002_EP IA Générative Multimodale (2025–2026), École Polytechnique  
> **Authors:** Yiheng Chen, André Dal Bosco  
> **Teachers:** Vicky Kalogeiton, Xi Wang

---

## Overview

This project implements **Rectified Flow** speech enhancement operating in the continuous latent space of two pre-trained audio tokenizers:

- **DAC** (Descript Audio Codec, 16 kHz → 50 Hz, d=1024): encodes both the noisy source and clean target.
- **MOSS-Audio-Tokenizer** (1.6B params, 24 kHz → 12.5 Hz): provides rich conditioning features from the noisy speech. ([Paper — arXiv 2602.10934](https://arxiv.org/abs/2602.10934))

A lightweight **Diffusion Transformer** (DiT, 6 layers, 28.4M base params) learns the flow field from noisy DAC latent to clean DAC latent, conditioned on MOSS features via Cross-Attention.

### Core Contribution

**Multi-layer tokenizer conditioning.** Instead of using only the last hidden layer of MOSS (standard practice), we extract features from **all 32 transformer layers** of MOSS Stage 3 and fuse them via:

1. **Static fusion:** A learnable softmax-normalised weight per layer (32 scalar parameters).
2. **Time-dependent fusion:** A lightweight MLP (1→64→32) maps the ODE timestep *t* to per-layer weights, allowing different denoising stages to emphasise different layers.

We compare **4 conditioning strategies** in a controlled ablation study.

### Architecture

```
Noisy Audio ──► DAC Encoder ──► X_0 (50Hz latent) ──┐
                                                      ├──► DiT (Rectified Flow) ──► X̂_1 ──► DAC Decoder ──► Enhanced Audio
Noisy Audio ──► MOSS Tokenizer ──► C (12.5Hz embed) ─┘
                                   (multi-layer)
```

---

## Results

| # | Condition              | PESQ ↑  | STOI ↑  | FAD ↓  |
|---|------------------------|---------|---------|--------|
| 1 | None (no MOSS)         | 1.6048  | 0.8527  | 2.9774 |
| 2 | Last layer only        | 1.6499  | 0.8589  | 2.6997 |
| 3 | Static multi-layer     | 1.6868  | 0.8642  | 2.3857 |
| 4 | Time-dep. multi-layer  | **1.6986** | **0.8647** | **2.3456** |

**Monotonic improvement** across all metrics: No conditioning < Last layer < Static multi-layer < Time-dependent multi-layer.

---

## Pre-trained Checkpoints & Outputs

| Resource | Link |
|----------|------|
| **Trained checkpoints** (all 4 conditions) | [Google Drive — checkpoints](https://drive.google.com/drive/folders/1siQa2Rlsx1A4DYlEXo6qyksoOM1poyU8?usp=share_link) |
| **Evaluation outputs** (enhanced audio, figures) | [Google Drive — outputs](https://drive.google.com/drive/folders/1uH9JNkk6FzoSQb3anljAYGUmRdeIPw4d?usp=share_link) |

Download checkpoints and place them as:
```text
checkpoints/
├── none/best.pt
├── last_layer/best.pt
├── multi_layer/best.pt
└── multi_layer_time/best.pt
```

---

## Repository Structure

```text
├── configs/
│   └── default.yaml               # Full training / model / evaluation config
├── src/
│   ├── data/
│   │   ├── mixer.py               # Mix clean + noise at target SNR
│   │   ├── extract_dac.py         # DAC 50Hz latent extraction
│   │   └── extract_moss.py        # MOSS embedding extraction (last / all layers)
│   ├── models/
│   │   ├── dit.py                 # Diffusion Transformer + Multi-Layer Fusion
│   │   └── flow_matching.py       # Rectified Flow (loss + Euler ODE solver)
│   └── utils/
│       ├── metrics.py             # FAD computation
│       └── viz.py                 # Mel-spectrogram visualisation
├── train.py                       # Training loop (loads pre-computed .pt features)
├── evaluate.py                    # Inference + PESQ / STOI / FAD evaluation
├── demo.py                        # Local demo: mix → enhance → visualise
├── notebooks/
│   ├── demo_analysis.ipynb        # ★ Analysis of pre-computed enhanced audio
│   ├── analysis.ipynb             # Results analysis and plotting
│   ├── inspection.ipynb           # Feature / weight inspection
│   ├── train_colab.ipynb          # Google Colab training notebook
│   ├── train_multi_snr_colab.ipynb # Multi-SNR Colab training
│   ├── plan_b_data_prep.ipynb     # Alternative data pipeline (Colab)
│   ├── plan_b_train.ipynb         # Alternative training (Colab)
│   ├── plan_c_small_model.ipynb   # Small-model experiments (Colab)
│   └── plan_d_anti_overfit.ipynb  # Anti-overfitting experiments (Colab)
├── report/
│   └── report.tex                 # CVPR-format project report (≤5 pages)
├── poster/                        # Poster source and figures
├── scripts/                       # Shell scripts for data prep, packaging
└── requirements.txt
```

---

## Quick Start

### Installation

```bash
git clone <repo-url>
cd speech-enhancement-project
pip install -r requirements.txt
```

### Demo Analysis (Recommended)

The easiest way to explore results is the **demo analysis notebook**:

1. Download the [evaluation outputs](https://drive.google.com/drive/folders/1uH9JNkk6FzoSQb3anljAYGUmRdeIPw4d?usp=share_link) and place them under `speech_enhancement_outputs/`.
2. Open `notebooks/demo_analysis.ipynb` in Jupyter.
3. Run all cells — the notebook loads pre-computed enhanced audio from all 4 conditions and produces spectrograms, waveform comparisons, PESQ/STOI scores, and frequency-band analysis.

### Colab Training

Upload to Google Colab and run `notebooks/train_colab.ipynb`. The notebook handles environment setup, data download, feature extraction, and training.

---

## Pipeline — Step by Step

### 1. Data Preparation

```bash
python -m src.data.mixer \
    --clean_dir data/raw/clean \
    --noise_dir data/raw/noise \
    --snr_list -5 0 5 10 15 --sr 16000
```

### 2. Feature Extraction (Offline)

```bash
# DAC latents (clean + noisy)
python -m src.data.extract_dac --audio_dir data/raw/clean     --out_dir data/features/clean_dac
python -m src.data.extract_dac --audio_dir data/mixed/snr_5dB  --out_dir data/features/noisy_dac

# MOSS embeddings
python -m src.data.extract_moss --audio_dir data/mixed/snr_5dB --out_dir data/features/moss_last
python -m src.data.extract_moss --audio_dir data/mixed/snr_5dB --out_dir data/features/moss_multi --save_all_layers
```

### 3. Training

```bash
python train.py --config configs/default.yaml --condition_type none
python train.py --config configs/default.yaml --condition_type last_layer
python train.py --config configs/default.yaml --condition_type multi_layer
python train.py --config configs/default.yaml --condition_type multi_layer_time
```

### 4. Evaluation

```bash
python evaluate.py --config configs/default.yaml \
    --checkpoint checkpoints/multi_layer_time/best.pt \
    --condition_type multi_layer_time

# Or compare all 4 conditions
python evaluate.py --config configs/default.yaml --compare
```

### 5. Local Demo

```bash
python demo.py                       # Random sample, 5 dB SNR
python demo.py --snr 0 --seed 123   # Specific SNR and seed
```

---

## Conditioning Variants

| Variant | `condition_type` | MOSS Input | Extra Parameters |
|---------|-----------------|------------|-----------------|
| Baseline 1 | `none` | — | 28.4M (base DiT) |
| Baseline 2 | `last_layer` | h₃₁ only (768-dim) | +10.1M (cross-attn) |
| Ours (static) | `multi_layer` | All 32 layers (1280-dim) | +32 scalar weights |
| Ours (time-dep.) | `multi_layer_time` | All 32 layers (1280-dim) | +2,208 (MLP 1→64→32) |

### Key Design Decisions

1. **Offline feature extraction** — DAC and MOSS encoders run once; training uses only pre-computed `.pt` files.
2. **Rectified Flow** — Straight-path ODE from noisy to clean latent (NOT from Gaussian noise). Simple MSE loss on the velocity field.
3. **ODE solver** — 50-step Euler method at inference.
4. **adaLN timestep injection** — Adaptive layer normalisation, standard in DiT architectures.
5. **Learnable layer fusion** — Softmax-normalised weights over 32 MOSS layers, with optional time-dependent variant.

---

## Configuration

All hyperparameters are in `configs/default.yaml`. Key settings:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `model.hidden_dim` | 512 | DiT hidden dimension |
| `model.num_layers` | 6 | Number of transformer blocks |
| `model.num_heads` | 8 | Attention heads |
| `model.condition_type` | `multi_layer` | Conditioning mode |
| `training.batch_size` | 16 | Batch size |
| `training.num_steps` | 50000 | Total training steps |
| `training.learning_rate` | 1e-4 | Learning rate |
| `evaluation.ode_steps` | 50 | Euler solver steps |

---

## Acknowledgements

- Training scripts in `notebooks/` were co-developed with **Claude Opus** (Anthropic).
- **DAC**: Kumar et al., *High-Fidelity Audio Compression with Improved RVQGAN*, NeurIPS 2024.
- **MOSS-Audio-Tokenizer**: Gong et al., *Scaling Audio Tokenizers for Future Audio Foundation Models*, arXiv:2602.10934, 2026.

---

## License

Academic course project — not intended for production use.
