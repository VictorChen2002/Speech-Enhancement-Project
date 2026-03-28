# Flow-Matching Speech Enhancement via Dynamic Multi-Layer Tokenizer Alignment

> **Course Project** — CSC_52002_EP IA Générative Multimodale (2025–2026), École Polytechnique  
> **Authors:** Yiheng Chen, André Dal Bosco  
> **Teachers:** Vicky Kalogeiton, Xi Wang

---

## Overview

This project implements **Rectified Flow** speech enhancement operating in the continuous latent space of two pre-trained audio tokenizers:

- **DAC** (Descript Audio Codec, 16 kHz → 50 Hz, d=1024): encodes both the noisy source and clean target.
- **MOSS-Audio-Tokenizer** (1.6B params, 24 kHz → 12.5 Hz): provides rich conditioning features from the noisy speech.

A lightweight **Diffusion Transformer** (DiT, 6 layers, 28.4M base params) learns the flow field from noisy DAC latent to clean DAC latent, conditioned on MOSS features via Cross-Attention.

### Core Contribution

**Multi-layer tokenizer conditioning.** Instead of using only the last hidden layer of MOSS (standard practice), we extract features from **all 32 transformer layers** of MOSS Stage 3 and fuse them via:

1. **Static fusion:** A learnable softmax-normalised weight per layer (32 scalar parameters).
2. **Time-dependent fusion:** A lightweight MLP (1→64→32) maps the ODE timestep t to per-layer weights, allowing different denoising stages to emphasise different layers.

We systematically compare 4 conditioning strategies in a controlled ablation study.

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

## Repository Structure

```text
├── configs/
│   └── default.yaml           # Full training/model/evaluation config
├── src/
│   ├── data/
│   │   ├── mixer.py           # Mix clean + noise at target SNR
│   │   ├── extract_dac.py     # DAC 50Hz latent extraction
│   │   └── extract_moss.py    # MOSS embedding extraction (last / all layers)
│   ├── models/
│   │   ├── dit.py             # Diffusion Transformer + Multi-Layer Fusion
│   │   └── flow_matching.py   # Rectified Flow (loss + Euler ODE solver)
│   └── utils/
│       ├── metrics.py         # FAD computation
│       └── viz.py             # Mel-spectrogram visualisation
├── train.py                   # Training loop (loads pre-computed .pt features)
├── evaluate.py                # Inference + PESQ/STOI/FAD evaluation
├── demo.py                    # Local demo: mix → extract → enhance → visualise
├── notebooks/
│   ├── analysis.ipynb         # Results analysis and plotting
│   └── inspection.ipynb       # Feature inspection
├── scripts/                   # Shell scripts for data prep, packaging
└── requirements.txt
```

---

## Setup

### Requirements
- Python 3.9+
- GPU recommended (CUDA or MPS)

```bash
pip install -r requirements.txt
```

### Checkpoints

Trained checkpoints are available on Google Drive:

**[Download checkpoints](https://drive.google.com/drive/folders/1siQa2Rlsx1A4DYlEXo6qyksoOM1poyU8?usp=share_link)**

Download and place them as:
```text
checkpoints/
├── none/best.pt
├── last_layer/best.pt
├── multi_layer/best.pt
└── multi_layer_time/best.pt
```

### Additional outputs (evaluation results, figures):

**[Download results](https://drive.google.com/drive/folders/1uH9JNkk6FzoSQb3anljAYGUmRdeIPw4d?usp=share_link)**

---

## Pipeline

### 1. Data Preparation

```bash
# Mix clean speech + noise at multiple SNR levels
python -m src.data.mixer \
    --clean_dir data/raw/clean \
    --noise_dir data/raw/noise \
    --snr_list -5 0 5 10 15 --sr 16000
```

### 2. Feature Extraction (offline)

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
# Single condition
python evaluate.py --config configs/default.yaml \
    --checkpoint checkpoints/multi_layer/best.pt --condition_type multi_layer

# Compare all 4 conditions
python evaluate.py --config configs/default.yaml --compare
```

### 5. Demo

```bash
python demo.py                       # Random sample, 5 dB SNR
python demo.py --snr 0 --seed 123   # Specific SNR and seed
```

The demo randomly samples clean + noise, mixes them, extracts features, runs all 4 models, and produces:
- `demo_output/clean.wav`, `noisy.wav`, `enhanced_{condition}.wav`
- `demo_output/mel_comparison.png`
- `demo_output/waveform_comparison.png`
- PESQ/STOI scores printed to stdout

---

## Conditioning Variants

| Variant | `condition_type` | MOSS Input | Extra Parameters |
|---------|-----------------|------------|-----------------|
| Baseline 1 | `none` | — | 28.4M (base DiT) |
| Baseline 2 | `last_layer` | h₃₁ only (768-dim) | +10.1M (cross-attn) |
| Ours (static) | `multi_layer` | All 32 layers (1280-dim) | +32 scalar weights |
| Ours (time-dep.) | `multi_layer_time` | All 32 layers (1280-dim) | +2,208 (MLP 1→64→32) |

---

## Key Design Decisions

1. **Offline feature extraction** — DAC and MOSS encoders run once; training uses only pre-computed `.pt` files.
2. **Rectified Flow** — Straight-path ODE from noisy to clean latent (NOT from Gaussian noise). Simple MSE loss on the velocity field.
3. **ODE solver** — 50-step Euler method at inference.
4. **adaLN timestep injection** — Adaptive layer normalisation, standard in DiT architectures.
5. **Learnable layer fusion** — Softmax-normalised weights over 32 MOSS layers, with optional time-dependent variant.
