#!/usr/bin/env python3
"""
demo.py — Local demo for Flow-Matching Speech Enhancement.

Randomly samples a clean utterance and a noise clip from local raw data,
mixes them at a target SNR, runs all 4 conditioning modes through the
trained models, and produces:
  1. Saved .wav files  (noisy / clean / enhanced x4)
  2. Mel-spectrogram comparison figure
  3. Per-sample PESQ / STOI scores printed to stdout

Prerequisites
-------------
1. Download checkpoints from Google Drive into checkpoints/:
       https://drive.google.com/drive/folders/1siQa2Rlsx1A4DYlEXo6qyksoOM1poyU8
   Expected layout:
       checkpoints/none/best.pt
       checkpoints/last_layer/best.pt
       checkpoints/multi_layer/best.pt
       checkpoints/multi_layer_time/best.pt

2. Raw data should be at:
       data/raw/clean/LibriSpeech/dev-clean/
       data/raw/noise/musan/noise/free-sound/

Usage:
    python demo.py                       # random sample, 5 dB SNR
    python demo.py --snr 0               # 0 dB SNR
    python demo.py --seed 123            # reproducible sample
    python demo.py --clean_file <path>   # use specific files
"""

import argparse
import os
import random
from pathlib import Path

import numpy as np
import torch
import torchaudio
import soundfile as sf
import matplotlib.pyplot as plt
import librosa
import librosa.display

# Project imports
from src.data.mixer import load_audio, mix_at_snr
from src.models.dit import DiffusionTransformer
from src.models.flow_matching import ode_solve
from evaluate import decode_dac_latent, compute_pesq, compute_stoi

import dac
from dac.utils import load_model as load_dac_model


# ── Utilities ────────────────────────────────────────────────────────────── #

CONDITION_TYPES = ["none", "last_layer", "multi_layer", "multi_layer_time"]

# Model hyper-parameters (must match training config)
MODEL_CFG = dict(
    dac_latent_dim=1024,
    hidden_dim=512,
    num_heads=8,
    num_layers=6,
    num_moss_layers=32,
)


def auto_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def find_audio_files(directory: str) -> list[Path]:
    """Recursively find audio files (.wav/.flac) under a directory."""
    d = Path(directory)
    exts = (".wav", ".flac")
    return sorted([f for ext in exts for f in d.rglob(f"*{ext}")])


def compute_mel(waveform: np.ndarray, sr: int = 16000) -> np.ndarray:
    """Compute log-Mel spectrogram for plotting."""
    mel = librosa.feature.melspectrogram(
        y=waveform, sr=sr, n_fft=1024, hop_length=256, n_mels=80
    )
    return librosa.power_to_db(mel, ref=np.max)


# ── Feature extraction (single-file, on-the-fly) ────────────────────────── #

@torch.no_grad()
def extract_dac_latent(waveform_16k: torch.Tensor, dac_model, device) -> torch.Tensor:
    """Encode a 16 kHz waveform to continuous DAC latent (T, 1024)."""
    wav = waveform_16k.unsqueeze(0).unsqueeze(0).to(device)   # (1, 1, T)
    z = dac_model.encoder(wav)                                 # (1, D, T_dac)
    return z.squeeze(0).permute(1, 0).cpu()                    # (T_dac, D)


@torch.no_grad()
def extract_moss_last(waveform_24k: torch.Tensor, moss_model, device) -> torch.Tensor:
    """Extract MOSS last-layer embedding (T_moss, 768)."""
    wav = waveform_24k.unsqueeze(0).unsqueeze(0).to(device)    # (1, 1, T)
    enc = moss_model.encode(wav, return_dict=True)
    return enc.encoder_hidden_states.squeeze(0).permute(1, 0).cpu()


@torch.no_grad()
def extract_moss_multi(waveform_24k: torch.Tensor, moss_model, device) -> list[torch.Tensor]:
    """Extract all 32 MOSS Stage-3 hidden layers → list of (T_moss, 1280)."""
    # Locate the last ProjectedTransformer (Stage 3)
    last_mod = None
    for module in moss_model.encoder:
        if hasattr(module, "transformer"):
            last_mod = module
    if last_mod is None:
        raise RuntimeError("Could not find ProjectedTransformer in MOSS encoder")

    layer_outputs: list[torch.Tensor] = []

    def _make_hook():
        def hook_fn(_module, _input, output):
            layer_outputs.append(output.detach().cpu())
        return hook_fn

    hooks = []
    for layer in last_mod.transformer.layers:
        hooks.append(layer.register_forward_hook(_make_hook()))

    wav = waveform_24k.unsqueeze(0).unsqueeze(0).to(device)
    layer_outputs.clear()
    moss_model.encode(wav, return_dict=True)

    for h in hooks:
        h.remove()

    return [h.squeeze(0) for h in layer_outputs]  # list of (T_moss, 1280)


# ── Model loading ────────────────────────────────────────────────────────── #

def load_enhancement_model(condition_type: str, checkpoint_path: str, device: torch.device):
    """Load a trained DiffusionTransformer from checkpoint."""
    moss_dim = 1280 if condition_type in ("multi_layer", "multi_layer_time") else 768
    if condition_type == "none":
        moss_dim = 768  # placeholder, unused

    model = DiffusionTransformer(
        dac_latent_dim=MODEL_CFG["dac_latent_dim"],
        moss_embed_dim=moss_dim,
        hidden_dim=MODEL_CFG["hidden_dim"],
        num_heads=MODEL_CFG["num_heads"],
        num_layers=MODEL_CFG["num_layers"],
        dropout=0.0,
        condition_type=condition_type,
        num_moss_layers=MODEL_CFG["num_moss_layers"],
    ).to(device)

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model


# ── Main demo pipeline ───────────────────────────────────────────────────── #

def run_demo(args):
    device = auto_device()
    print(f"Device: {device}")

    sr = 16000
    max_seq_len = 200  # 4 s at 50 Hz DAC
    max_cond_len = 50  # 4 s at 12.5 Hz MOSS
    ode_steps = 50

    # ── 1. Select audio files ────────────────────────────────────────────
    random.seed(args.seed)
    np.random.seed(args.seed)

    if args.clean_file:
        clean_path = args.clean_file
    else:
        clean_files = find_audio_files(args.clean_dir)
        clean_path = str(random.choice(clean_files))

    if args.noise_file:
        noise_path = args.noise_file
    else:
        noise_files = find_audio_files(args.noise_dir)
        noise_path = str(random.choice(noise_files))

    print(f"\nClean: {clean_path}")
    print(f"Noise: {noise_path}")
    print(f"SNR:   {args.snr} dB\n")

    # ── 2. Load and mix ──────────────────────────────────────────────────
    clean_audio = load_audio(clean_path, sr=sr)
    noise_audio = load_audio(noise_path, sr=sr)
    noisy_audio = mix_at_snr(clean_audio, noise_audio, args.snr)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    sf.write(str(out_dir / "clean.wav"), clean_audio, sr)
    sf.write(str(out_dir / "noisy.wav"), noisy_audio, sr)
    print(f"Saved clean.wav and noisy.wav ({len(clean_audio)/sr:.2f}s)")

    # ── 3. Load tokenizers (DAC + MOSS) ──────────────────────────────────
    print("\nLoading DAC model (16 kHz) ...")
    dac_model = load_dac_model(model_type="16khz").to(device).eval()

    print("Loading MOSS-Audio-Tokenizer ...")
    from transformers import AutoModel
    moss_model = AutoModel.from_pretrained(
        "OpenMOSS-Team/MOSS-Audio-Tokenizer", trust_remote_code=True
    ).to(device).eval()

    # ── 4. Extract features ──────────────────────────────────────────────
    clean_t = torch.from_numpy(clean_audio).float()
    noisy_t = torch.from_numpy(noisy_audio).float()

    # DAC latents (16 kHz input)
    z_clean = extract_dac_latent(clean_t, dac_model, device)  # (T_dac, 1024)
    z_noisy = extract_dac_latent(noisy_t, dac_model, device)

    # Truncate to max_seq_len
    z_clean = z_clean[:max_seq_len]
    z_noisy = z_noisy[:max_seq_len]

    # Resample to 24 kHz for MOSS
    noisy_24k = torchaudio.functional.resample(noisy_t.unsqueeze(0), sr, 24000).squeeze(0)

    # MOSS features
    print("Extracting MOSS features ...")
    moss_last = extract_moss_last(noisy_24k, moss_model, device)[:max_cond_len]
    moss_multi = extract_moss_multi(noisy_24k, moss_model, device)
    moss_multi = [layer[:max_cond_len] for layer in moss_multi]

    print(f"  DAC latent shape:      {z_noisy.shape}")
    print(f"  MOSS last-layer shape: {moss_last.shape}")
    print(f"  MOSS multi-layer:      {len(moss_multi)} × {moss_multi[0].shape}")

    # ── 5. Run enhancement for all 4 conditions ─────────────────────────
    ckpt_base = Path(args.checkpoint_dir)
    results = {}

    for ct in CONDITION_TYPES:
        ckpt_path = ckpt_base / ct / "best.pt"
        if not ckpt_path.exists():
            print(f"\n⚠  Checkpoint not found: {ckpt_path}, skipping {ct}")
            continue

        print(f"\n── Enhancing with condition = {ct} ──")
        model = load_enhancement_model(ct, str(ckpt_path), device)

        # Prepare inputs
        x0 = z_noisy.unsqueeze(0).to(device)
        cond = None
        cond_layers = None

        if ct == "last_layer":
            cond = moss_last.unsqueeze(0).to(device)
        elif ct in ("multi_layer", "multi_layer_time"):
            cond_layers = [layer.unsqueeze(0).to(device) for layer in moss_multi]

        # ODE solve
        x1_pred = ode_solve(model, x0, num_steps=ode_steps, cond=cond, cond_layers=cond_layers)
        z_pred = x1_pred.squeeze(0).cpu()

        # Decode to waveform
        enhanced_wav = decode_dac_latent(z_pred, dac_model, device=str(device))
        wav_np = enhanced_wav.squeeze(0).numpy()

        out_path = out_dir / f"enhanced_{ct}.wav"
        sf.write(str(out_path), wav_np, sr)

        # Metrics against clean DAC reconstruction (consistent with evaluate.py)
        clean_wav_from_dac = decode_dac_latent(z_clean, dac_model, device=str(device))

        pesq_val = compute_pesq(clean_wav_from_dac, enhanced_wav, sr=sr)
        stoi_val = compute_stoi(clean_wav_from_dac, enhanced_wav, sr=sr)

        results[ct] = {"PESQ": pesq_val, "STOI": stoi_val, "waveform": wav_np}
        print(f"  PESQ = {pesq_val:.4f}   STOI = {stoi_val:.4f}")

        # Free GPU memory
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # ── 6. Summary table ─────────────────────────────────────────────────
    print(f"\n{'='*55}")
    print(f"  {'Condition':<20s}  {'PESQ':>8s}  {'STOI':>8s}")
    print(f"  {'-'*20}  {'-'*8}  {'-'*8}")
    for ct in CONDITION_TYPES:
        if ct in results:
            r = results[ct]
            print(f"  {ct:<20s}  {r['PESQ']:8.4f}  {r['STOI']:8.4f}")
    print(f"{'='*55}")

    # ── 7. Mel-spectrogram figure ────────────────────────────────────────
    num_panels = 2 + len(results)  # noisy + clean + enhanced variants
    fig, axes = plt.subplots(1, num_panels, figsize=(5 * num_panels, 4))

    panels = [("Noisy", noisy_audio), ("Clean", clean_audio)]
    for ct in CONDITION_TYPES:
        if ct in results:
            panels.append((f"Enhanced\n({ct})", results[ct]["waveform"]))

    for ax, (label, wav_np) in zip(axes, panels):
        mel_db = compute_mel(wav_np, sr=sr)
        img = librosa.display.specshow(
            mel_db, sr=sr, hop_length=256, x_axis="time", y_axis="mel", ax=ax
        )
        ax.set_title(label, fontsize=10)
        fig.colorbar(img, ax=ax, format="%+2.0f dB")

    fig.suptitle(
        f"Flow-Matching Speech Enhancement — Tokenizer-Layer Comparison (SNR={args.snr} dB)",
        fontsize=13,
    )
    plt.tight_layout()

    fig_path = out_dir / "mel_comparison.png"
    plt.savefig(str(fig_path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nMel-spectrogram figure saved to {fig_path}")

    # ── 8. Waveform overlay figure ───────────────────────────────────────
    fig2, axes2 = plt.subplots(num_panels, 1, figsize=(12, 3 * num_panels), sharex=True)

    for ax, (label, wav_np) in zip(axes2, panels):
        t_axis = np.arange(len(wav_np)) / sr
        ax.plot(t_axis, wav_np, linewidth=0.4)
        ax.set_ylabel(label, fontsize=9)
        ax.set_xlim(0, t_axis[-1])

    axes2[-1].set_xlabel("Time (s)")
    fig2.suptitle("Waveform Comparison", fontsize=13)
    plt.tight_layout()

    fig2_path = out_dir / "waveform_comparison.png"
    plt.savefig(str(fig2_path), dpi=150, bbox_inches="tight")
    plt.close(fig2)
    print(f"Waveform figure saved to {fig2_path}")

    print("\nDemo complete! Output files:")
    for f in sorted(out_dir.iterdir()):
        print(f"  {f.name}")


# ── CLI ──────────────────────────────────────────────────────────────────── #

def main():
    parser = argparse.ArgumentParser(
        description="Local demo: mix audio, enhance with all 4 conditions, visualise"
    )
    parser.add_argument(
        "--clean_dir", type=str,
        default="data/raw/clean/LibriSpeech/dev-clean",
        help="Directory of clean speech files",
    )
    parser.add_argument(
        "--noise_dir", type=str,
        default="data/raw/noise/musan/noise/free-sound",
        help="Directory of noise files",
    )
    parser.add_argument("--clean_file", type=str, default=None, help="Specific clean file (overrides --clean_dir)")
    parser.add_argument("--noise_file", type=str, default=None, help="Specific noise file (overrides --noise_dir)")
    parser.add_argument("--snr", type=float, default=5.0, help="Mixing SNR in dB (default: 5)")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints", help="Base dir for model checkpoints")
    parser.add_argument("--output_dir", type=str, default="demo_output", help="Output directory")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    args = parser.parse_args()

    run_demo(args)


if __name__ == "__main__":
    main()
