"""
evaluate.py — Inference & evaluation for Flow-Matching Speech Enhancement.

Pipeline:
  1. Load a trained checkpoint.
  2. For each sample, load noisy DAC latent (X_0) and optional MOSS condition.
  3. Run ODE solver (Euler) from X_0 → predicted X_1.
  4. Decode predicted X_1 back to waveform via DAC decoder.
  5. Compute FAD and save Mel-spectrogram visualisations.

Usage:
    python evaluate.py \
        --config configs/default.yaml \
        --checkpoint checkpoints/multi_layer/step_50000.pt \
        --condition_type multi_layer
"""

import argparse
import os
from pathlib import Path
from typing import Optional

import torch
import torchaudio
import yaml
from tqdm import tqdm

import dac
from dac.utils import load_model as load_dac_model

from src.models.dit import DiffusionTransformer
from src.models.flow_matching import ode_solve
from src.utils.metrics import compute_fad
from src.utils.viz import plot_mel_comparison


# --------------------------------------------------------------------------- #
#  Decode DAC latent → waveform                                                #
# --------------------------------------------------------------------------- #

@torch.no_grad()
def decode_dac_latent(
    z: torch.Tensor,
    dac_model: torch.nn.Module,
    device: str = "cuda",
) -> torch.Tensor:
    """
    Decode a continuous DAC latent back to a waveform.

    Parameters
    ----------
    z         : (T, D) continuous latent
    dac_model : pre-loaded DAC model
    device    : "cuda" or "cpu"

    Returns
    -------
    waveform : (1, num_samples) tensor
    """
    # DAC decoder expects (B, D, T)
    z = z.unsqueeze(0).permute(0, 2, 1).to(device)  # (1, D, T)

    # Quantise and decode (pass through VQ then decode)
    # For a clean pass-through, we can use the decoder directly on z
    # since z is the continuous latent (pre-quantisation representation).
    z_q, *_ = dac_model.quantizer(z)
    waveform = dac_model.decode(z_q)  # (1, 1, num_samples)
    return waveform.squeeze(0).cpu()


# --------------------------------------------------------------------------- #
#  Main evaluation                                                             #
# --------------------------------------------------------------------------- #

def _auto_device(cfg_device: str = "auto") -> torch.device:
    """Resolve 'auto' or explicit device string to a torch.device."""
    if cfg_device == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(cfg_device)


def evaluate(
    config: dict,
    checkpoint_path: str,
    condition_type_override: Optional[str] = None,
):
    cfg_data = config["data"]
    cfg_model = config["model"]
    cfg_eval = config["evaluation"]

    condition_type = condition_type_override or cfg_model["condition_type"]
    device = _auto_device(config["training"].get("device", "auto"))

    print(f"{'='*60}")
    print(f"  Flow-Matching Speech Enhancement — Evaluation")
    print(f"  condition_type = {condition_type}")
    print(f"  checkpoint     = {checkpoint_path}")
    print(f"  device         = {device}")
    print(f"{'='*60}")

    # ---- Load model ----
    # Auto-detect num_moss_layers and moss_embed_dim from saved data
    num_moss_layers = cfg_model.get("num_moss_layers", 32)
    moss_embed_dim = cfg_model.get("moss_embed_dim", "auto")

    if condition_type == "multi_layer":
        moss_multi_dir = Path(cfg_data["features_dir"]) / "moss_multi"
        sample_files = sorted(moss_multi_dir.glob("*.pt"))
        if sample_files:
            sample_layers = torch.load(str(sample_files[0]), map_location="cpu", weights_only=False)
            num_moss_layers = len(sample_layers)
            if moss_embed_dim == "auto":
                moss_embed_dim = sample_layers[0].shape[-1]
            print(f"[Auto-detect] MOSS multi-layer: {num_moss_layers} layers, dim={moss_embed_dim}")
    elif condition_type == "last_layer":
        moss_last_dir = Path(cfg_data["features_dir"]) / "moss_last"
        sample_files = sorted(moss_last_dir.glob("*.pt"))
        if sample_files and moss_embed_dim == "auto":
            sample_tensor = torch.load(str(sample_files[0]), map_location="cpu", weights_only=False)
            moss_embed_dim = sample_tensor.shape[-1]
            print(f"[Auto-detect] MOSS last-layer dim={moss_embed_dim}")

    if moss_embed_dim == "auto":
        moss_embed_dim = 768

    model = DiffusionTransformer(
        dac_latent_dim=cfg_model["dac_latent_dim"],
        moss_embed_dim=moss_embed_dim,
        hidden_dim=cfg_model["hidden_dim"],
        num_heads=cfg_model["num_heads"],
        num_layers=cfg_model["num_layers"],
        dropout=0.0,  # no dropout at inference
        condition_type=condition_type,
        num_moss_layers=num_moss_layers,
    ).to(device)

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    print("Model loaded.")

    # ---- Load DAC decoder ----
    dac_model = load_dac_model(model_type="16khz").to(device).eval()
    print("DAC decoder loaded.")

    # ---- Discover test samples ----
    features_dir = Path(cfg_data["features_dir"])
    noisy_dac_dir = features_dir / "noisy_dac"
    stems = sorted([f.stem for f in noisy_dac_dir.glob("*.pt")])
    print(f"Found {len(stems)} test samples.")

    # ---- Output dirs ----
    out_dir = Path(cfg_eval["output_dir"]) / condition_type
    wav_dir = out_dir / "wavs"
    mel_dir = out_dir / "mels"
    wav_dir.mkdir(parents=True, exist_ok=True)
    mel_dir.mkdir(parents=True, exist_ok=True)

    # ---- Inference loop ----
    max_seq_len = cfg_data["max_seq_len"]
    max_cond_len = max_seq_len // 4
    ode_steps = cfg_eval["ode_steps"]

    for stem in tqdm(stems, desc="Inference"):
        # Load noisy DAC latent
        x0 = torch.load(noisy_dac_dir / f"{stem}.pt", map_location="cpu", weights_only=False)
        if x0.shape[0] > max_seq_len:
            x0 = x0[:max_seq_len]
        x0 = x0.unsqueeze(0).to(device)  # (1, T, D)

        # Load condition
        cond = None
        cond_layers = None

        if condition_type == "last_layer":
            c = torch.load(features_dir / "moss_last" / f"{stem}.pt", map_location="cpu", weights_only=False)
            if c.shape[0] > max_cond_len:
                c = c[:max_cond_len]
            cond = c.unsqueeze(0).to(device)  # (1, T_c, D_c)

        elif condition_type == "multi_layer":
            cl = torch.load(features_dir / "moss_multi" / f"{stem}.pt", map_location="cpu", weights_only=False)
            cl = [layer[:max_cond_len] if layer.shape[0] > max_cond_len else layer for layer in cl]
            cond_layers = [layer.unsqueeze(0).to(device) for layer in cl]  # list of (1, T_c, D_c)

        # ODE solve: X_0 (noisy) -> X_1 (enhanced)
        x1_pred = ode_solve(
            model, x0,
            num_steps=ode_steps,
            cond=cond,
            cond_layers=cond_layers,
        )  # (1, T, D)

        # Decode to waveform
        z_pred = x1_pred.squeeze(0).cpu()  # (T, D)
        waveform = decode_dac_latent(z_pred, dac_model, device=str(device))

        # Save waveform
        wav_path = wav_dir / f"{stem}.wav"
        torchaudio.save(str(wav_path), waveform, cfg_data["sample_rate"])

    print(f"\nEnhanced wavs saved to {wav_dir}")

    # ---- Compute FAD ----
    # Assumes clean reference wavs exist in data/raw/clean
    clean_wav_dir = "data/raw/clean"
    if Path(clean_wav_dir).exists():
        print("\nComputing FAD ...")
        fad_score = compute_fad(str(wav_dir), clean_wav_dir, sr=cfg_data["sample_rate"])
        print(f"FAD score: {fad_score:.4f}")

        # Save score
        with open(out_dir / "fad_score.txt", "w") as f:
            f.write(f"condition_type: {condition_type}\n")
            f.write(f"fad: {fad_score:.4f}\n")
    else:
        print(f"[WARN] Clean wav dir '{clean_wav_dir}' not found — skipping FAD.")

    # ---- Mel-spectrogram visualisation (first 5 samples) ----
    noisy_wav_dir = "data/mixed"
    # Try to find noisy wavs — look for the first SNR subfolder
    noisy_wav_base = Path(noisy_wav_dir)
    if noisy_wav_base.exists():
        snr_dirs = sorted([d for d in noisy_wav_base.iterdir() if d.is_dir()])
        if snr_dirs:
            noisy_wav_source = snr_dirs[0]  # use first SNR level
        else:
            noisy_wav_source = noisy_wav_base
    else:
        noisy_wav_source = None

    if noisy_wav_source:
        for stem in stems[:5]:
            noisy_path = noisy_wav_source / f"{stem}.wav"
            enhanced_path = wav_dir / f"{stem}.wav"
            clean_path = Path(clean_wav_dir) / f"{stem}.wav"

            if noisy_path.exists() and enhanced_path.exists():
                plot_mel_comparison(
                    noisy_path=str(noisy_path),
                    enhanced_path=str(enhanced_path),
                    clean_path=str(clean_path) if clean_path.exists() else None,
                    sr=cfg_data["sample_rate"],
                    save_path=str(mel_dir / f"{stem}_mel.png"),
                    title=f"{stem} — {condition_type}",
                )

    print(f"\nEvaluation complete. Results in {out_dir}")


# --------------------------------------------------------------------------- #
#  CLI                                                                         #
# --------------------------------------------------------------------------- #

def main():
    parser = argparse.ArgumentParser(description="Evaluate Flow-Matching Speech Enhancement")
    parser.add_argument(
        "--config", type=str, default="configs/default.yaml", help="Path to YAML config"
    )
    parser.add_argument(
        "--checkpoint", type=str, required=True, help="Path to model checkpoint"
    )
    parser.add_argument(
        "--condition_type",
        type=str,
        default=None,
        choices=["none", "last_layer", "multi_layer"],
        help="Override condition_type from config",
    )
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    evaluate(config, args.checkpoint, condition_type_override=args.condition_type)


if __name__ == "__main__":
    main()
