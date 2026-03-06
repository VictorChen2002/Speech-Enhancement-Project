"""
evaluate.py — Inference & evaluation for Flow-Matching Speech Enhancement.

Pipeline:
  1. Load a trained checkpoint (best.pt).
  2. For each TEST sample, load noisy DAC latent (X_0) and optional MOSS condition.
  3. Run ODE solver (Euler) from X_0 → predicted X_1.
  4. Decode predicted X_1 back to waveform via DAC decoder.
  5. Compute PESQ, STOI, FAD and save Mel-spectrogram visualisations.

Supports:
  * Single condition type evaluation
  * Compare mode: evaluate all 3 condition types and print a summary table

Usage:
    # Evaluate one condition type
    python evaluate.py --config configs/colab.yaml \\
        --checkpoint checkpoints/multi_layer/best.pt \\
        --condition_type multi_layer

    # Compare all 3 condition types (checkpoints auto-discovered)
    python evaluate.py --config configs/colab.yaml --compare
"""

import argparse
import glob
import json
import os
from pathlib import Path
from typing import Optional, List, Dict

import numpy as np
import torch
import torchaudio
import yaml
from tqdm import tqdm

import dac
from dac.utils import load_model as load_dac_model

from src.models.dit import DiffusionTransformer
from src.models.flow_matching import ode_solve


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
    z = z.unsqueeze(0).permute(0, 2, 1).to(device)  # (1, D, T)
    z_q, *_ = dac_model.quantizer(z)
    waveform = dac_model.decode(z_q)  # (1, 1, num_samples)
    return waveform.squeeze(0).cpu()


# --------------------------------------------------------------------------- #
#  Per-sample metrics                                                          #
# --------------------------------------------------------------------------- #

def compute_pesq(ref_wav: torch.Tensor, deg_wav: torch.Tensor, sr: int = 16000) -> float:
    """Compute PESQ (Perceptual Evaluation of Speech Quality)."""
    from pesq import pesq
    ref = ref_wav.squeeze().numpy()
    deg = deg_wav.squeeze().numpy()
    min_len = min(len(ref), len(deg))
    ref, deg = ref[:min_len], deg[:min_len]
    try:
        score = pesq(sr, ref, deg, "wb")
    except Exception:
        score = float("nan")
    return score


def compute_stoi(ref_wav: torch.Tensor, deg_wav: torch.Tensor, sr: int = 16000) -> float:
    """Compute STOI (Short-Time Objective Intelligibility)."""
    from pystoi import stoi
    ref = ref_wav.squeeze().numpy()
    deg = deg_wav.squeeze().numpy()
    min_len = min(len(ref), len(deg))
    ref, deg = ref[:min_len], deg[:min_len]
    try:
        score = stoi(ref, deg, sr, extended=False)
    except Exception:
        score = float("nan")
    return score


# --------------------------------------------------------------------------- #
#  Helpers                                                                     #
# --------------------------------------------------------------------------- #

def _auto_device(cfg_device: str = "auto") -> torch.device:
    if cfg_device == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(cfg_device)


def _get_test_stems(config: dict) -> List[str]:
    """Load test stems from the split file."""
    split_file = config["data"].get("split_file", "data/split.json")
    if Path(split_file).exists():
        with open(split_file) as f:
            split = json.load(f)
        return split["test"]
    features_dir = Path(config["data"]["features_dir"])
    return sorted([f.stem for f in (features_dir / "noisy_dac").glob("*.pt")])


def _find_checkpoint(ckpt_base: str, condition_type: str, drive_ckpt_dir: Optional[str] = None) -> Optional[str]:
    """Find best.pt or latest step checkpoint."""
    search_dirs = [ckpt_base]
    if drive_ckpt_dir:
        search_dirs.insert(0, drive_ckpt_dir)

    for root in search_dirs:
        best = Path(root) / condition_type / "best.pt"
        if best.exists():
            return str(best)
        pattern = str(Path(root) / condition_type / "step_*.pt")
        ckpts = sorted(glob.glob(pattern))
        if ckpts:
            return ckpts[-1]
    return None


# --------------------------------------------------------------------------- #
#  Main evaluation                                                             #
# --------------------------------------------------------------------------- #

def evaluate(
    config: dict,
    checkpoint_path: str,
    condition_type: str,
    output_dir: Optional[str] = None,
) -> Dict[str, float]:
    """
    Evaluate a single condition type on the test set.
    Returns dict with PESQ, STOI, FAD scores.
    """
    cfg_data = config["data"]
    cfg_model = config["model"]
    cfg_eval = config["evaluation"]

    device = _auto_device(config["training"].get("device", "auto"))

    print(f"\n{'='*60}")
    print(f"  Evaluation — {condition_type}")
    print(f"  checkpoint = {checkpoint_path}")
    print(f"  device     = {device}")
    print(f"{'='*60}")

    # ---- Load model ----
    num_moss_layers = cfg_model.get("num_moss_layers", 32)
    moss_embed_dim = cfg_model.get("moss_embed_dim", "auto")
    features_dir = Path(cfg_data["features_dir"])

    if condition_type in ("multi_layer", "multi_layer_time"):
        moss_multi_dir = features_dir / "moss_multi"
        sample_files = sorted(moss_multi_dir.glob("*.pt"))
        if sample_files:
            sample_layers = torch.load(str(sample_files[0]), map_location="cpu", weights_only=False)
            num_moss_layers = len(sample_layers)
            if moss_embed_dim == "auto":
                moss_embed_dim = sample_layers[0].shape[-1]
    elif condition_type == "last_layer":
        moss_last_dir = features_dir / "moss_last"
        sample_files = sorted(moss_last_dir.glob("*.pt"))
        if sample_files and moss_embed_dim == "auto":
            sample_tensor = torch.load(str(sample_files[0]), map_location="cpu", weights_only=False)
            moss_embed_dim = sample_tensor.shape[-1]

    if moss_embed_dim == "auto":
        moss_embed_dim = 768

    model = DiffusionTransformer(
        dac_latent_dim=cfg_model["dac_latent_dim"],
        moss_embed_dim=moss_embed_dim,
        hidden_dim=cfg_model["hidden_dim"],
        num_heads=cfg_model["num_heads"],
        num_layers=cfg_model["num_layers"],
        dropout=0.0,
        condition_type=condition_type,
        num_moss_layers=num_moss_layers,
    ).to(device)

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    print(f"Model loaded (step={ckpt.get('step', '?')}, epoch={ckpt.get('epoch', '?')})")

    # ---- Load DAC decoder ----
    dac_model = load_dac_model(model_type="16khz").to(device).eval()

    # ---- Test samples ----
    test_stems = _get_test_stems(config)
    print(f"Test samples: {len(test_stems)}")

    # ---- Output dirs ----
    out_root = Path(output_dir or cfg_eval["output_dir"])
    out_dir = out_root / condition_type
    wav_dir = out_dir / "wavs"
    clean_wav_dir = out_dir / "clean_wavs"
    wav_dir.mkdir(parents=True, exist_ok=True)
    clean_wav_dir.mkdir(parents=True, exist_ok=True)

    # ---- Inference + per-sample metrics ----
    max_seq_len = cfg_data["max_seq_len"]
    max_cond_len = max_seq_len // 4
    ode_steps = cfg_eval["ode_steps"]
    sr = cfg_data["sample_rate"]

    pesq_scores = []
    stoi_scores = []

    for stem in tqdm(test_stems, desc=f"Eval {condition_type}"):
        x0 = torch.load(features_dir / "noisy_dac" / f"{stem}.pt", map_location="cpu", weights_only=False)
        if x0.shape[0] > max_seq_len:
            x0 = x0[:max_seq_len]
        x0_batch = x0.unsqueeze(0).to(device)

        cond = None
        cond_layers = None

        if condition_type == "last_layer":
            c = torch.load(features_dir / "moss_last" / f"{stem}.pt", map_location="cpu", weights_only=False)
            if c.shape[0] > max_cond_len:
                c = c[:max_cond_len]
            cond = c.unsqueeze(0).to(device)

        elif condition_type in ("multi_layer", "multi_layer_time"):
            cl = torch.load(features_dir / "moss_multi" / f"{stem}.pt", map_location="cpu", weights_only=False)
            cl = [layer[:max_cond_len] if layer.shape[0] > max_cond_len else layer for layer in cl]
            cond_layers = [layer.unsqueeze(0).to(device) for layer in cl]

        x1_pred = ode_solve(model, x0_batch, num_steps=ode_steps, cond=cond, cond_layers=cond_layers)
        z_pred = x1_pred.squeeze(0).cpu()

        enhanced_wav = decode_dac_latent(z_pred, dac_model, device=str(device))
        torchaudio.save(str(wav_dir / f"{stem}.wav"), enhanced_wav, sr)

        x1_clean = torch.load(features_dir / "clean_dac" / f"{stem}.pt", map_location="cpu", weights_only=False)
        if x1_clean.shape[0] > max_seq_len:
            x1_clean = x1_clean[:max_seq_len]
        clean_wav = decode_dac_latent(x1_clean, dac_model, device=str(device))
        torchaudio.save(str(clean_wav_dir / f"{stem}.wav"), clean_wav, sr)

        pesq_val = compute_pesq(clean_wav, enhanced_wav, sr=sr)
        stoi_val = compute_stoi(clean_wav, enhanced_wav, sr=sr)
        pesq_scores.append(pesq_val)
        stoi_scores.append(stoi_val)

    # ---- Aggregate metrics ----
    pesq_scores = [s for s in pesq_scores if not np.isnan(s)]
    stoi_scores = [s for s in stoi_scores if not np.isnan(s)]

    avg_pesq = float(np.mean(pesq_scores)) if pesq_scores else float("nan")
    avg_stoi = float(np.mean(stoi_scores)) if stoi_scores else float("nan")

    try:
        from src.utils.metrics import compute_fad
        fad_score = compute_fad(str(wav_dir), str(clean_wav_dir), sr=sr)
    except Exception as e:
        print(f"[WARN] FAD computation failed: {e}")
        fad_score = float("nan")

    results = {
        "condition_type": condition_type,
        "checkpoint": checkpoint_path,
        "num_samples": len(test_stems),
        "PESQ": round(avg_pesq, 4),
        "STOI": round(avg_stoi, 4),
        "FAD": round(fad_score, 4) if not np.isnan(fad_score) else "N/A",
    }

    print(f"\n[Results — {condition_type}]")
    print(f"  PESQ: {avg_pesq:.4f}")
    print(f"  STOI: {avg_stoi:.4f}")
    print(f"  FAD:  {fad_score:.4f}" if not np.isnan(fad_score) else "  FAD:  N/A")

    with open(out_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)

    return results


def compare_all(config: dict, drive_ckpt_dir: Optional[str] = None):
    """Evaluate all 3 condition types and print comparison table."""
    ckpt_base = config["training"]["checkpoint_dir"]
    all_results = []

    for ct in ["none", "last_layer", "multi_layer", "multi_layer_time"]:
        ckpt_path = _find_checkpoint(ckpt_base, ct, drive_ckpt_dir)
        if ckpt_path is None:
            print(f"\n⚠️  No checkpoint found for '{ct}', skipping.")
            continue
        try:
            results = evaluate(config, ckpt_path, ct)
            all_results.append(results)
        except Exception as e:
            print(f"\n❌ Evaluation failed for '{ct}': {e}")

    if not all_results:
        print("\nNo results to compare.")
        return

    print(f"\n{'='*70}")
    print(f"  COMPARISON TABLE — Test Set ({all_results[0]['num_samples']} samples)")
    print(f"{'='*70}")
    print(f"  {'Condition':<15s}  {'PESQ':>8s}  {'STOI':>8s}  {'FAD':>8s}")
    print(f"  {'-'*15}  {'-'*8}  {'-'*8}  {'-'*8}")
    for r in all_results:
        pesq_s = f"{r['PESQ']:.4f}" if isinstance(r['PESQ'], float) else str(r['PESQ'])
        stoi_s = f"{r['STOI']:.4f}" if isinstance(r['STOI'], float) else str(r['STOI'])
        fad_s = f"{r['FAD']:.4f}" if isinstance(r['FAD'], (int, float)) else str(r['FAD'])
        print(f"  {r['condition_type']:<15s}  {pesq_s:>8s}  {stoi_s:>8s}  {fad_s:>8s}")
    print(f"{'='*70}")

    pesq_results = [(r['condition_type'], r['PESQ']) for r in all_results if isinstance(r['PESQ'], float)]
    if pesq_results:
        best = max(pesq_results, key=lambda x: x[1])
        print(f"\n  Best PESQ: {best[0]} ({best[1]:.4f})")

    out_dir = Path(config["evaluation"]["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "comparison.json", "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"  Results saved to {out_dir / 'comparison.json'}")


# --------------------------------------------------------------------------- #
#  CLI                                                                         #
# --------------------------------------------------------------------------- #

def main():
    parser = argparse.ArgumentParser(description="Evaluate Flow-Matching Speech Enhancement")
    parser.add_argument(
        "--config", type=str, default="configs/default.yaml", help="Path to YAML config"
    )
    parser.add_argument(
        "--checkpoint", type=str, default=None, help="Path to model checkpoint"
    )
    parser.add_argument(
        "--condition_type",
        type=str,
        default=None,
        choices=["none", "last_layer", "multi_layer", "multi_layer_time"],
        help="Which condition type to evaluate",
    )
    parser.add_argument(
        "--compare", action="store_true",
        help="Compare all 3 condition types (auto-discover checkpoints)",
    )
    parser.add_argument(
        "--drive_ckpt_dir", type=str, default=None,
        help="Google Drive checkpoint directory (for compare mode)",
    )
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    if args.compare:
        compare_all(config, drive_ckpt_dir=args.drive_ckpt_dir)
    else:
        if not args.checkpoint:
            parser.error("--checkpoint is required when not using --compare")
        ct = args.condition_type or config["model"]["condition_type"]
        evaluate(config, args.checkpoint, ct)


if __name__ == "__main__":
    main()
