"""
train.py — Main training loop for Flow-Matching Speech Enhancement.

Loads pre-computed .pt features (DAC latents & MOSS embeddings) from disk.
NO encoder runs during training — all features are offline.

Usage:
    python train.py --config configs/default.yaml
    python train.py --config configs/default.yaml --condition_type none
    python train.py --config configs/default.yaml --condition_type last_layer
    python train.py --config configs/default.yaml --condition_type multi_layer
"""

import argparse
import math
import os
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
import yaml
from tqdm import tqdm

from src.models.dit import DiffusionTransformer
from src.models.flow_matching import RectifiedFlow


# --------------------------------------------------------------------------- #
#  Dataset                                                                     #
# --------------------------------------------------------------------------- #

class OfflineFeatureDataset(Dataset):
    """
    Loads pre-computed .pt files:
        - clean DAC latents  (X_1)  : data/features/clean_dac/<stem>.pt   -> (T, D)
        - noisy DAC latents  (X_0)  : data/features/noisy_dac/<stem>.pt   -> (T, D)
        - MOSS last-layer    (C)    : data/features/moss_last/<stem>.pt   -> (T_c, D_c)
        - MOSS multi-layer   (C_ml) : data/features/moss_multi/<stem>.pt  -> list[(T_c, D_c)]

    All tensors are truncated / padded to ``max_seq_len`` DAC frames.
    MOSS tensors are scaled to ``max_seq_len // 4`` (50 Hz / 12.5 Hz = 4).
    """

    def __init__(
        self,
        features_dir: str,
        condition_type: str = "none",
        max_seq_len: int = 200,
    ):
        self.features_dir = Path(features_dir)
        self.condition_type = condition_type
        self.max_seq_len = max_seq_len
        self.max_cond_len = max_seq_len // 4  # 50 Hz -> 12.5 Hz ratio

        # Discover samples by the clean DAC directory
        clean_dac_dir = self.features_dir / "clean_dac"
        self.stems = sorted([f.stem for f in clean_dac_dir.glob("*.pt")])
        if len(self.stems) == 0:
            raise FileNotFoundError(
                f"No .pt files found in {clean_dac_dir}. "
                "Run extract_dac.py first."
            )
        print(f"[Dataset] Found {len(self.stems)} samples, condition_type={condition_type}")

    def _pad_or_truncate(self, tensor: torch.Tensor, max_len: int) -> torch.Tensor:
        """Pad (zero) or truncate along the time axis (dim 0)."""
        T = tensor.shape[0]
        if T >= max_len:
            return tensor[:max_len]
        pad = torch.zeros(max_len - T, *tensor.shape[1:], dtype=tensor.dtype)
        return torch.cat([tensor, pad], dim=0)

    def __len__(self):
        return len(self.stems)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        stem = self.stems[idx]

        # Load clean & noisy DAC latents
        x1 = torch.load(self.features_dir / "clean_dac" / f"{stem}.pt", map_location="cpu", weights_only=False)
        x0 = torch.load(self.features_dir / "noisy_dac" / f"{stem}.pt", map_location="cpu", weights_only=False)

        x1 = self._pad_or_truncate(x1, self.max_seq_len)
        x0 = self._pad_or_truncate(x0, self.max_seq_len)

        sample = {"x0": x0, "x1": x1}

        # Load MOSS condition if needed
        if self.condition_type == "last_layer":
            cond = torch.load(
                self.features_dir / "moss_last" / f"{stem}.pt", map_location="cpu", weights_only=False
            )
            cond = self._pad_or_truncate(cond, self.max_cond_len)
            sample["cond"] = cond

        elif self.condition_type == "multi_layer":
            cond_layers = torch.load(
                self.features_dir / "moss_multi" / f"{stem}.pt", map_location="cpu", weights_only=False
            )
            # cond_layers is a list of (T_c, D) tensors
            cond_layers = [self._pad_or_truncate(c, self.max_cond_len) for c in cond_layers]
            # Stack into (L, T_c, D) for easier batching
            sample["cond_layers"] = torch.stack(cond_layers, dim=0)

        return sample


def collate_fn(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    """Custom collate to handle optional keys."""
    out: Dict[str, torch.Tensor] = {}
    out["x0"] = torch.stack([b["x0"] for b in batch])
    out["x1"] = torch.stack([b["x1"] for b in batch])

    if "cond" in batch[0]:
        out["cond"] = torch.stack([b["cond"] for b in batch])
    if "cond_layers" in batch[0]:
        # (B, L, T_c, D)
        out["cond_layers"] = torch.stack([b["cond_layers"] for b in batch])

    return out


# --------------------------------------------------------------------------- #
#  Training Loop                                                               #
# --------------------------------------------------------------------------- #

def _auto_device(cfg_device: str) -> torch.device:
    """Resolve 'auto' or explicit device string to a torch.device."""
    if cfg_device == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(cfg_device)


def train(config: dict, condition_type_override: Optional[str] = None):
    # ---- Config ----
    cfg_data = config["data"]
    cfg_model = config["model"]
    cfg_train = config["training"]

    condition_type = condition_type_override or cfg_model["condition_type"]
    device = _auto_device(cfg_train["device"])

    # Seed
    seed = cfg_train["seed"]
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    print(f"{'='*60}")
    print(f"  Flow-Matching Speech Enhancement — Training")
    print(f"  condition_type = {condition_type}")
    print(f"  device         = {device}")
    print(f"{'='*60}")

    # ---- Dataset & DataLoader ----
    dataset = OfflineFeatureDataset(
        features_dir=cfg_data["features_dir"],
        condition_type=condition_type,
        max_seq_len=cfg_data["max_seq_len"],
    )
    loader = DataLoader(
        dataset,
        batch_size=cfg_train["batch_size"],
        shuffle=True,
        num_workers=2,
        pin_memory=True,
        collate_fn=collate_fn,
        drop_last=True,
    )

    # ---- Model ----
    # Auto-detect num_moss_layers and moss_embed_dim from the first .pt file
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

    # Fallback if still 'auto'
    if moss_embed_dim == "auto":
        moss_embed_dim = 768
        print(f"[Fallback] moss_embed_dim={moss_embed_dim}")

    model = DiffusionTransformer(
        dac_latent_dim=cfg_model["dac_latent_dim"],
        moss_embed_dim=moss_embed_dim,
        hidden_dim=cfg_model["hidden_dim"],
        num_heads=cfg_model["num_heads"],
        num_layers=cfg_model["num_layers"],
        dropout=cfg_model["dropout"],
        condition_type=condition_type,
        num_moss_layers=num_moss_layers,
    ).to(device)

    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {num_params:,}")

    # ---- Optimiser & Scheduler ----
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg_train["learning_rate"],
        weight_decay=cfg_train["weight_decay"],
    )
    # Linear warmup + cosine decay
    warmup_steps = cfg_train.get("warmup_steps", 0)
    total_steps = cfg_train["num_steps"]

    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(warmup_steps, 1)
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # ---- Rectified Flow ----
    flow = RectifiedFlow()

    # ---- Logging ----
    ckpt_dir = Path(cfg_train["checkpoint_dir"]) / condition_type
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=str(ckpt_dir / "logs"))

    # ---- Training ----
    model.train()
    global_step = 0
    epoch = 0

    while global_step < cfg_train["num_steps"]:
        epoch += 1
        for batch in loader:
            if global_step >= cfg_train["num_steps"]:
                break

            x0 = batch["x0"].to(device)
            x1 = batch["x1"].to(device)
            cond = batch.get("cond")
            cond_layers_batch = batch.get("cond_layers")

            if cond is not None:
                cond = cond.to(device)

            cond_layers = None
            if cond_layers_batch is not None:
                # (B, L, T_c, D) -> list of L tensors each (B, T_c, D)
                cond_layers_batch = cond_layers_batch.to(device)
                cond_layers = [cond_layers_batch[:, i] for i in range(cond_layers_batch.shape[1])]

            # Compute loss
            loss = flow.compute_loss(model, x0, x1, cond=cond, cond_layers=cond_layers)

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), cfg_train["gradient_clip"])
            optimizer.step()
            scheduler.step()

            global_step += 1

            # Logging
            if global_step % cfg_train["log_every"] == 0:
                lr = optimizer.param_groups[0]["lr"]
                print(
                    f"[Step {global_step:>6d}/{cfg_train['num_steps']}]  "
                    f"loss={loss.item():.6f}  lr={lr:.2e}"
                )
                writer.add_scalar("train/loss", loss.item(), global_step)
                writer.add_scalar("train/lr", lr, global_step)

            # Save checkpoint
            if global_step % cfg_train["save_every"] == 0:
                ckpt_path = ckpt_dir / f"step_{global_step}.pt"
                torch.save(
                    {
                        "step": global_step,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "config": config,
                        "condition_type": condition_type,
                    },
                    str(ckpt_path),
                )
                print(f"  -> Saved checkpoint: {ckpt_path}")

    writer.close()
    print(f"\nTraining complete. Final step: {global_step}")


# --------------------------------------------------------------------------- #
#  CLI                                                                         #
# --------------------------------------------------------------------------- #

def main():
    parser = argparse.ArgumentParser(description="Train Flow-Matching Speech Enhancement")
    parser.add_argument(
        "--config", type=str, default="configs/default.yaml", help="Path to YAML config"
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

    train(config, condition_type_override=args.condition_type)


if __name__ == "__main__":
    main()
