"""
train.py — Main training loop for Flow-Matching Speech Enhancement.

Loads pre-computed .pt features (DAC latents & MOSS embeddings) from disk.
NO encoder runs during training — all features are offline.

Features:
  * Train / validation split (via JSON split file)
  * Weights & Biases (wandb) logging
  * Resume from checkpoint (--resume)
  * Immediate checkpoint-to-Drive sync (--drive_ckpt_dir)
  * Periodic validation loss logging

Usage:
    python train.py --config configs/default.yaml
    python train.py --config configs/colab.yaml --condition_type multi_layer \\
        --wandb --drive_ckpt_dir /content/drive/MyDrive/speech_enhancement_checkpoints
    python train.py --config configs/colab.yaml --resume checkpoints/multi_layer/step_5000.pt
"""

import argparse
import json
import math
import os
import random
import shutil
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
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
        stems: Optional[List[str]] = None,
    ):
        self.features_dir = Path(features_dir)
        self.condition_type = condition_type
        self.max_seq_len = max_seq_len
        self.max_cond_len = max_seq_len // 4  # 50 Hz -> 12.5 Hz ratio

        if stems is not None:
            self.stems = stems
        else:
            # Discover samples by the clean DAC directory
            clean_dac_dir = self.features_dir / "clean_dac"
            self.stems = sorted([f.stem for f in clean_dac_dir.glob("*.pt")])

        if len(self.stems) == 0:
            raise FileNotFoundError(
                f"No .pt files found. Check features_dir={features_dir}"
            )

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

        elif self.condition_type in ("multi_layer", "multi_layer_time"):
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
#  Helpers                                                                     #
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


def _load_or_create_split(features_dir: str, split_file: str, seed: int = 42):
    """
    Load an existing train/valid/test split JSON, or create one with
    80/10/10 ratio and save it for reproducibility.

    Returns:  (train_stems, valid_stems, test_stems)
    """
    split_path = Path(split_file)
    if split_path.exists():
        with open(split_path) as f:
            split = json.load(f)
        return split["train"], split["valid"], split["test"]

    # Discover all stems
    clean_dac_dir = Path(features_dir) / "clean_dac"
    all_stems = sorted([f.stem for f in clean_dac_dir.glob("*.pt")])

    rng = random.Random(seed)
    rng.shuffle(all_stems)

    n = len(all_stems)
    n_test = max(1, int(n * 0.10))
    n_valid = max(1, int(n * 0.10))
    n_train = n - n_valid - n_test

    split = {
        "train": sorted(all_stems[:n_train]),
        "valid": sorted(all_stems[n_train:n_train + n_valid]),
        "test":  sorted(all_stems[n_train + n_valid:]),
    }

    split_path.parent.mkdir(parents=True, exist_ok=True)
    with open(split_path, "w") as f:
        json.dump(split, f, indent=2)

    return split["train"], split["valid"], split["test"]


def _detect_moss_dims(features_dir: str, condition_type: str, cfg_model: dict):
    """Auto-detect num_moss_layers and moss_embed_dim from the first .pt file."""
    num_moss_layers = cfg_model.get("num_moss_layers", 32)
    moss_embed_dim = cfg_model.get("moss_embed_dim", "auto")

    if condition_type in ("multi_layer", "multi_layer_time"):
        moss_multi_dir = Path(features_dir) / "moss_multi"
        sample_files = sorted(moss_multi_dir.glob("*.pt"))
        if sample_files:
            sample_layers = torch.load(str(sample_files[0]), map_location="cpu", weights_only=False)
            num_moss_layers = len(sample_layers)
            if moss_embed_dim == "auto":
                moss_embed_dim = sample_layers[0].shape[-1]
            print(f"[Auto-detect] MOSS multi-layer: {num_moss_layers} layers, dim={moss_embed_dim}")
    elif condition_type == "last_layer":
        moss_last_dir = Path(features_dir) / "moss_last"
        sample_files = sorted(moss_last_dir.glob("*.pt"))
        if sample_files and moss_embed_dim == "auto":
            sample_tensor = torch.load(str(sample_files[0]), map_location="cpu", weights_only=False)
            moss_embed_dim = sample_tensor.shape[-1]
            print(f"[Auto-detect] MOSS last-layer dim={moss_embed_dim}")

    if moss_embed_dim == "auto":
        moss_embed_dim = 768
        print(f"[Fallback] moss_embed_dim={moss_embed_dim}")

    return num_moss_layers, moss_embed_dim


@torch.no_grad()
def _validate(model, val_loader, flow, device, condition_type):
    """Run one pass over the validation set and return average loss."""
    model.eval()
    total_loss = 0.0
    n_batches = 0
    for batch in val_loader:
        x0 = batch["x0"].to(device)
        x1 = batch["x1"].to(device)
        cond = batch.get("cond")
        cond_layers_batch = batch.get("cond_layers")

        if cond is not None:
            cond = cond.to(device)

        cond_layers = None
        if cond_layers_batch is not None:
            cond_layers_batch = cond_layers_batch.to(device)
            cond_layers = [cond_layers_batch[:, i] for i in range(cond_layers_batch.shape[1])]

        loss = flow.compute_loss(model, x0, x1, cond=cond, cond_layers=cond_layers)
        total_loss += loss.item()
        n_batches += 1

    model.train()
    return total_loss / max(n_batches, 1)


def _save_checkpoint(
    model, optimizer, scheduler, global_step, epoch, config, condition_type,
    ckpt_dir, drive_ckpt_dir=None,
    best_val_loss=float("inf"), no_improve_count=0,
):
    """Save checkpoint locally and optionally copy to Drive."""
    ckpt_path = ckpt_dir / f"step_{global_step}.pt"
    payload = {
        "step": global_step,
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "config": config,
        "condition_type": condition_type,
        "best_val_loss": best_val_loss,
        "no_improve_count": no_improve_count,
    }
    torch.save(payload, str(ckpt_path))
    print(f"  -> Saved checkpoint: {ckpt_path}")

    # Copy to Drive immediately
    if drive_ckpt_dir:
        drive_ct_dir = Path(drive_ckpt_dir) / condition_type
        drive_ct_dir.mkdir(parents=True, exist_ok=True)
        dst = drive_ct_dir / f"step_{global_step}.pt"
        shutil.copy2(str(ckpt_path), str(dst))
        print(f"  -> Synced to Drive: {dst}")

    return ckpt_path


# --------------------------------------------------------------------------- #
#  Training Loop                                                               #
# --------------------------------------------------------------------------- #

def train(
    config: dict,
    condition_type_override: Optional[str] = None,
    resume_path: Optional[str] = None,
    use_wandb: bool = False,
    wandb_project: str = "speech-enhancement",
    wandb_run_name: Optional[str] = None,
    drive_ckpt_dir: Optional[str] = None,
):
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
    print(f"  resume         = {resume_path or 'None'}")
    print(f"  wandb          = {use_wandb}")
    print(f"  drive_ckpt_dir = {drive_ckpt_dir or 'None'}")
    print(f"{'='*60}")

    # ---- Data split ----
    features_dir = cfg_data["features_dir"]
    split_file = cfg_data.get("split_file", "data/split.json")
    train_stems, valid_stems, test_stems = _load_or_create_split(
        features_dir, split_file, seed=seed
    )
    print(f"[Split] train={len(train_stems)}, valid={len(valid_stems)}, test={len(test_stems)}")

    # ---- Datasets & DataLoaders ----
    max_seq_len = cfg_data["max_seq_len"]

    train_dataset = OfflineFeatureDataset(
        features_dir=features_dir,
        condition_type=condition_type,
        max_seq_len=max_seq_len,
        stems=train_stems,
    )
    valid_dataset = OfflineFeatureDataset(
        features_dir=features_dir,
        condition_type=condition_type,
        max_seq_len=max_seq_len,
        stems=valid_stems,
    )

    print(f"[Dataset] train={len(train_dataset)}, valid={len(valid_dataset)}")

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg_train["batch_size"],
        shuffle=True,
        num_workers=2,
        pin_memory=True,
        collate_fn=collate_fn,
        drop_last=True,
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=cfg_train["batch_size"],
        shuffle=False,
        num_workers=2,
        pin_memory=True,
        collate_fn=collate_fn,
        drop_last=False,
    )

    # ---- Model ----
    num_moss_layers, moss_embed_dim = _detect_moss_dims(
        features_dir, condition_type, cfg_model
    )

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
    warmup_steps = cfg_train.get("warmup_steps", 0)
    num_epochs = cfg_train.get("num_epochs", None)
    num_steps_cfg = cfg_train.get("num_steps", None)
    steps_per_epoch = math.ceil(len(train_dataset) / cfg_train["batch_size"])
    if num_epochs is not None:
        total_steps = num_epochs * steps_per_epoch
    elif num_steps_cfg is not None:
        total_steps = num_steps_cfg
        num_epochs = math.ceil(total_steps / max(steps_per_epoch, 1))
    else:
        raise ValueError("Config must specify either 'num_epochs' or 'num_steps'")
    patience = cfg_train.get("patience", 0)  # 0 = no early stopping

    print(f"[Schedule] {num_epochs} epochs × {steps_per_epoch} steps/epoch = {total_steps} total steps")
    if patience > 0:
        print(f"[Schedule] Early stopping patience = {patience} epochs")

    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(warmup_steps, 1)
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # ---- Resume from checkpoint ----
    start_step = 0
    start_epoch = 0

    if resume_path and Path(resume_path).exists():
        print(f"Resuming from {resume_path} ...")
        ckpt = torch.load(resume_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        if "scheduler_state_dict" in ckpt:
            scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        else:
            # Manually step scheduler to catch up
            for _ in range(ckpt["step"]):
                scheduler.step()
        start_step = ckpt["step"]
        start_epoch = ckpt.get("epoch", 0)
        best_val_loss = ckpt.get("best_val_loss", float("inf"))
        no_improve_count = ckpt.get("no_improve_count", 0)
        print(f"  Resumed at step={start_step}, epoch={start_epoch}")
        print(f"  best_val_loss={best_val_loss:.6f}, no_improve_count={no_improve_count}")

    # ---- Rectified Flow ----
    flow = RectifiedFlow()

    # ---- Logging ----
    ckpt_dir = Path(cfg_train["checkpoint_dir"]) / condition_type
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # TensorBoard
    try:
        from torch.utils.tensorboard import SummaryWriter
        writer = SummaryWriter(log_dir=str(ckpt_dir / "logs"))
    except ImportError:
        writer = None

    # wandb
    wandb_run = None
    if use_wandb:
        try:
            import wandb
            run_name = wandb_run_name or f"{condition_type}"
            wandb_run = wandb.init(
                project=wandb_project,
                name=run_name,
                config={
                    **config,
                    "condition_type": condition_type,
                    "num_params": num_params,
                    "num_moss_layers": num_moss_layers,
                    "moss_embed_dim": moss_embed_dim,
                    "train_samples": len(train_dataset),
                    "valid_samples": len(valid_dataset),
                },
                resume="allow",
            )
            print(f"[wandb] Run: {wandb_run.url}")
        except ImportError:
            print("[WARN] wandb not installed, skipping wandb logging")
            use_wandb = False

    # ---- Training ----
    model.train()
    global_step = start_step
    if not (resume_path and Path(resume_path).exists()):
        best_val_loss = float("inf")
        no_improve_count = 0

    for epoch in range(start_epoch + 1, num_epochs + 1):
        if global_step >= total_steps:
            break

        epoch_loss = 0.0
        epoch_batches = 0

        for batch in train_loader:
            if global_step >= total_steps:
                break

            x0 = batch["x0"].to(device)
            x1 = batch["x1"].to(device)
            cond = batch.get("cond")
            cond_layers_batch = batch.get("cond_layers")

            if cond is not None:
                cond = cond.to(device)

            cond_layers = None
            if cond_layers_batch is not None:
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
            epoch_loss += loss.item()
            epoch_batches += 1

            # Step-level logging
            if global_step % cfg_train["log_every"] == 0:
                lr = optimizer.param_groups[0]["lr"]
                print(
                    f"[Step {global_step:>6d}/{total_steps}]  "
                    f"loss={loss.item():.6f}  lr={lr:.2e}  epoch={epoch}"
                )
                if writer:
                    writer.add_scalar("train/loss", loss.item(), global_step)
                    writer.add_scalar("train/lr", lr, global_step)
                if use_wandb:
                    import wandb
                    wandb.log({
                        "train/loss": loss.item(),
                        "train/lr": lr,
                        "train/epoch": epoch,
                    }, step=global_step)

        # ---- End of epoch: validate & checkpoint ----
        avg_train_loss = epoch_loss / max(epoch_batches, 1)
        val_loss = _validate(model, valid_loader, flow, device, condition_type)
        lr = optimizer.param_groups[0]["lr"]

        print(
            f"\n[Epoch {epoch:>3d}/{num_epochs}]  "
            f"train_loss={avg_train_loss:.6f}  val_loss={val_loss:.6f}  "
            f"lr={lr:.2e}  step={global_step}"
        )

        if writer:
            writer.add_scalar("epoch/train_loss", avg_train_loss, epoch)
            writer.add_scalar("epoch/val_loss", val_loss, epoch)
            writer.add_scalar("epoch/lr", lr, epoch)
        if use_wandb:
            import wandb
            wandb.log({
                "epoch/train_loss": avg_train_loss,
                "epoch/val_loss": val_loss,
                "epoch/lr": lr,
                "epoch": epoch,
            }, step=global_step)

        # Save checkpoint every epoch
        _save_checkpoint(
            model, optimizer, scheduler, global_step, epoch,
            config, condition_type, ckpt_dir, drive_ckpt_dir,
            best_val_loss=best_val_loss, no_improve_count=no_improve_count,
        )

        # Track best
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            no_improve_count = 0
            best_dst = ckpt_dir / "best.pt"
            src_path = ckpt_dir / f"step_{global_step}.pt"
            shutil.copy2(str(src_path), str(best_dst))
            if drive_ckpt_dir:
                drive_best = Path(drive_ckpt_dir) / condition_type / "best.pt"
                shutil.copy2(str(src_path), str(drive_best))
            print(f"  ★ New best val_loss={val_loss:.6f} (epoch {epoch})")
        else:
            no_improve_count += 1
            print(f"  val_loss did not improve ({no_improve_count}/{patience if patience > 0 else '∞'})")

        # Early stopping
        if patience > 0 and no_improve_count >= patience:
            print(f"\n⏹  Early stopping at epoch {epoch} (no improvement for {patience} epochs)")
            break

    if writer:
        writer.close()
    if use_wandb and wandb_run:
        import wandb
        wandb.finish()

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
        choices=["none", "last_layer", "multi_layer", "multi_layer_time"],
        help="Override condition_type from config",
    )
    parser.add_argument(
        "--resume", type=str, default=None,
        help="Path to checkpoint to resume training from",
    )
    parser.add_argument(
        "--wandb", action="store_true",
        help="Enable Weights & Biases logging",
    )
    parser.add_argument(
        "--wandb_project", type=str, default="speech-enhancement",
        help="wandb project name",
    )
    parser.add_argument(
        "--wandb_run_name", type=str, default=None,
        help="wandb run name (default: condition_type)",
    )
    parser.add_argument(
        "--drive_ckpt_dir", type=str, default=None,
        help="Google Drive directory for checkpoint backup "
             "(e.g., /content/drive/MyDrive/speech_enhancement_checkpoints)",
    )
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    train(
        config,
        condition_type_override=args.condition_type,
        resume_path=args.resume,
        use_wandb=args.wandb,
        wandb_project=args.wandb_project,
        wandb_run_name=args.wandb_run_name,
        drive_ckpt_dir=args.drive_ckpt_dir,
    )


if __name__ == "__main__":
    main()
