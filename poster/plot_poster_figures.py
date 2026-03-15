"""
plot_poster_figures.py — Generate 4 publication-quality figures for the poster.

Run on Google Colab after mounting Drive with checkpoints.

Generates:
  1. metrics_comparison.png  — PESQ / STOI / FAD grouped bar charts
  2. fad_over_epochs.png     — FAD vs epoch for all 4 conditions
  3. static_weights.png      — Exp 3 static fusion weights (bar chart)
  4. timedep_weights.png     — Exp 4 time-dependent fusion weight heatmap

Usage (Colab):
    %run poster/plot_poster_figures.py --drive_dir /content/drive/MyDrive/speech_enhancement_output
"""

import argparse
import json
import os

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import torch
import torch.nn.functional as F

# ─── Configuration ──────────────────────────────────────────────────────────

# Colors for the 4 experiments (consistent across figures)
COLORS = {
    "none": "#E74C3C",            # red
    "last_layer": "#3498DB",      # blue
    "multi_layer": "#27AE60",     # green
    "multi_layer_time": "#8E44AD" # purple
}
LABELS = {
    "none": "Exp 1: No Conditioning",
    "last_layer": "Exp 2: Last Layer",
    "multi_layer": "Exp 3: Static Multi-Layer",
    "multi_layer_time": "Exp 4: Time-Dep. Multi-Layer"
}
SHORT_LABELS = {
    "none": "No Cond.",
    "last_layer": "Last Layer",
    "multi_layer": "Static ML",
    "multi_layer_time": "Time-Dep."
}

# Hardcoded results from evaluation
RESULTS = {
    "none":             {"PESQ": 1.6048, "STOI": 0.8527, "FAD": 2.9774},
    "last_layer":       {"PESQ": 1.6499, "STOI": 0.8589, "FAD": 2.6997},
    "multi_layer":      {"PESQ": 1.6868, "STOI": 0.8642, "FAD": 2.3857},
    "multi_layer_time": {"PESQ": 1.6986, "STOI": 0.8647, "FAD": 2.3456},
}

CONDITION_ORDER = ["none", "last_layer", "multi_layer", "multi_layer_time"]


# ─── Figure 1: Metrics Comparison ──────────────────────────────────────────

def plot_metrics_comparison(save_path: str):
    """3 grouped bar charts: PESQ, STOI, FAD with metric descriptions."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.subplots_adjust(wspace=0.35)

    metrics = [
        ("PESQ", "higher is better", "Perceptual speech quality\n(range: −0.5 to 4.5)", True),
        ("STOI", "higher is better", "Short-time intelligibility\n(range: 0 to 1)", True),
        ("FAD", "lower is better", "Fréchet Audio Distance\n(lower = more natural)", False),
    ]

    baseline = RESULTS["none"]

    for ax, (metric, direction, description, higher_better) in zip(axes, metrics):
        values = [RESULTS[c][metric] for c in CONDITION_ORDER]
        colors = [COLORS[c] for c in CONDITION_ORDER]
        labels = [SHORT_LABELS[c] for c in CONDITION_ORDER]

        bars = ax.bar(labels, values, color=colors, edgecolor="black", linewidth=0.8,
                      width=0.65, zorder=3)

        # Highlight best bar
        best_idx = np.argmax(values) if higher_better else np.argmin(values)
        bars[best_idx].set_edgecolor("#FFD700")
        bars[best_idx].set_linewidth(3)

        # Value labels on top
        for i, (bar, val) in enumerate(zip(bars, values)):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                    f"{val:.4f}", ha="center", va="bottom", fontsize=10, fontweight="bold")

        # % improvement annotations (vs baseline)
        for i, c in enumerate(CONDITION_ORDER):
            if c == "none":
                continue
            if higher_better:
                pct = (RESULTS[c][metric] - baseline[metric]) / baseline[metric] * 100
                sign = "+"
            else:
                pct = (baseline[metric] - RESULTS[c][metric]) / baseline[metric] * 100
                sign = "−"
            ax.text(i, min(values) * 0.97 if not higher_better else max(values) * 1.03,
                    f"{sign}{pct:.1f}%", ha="center", va="top" if not higher_better else "bottom",
                    fontsize=9, color=COLORS[c], fontstyle="italic")

        ax.set_title(f"{metric} ({direction})", fontsize=14, fontweight="bold")
        ax.set_xlabel(description, fontsize=10, color="gray")
        ax.grid(axis="y", alpha=0.3, zorder=0)
        ax.tick_params(axis="x", labelsize=9)

        # Tighten y-axis
        if higher_better:
            ax.set_ylim(min(values) * 0.97, max(values) * 1.06)
        else:
            ax.set_ylim(min(values) * 0.90, max(values) * 1.05)

    fig.suptitle("Evaluation Metrics — All Conditions", fontsize=16, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved: {save_path}")


# ─── Figure 2: FAD Over Epochs ─────────────────────────────────────────────

def plot_fad_over_epochs(fad_curves_path: str, save_path: str):
    """Line plot of FAD vs epoch for all 4 conditions."""
    with open(fad_curves_path) as f:
        fad_curves = json.load(f)

    fig, ax = plt.subplots(figsize=(10, 6))

    for ct in CONDITION_ORDER:
        if ct not in fad_curves:
            continue
        epochs = fad_curves[ct]["epoch"]
        fads = fad_curves[ct]["fad"]

        ax.plot(epochs, fads, "o-", color=COLORS[ct], label=LABELS[ct],
                linewidth=2, markersize=5, zorder=3)

        # Mark best FAD point
        best_idx = int(np.argmin(fads))
        ax.plot(epochs[best_idx], fads[best_idx], "*", color=COLORS[ct],
                markersize=18, zorder=4, markeredgecolor="black", markeredgewidth=0.8)
        ax.annotate(f"{fads[best_idx]:.2f}\n(ep {epochs[best_idx]})",
                    xy=(epochs[best_idx], fads[best_idx]),
                    xytext=(10, 10), textcoords="offset points",
                    fontsize=9, color=COLORS[ct], fontweight="bold",
                    arrowprops=dict(arrowstyle="-", color=COLORS[ct], lw=0.8))

    ax.set_xlabel("Epoch", fontsize=13)
    ax.set_ylabel("FAD (↓ lower is better)", fontsize=13)
    ax.set_title("Fréchet Audio Distance Over Training Epochs", fontsize=15, fontweight="bold")
    ax.legend(fontsize=11, loc="upper right")
    ax.grid(alpha=0.3, zorder=0)
    ax.tick_params(labelsize=11)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved: {save_path}")


# ─── Figure 3: Static Weights (Exp 3) ──────────────────────────────────────

def plot_static_weights(checkpoint_path: str, save_path: str):
    """Bar chart of Exp 3 static fusion weights showing near-uniform distribution."""
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    state = ckpt.get("model_state_dict", ckpt)

    # Find the layer_weights key
    raw_weights = None
    for key in state:
        if "layer_weights" in key or "fusion_weights" in key:
            raw_weights = state[key]
            break

    if raw_weights is None:
        print("WARNING: Could not find static fusion weights in checkpoint")
        return

    softmax_weights = F.softmax(raw_weights.float(), dim=0).numpy()
    num_layers = len(softmax_weights)
    uniform = 1.0 / num_layers

    fig, ax = plt.subplots(figsize=(14, 5))

    # Color by above/below uniform
    colors = ["#27AE60" if w >= uniform else "#E74C3C" for w in softmax_weights]
    bars = ax.bar(range(num_layers), softmax_weights, color=colors, edgecolor="black",
                  linewidth=0.5, width=0.8, zorder=3)

    # Uniform baseline
    ax.axhline(y=uniform, color="black", linestyle="--", linewidth=1.5, label=f"Uniform = 1/{num_layers} = {uniform:.5f}", zorder=4)

    # Statistics
    gini = (np.sum(np.abs(np.subtract.outer(softmax_weights, softmax_weights))) /
            (2 * num_layers * np.sum(softmax_weights)))
    entropy = -np.sum(softmax_weights * np.log(softmax_weights + 1e-12))
    max_entropy = np.log(num_layers)

    ax.set_xlabel("MOSS Stage-3 Transformer Layer", fontsize=13)
    ax.set_ylabel("Fusion Weight (softmax)", fontsize=13)
    ax.set_title(
        f"Exp 3: Static Fusion Weights — Nearly Uniform Distribution\n"
        f"Gini = {gini:.4f}  |  Entropy ratio = {entropy/max_entropy:.4f}  |  "
        f"All weights within {(softmax_weights.max()/uniform - 1)*100:.1f}% of uniform",
        fontsize=13, fontweight="bold"
    )
    ax.set_xticks(range(num_layers))
    ax.set_xticklabels(range(num_layers), fontsize=9)
    ax.legend(fontsize=11)
    ax.grid(axis="y", alpha=0.3, zorder=0)
    ax.tick_params(labelsize=10)

    # Tighten y-axis around data
    ymin = softmax_weights.min() * 0.95
    ymax = softmax_weights.max() * 1.05
    ax.set_ylim(ymin, ymax)

    # Add annotation
    above_avg = [i for i in range(num_layers) if softmax_weights[i] >= uniform]
    below_avg = [i for i in range(num_layers) if softmax_weights[i] < uniform]
    ax.text(0.98, 0.02,
            f"Above uniform (green): layers {above_avg[0]}–{above_avg[-1] if len(above_avg) > 1 else above_avg[0]}"
            f"  |  Below uniform (red): layers {below_avg[0]}–{below_avg[-1] if len(below_avg) > 1 else below_avg[0]}",
            transform=ax.transAxes, fontsize=9, ha="right", va="bottom",
            color="gray", style="italic")

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved: {save_path}")


# ─── Figure 4: Time-Dep Weights Heatmap (Exp 4) ────────────────────────────

def plot_timedep_weights(checkpoint_path: str, save_path: str, num_timesteps: int = 51):
    """Heatmap of Exp 4 time-dependent fusion weights (layers × timesteps)."""
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    state = ckpt.get("model_state_dict", ckpt)

    # Extract MLP weights
    mlp_keys = sorted([k for k in state if "weight_mlp" in k])
    if not mlp_keys:
        print("WARNING: Could not find weight_mlp in checkpoint")
        return

    # Reconstruct MLP: Linear(1, 64) -> SiLU -> Linear(64, 32)
    w0 = state[[k for k in mlp_keys if "0.weight" in k][0]].float()
    b0 = state[[k for k in mlp_keys if "0.bias" in k][0]].float()
    w2 = state[[k for k in mlp_keys if "2.weight" in k][0]].float()
    b2 = state[[k for k in mlp_keys if "2.bias" in k][0]].float()

    num_layers = w2.shape[0]

    # Compute weights at each timestep
    timesteps = np.linspace(0, 1, num_timesteps)
    weight_matrix = np.zeros((num_timesteps, num_layers))

    for i, t in enumerate(timesteps):
        t_tensor = torch.tensor([t], dtype=torch.float32)
        h = F.silu(F.linear(t_tensor.unsqueeze(-1), w0, b0))
        logits = F.linear(h, w2, b2)
        weights = F.softmax(logits, dim=-1)
        weight_matrix[i] = weights.detach().numpy().flatten()

    uniform = 1.0 / num_layers

    # Create figure with heatmap + side panel + bottom panel
    fig = plt.figure(figsize=(16, 8))
    gs = fig.add_gridspec(2, 2, width_ratios=[4, 1], height_ratios=[3, 1],
                           hspace=0.3, wspace=0.15)

    # Main heatmap
    ax_heat = fig.add_subplot(gs[0, 0])
    im = ax_heat.imshow(weight_matrix, aspect="auto", cmap="viridis",
                         extent=[-0.5, num_layers - 0.5, 0, 1],
                         origin="lower", interpolation="nearest")
    ax_heat.set_xlabel("MOSS Stage-3 Transformer Layer", fontsize=12)
    ax_heat.set_ylabel("ODE Timestep t", fontsize=12)
    ax_heat.set_title("Exp 4: Time-Dependent Fusion Weights", fontsize=14, fontweight="bold")
    ax_heat.set_xticks(range(0, num_layers, 2))
    ax_heat.set_xticklabels(range(0, num_layers, 2), fontsize=8)
    cbar = fig.colorbar(im, ax=ax_heat, fraction=0.02, pad=0.02)
    cbar.set_label("Weight", fontsize=10)

    # Side panel: weight std vs timestep
    ax_side = fig.add_subplot(gs[0, 1])
    stds = weight_matrix.std(axis=1)
    ax_side.plot(stds, timesteps, color="#8E44AD", linewidth=2)
    ax_side.set_xlabel("Std across layers", fontsize=10)
    ax_side.set_ylabel("Timestep t", fontsize=10)
    ax_side.set_title("Selectivity", fontsize=11, fontweight="bold")
    ax_side.grid(alpha=0.3)
    ax_side.tick_params(labelsize=9)
    # Annotate endpoints
    ax_side.annotate(f"{stds[0]:.4f}", xy=(stds[0], 0), fontsize=9,
                     xytext=(5, -10), textcoords="offset points", color="#8E44AD")
    ax_side.annotate(f"{stds[-1]:.4f}", xy=(stds[-1], 1), fontsize=9,
                     xytext=(5, 5), textcoords="offset points", color="#8E44AD")

    # Bottom panel: mean weight per layer
    ax_bot = fig.add_subplot(gs[1, 0])
    mean_weights = weight_matrix.mean(axis=0)
    colors_bot = ["#27AE60" if w >= uniform else "#E74C3C" for w in mean_weights]
    ax_bot.bar(range(num_layers), mean_weights, color=colors_bot, edgecolor="black",
               linewidth=0.3, width=0.8)
    ax_bot.axhline(y=uniform, color="black", linestyle="--", linewidth=1.2,
                   label=f"Uniform = {uniform:.5f}")
    ax_bot.set_xlabel("Layer", fontsize=10)
    ax_bot.set_ylabel("Mean weight", fontsize=10)
    ax_bot.set_title("Mean Weight Across All Timesteps", fontsize=11, fontweight="bold")
    ax_bot.set_xticks(range(0, num_layers, 2))
    ax_bot.set_xticklabels(range(0, num_layers, 2), fontsize=8)
    ax_bot.legend(fontsize=9)
    ax_bot.grid(axis="y", alpha=0.3)
    ymin = mean_weights.min() * 0.95
    ymax = mean_weights.max() * 1.05
    ax_bot.set_ylim(ymin, ymax)

    # Annotation in empty bottom-right cell
    ax_text = fig.add_subplot(gs[1, 1])
    ax_text.axis("off")
    top3 = np.argsort(weight_matrix[-1])[::-1][:3]  # top 3 at t=1
    ax_text.text(0.1, 0.7,
                 f"Top-3 layers (all t):\n  [{top3[0]}, {top3[1]}, {top3[2]}]\n\n"
                 f"Std increase:\n  {stds[0]:.4f} → {stds[-1]:.4f}\n  ({stds[-1]/stds[0]:.1f}× more selective)\n\n"
                 f"Early layers (0–15)\n  gain: {weight_matrix[0,:16].sum():.3f} → {weight_matrix[-1,:16].sum():.3f}",
                 transform=ax_text.transAxes, fontsize=10, va="top",
                 fontfamily="monospace",
                 bbox=dict(boxstyle="round,pad=0.5", facecolor="#F0E6F6", alpha=0.8))

    plt.savefig(save_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved: {save_path}")


# ─── Main ──────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Generate poster figures")
    parser.add_argument("--drive_dir", type=str,
                        default="/content/drive/MyDrive/speech_enhancement_output",
                        help="Path to output directory on Google Drive")
    parser.add_argument("--out_dir", type=str, default=None,
                        help="Output directory for figures (default: same as drive_dir)")
    args = parser.parse_args()

    drive_dir = args.drive_dir
    out_dir = args.out_dir or os.path.join(drive_dir, "poster_figures")
    os.makedirs(out_dir, exist_ok=True)

    # Figure 1: Metrics comparison (no checkpoint needed)
    print("\n=== Figure 1: Metrics Comparison ===")
    plot_metrics_comparison(os.path.join(out_dir, "metrics_comparison.png"))

    # Figure 2: FAD over epochs
    print("\n=== Figure 2: FAD Over Epochs ===")
    fad_path = os.path.join(drive_dir, "fad_curves.json")
    if os.path.exists(fad_path):
        plot_fad_over_epochs(fad_path, os.path.join(out_dir, "fad_over_epochs.png"))
    else:
        print(f"WARNING: {fad_path} not found, skipping")

    # Figure 3: Static weights (Exp 3)
    print("\n=== Figure 3: Static Weights (Exp 3) ===")
    ckpt3 = os.path.join(drive_dir, "checkpoints", "multi_layer", "best_model.pt")
    if os.path.exists(ckpt3):
        plot_static_weights(ckpt3, os.path.join(out_dir, "static_weights.png"))
    else:
        print(f"WARNING: {ckpt3} not found, skipping")

    # Figure 4: Time-dep weights heatmap (Exp 4)
    print("\n=== Figure 4: Time-Dep Weights Heatmap (Exp 4) ===")
    ckpt4 = os.path.join(drive_dir, "checkpoints", "multi_layer_time", "best_model.pt")
    if os.path.exists(ckpt4):
        plot_timedep_weights(ckpt4, os.path.join(out_dir, "timedep_weights.png"))
    else:
        print(f"WARNING: {ckpt4} not found, skipping")

    print(f"\n✅ All figures saved to: {out_dir}")


if __name__ == "__main__":
    main()
