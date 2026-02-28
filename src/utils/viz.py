"""
viz.py — Visualisation utilities for speech enhancement.

Provides:
  * ``plot_mel_comparison`` – side-by-side Mel-spectrogram plots of
    noisy / enhanced / clean audio.
"""

import argparse
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import torch
import torchaudio
import librosa
import librosa.display


def compute_mel_spectrogram(
    waveform: torch.Tensor,
    sr: int = 16000,
    n_fft: int = 1024,
    hop_length: int = 256,
    n_mels: int = 80,
) -> np.ndarray:
    """
    Compute log-Mel spectrogram from a 1-D waveform tensor.

    Returns
    -------
    mel_db : np.ndarray of shape (n_mels, T_frames)
    """
    wav_np = waveform.numpy()
    mel = librosa.feature.melspectrogram(
        y=wav_np, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels
    )
    mel_db = librosa.power_to_db(mel, ref=np.max)
    return mel_db


def plot_mel_comparison(
    noisy_path: str,
    enhanced_path: str,
    clean_path: Optional[str] = None,
    sr: int = 16000,
    save_path: Optional[str] = None,
    title: str = "Mel-Spectrogram Comparison",
):
    """
    Plot side-by-side Mel-spectrograms.

    Parameters
    ----------
    noisy_path    : path to noisy audio
    enhanced_path : path to enhanced audio
    clean_path    : path to clean reference (optional)
    sr            : sample rate
    save_path     : if given, save figure to this path
    title         : overall figure title
    """
    num_plots = 3 if clean_path else 2
    fig, axes = plt.subplots(1, num_plots, figsize=(6 * num_plots, 4))

    labels_paths = [("Noisy", noisy_path), ("Enhanced", enhanced_path)]
    if clean_path:
        labels_paths.append(("Clean", clean_path))

    for ax, (label, path) in zip(axes, labels_paths):
        wav, orig_sr = torchaudio.load(path)
        if orig_sr != sr:
            wav = torchaudio.functional.resample(wav, orig_sr, sr)
        if wav.shape[0] > 1:
            wav = wav.mean(dim=0)
        else:
            wav = wav.squeeze(0)

        mel_db = compute_mel_spectrogram(wav, sr=sr)
        img = librosa.display.specshow(
            mel_db, sr=sr, hop_length=256, x_axis="time", y_axis="mel", ax=ax
        )
        ax.set_title(label)
        fig.colorbar(img, ax=ax, format="%+2.0f dB")

    fig.suptitle(title, fontsize=14)
    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved figure to {save_path}")
    else:
        plt.show()

    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Plot Mel-spectrogram comparisons")
    parser.add_argument("--noisy", type=str, required=True, help="Noisy audio path")
    parser.add_argument("--enhanced", type=str, required=True, help="Enhanced audio path")
    parser.add_argument("--clean", type=str, default=None, help="Clean reference audio path")
    parser.add_argument("--sr", type=int, default=16000)
    parser.add_argument("--save", type=str, default=None, help="Save figure path")
    args = parser.parse_args()

    plot_mel_comparison(
        noisy_path=args.noisy,
        enhanced_path=args.enhanced,
        clean_path=args.clean,
        sr=args.sr,
        save_path=args.save,
    )


if __name__ == "__main__":
    main()
