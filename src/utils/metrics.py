"""
metrics.py — Evaluation metrics for speech enhancement.

Provides:
  * FAD (Fréchet Audio Distance) between two directories of audio files.
"""

import argparse
from pathlib import Path

import torch
import torchaudio
from frechet_audio_distance import FrechetAudioDistance


def compute_fad(
    generated_dir: str,
    reference_dir: str,
    sr: int = 16000,
    model_name: str = "vggish",
) -> float:
    """
    Compute Fréchet Audio Distance (FAD) between generated and reference
    audio directories.

    Parameters
    ----------
    generated_dir : str
        Directory with generated / enhanced audio .wav files.
    reference_dir : str
        Directory with ground-truth clean audio .wav files.
    sr : int
        Sample rate (files will be resampled if needed).
    model_name : str
        Embedding model to use ("vggish" or "encodec").

    Returns
    -------
    fad_score : float
    """
    fad = FrechetAudioDistance(
        model_name=model_name,
        sample_rate=sr,
        use_pca=False,
        use_activation=False,
        verbose=True,
    )
    score = fad.score(generated_dir, reference_dir)
    return score


def main():
    parser = argparse.ArgumentParser(description="Compute FAD between two audio directories")
    parser.add_argument("--gen_dir", type=str, required=True, help="Generated audio directory")
    parser.add_argument("--ref_dir", type=str, required=True, help="Reference (clean) audio directory")
    parser.add_argument("--sr", type=int, default=16000, help="Sample rate")
    args = parser.parse_args()

    score = compute_fad(args.gen_dir, args.ref_dir, args.sr)
    print(f"FAD score: {score:.4f}")


if __name__ == "__main__":
    main()
