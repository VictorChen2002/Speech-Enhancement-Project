"""
mixer.py — Dynamically mix clean speech + noise at target SNR levels.

Usage:
    python -m src.data.mixer \
        --clean_dir data/raw/clean \
        --noise_dir data/raw/noise \
        --out_dir   data/mixed \
        --snr_list  0 5 10 \
        --sr        16000
"""

import argparse
import os
import random
from pathlib import Path

import numpy as np
import soundfile as sf
import torchaudio


def load_audio(path: str, sr: int = 16000) -> np.ndarray:
    """Load an audio file and resample to `sr` if needed. Returns mono numpy array."""
    waveform, orig_sr = torchaudio.load(path)
    if orig_sr != sr:
        waveform = torchaudio.functional.resample(waveform, orig_sr, sr)
    # Convert to mono
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    return waveform.squeeze(0).numpy()


def mix_at_snr(clean: np.ndarray, noise: np.ndarray, snr_db: float) -> np.ndarray:
    """
    Mix clean speech with noise at the given SNR (dB).

    If noise is shorter than clean, it is repeated (tiled). 
    If noise is longer, it is randomly cropped.
    """
    clean_len = len(clean)

    # Adjust noise length to match clean
    if len(noise) < clean_len:
        repeats = int(np.ceil(clean_len / len(noise)))
        noise = np.tile(noise, repeats)
    if len(noise) > clean_len:
        start = random.randint(0, len(noise) - clean_len)
        noise = noise[start : start + clean_len]

    # Compute power and scale noise
    clean_power = np.mean(clean ** 2) + 1e-10
    noise_power = np.mean(noise ** 2) + 1e-10
    snr_linear = 10 ** (snr_db / 10.0)
    scale = np.sqrt(clean_power / (noise_power * snr_linear))

    mixed = clean + scale * noise
    # Prevent clipping
    peak = np.max(np.abs(mixed))
    if peak > 1.0:
        mixed /= peak
    return mixed


def main():
    parser = argparse.ArgumentParser(description="Mix clean speech with noise at target SNRs")
    parser.add_argument("--clean_dir", type=str, required=True, help="Directory of clean speech WAVs")
    parser.add_argument("--noise_dir", type=str, required=True, help="Directory of noise WAVs")
    parser.add_argument("--out_dir", type=str, default="data/mixed", help="Output directory for mixed WAVs")
    parser.add_argument("--snr_list", type=float, nargs="+", default=[0, 5, 10], help="List of SNR levels (dB)")
    parser.add_argument("--sr", type=int, default=16000, help="Sample rate")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    clean_dir = Path(args.clean_dir)
    noise_dir = Path(args.noise_dir)
    out_dir = Path(args.out_dir)

    clean_files = sorted([f for f in clean_dir.iterdir() if f.suffix in (".wav", ".flac", ".mp3")])
    noise_files = sorted([f for f in noise_dir.iterdir() if f.suffix in (".wav", ".flac", ".mp3")])

    if not clean_files:
        raise FileNotFoundError(f"No audio files found in {clean_dir}")
    if not noise_files:
        raise FileNotFoundError(f"No audio files found in {noise_dir}")

    print(f"Found {len(clean_files)} clean files, {len(noise_files)} noise files")
    print(f"SNR levels: {args.snr_list} dB")

    for snr_db in args.snr_list:
        snr_dir = out_dir / f"snr_{int(snr_db)}dB"
        snr_dir.mkdir(parents=True, exist_ok=True)

        for clean_path in clean_files:
            noise_path = random.choice(noise_files)
            clean_audio = load_audio(str(clean_path), sr=args.sr)
            noise_audio = load_audio(str(noise_path), sr=args.sr)

            mixed = mix_at_snr(clean_audio, noise_audio, snr_db)

            out_path = snr_dir / clean_path.name
            sf.write(str(out_path), mixed, args.sr)

        print(f"  [SNR {snr_db:+.0f} dB] Written {len(clean_files)} files -> {snr_dir}")

    print("Done mixing.")


if __name__ == "__main__":
    main()
