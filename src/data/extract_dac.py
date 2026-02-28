"""
extract_dac.py — Extract continuous latent representations from DAC (50 Hz).

Saves a .pt file per utterance containing the continuous latent tensor
(before quantisation) with shape [T_dac, D_dac].

Usage:
    python -m src.data.extract_dac \
        --audio_dir  data/mixed/snr_5dB \
        --out_dir    data/features/noisy_dac \
        --sr         16000
"""

import argparse
from pathlib import Path

import torch
import torchaudio
from tqdm import tqdm

# DAC imports
import dac
from dac.utils import load_model


@torch.no_grad()
def extract_dac_latents(audio_dir: str, out_dir: str, sr: int = 16000, device: str = "cuda"):
    """
    For every .wav in `audio_dir`, encode with DAC and save the continuous
    latent (pre-quantisation) tensor to `out_dir/<stem>.pt`.
    """
    audio_dir = Path(audio_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load the pre-trained DAC model (16 kHz variant)
    model_path = dac.utils.download(model_type="16khz")
    model = load_model(model_path)
    model = model.to(device).eval()

    audio_files = sorted(
        [f for f in audio_dir.iterdir() if f.suffix in (".wav", ".flac", ".mp3")]
    )
    print(f"Extracting DAC latents for {len(audio_files)} files from {audio_dir}")

    for fpath in tqdm(audio_files, desc="DAC extraction"):
        waveform, orig_sr = torchaudio.load(str(fpath))
        if orig_sr != sr:
            waveform = torchaudio.functional.resample(waveform, orig_sr, sr)
        # DAC expects (B, 1, T) input
        waveform = waveform.unsqueeze(0).to(device)  # (1, C, T)
        if waveform.shape[1] > 1:
            waveform = waveform.mean(dim=1, keepdim=True)

        # Encode — get continuous latent *before* quantisation
        # model.encode returns (z, codes, latents, commitment_loss, codebook_loss)
        z, codes, latents, _, _ = model.encode(waveform)
        # z shape: (1, D, T_dac)  — continuous latent
        z = z.squeeze(0).permute(1, 0).cpu()  # -> (T_dac, D)

        out_path = out_dir / f"{fpath.stem}.pt"
        torch.save(z, str(out_path))

    print(f"Saved {len(audio_files)} .pt files to {out_dir}")


def main():
    parser = argparse.ArgumentParser(description="Extract DAC continuous latents")
    parser.add_argument("--audio_dir", type=str, required=True, help="Input audio directory")
    parser.add_argument("--out_dir", type=str, required=True, help="Output directory for .pt files")
    parser.add_argument("--sr", type=int, default=16000, help="Sample rate")
    parser.add_argument("--device", type=str, default="cuda", help="Device (cuda/cpu)")
    args = parser.parse_args()

    extract_dac_latents(args.audio_dir, args.out_dir, args.sr, args.device)


if __name__ == "__main__":
    main()
