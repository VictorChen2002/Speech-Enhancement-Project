"""
extract_moss.py — Extract continuous embeddings from MOSS-Audio-Tokenizer (12.5 Hz).

Saves per-utterance .pt files.  When ``--save_all_layers`` is set the file
contains a list of tensors (one per transformer layer) so that the
multi-layer conditioning ablation can be run later without re-extracting.

Usage (last layer only):
    python -m src.data.extract_moss \
        --audio_dir  data/mixed/snr_5dB \
        --out_dir    data/features/moss_last \
        --sr         16000

Usage (all hidden layers for multi-layer ablation):
    python -m src.data.extract_moss \
        --audio_dir  data/mixed/snr_5dB \
        --out_dir    data/features/moss_multi \
        --sr         16000 \
        --save_all_layers
"""

import argparse
from pathlib import Path

import torch
import torchaudio
from tqdm import tqdm
from transformers import AutoModel, AutoProcessor


MOSS_MODEL_NAME = "fnlp/MOSS-audio-tokenizer"


@torch.no_grad()
def extract_moss_embeddings(
    audio_dir: str,
    out_dir: str,
    sr: int = 16000,
    device: str = "cuda",
    save_all_layers: bool = False,
):
    """
    Extract MOSS-Audio-Tokenizer embeddings.

    Parameters
    ----------
    audio_dir : str
        Directory with .wav files.
    out_dir : str
        Where to save .pt tensors.
    sr : int
        Target sample rate (MOSS expects 16 kHz).
    device : str
        "cuda" or "cpu".
    save_all_layers : bool
        If True, save all hidden-layer outputs as a list of tensors
        (needed for ``condition_type="multi_layer"``).
        If False, save only the last hidden state (for ``condition_type="last_layer"``).
    """
    audio_dir = Path(audio_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load MOSS model and processor
    print(f"Loading MOSS model: {MOSS_MODEL_NAME}")
    processor = AutoProcessor.from_pretrained(MOSS_MODEL_NAME, trust_remote_code=True)
    model = AutoModel.from_pretrained(
        MOSS_MODEL_NAME,
        trust_remote_code=True,
        output_hidden_states=save_all_layers,
    ).to(device).eval()

    audio_files = sorted(
        [f for f in audio_dir.iterdir() if f.suffix in (".wav", ".flac", ".mp3")]
    )
    print(f"Extracting MOSS embeddings for {len(audio_files)} files (all_layers={save_all_layers})")

    for fpath in tqdm(audio_files, desc="MOSS extraction"):
        waveform, orig_sr = torchaudio.load(str(fpath))
        if orig_sr != sr:
            waveform = torchaudio.functional.resample(waveform, orig_sr, sr)
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        # Prepare inputs through processor
        inputs = processor(
            waveform.squeeze(0).numpy(),
            sampling_rate=sr,
            return_tensors="pt",
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}

        outputs = model(**inputs)

        if save_all_layers:
            # Save all hidden states as list[Tensor] each of shape (T_moss, D)
            hidden_states = [h.squeeze(0).cpu() for h in outputs.hidden_states]
            torch.save(hidden_states, str(out_dir / f"{fpath.stem}.pt"))
        else:
            # Save only last hidden state: (T_moss, D)
            last_hidden = outputs.last_hidden_state.squeeze(0).cpu()
            torch.save(last_hidden, str(out_dir / f"{fpath.stem}.pt"))

    print(f"Saved {len(audio_files)} .pt files to {out_dir}")


def main():
    parser = argparse.ArgumentParser(description="Extract MOSS-Audio-Tokenizer embeddings")
    parser.add_argument("--audio_dir", type=str, required=True, help="Input audio directory")
    parser.add_argument("--out_dir", type=str, required=True, help="Output directory for .pt files")
    parser.add_argument("--sr", type=int, default=16000, help="Sample rate")
    parser.add_argument("--device", type=str, default="cuda", help="Device (cuda/cpu)")
    parser.add_argument(
        "--save_all_layers",
        action="store_true",
        help="Save all hidden-layer outputs (for multi-layer ablation)",
    )
    args = parser.parse_args()

    extract_moss_embeddings(
        args.audio_dir, args.out_dir, args.sr, args.device, args.save_all_layers
    )


if __name__ == "__main__":
    main()
