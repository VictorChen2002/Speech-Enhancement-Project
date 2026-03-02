"""
extract_moss.py — Extract continuous embeddings from MOSS-Audio-Tokenizer (12.5 Hz).

Saves per-utterance .pt files.  When ``--save_all_layers`` is set the file
contains a list of tensors (one per transformer layer) so that the
multi-layer conditioning ablation can be run later without re-extracting.

Architecture notes (from MOSS-Audio-Tokenizer config):
    Encoder = 4 stages of [PatchedPretransform → Transformer]:
        Stage 0: PatchedPretransform(240) → Transformer(12L, d=768)  → output 384-dim
        Stage 1: PatchedPretransform(2)   → Transformer(12L, d=768)  → output 384-dim
        Stage 2: PatchedPretransform(2)   → Transformer(12L, d=768)  → output 640-dim
        Stage 3: PatchedPretransform(2)   → Transformer(32L, d=1280) → output 768-dim
    Total downsample: 240×2×2×2 = 1920.  At 24 kHz → 12.5 Hz.
    Final encoder output (encoder_hidden_states): (B, 768, T) at 12.5 Hz — pre-VQ.
    Multi-layer hooks target the 32-layer Stage-3 transformer (d_model=1280).

Usage (last layer only):
    python -m src.data.extract_moss \\
        --audio_dir  data/mixed/snr_5dB \\
        --out_dir    data/features/moss_last

Usage (all hidden layers for multi-layer ablation):
    python -m src.data.extract_moss \\
        --audio_dir  data/mixed/snr_5dB \\
        --out_dir    data/features/moss_multi \\
        --save_all_layers
"""

import argparse
from pathlib import Path

import torch
import torchaudio
from tqdm import tqdm
from transformers import AutoModel


MOSS_MODEL_NAME = "OpenMOSS-Team/MOSS-Audio-Tokenizer"
MOSS_SAMPLE_RATE = 24_000  # MOSS expects 24 kHz input


def auto_device() -> str:
    """Pick the best available device: cuda > mps > cpu."""
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _find_last_projected_transformer(encoder):
    """
    Return (index, module) for the last ProjectedTransformer in the
    encoder ModuleList (the one containing the deepest transformer).
    """
    last_idx, last_mod = None, None
    for i, module in enumerate(encoder):
        if hasattr(module, "transformer"):  # ProjectedTransformer has .transformer
            last_idx, last_mod = i, module
    if last_mod is None:
        raise RuntimeError("Could not locate a ProjectedTransformer in model.encoder")
    return last_idx, last_mod


@torch.no_grad()
def extract_moss_embeddings(
    audio_dir: str,
    out_dir: str,
    sr: int = 16000,
    device: str = "auto",
    save_all_layers: bool = False,
):
    """
    Extract MOSS-Audio-Tokenizer embeddings.

    Parameters
    ----------
    audio_dir : str
        Directory with audio files (wav/flac/mp3).
    out_dir : str
        Where to save .pt tensors.
    sr : int
        Sample rate of the *input* files (they will be resampled to 24 kHz
        internally because MOSS expects 24 kHz).
    device : str
        "auto" (detect), "cuda", "mps", or "cpu".
    save_all_layers : bool
        If True, save all hidden-layer outputs from the last encoder
        transformer stage as a list of tensors, each (T_moss, 1280).
        If False, save only the final encoder output (T_moss, 768).
    """
    if device == "auto":
        device = auto_device()

    audio_dir = Path(audio_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Load model (no AutoProcessor — MOSS takes raw waveforms directly)
    # ------------------------------------------------------------------
    print(f"Loading MOSS model: {MOSS_MODEL_NAME}  (device={device})")
    model = AutoModel.from_pretrained(
        MOSS_MODEL_NAME,
        trust_remote_code=True,
    ).to(device).eval()

    # ------------------------------------------------------------------
    # (multi-layer) Locate the last encoder transformer and attach hooks
    # ------------------------------------------------------------------
    layer_outputs: list[torch.Tensor] = []
    hooks = []

    if save_all_layers:
        _, last_proj_transformer = _find_last_projected_transformer(model.encoder)
        transformer_layers = last_proj_transformer.transformer.layers
        num_layers = len(transformer_layers)
        print(f"  Hooking {num_layers} transformer layers (d_model="
              f"{transformer_layers[0].self_attn.embed_dim})")

        def _make_hook():
            def hook_fn(_module, _input, output):
                # output: (B, T, d_model)
                layer_outputs.append(output.detach().cpu())
            return hook_fn

        for layer in transformer_layers:
            hooks.append(layer.register_forward_hook(_make_hook()))

    # ------------------------------------------------------------------
    # Discover audio files
    # ------------------------------------------------------------------
    audio_exts = (".wav", ".flac", ".mp3")
    audio_files = sorted(
        [f for ext in audio_exts for f in audio_dir.rglob(f"*{ext}")]
    )
    if not audio_files:
        raise FileNotFoundError(f"No audio files found in {audio_dir}")

    print(f"Extracting MOSS embeddings for {len(audio_files)} files "
          f"(all_layers={save_all_layers})")

    # ------------------------------------------------------------------
    # Main extraction loop
    # ------------------------------------------------------------------
    skipped = 0
    for fpath in tqdm(audio_files, desc="MOSS extraction"):
        # Collision-safe filename from relative path
        safe_name = "_".join(fpath.relative_to(audio_dir).with_suffix(".pt").parts)
        out_path = out_dir / safe_name

        # Skip if already extracted (resume support)
        if out_path.exists():
            skipped += 1
            continue

        waveform, orig_sr = torchaudio.load(str(fpath))

        # Resample to 24 kHz (MOSS native rate)
        if orig_sr != MOSS_SAMPLE_RATE:
            waveform = torchaudio.functional.resample(waveform, orig_sr, MOSS_SAMPLE_RATE)

        # Mono
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        # MOSS encode expects (B, channels, T) or (B, T)
        wav_input = waveform.unsqueeze(0).to(device)  # (1, 1, T_24k)

        if save_all_layers:
            layer_outputs.clear()
            # Encode triggers hooks on each transformer layer
            model.encode(wav_input, return_dict=True)
            # layer_outputs: list of 32 tensors, each (1, T_moss, 1280)
            hidden_states = [h.squeeze(0) for h in layer_outputs]  # list[(T_moss, 1280)]
            torch.save(hidden_states, str(out_path))
        else:
            enc = model.encode(wav_input, return_dict=True)
            # encoder_hidden_states: (1, 768, T_moss)
            last_hidden = enc.encoder_hidden_states.squeeze(0).permute(1, 0).cpu()  # (T_moss, 768)
            torch.save(last_hidden, str(out_path))

    # Clean up hooks
    for h in hooks:
        h.remove()

    print(f"Saved {len(audio_files) - skipped} new .pt files to {out_dir} (skipped {skipped} existing)")


def main():
    parser = argparse.ArgumentParser(description="Extract MOSS-Audio-Tokenizer embeddings")
    parser.add_argument("--audio_dir", type=str, required=True, help="Input audio directory")
    parser.add_argument("--out_dir", type=str, required=True, help="Output directory for .pt files")
    parser.add_argument("--sr", type=int, default=16000,
                        help="Sample rate of input files (resampled to 24 kHz for MOSS)")
    parser.add_argument("--device", type=str, default="auto",
                        help="Device (auto/cuda/mps/cpu)")
    parser.add_argument(
        "--save_all_layers",
        action="store_true",
        help="Save all transformer-layer outputs from the last encoder stage "
             "(for multi-layer ablation). Each file contains list[(T, 1280)].",
    )
    args = parser.parse_args()

    extract_moss_embeddings(
        args.audio_dir, args.out_dir, args.sr, args.device, args.save_all_layers
    )


if __name__ == "__main__":
    main()
