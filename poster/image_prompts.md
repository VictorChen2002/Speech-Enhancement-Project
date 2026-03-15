# Image Generation Prompts for Poster Diagrams

We use **2 AI-generated diagrams** (Prompts 1 & 2) and **4 Python-generated plots** (see `plot_poster_figures.py`).

---

## Prompt 1: Overall Pipeline (AI-generated)

```
Create a clean technical diagram for a speech enhancement pipeline. Show TWO paths side by side:

=== TRAINING (left half) ===
Two inputs at the top:
- "Noisy Speech" waveform → a single box labeled "MOSS Tokenizer" → arrow labeled "condition c"
- "Clean Speech" waveform → a single box labeled "DAC Encoder" → arrow labeled "x₁ (clean latent)"

In the center: a straight arrow from "x₀ ~ N(0,I)" (noise) to "x₁" (clean latent), labeled "Rectified Flow trajectory". An intermediate point "xₜ" sits on this line.

A large box labeled "DiT (Diffusion Transformer)" receives three inputs: xₜ, timestep t, and condition c. It outputs "predicted velocity v_θ".

Below: "Loss = MSE(v_θ, x₁ − x₀)"

=== INFERENCE (right half) ===
- "Noisy Speech" → "MOSS Tokenizer" → "condition c"
- "x₀ ~ N(0,I)" → "50-step Euler ODE" box (with arrows for t and c entering) → "x̂₁"
- "x̂₁" → "DAC Decoder" → "Enhanced Speech" waveform

Keep boxes as opaque blocks — do NOT show internal layers or sub-components of MOSS, DiT, or DAC.

Style: flat modern diagram, white background, blue/teal color scheme, thin arrows, sans-serif labels, academic poster quality. No shadows, no 3D effects.
```

---

## Prompt 2: MOSS Tokenizer & 4 Experiment Variants (AI-generated)

```
Create a technical diagram with two parts, arranged top-to-bottom:

=== TOP: MOSS Tokenizer ===
A single tall box labeled "MOSS Audio Tokenizer". Inside, show a simple vertical stack of 4 colored blocks labeled "Stage 0", "Stage 1", "Stage 2", "Stage 3" (do NOT show internal details like conv layers or dimensions).

From the bottom of Stage 3, draw 32 horizontal arrows fanning out to the right, labeled "Layer 0", "Layer 1", ..., "Layer 31" (show first few and last few with "..." in the middle).

=== BOTTOM: 4 Experiment Variants ===
Show 4 small diagrams side by side, each showing how the 32 layer arrows are used:

(A) "Exp 1 — No conditioning":
- All 32 arrows are grayed out / crossed. Label: "No MOSS features used"
- A flow trajectory arrow from noise to clean with NO condition input

(B) "Exp 2 — Last Layer":
- Only Layer 31 arrow is active (colored), all others grayed out
- Arrow feeds into the flow trajectory as "c = h₃₁"

(C) "Exp 3 — Static Multi-Layer":
- All 32 arrows are active, converging into a "Σ softmax(w) · hᵢ" fusion box
- Small weights w₀...w₃₁ shown as fixed scalars
- Output "c" feeds into the flow

(D) "Exp 4 — Time-Dep. Multi-Layer":
- All 32 arrows are active, converging into a fusion box
- A side arrow labeled "timestep t" enters a small "MLP" box that outputs weights "w(t)"
- "Σ softmax(MLP(t)) · hᵢ" shown
- Output "c(t)" feeds into the flow — emphasize that the condition changes with t

For each variant, show a small curved arrow labeled "injecting condition into the flow" pointing from c toward the flow trajectory.

Style: clean, white background, blue gradient for MOSS stages, 4 mini-diagrams in a row for variants, sans-serif, academic poster quality, no clutter.
```

---

## Python-Generated Figures (see `plot_poster_figures.py`)

The following 4 figures are generated precisely from real data using Python/matplotlib. Run the script on Colab:

```bash
%run poster/plot_poster_figures.py --drive_dir /content/drive/MyDrive/speech_enhancement_output
```

1. **metrics_comparison.png** — PESQ / STOI / FAD bar charts with metric descriptions and % improvements
2. **fad_over_epochs.png** — FAD vs training epoch for all 4 conditions (professor's requirement)
3. **static_weights.png** — Exp 3 static fusion weights showing near-uniform distribution (Gini = 0.007)
4. **timedep_weights.png** — Exp 4 time-dependent fusion weights heatmap (layers × timesteps)

---

## Notes

- AI-generated images: target 2000×1500 px, white background, flat modern style
- Python-generated figures: 300 DPI, saved automatically by the script
- Color consistency: red (Exp 1), blue (Exp 2), green (Exp 3), purple (Exp 4)
