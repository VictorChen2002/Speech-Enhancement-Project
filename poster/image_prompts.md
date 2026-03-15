# Image Generation Prompts for Poster Diagrams

We use **4 AI-generated diagrams** (Prompts 1–4) and **4 Python-generated plots** (see `plot_poster_figures.py`).

---

## Prompt 1: Training Pipeline (AI-generated)

```
Create a clean technical block-diagram showing the TRAINING pipeline of a speech enhancement system. Layout flows top-to-bottom.

=== TOP ROW — Three parallel encoders ===
Three inputs at the very top, each with a waveform icon:
- LEFT: "Clean Speech" waveform → a single opaque box labeled "DAC Encoder" → output arrow labeled "x₁ (clean DAC latent, 50 Hz, d=1024)"
- CENTER: "Noisy Speech" waveform → a single opaque box labeled "DAC Encoder" (same style as left) → output arrow labeled "x₀ (noisy DAC latent, 50 Hz, d=1024)"
- RIGHT: "Noisy Speech" waveform (same input as center) → a single opaque box labeled "MOSS Tokenizer" → output arrow labeled "condition c (12.5 Hz, d=1280)"
  Add a small note: "(how c is built varies — see Image 4)"

=== MIDDLE — Interpolation ===
Below x₀ and x₁, show a formula box:
"Sample t ~ U(0,1)"
"xₜ = t · x₁ + (1−t) · x₀" (linear interpolation from noisy to clean)

=== CENTER — DiT (Diffusion Transformer) ===
A large box labeled "DiT — Diffusion Transformer (×6 layers)"
Inside the box, show ONE representative layer as a vertical stack of 4 sub-blocks:
  1. "AdaLayerNorm" — with a side arrow labeled "timestep t" entering it
  2. "Self-Attention"
  3. "Cross-Attention" — with a side arrow labeled "condition c" entering as K, V. A small label says "Q from latent, K/V from condition". Highlight this block in a different color (e.g., light orange border) to emphasize it.
  4. "Feed-Forward Network (FFN)"
Show residual connections (curved arrows) around each sub-block.
Labeled "×6" on the side to indicate 6 identical layers.

Three inputs enter the DiT box: xₜ (from above), t (from a "timestep" circle), c (from MOSS, on the right side).
One output exits the bottom: "v_θ (predicted velocity)"

=== BOTTOM — Loss ===
Below the DiT output:
A box showing: "Loss = MSE( v_θ , x₁ − x₀ )"
With "x₁ − x₀" labeled as "true velocity (target direction)"

Do NOT show inference, ODE solving, or DAC decoding — that is in Image 2.
Do NOT show what is inside MOSS or DAC — they are opaque boxes.

Style: flat modern block diagram, white background, blue/teal color scheme for main boxes, light orange highlight on the cross-attention block, thin black arrows, sans-serif labels, academic poster quality. No shadows, no 3D.
```

---

## Prompt 2: Inference Pipeline (AI-generated)

```
Create a clean technical block-diagram showing the INFERENCE pipeline of a speech enhancement system. Layout flows left-to-right.

=== LEFT — Input ===
"Noisy Speech" waveform icon
Two arrows branch out from it:
  - Arrow DOWN to a box labeled "DAC Encoder" → output "x₀ (noisy DAC latent)"
  - Arrow RIGHT to a box labeled "MOSS Tokenizer" → output "condition c"

=== CENTER — ODE Solver ===
A large rounded box labeled "50-step Euler ODE Solver"
Three inputs enter this box:
  - x₀ from the left (starting point)
  - c from above (condition, entering every step)
  - A clock/loop icon labeled "t = 0/50, 1/50, ..., 49/50"

Inside the box, show a simple loop diagram:
  "For each step i = 0 to 49:"
  "  t = i / 50"
  "  v = DiT(xₜ, t, c)"     ← the DiT is used here as a black box
  "  xₜ₊₁ = xₜ + v × Δt"

One output exits to the right: "x̂₁ (predicted clean latent)"

=== RIGHT — Output ===
"x̂₁" → a box labeled "DAC Decoder" → "Enhanced Speech" waveform icon

Add a contrast note at the bottom:
"Compared to Training: no clean speech needed. We start from x₀ (noisy) and iteratively walk toward clean."

Do NOT show DiT internals — that was in Image 1.
Do NOT show MOSS internals — just an opaque box.

Style: flat modern diagram, white background, blue/teal color scheme matching Image 1, thin arrows, sans-serif labels, academic poster quality. No shadows, no 3D.
```

---

## Prompt 3: Rectified Flow Illustration (AI-generated)

```
Create a clean conceptual diagram illustrating Rectified Flow for speech enhancement.

MAIN ELEMENT — A straight line in a latent space:
- LEFT endpoint: a point labeled "x₀ (noisy DAC latent)" with a small noisy waveform icon, colored in red/orange
- RIGHT endpoint: a point labeled "x₁ (clean DAC latent)" with a small clean waveform icon, colored in green
- The line between them represents the straight-path interpolation

Along the line, show 5 evenly spaced points at t = 0, 0.25, 0.5, 0.75, 1.0:
- At each intermediate point, label it "xₜ"
- At each point, draw a small tangent arrow (velocity vector) pointing from left to right along the line
- Label these arrows "v_θ(xₜ, t, c)"
- One annotation says: "Target velocity = x₁ − x₀ (constant along the path)"

BELOW the line, show two panels side by side:

Left panel — "TRAINING":
- "1. Sample random t ~ U(0,1)"
- "2. Compute xₜ = t · x₁ + (1−t) · x₀"
- "3. DiT predicts velocity v_θ(xₜ, t, c)"
- "4. Loss = MSE(v_θ, x₁ − x₀)"

Right panel — "INFERENCE":
- "1. Start at x₀ (noisy DAC latent)"
- "2. 50 Euler steps: xₜ₊Δₜ = xₜ + Δt · v_θ"
- "3. Arrive at x̂₁ ≈ x₁ (predicted clean latent)"
- "4. Decode x̂₁ with DAC → enhanced speech"

IMPORTANT: The line does NOT start from Gaussian noise. x₀ is the noisy DAC latent (a meaningful signal, not random noise). This is what makes it different from standard diffusion: we learn a direct path from noisy to clean.

A subtle annotation: "Unlike standard diffusion, x₀ is the noisy encoding — not random noise."

Style: clean mathematical figure, white background, pastel colors for endpoints (red→green gradient along the line), thin arrows for velocity, sans-serif labels, academic poster quality.
```

---

## Prompt 4: Ablation — 4 Experiment Conditioning Variants (AI-generated)

```
Create a technical diagram with two parts, top and bottom.

=== TOP — Shared component: MOSS Tokenizer ===
A tall box labeled "MOSS Audio Tokenizer" with "Noisy Speech" waveform entering at the top.
Inside, show a simple vertical stack of 4 colored blocks (blue gradient, lightest at top):
  - "Stage 0" (light blue)
  - "Stage 1" (medium blue)
  - "Stage 2" (darker blue)
  - "Stage 3 — 32 Transformer Layers, d=1280" (darkest blue)

From Stage 3, draw 32 horizontal arrows fanning out to the right, labeled:
  "Layer 0", "Layer 1", "Layer 2", "...", "Layer 30", "Layer 31"
  Each arrow carries "hᵢ ∈ ℝ¹²⁸⁰"

Title above: "What all experiments share: MOSS extracts 32 layers of features"

=== BOTTOM — 4 experiment variants, side by side ===
Show 4 mini-diagrams in a horizontal row. Each shows how the 32 layer arrows are turned into a condition c, which then enters a small box labeled "→ DiT (Cross-Attention K/V)".

(A) "Exp 1 — No Conditioning":
- All 32 arrows are grayed out with an "✗" symbol
- The DiT box has NO condition input — label: "c = ∅"
- A note: "No cross-attention layers (28.4M params)"

(B) "Exp 2 — Last Layer Only":
- Only the arrow from Layer 31 is active (colored blue), all others grayed out
- Arrow goes directly into the DiT box
- Formula next to it: "c = h₃₁"
- A note: "+10.1M cross-attention params"

(C) "Exp 3 — Static Multi-Layer":
- All 32 arrows are active (colored green)
- They converge into a fusion box labeled "Weighted Sum"
- Next to the fusion box, show the formula: "c = Σᵢ softmax(wᵢ) · hᵢ"
- Small text: "32 learnable scalar weights, fixed across all timesteps"
- Output c goes into the DiT box
- A note: "+32 extra params"

(D) "Exp 4 — Time-Dependent Multi-Layer":
- All 32 arrows are active (colored purple)
- They converge into a fusion box
- A side arrow labeled "timestep t" enters a small box labeled "MLP (1→64→32)"
- The MLP outputs "w(t)" which enters the fusion box
- Formula: "c(t) = Σᵢ softmax(MLP(t))ᵢ · hᵢ"
- Small text: "Weights change with ODE timestep — different layers emphasized at different stages of denoising"
- Output c(t) goes into the DiT box
- A note: "+2,208 extra MLP params"

Below all 4 variants, a shared arrow pointing down to a single box:
"All conditions c are injected into DiT via Cross-Attention (Q from latent, K/V from c)"
"(DiT architecture shown in Image 1)"

Style: clean, white background, blue gradient for MOSS stages, each experiment variant in its own color (red=Exp1, blue=Exp2, green=Exp3, purple=Exp4), sans-serif, academic poster quality, no clutter.
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
