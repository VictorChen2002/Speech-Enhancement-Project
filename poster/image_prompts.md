# Image Generation Prompts for Poster Diagrams

Use these prompts with Nano Banana (or similar) to generate architecture diagrams for the poster.

---

## Prompt 1: Overall Pipeline — Flow-Matching Speech Enhancement

```
Create a clean, professional technical diagram showing a speech enhancement pipeline with these components, laid out left-to-right:

LEFT SIDE — Input:
- A waveform labeled "Noisy Speech (16 kHz)" enters the system

TOP BRANCH — Condition Extraction:
- The noisy waveform goes into a box labeled "MOSS Audio Tokenizer"
- Inside MOSS, show 4 stacked stages: Stage 0 (240↓, 12 layers), Stage 1 (2↓, 12 layers), Stage 2 (2↓, 12 layers), Stage 3 (2↓, 32 layers, d=1280)
- An arrow from MOSS outputs "Multi-Layer Features (32 layers × d=1280, 12.5 Hz)"
- A small box labeled "Fusion Module" combines the 32 layer outputs using learned weights
- Output: single condition vector "c ∈ ℝ^1280"

BOTTOM BRANCH — Target:  
- A clean waveform labeled "Clean Speech" goes into a box labeled "DAC Encoder"
- Output: "x₁ ∈ ℝ^1024 (50 Hz)"

CENTER — Flow Matching:
- A box labeled "Rectified Flow ODE" shows a straight-line trajectory from "x₀ ~ N(0,I)" (Gaussian noise) to "x₁" (clean latent)
- An intermediate point "xₜ = t·x₁ + (1-t)·x₀" is shown on the line
- The velocity field "v_θ(xₜ, t, c)" is predicted by:

CENTER — DiT Backbone:
- A tall box labeled "Diffusion Transformer (DiT)" with 6 stacked layers
- Each layer contains: AdaLayerNorm → Self-Attention → Cross-Attention (with condition c as K,V) → FFN
- Timestep t enters via sinusoidal embedding → AdaLayerNorm
- The condition c enters via Cross-Attention (Q from latent, K/V from condition)

RIGHT SIDE — Inference:
- "50-step Euler ODE solver" box produces "x̂₁" (predicted clean latent)
- Goes into "DAC Decoder" box
- Output: "Enhanced Speech (16 kHz)" waveform

Style: flat, modern, white background, blue/teal/green color scheme, sans-serif labels, thin arrows, no shadows, suitable for an academic poster. Similar to diagrams in ICLR/NeurIPS papers.
```

---

## Prompt 2: MOSS Multi-Layer Feature Extraction Detail

```
Create a detailed technical diagram showing multi-layer feature extraction from the MOSS Audio Tokenizer, vertical layout:

TOP: Input waveform "Noisy Speech (24 kHz → resampled)"

VERTICAL STACK of 4 stages, each as a colored block:
- Stage 0: Light blue block, "Conv 240↓ + 12 Transformer Layers, d=768"
  Arrow showing "100 Hz output"
- Stage 1: Medium blue block, "Conv 2↓ + 12 Transformer Layers, d=768"
  Arrow showing "50 Hz output"
- Stage 2: Darker blue block, "Conv 2↓ + 12 Transformer Layers, d=768"
  Arrow showing "25 Hz output"
- Stage 3: Dark blue/navy block, "Conv 2↓ + 32 Transformer Layers, d=1280"
  Arrow showing "12.5 Hz output"

FROM Stage 3, show 32 horizontal arrows coming out of each transformer layer (labeled Layer 0, Layer 1, ..., Layer 31), each carrying a vector of dimension 1280.

These 32 arrows converge into a FUSION MODULE (shown as a highlighted box) with 3 variants side by side:

Variant A — "Last Layer Only (Exp 2)":
- Only Layer 31's arrow connects, others are grayed out
- Output: "c = h₃₁"

Variant B — "Static Multi-Layer (Exp 3)":
- All 32 arrows connect with learned scalar weights w₀, w₁, ..., w₃₁
- A softmax symbol normalizes them
- Formula: "c = Σᵢ softmax(wᵢ) · hᵢ"
- Show a small bar chart of nearly uniform weights

Variant C — "Time-Dependent Multi-Layer (Exp 4)":
- All 32 arrows connect with weights that depend on timestep t
- An MLP box: "t → MLP(1→64→32) → softmax → w(t)"
- Formula: "c(t) = Σᵢ softmax(MLP(t))ᵢ · hᵢ"
- Show a small heatmap suggesting weights vary with t

Style: clean technical diagram, white background, blue color gradient for stages, academic poster quality, sans-serif font, clear labels with dimensions.
```

---

## Prompt 3: Rectified Flow ODE Visualization

```
Create a clean mathematical diagram illustrating Rectified Flow for speech enhancement:

LEFT: A cloud/distribution labeled "x₀ ~ N(0, I)" representing Gaussian noise, colored in light red/orange

RIGHT: A distribution labeled "x₁ ~ p_data" representing clean DAC latents, colored in green

BETWEEN them: Multiple straight lines (3-4 parallel trajectories) connecting paired samples from x₀ to x₁, showing the "straight path" interpolation:
- Each line has intermediate points at t=0.2, 0.4, 0.6, 0.8
- The interpolation formula "xₜ = t·x₁ + (1−t)·x₀" is displayed
- Small velocity arrows v_θ tangent to each trajectory, pointing from x₀ toward x₁
- The arrows are labeled "v_θ(xₜ, t, c) ≈ x₁ − x₀"

BELOW the trajectories: A horizontal timeline from t=0 to t=1
- At t=0: "Pure Noise"
- At t=1: "Clean Latent"
- At intermediate points: show small spectrograms transitioning from noise to speech

TRAINING box (bottom-left):
- "Loss = MSE(v_θ, x₁ − x₀)"
- "Random t ~ U(0,1) each step"

INFERENCE box (bottom-right):
- "50-step Euler solver"
- "xₜ₊ₐₜ = xₜ + Δt · v_θ(xₜ, t, c)"
- "Start from x₀ ~ N(0,I), arrive at x̂₁"

Style: mathematical figure style, white background, clean arrows, pastel colors for distributions, similar to figures in flow matching papers (Lipman et al., 2023). Academic poster quality.
```

---

## Prompt 4: DiT Block Architecture Detail

```
Create a detailed block diagram of a single Diffusion Transformer (DiT) block:

INPUT arrow from bottom labeled "h ∈ ℝ^(T×512)" (hidden representation)

VERTICAL STACK of operations (bottom to top):

1. Box: "Adaptive Layer Norm" 
   Side input: arrow from "Timestep Embedding" (sinusoidal, dim=512)
   Shows: γ, β learned from timestep via MLP
   Formula: "AdaLN(h, t) = γ(t) · LayerNorm(h) + β(t)"

2. Box: "Multi-Head Self-Attention (8 heads)"
   Shows Q, K, V all from h
   Residual connection (skip arrow) around this block

3. Box: "Multi-Head Cross-Attention (8 heads)"
   Shows: Q from h, K/V from condition c
   Side input: arrow from "MOSS Condition c ∈ ℝ^(S×1280)" 
   (with a note: "projected to 512 via linear layer")
   Residual connection around this block

4. Box: "Feed-Forward Network"
   "Linear(512→2048) → GELU → Linear(2048→512)"
   Residual connection around this block

OUTPUT arrow to top labeled "h' ∈ ℝ^(T×512)"

SIDE NOTE: "×6 layers stacked"

Show all residual connections as curved arrows on the right side.
Show dropout (p=0.1) after each sub-layer.

Style: vertical flowchart, clean boxes with rounded corners, light blue/gray color scheme, academic poster quality, sans-serif labels.
```

---

## Prompt 5: Results Comparison — Bar Chart Style

```
Create a clean results comparison figure with 3 grouped bar charts side by side:

Chart 1 — PESQ (↑ higher is better):
- 4 bars: Exp1 (red, 1.6048), Exp2 (blue, 1.6499), Exp3 (green, 1.6868), Exp4 (purple, 1.6986)
- Exp4 bar highlighted with gold border (best)
- Values displayed on top of each bar

Chart 2 — STOI (↑ higher is better):
- Same 4 bars: 0.8527, 0.8589, 0.8642, 0.8647
- Exp4 highlighted

Chart 3 — FAD (↓ lower is better):
- Same 4 bars: 2.9774, 2.6997, 2.3857, 2.3456
- Exp4 highlighted (lowest)

Legend at bottom:
- Exp1: No Conditioning (red)
- Exp2: Last Layer Only (blue)
- Exp3: Static Multi-Layer (green)
- Exp4: Time-Dependent Multi-Layer (purple)

Below charts, a text box: "Consistent improvement across all metrics: Exp4 achieves best PESQ (+5.8%), STOI (+1.4%), and FAD (−21.2%) vs baseline"

Style: clean, white background, bold colors, academic poster quality, large readable labels, suitable for A3 poster printing at 300 DPI.
```

---

## Prompt 6: Time-Dependent Fusion Weight Heatmap

```
Create a heatmap visualization showing how layer importance changes with ODE timestep:

MAIN FIGURE — Heatmap:
- X-axis: "MOSS Transformer Layer" (0 to 31)
- Y-axis: "ODE Timestep t" (0.0 at bottom to 1.0 at top, 11 rows)
- Color: viridis or plasma colormap, representing fusion weight magnitude
- Key pattern to show: 
  - At t=0.0 (bottom): nearly uniform weights across all layers
  - At t=1.0 (top): early layers (0-15) slightly brighter (higher weight), middle layers (16-30) slightly darker
  - Layer 31 (last) always slightly brighter than middle layers
  - The variation is subtle but visible

SIDE PANEL (right) — Line plot:
- Same Y-axis (timestep 0 to 1)
- X-axis: "Weight variance across layers"
- Shows variance increasing from t=0 to t=1
- Title: "More selective at t→1"

BOTTOM PANEL — Bar chart:
- X-axis: layers 0-31
- Y-axis: "Mean weight across all t"
- Horizontal dashed line at 1/32 = 0.03125 labeled "uniform"
- Bars colored by whether above/below uniform

ANNOTATION: "Interpretation: Early in denoising (t≈0), the model uses all MOSS layers uniformly. Near completion (t≈1), it relies more on early layers (fine-grained features) while reducing middle layer contribution."

Style: scientific figure, white background, colorbar, clear axis labels, suitable for academic poster. Matplotlib/seaborn aesthetic.
```

---

## Notes for Image Generation

- Target resolution: at least 2000×1500 pixels (for A3 poster at 300 DPI)
- Color scheme consistency: use blue, green, purple, red/orange across all figures
- Font: sans-serif, readable at poster viewing distance (~1m)
- All mathematical notation should use proper symbols (subscripts, Greek letters)
- Prefer flat design over 3D effects
- Keep backgrounds white or very light gray
