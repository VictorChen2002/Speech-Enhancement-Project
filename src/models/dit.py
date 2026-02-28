"""
dit.py — Vanilla Diffusion Transformer (DiT) with Cross-Attention conditioning.

Supports three conditioning modes controlled by ``condition_type``:
    * "none"        – no external conditioning (timestep only)
    * "last_layer"  – cross-attention with MOSS last-layer embeddings
    * "multi_layer" – cross-attention with learnable weighted-sum of
                      multiple MOSS hidden layers
"""

import math
from typing import Optional, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


# --------------------------------------------------------------------------- #
#  Sinusoidal timestep embedding                                               #
# --------------------------------------------------------------------------- #

class SinusoidalTimestepEmbedding(nn.Module):
    """Map scalar timestep t ∈ [0, 1] to a vector."""

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.SiLU(),
            nn.Linear(dim * 4, dim),
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        t : (B,) float tensor in [0, 1]

        Returns
        -------
        (B, dim) embedding
        """
        half_dim = self.dim // 2
        freqs = torch.exp(
            -math.log(10000.0) * torch.arange(half_dim, device=t.device) / half_dim
        )
        args = t[:, None] * freqs[None, :]  # (B, half_dim)
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)  # (B, dim)
        return self.mlp(emb)


# --------------------------------------------------------------------------- #
#  Adaptive Layer Normalization (adaLN)                                        #
# --------------------------------------------------------------------------- #

class AdaLayerNorm(nn.Module):
    """Layer norm whose scale & shift are predicted from the time embedding."""

    def __init__(self, dim: int):
        super().__init__()
        self.norm = nn.LayerNorm(dim, elementwise_affine=False)
        # Predict scale (gamma) and shift (beta) from time embedding
        self.proj = nn.Sequential(
            nn.SiLU(),
            nn.Linear(dim, dim * 2),
        )

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x     : (B, T, D)
        t_emb : (B, D) — time embedding
        """
        scale_shift = self.proj(t_emb).unsqueeze(1)  # (B, 1, 2D)
        scale, shift = scale_shift.chunk(2, dim=-1)   # each (B, 1, D)
        return self.norm(x) * (1 + scale) + shift


# --------------------------------------------------------------------------- #
#  Multi-Head Self-Attention                                                   #
# --------------------------------------------------------------------------- #

class SelfAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int = 8, dropout: float = 0.0):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape
        qkv = self.qkv(x).reshape(B, T, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B, H, T, d)
        q, k, v = qkv.unbind(0)
        attn = F.scaled_dot_product_attention(q, k, v)
        out = rearrange(attn, "b h t d -> b t (h d)")
        return self.dropout(self.proj(out))


# --------------------------------------------------------------------------- #
#  Multi-Head Cross-Attention                                                  #
# --------------------------------------------------------------------------- #

class CrossAttention(nn.Module):
    """Standard cross-attention: Q comes from DiT, K/V from condition."""

    def __init__(self, dim: int, cond_dim: int, num_heads: int = 8, dropout: float = 0.0):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.q_proj = nn.Linear(dim, dim)
        self.kv_proj = nn.Linear(cond_dim, dim * 2)
        self.out_proj = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x    : (B, T_x, D)   — DiT latent sequence
        cond : (B, T_c, D_c) — condition (MOSS embeddings)
        """
        B, T_x, D = x.shape
        T_c = cond.shape[1]

        q = self.q_proj(x).reshape(B, T_x, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        kv = self.kv_proj(cond).reshape(B, T_c, 2, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        k, v = kv.unbind(0)

        attn = F.scaled_dot_product_attention(q, k, v)
        out = rearrange(attn, "b h t d -> b t (h d)")
        return self.dropout(self.out_proj(out))


# --------------------------------------------------------------------------- #
#  Feed-Forward Network                                                        #
# --------------------------------------------------------------------------- #

class FeedForward(nn.Module):
    def __init__(self, dim: int, mult: int = 4, dropout: float = 0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim * mult),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * mult, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# --------------------------------------------------------------------------- #
#  DiT Block                                                                   #
# --------------------------------------------------------------------------- #

class DiTBlock(nn.Module):
    """
    One Transformer block with:
        1. adaLN → Self-Attention
        2. (optional) adaLN → Cross-Attention (when use_cross_attn=True)
        3. adaLN → Feed-Forward
    """

    def __init__(
        self,
        dim: int,
        cond_dim: int,
        num_heads: int = 8,
        dropout: float = 0.0,
        use_cross_attn: bool = True,
    ):
        super().__init__()
        # Self-attention
        self.norm1 = AdaLayerNorm(dim)
        self.self_attn = SelfAttention(dim, num_heads, dropout)

        # Cross-attention (conditional)
        self.use_cross_attn = use_cross_attn
        if use_cross_attn:
            self.norm2 = AdaLayerNorm(dim)
            self.cross_attn = CrossAttention(dim, cond_dim, num_heads, dropout)

        # Feed-forward
        self.norm3 = AdaLayerNorm(dim)
        self.ff = FeedForward(dim, dropout=dropout)

    def forward(
        self,
        x: torch.Tensor,
        t_emb: torch.Tensor,
        cond: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # Self-attention
        x = x + self.self_attn(self.norm1(x, t_emb))
        # Cross-attention
        if self.use_cross_attn and cond is not None:
            x = x + self.cross_attn(self.norm2(x, t_emb), cond)
        # Feed-forward
        x = x + self.ff(self.norm3(x, t_emb))
        return x


# --------------------------------------------------------------------------- #
#  Multi-Layer Condition Fusion                                                #
# --------------------------------------------------------------------------- #

class MultiLayerConditionFusion(nn.Module):
    """
    Learnable weighted sum of multiple MOSS hidden-layer outputs.

    Given a list of L tensors each of shape (B, T_c, D_moss), produce a
    single (B, T_c, D_moss) tensor via softmax-normalised scalar weights.
    """

    def __init__(self, num_layers: int):
        super().__init__()
        # Learnable scalar weight per layer (initialised uniformly)
        self.layer_weights = nn.Parameter(torch.zeros(num_layers))

    def forward(self, layer_outputs: List[torch.Tensor]) -> torch.Tensor:
        """
        Parameters
        ----------
        layer_outputs : list of (B, T_c, D) tensors, length L

        Returns
        -------
        (B, T_c, D) weighted-sum tensor
        """
        weights = F.softmax(self.layer_weights, dim=0)  # (L,)
        stacked = torch.stack(layer_outputs, dim=0)      # (L, B, T_c, D)
        # Weighted sum along layer dimension
        fused = (weights[:, None, None, None] * stacked).sum(dim=0)  # (B, T_c, D)
        return fused


# --------------------------------------------------------------------------- #
#  Full DiT Model                                                              #
# --------------------------------------------------------------------------- #

class DiffusionTransformer(nn.Module):
    """
    Diffusion Transformer for flow-matching speech enhancement.

    Parameters
    ----------
    dac_latent_dim : int
        Dimensionality of DAC continuous latent vectors.
    moss_embed_dim : int
        Dimensionality of MOSS condition embeddings.
    hidden_dim : int
        Internal hidden size of the transformer.
    num_heads : int
        Number of attention heads.
    num_layers : int
        Number of DiT blocks.
    dropout : float
        Dropout rate.
    condition_type : str
        One of "none", "last_layer", "multi_layer".
    num_moss_layers : int
        Number of MOSS layers (only used when condition_type == "multi_layer").
    """

    def __init__(
        self,
        dac_latent_dim: int = 1024,
        moss_embed_dim: int = 1024,
        hidden_dim: int = 512,
        num_heads: int = 8,
        num_layers: int = 6,
        dropout: float = 0.1,
        condition_type: str = "multi_layer",
        num_moss_layers: int = 4,
    ):
        super().__init__()
        self.condition_type = condition_type

        # Input projection: DAC latent dim -> hidden dim
        self.input_proj = nn.Linear(dac_latent_dim, hidden_dim)

        # Timestep embedding
        self.time_embed = SinusoidalTimestepEmbedding(hidden_dim)

        # Condition projection(s)
        use_cross_attn = condition_type != "none"
        if use_cross_attn:
            self.cond_proj = nn.Linear(moss_embed_dim, hidden_dim)

        if condition_type == "multi_layer":
            self.multi_layer_fusion = MultiLayerConditionFusion(num_moss_layers)

        # Transformer blocks
        self.blocks = nn.ModuleList([
            DiTBlock(
                dim=hidden_dim,
                cond_dim=hidden_dim,  # after projection
                num_heads=num_heads,
                dropout=dropout,
                use_cross_attn=use_cross_attn,
            )
            for _ in range(num_layers)
        ])

        # Output projection: hidden dim -> DAC latent dim (predict vector field)
        self.output_norm = nn.LayerNorm(hidden_dim)
        self.output_proj = nn.Linear(hidden_dim, dac_latent_dim)

        self._init_weights()

    def _init_weights(self):
        """Initialize weights with small values for stable training."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        cond: Optional[torch.Tensor] = None,
        cond_layers: Optional[List[torch.Tensor]] = None,
    ) -> torch.Tensor:
        """
        Predict the vector field V at (x_t, t) optionally conditioned on MOSS.

        Parameters
        ----------
        x_t        : (B, T, D_dac) – interpolated latent at time t
        t          : (B,) – timestep in [0, 1]
        cond       : (B, T_c, D_moss) – MOSS last-layer embedding
                     (used when condition_type == "last_layer")
        cond_layers: list of (B, T_c, D_moss) – MOSS multi-layer embeddings
                     (used when condition_type == "multi_layer")

        Returns
        -------
        v_pred : (B, T, D_dac) – predicted vector field
        """
        # Project input
        h = self.input_proj(x_t)             # (B, T, hidden)
        t_emb = self.time_embed(t)           # (B, hidden)

        # Prepare condition
        c = None
        if self.condition_type == "last_layer" and cond is not None:
            c = self.cond_proj(cond)         # (B, T_c, hidden)
        elif self.condition_type == "multi_layer" and cond_layers is not None:
            fused = self.multi_layer_fusion(cond_layers)  # (B, T_c, D_moss)
            c = self.cond_proj(fused)                     # (B, T_c, hidden)

        # Transformer blocks
        for block in self.blocks:
            h = block(h, t_emb, c)

        # Output
        h = self.output_norm(h)
        v_pred = self.output_proj(h)         # (B, T, D_dac)
        return v_pred
