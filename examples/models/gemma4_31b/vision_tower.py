# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Gemma 4 31B vision tower — self-contained PyTorch port.

This module mirrors the vision tower of HuggingFace's
``transformers.models.gemma4.modeling_gemma4`` (Gemma4VisionModel +
Gemma4MultimodalEmbedder), plus the standardize step and the post-pool soft-token
projection. It contains no transformers imports at runtime, so it is safe to
``torch.export(strict=True)`` and ship in the ExecuTorch binary.

Final 4-method export contract (locked-in by orchestrator, pin #4)
==================================================================

The exported .pte ships with **4 methods**. ``forward`` (used by the exported
``prefill`` method) takes pre-computed embeddings, so a single code path covers
both text-only and image+text. The runner stitches the inputs together using
``embed_text`` (and, for images, ``vision_encoder``) before calling ``prefill``.

  1. ``embed_text(tokens) -> embeds [B,T,5376] bf16``
       Pure embed_tokens lookup + ``sqrt(hidden_size)`` scale, returned as bf16.
  2. ``vision_encoder(pixel_values [B,P,768] f32, pixel_position_ids [B,P,2] i64)
        -> (image_embeds [B,N,5376] bf16, mask [B,N] bool)``
       This module's ``Gemma4_31BVisionTower``.
  3. ``prefill(inputs_embeds [B,T,5376] bf16, input_pos [T] i64, temperature [1] f32)
        -> sampled [B,1] f32``
       UNIFIED. Used for BOTH text-only and image+text. Maps to
       ``Gemma4_31B.forward``.
  4. ``decode(tokens [B,1] i64, input_pos [1] i64, temperature [1] f32)
        -> sampled [B,1] f32``
       Single-token decode, token-input. Maps to ``Gemma4_31B.decode_forward``.

Multimodal prefill flow (runner-side):

    text_embeds       = embed_text(tokens)                              # [1, T, 5376] bf16
    image_embeds, mask = vision_encoder(pixel_values, pixel_position_ids)
    # In-place splice: for every i where tokens[i] == image_token_id (258880),
    # overwrite text_embeds[:, i, :] with the next valid row of image_embeds
    # (skipping rows where mask is False — those are padding soft tokens).
    sampled = prefill(text_embeds, input_pos, temperature)
    # then per-token decode loop using `decode(tokens, input_pos, temperature)`.

Vision tower output is NOT pre-scaled by ``sqrt(hidden_size)`` (matches HF). Only
``embed_text`` applies that scale, so text rows of ``inputs_embeds`` are scaled
and image rows are not — same convention HF uses.

Numerical contract
==================

For a fp32 reference run:

    cosine_sim( hf_wrapper(pixel_values, pixel_position_ids),
                Gemma4_31BVisionTower(pixel_values, pixel_position_ids) ) > 0.99999

(See ``tests/test_vision_tower.py``.)
"""

from __future__ import annotations

import json
from dataclasses import dataclass

import torch
import torch.nn as nn

# Reuse the gemma4 text-decoder primitives that are numerically identical
# between the LM and the vision tower: Gemma4MLP (same SwiGLU GELU-tanh block,
# same gate/up/down submodule names) and rotate_half (same HF-style rotary
# helper). RMSNorm uses torch's nn.RMSNorm directly -- numerically identical to
# HF's Gemma4RMSNorm (float32 upcast + pow(mean_squared, -0.5)); the weightless
# V-norm / pre-projection norm use nn.RMSNorm(..., elementwise_affine=False).
from executorch.examples.models.gemma4.text_decoder import Gemma4MLP, rotate_half
from torch.nn import functional as F


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass
class Gemma4VisionConfig:
    """Mirror of HF ``Gemma4VisionConfig`` for the bits we actually use."""

    hidden_size: int = 1152
    intermediate_size: int = 4304
    num_hidden_layers: int = 27
    num_attention_heads: int = 16
    num_key_value_heads: int = 16
    head_dim: int = 72
    hidden_activation: str = "gelu_pytorch_tanh"
    rms_norm_eps: float = 1e-6
    patch_size: int = 16
    pooling_kernel_size: int = 3
    position_embedding_size: int = 10240
    max_position_embeddings: int = 131072
    rope_theta: float = 100.0
    standardize: bool = True
    use_clipped_linears: bool = (
        False  # 31B doesn't clip — checkpoint has no clamp params.
    )
    default_output_length: int = 280  # vision_soft_tokens_per_image at top level

    # Channels per spatial axis in the patchified input (RGB * patch_size^2).
    in_channels: int = 3

    @staticmethod
    def from_hf_config(hf_cfg_path_or_dict) -> "Gemma4VisionConfig":
        """Build from the top-level HF config dict OR a path to ``config.json``.

        Reads the ``vision_config`` block (and ``vision_soft_tokens_per_image``).
        Returns ``None`` if there is no vision_config in the file.
        """
        if isinstance(hf_cfg_path_or_dict, str):
            with open(hf_cfg_path_or_dict, "r") as f:
                top = json.load(f)
        else:
            top = hf_cfg_path_or_dict

        vc = top.get("vision_config", None)
        if vc is None:
            return None

        rope_params = vc.get("rope_parameters", {}) or {}
        rope_theta = rope_params.get("rope_theta", 100.0)

        default_output_length = vc.get(
            "default_output_length",
            top.get("vision_soft_tokens_per_image", 280),
        )

        return Gemma4VisionConfig(
            hidden_size=vc.get("hidden_size", 1152),
            intermediate_size=vc.get("intermediate_size", 4304),
            num_hidden_layers=vc.get("num_hidden_layers", 27),
            num_attention_heads=vc.get("num_attention_heads", 16),
            num_key_value_heads=vc.get("num_key_value_heads", 16),
            head_dim=vc.get("head_dim", 72),
            hidden_activation=vc.get("hidden_activation", "gelu_pytorch_tanh"),
            rms_norm_eps=vc.get("rms_norm_eps", 1e-6),
            patch_size=vc.get("patch_size", 16),
            pooling_kernel_size=vc.get("pooling_kernel_size", 3),
            position_embedding_size=vc.get("position_embedding_size", 10240),
            max_position_embeddings=vc.get("max_position_embeddings", 131072),
            rope_theta=rope_theta,
            standardize=vc.get("standardize", True),
            use_clipped_linears=vc.get("use_clipped_linears", False),
            default_output_length=default_output_length,
        )

    @property
    def patch_dim(self) -> int:
        return self.in_channels * self.patch_size * self.patch_size


# ---------------------------------------------------------------------------
# Patch embedder
# ---------------------------------------------------------------------------


class Gemma4VisionPatchEmbedder(nn.Module):
    """HF ``Gemma4VisionPatchEmbedder``: rescale → linear → 2D position lookup."""

    def __init__(self, config: Gemma4VisionConfig):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.position_embedding_size = config.position_embedding_size

        # Linear from patch_dim (3*16*16=768) to hidden_size, no bias.
        self.input_proj = nn.Linear(config.patch_dim, self.hidden_size, bias=False)
        # 2 axes × position_embedding_size positions × hidden_size.
        self.position_embedding_table = nn.Parameter(
            torch.zeros(2, self.position_embedding_size, self.hidden_size)
        )

    def _position_embeddings(
        self,
        pixel_position_ids: torch.Tensor,  # [B, P, 2]
        padding_positions: torch.Tensor,  # [B, P] (True = padding)
    ) -> torch.Tensor:
        """2D positional lookup. Numerically identical to HF's one_hot @ table form,
        but uses ``F.embedding`` for clarity / speed."""
        clamped = pixel_position_ids.clamp(min=0).long()  # [B, P, 2]
        # axis 0 is x (column), axis 1 is y (row).
        emb_x = F.embedding(clamped[..., 0], self.position_embedding_table[0])
        emb_y = F.embedding(clamped[..., 1], self.position_embedding_table[1])
        pos_emb = emb_x + emb_y
        # Zero-out padding patches.
        pos_emb = torch.where(
            padding_positions.unsqueeze(-1), torch.zeros_like(pos_emb), pos_emb
        )
        return pos_emb

    def forward(
        self,
        pixel_values: torch.Tensor,  # [B, P, patch_dim]
        pixel_position_ids: torch.Tensor,  # [B, P, 2]
        padding_positions: torch.Tensor,  # [B, P]
    ) -> torch.Tensor:
        # Rescale [0,1] → [-1,1] (HF does ``2*(x-0.5)``).
        pixel_values = 2 * (pixel_values - 0.5)
        hidden_states = self.input_proj(pixel_values.to(self.input_proj.weight.dtype))
        position_embeddings = self._position_embeddings(
            pixel_position_ids, padding_positions
        )
        return hidden_states + position_embeddings


# ---------------------------------------------------------------------------
# Pooler
# ---------------------------------------------------------------------------


class Gemma4VisionPooler(nn.Module):
    """HF ``Gemma4VisionPooler``: zero out padding, optional 2D avg-pool, * sqrt(d)."""

    def __init__(self, config: Gemma4VisionConfig):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.root_hidden_size = self.hidden_size**0.5

    def _avg_pool_by_positions(
        self,
        hidden_states: torch.Tensor,  # [B, P, D]
        pixel_position_ids: torch.Tensor,  # [B, P, 2]
        length: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        input_seq_len = hidden_states.shape[1]
        k = int((input_seq_len // length) ** 0.5)
        k_squared = k * k
        if k_squared * length != input_seq_len:
            raise ValueError(
                f"Cannot pool {hidden_states.shape} to {length}: k={k}^2 * length={length} "
                f"must equal {input_seq_len}."
            )

        # Padding patches contribute zero (their hidden states are masked to zero
        # before this is called). Clamp -1's so one_hot doesn't explode.
        clamped_positions = pixel_position_ids.clamp(min=0)
        max_x = clamped_positions[..., 0].max(dim=-1, keepdim=True)[0] + 1
        kernel_idxs = torch.div(clamped_positions, k, rounding_mode="floor")
        kernel_idxs = kernel_idxs[..., 0] + (max_x // k) * kernel_idxs[..., 1]
        weights = F.one_hot(kernel_idxs.long(), length).float() / k_squared
        output = weights.transpose(1, 2) @ hidden_states.float()
        mask = torch.logical_not((weights == 0).all(dim=1))
        return output.to(hidden_states.dtype), mask

    def forward(
        self,
        hidden_states: torch.Tensor,  # [B, P, D]
        pixel_position_ids: torch.Tensor,  # [B, P, 2]
        padding_positions: torch.Tensor,  # [B, P]
        output_length: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if output_length > hidden_states.shape[1]:
            raise ValueError(
                f"Cannot output more soft tokens (requested {output_length}) than there are "
                f"patches ({hidden_states.shape[1]})."
            )

        hidden_states = hidden_states.masked_fill(padding_positions.unsqueeze(-1), 0.0)

        if hidden_states.shape[1] != output_length:
            hidden_states, padding_positions = self._avg_pool_by_positions(
                hidden_states, pixel_position_ids, output_length
            )
        # If no pooling is needed, padding_positions is already True=padding;
        # the wrapper expects pooler_mask = True=valid, so flip below.
        else:
            padding_positions = ~padding_positions  # now True = valid

        hidden_states = hidden_states * self.root_hidden_size
        return hidden_states, padding_positions


# ---------------------------------------------------------------------------
# Attention (with 2D RoPE)
#
# The SwiGLU GELU-tanh MLP is shared with the text decoder (Gemma4MLP) and the
# rotary helper is the shared ``rotate_half`` -- both imported above. The vision
# config always uses ``gelu_pytorch_tanh``, which is exactly what Gemma4MLP
# implements.
# ---------------------------------------------------------------------------


def _apply_rotary_pos_emb(
    x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, unsqueeze_dim: int = 2
) -> torch.Tensor:
    """HF ``apply_rotary_pos_emb`` for a single spatial axis.

    Input:
        x:   [B, P, H, head_dim/ndim]
        cos: [B, P, head_dim/ndim]
        sin: [B, P, head_dim/ndim]
    Returns: same shape as x.
    """
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    return (x * cos) + (rotate_half(x) * sin)


def _apply_multidimensional_rope(
    x: torch.Tensor,  # [B, P, H, head_dim]
    cos: torch.Tensor,  # [B, P, head_dim]
    sin: torch.Tensor,  # [B, P, head_dim]
    position_ids: torch.Tensor,  # [B, P, ndim] (unused except to read ndim)
) -> torch.Tensor:
    """Mirror of HF ``apply_multidimensional_rope`` for ndim=2 (image x,y).

    Splits ``x`` (and cos/sin) into ``ndim`` equal chunks along last dim, applies
    RoPE per chunk with the corresponding (cos, sin) chunk, concatenates.
    """
    ndim = position_ids.shape[-1]
    num_input_channels = x.shape[-1]
    num_rotated_channels_per_dim = 2 * (num_input_channels // (2 * ndim))
    if num_rotated_channels_per_dim <= 0:
        raise ValueError(
            f"num_rotated_channels_per_dim must be > 0; got "
            f"num_input_channels={num_input_channels} ndim={ndim}"
        )
    split_sizes = [num_rotated_channels_per_dim] * ndim
    x_parts = torch.split(x, split_sizes, dim=-1)
    cos_parts = torch.split(cos, split_sizes, dim=-1)
    sin_parts = torch.split(sin, split_sizes, dim=-1)
    y_parts = [
        _apply_rotary_pos_emb(x_parts[k], cos_parts[k], sin_parts[k], unsqueeze_dim=2)
        for k in range(ndim)
    ]
    return torch.cat(y_parts, dim=-1)


class Gemma4VisionRotaryEmbedding(nn.Module):
    """HF ``Gemma4VisionRotaryEmbedding`` (default RoPE only — vision uses theta=100).

    Computes (cos, sin) per spatial axis and concatenates them so each half of
    the head_dim gets rotated by its own axis.
    """

    inv_freq: torch.Tensor  # for type checkers

    def __init__(self, config: Gemma4VisionConfig):
        super().__init__()
        self.head_dim = config.head_dim
        self.rope_theta = config.rope_theta
        # head_dim is split across 2 spatial axes → spatial_dim = head_dim // 2.
        spatial_dim = self.head_dim // 2
        # range(0, spatial_dim, 2) gives spatial_dim // 2 frequencies.
        # NOTE: HF divides by ``spatial_dim`` (not head_dim) for the exponent.
        inv_freq = 1.0 / (
            self.rope_theta
            ** (
                torch.arange(0, spatial_dim, 2, dtype=torch.int64).float() / spatial_dim
            )
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(
        self,
        x: torch.Tensor,
        position_ids: torch.Tensor,  # [B, P, 2]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        all_cos: list[torch.Tensor] = []
        all_sin: list[torch.Tensor] = []
        # [n_freqs] -> [B, n_freqs, 1]
        inv_freq_expanded = (
            self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1)
        )
        for i in range(2):
            # [B, P]
            dim_position_ids = position_ids[:, :, i]
            # [B, 1, P]
            dim_position_ids_expanded = dim_position_ids[:, None, :].float()
            # [B, n_freqs, P] -> [B, P, n_freqs]
            freqs = (inv_freq_expanded @ dim_position_ids_expanded).transpose(1, 2)
            emb = torch.cat(
                (freqs, freqs), dim=-1
            )  # [B, P, 2*n_freqs] = [B, P, head_dim/2]
            all_cos.append(emb.cos())
            all_sin.append(emb.sin())
        cos = torch.cat(all_cos, dim=-1).to(dtype=x.dtype)  # [B, P, head_dim]
        sin = torch.cat(all_sin, dim=-1).to(dtype=x.dtype)
        return cos, sin


class Gemma4VisionAttention(nn.Module):
    """Multi-head bidirectional attention with QK-norm and 2D RoPE. ``scaling=1``."""

    def __init__(self, config: Gemma4VisionConfig):
        super().__init__()
        self.head_dim = config.head_dim
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.hidden_size = config.hidden_size
        self.scaling = 1.0  # QK-norm absorbs 1/sqrt(d) — identical to HF.

        self.q_proj = nn.Linear(
            self.hidden_size, self.num_heads * self.head_dim, bias=False
        )
        self.k_proj = nn.Linear(
            self.hidden_size, self.num_kv_heads * self.head_dim, bias=False
        )
        self.v_proj = nn.Linear(
            self.hidden_size, self.num_kv_heads * self.head_dim, bias=False
        )
        self.o_proj = nn.Linear(
            self.num_heads * self.head_dim, self.hidden_size, bias=False
        )

        # Q/K-norm have a learnable scale, V-norm does not.
        self.q_norm = nn.RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = nn.RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.v_norm = nn.RMSNorm(
            self.head_dim, eps=config.rms_norm_eps, elementwise_affine=False
        )

    def forward(
        self,
        hidden_states: torch.Tensor,  # [B, P, D]
        cos: torch.Tensor,  # [B, P, head_dim]
        sin: torch.Tensor,  # [B, P, head_dim]
        position_ids: torch.Tensor,  # [B, P, 2]
        attention_mask: torch.Tensor,  # [B, 1, T_q, T_kv] (additive, fp32-able)
    ) -> torch.Tensor:
        B, P, _ = hidden_states.shape

        # Project & reshape to [B, P, H, head_dim]
        q = self.q_proj(hidden_states).view(B, P, self.num_heads, self.head_dim)
        q = self.q_norm(q)
        q = _apply_multidimensional_rope(q, cos, sin, position_ids)
        q = q.transpose(1, 2)  # [B, H, P, head_dim]

        k = self.k_proj(hidden_states).view(B, P, self.num_kv_heads, self.head_dim)
        k = self.k_norm(k)
        k = _apply_multidimensional_rope(k, cos, sin, position_ids)
        k = k.transpose(1, 2)

        v = self.v_proj(hidden_states).view(B, P, self.num_kv_heads, self.head_dim)
        v = self.v_norm(v)
        v = v.transpose(1, 2)

        attn_out = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=attention_mask,
            dropout_p=0.0,
            is_causal=False,
            scale=self.scaling,
            enable_gqa=(self.num_heads != self.num_kv_heads),
        )
        # [B, H, P, head_dim] -> [B, P, H*head_dim]
        attn_out = (
            attn_out.transpose(1, 2)
            .contiguous()
            .view(B, P, self.num_heads * self.head_dim)
        )
        return self.o_proj(attn_out)


# ---------------------------------------------------------------------------
# Encoder layer / encoder
# ---------------------------------------------------------------------------


class Gemma4VisionEncoderLayer(nn.Module):
    """Norm-sandwich encoder block (same skeleton as the LM): pre/post norms
    around both self-attn and the SwiGLU MLP."""

    def __init__(self, config: Gemma4VisionConfig):
        super().__init__()
        self.self_attn = Gemma4VisionAttention(config)
        self.mlp = Gemma4MLP(config.hidden_size, config.intermediate_size)
        self.input_layernorm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = nn.RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.pre_feedforward_layernorm = nn.RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.post_feedforward_layernorm = nn.RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        position_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        residual = hidden_states
        h = self.input_layernorm(hidden_states)
        h = self.self_attn(h, cos, sin, position_ids, attention_mask)
        h = self.post_attention_layernorm(h)
        hidden_states = residual + h

        residual = hidden_states
        h = self.pre_feedforward_layernorm(hidden_states)
        h = self.mlp(h)
        h = self.post_feedforward_layernorm(h)
        hidden_states = residual + h

        return hidden_states


class Gemma4VisionEncoder(nn.Module):
    """Stack of N encoder layers. Builds (cos, sin) once and a bidirectional
    additive attention mask from ``padding_positions`` (True = padding)."""

    def __init__(self, config: Gemma4VisionConfig):
        super().__init__()
        self.rotary_emb = Gemma4VisionRotaryEmbedding(config)
        self.layers = nn.ModuleList(
            [Gemma4VisionEncoderLayer(config) for _ in range(config.num_hidden_layers)]
        )

    def _build_attention_mask(
        self,
        valid_mask: torch.Tensor,  # [B, P], True = valid
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """Bidirectional bool mask: each query attends to all valid key positions.

        Returns a BOOL mask of shape ``[B, 1, P, P]`` where ``True == attend``.

        Shape note: the executorch CUDA-backend triton SDPA kernel
        (``executorch/backends/cuda/triton/kernels/sdpa.py:_prepare_mask_params``)
        rejects masks that are NOT bool or NOT exactly ``[B, 1, L_q, L_kv]``
        (no broadcast over the L_q dim is allowed). We therefore materialize
        the L_q dim with ``expand`` here even though, mathematically, every
        query has the same per-key attendance (bidirectional encoder).

        ``dtype`` is accepted for backwards-API symmetry but is unused — the
        mask is always bool.
        """
        del dtype  # mask is always bool — see docstring
        B, P = valid_mask.shape
        # [B, P] -> [B, 1, 1, P] -> [B, 1, P, P] (materialized, no broadcast).
        kv_valid = valid_mask[:, None, None, :].expand(B, 1, P, P).contiguous()
        return kv_valid.to(torch.bool)

    def forward(
        self,
        inputs_embeds: torch.Tensor,  # [B, P, D]
        valid_mask: torch.Tensor,  # [B, P], True = valid
        pixel_position_ids: torch.Tensor,  # [B, P, 2]
    ) -> torch.Tensor:
        attention_mask = self._build_attention_mask(valid_mask, inputs_embeds.dtype)
        cos, sin = self.rotary_emb(inputs_embeds, pixel_position_ids)
        hidden_states = inputs_embeds
        for layer in self.layers:
            hidden_states = layer(
                hidden_states, cos, sin, pixel_position_ids, attention_mask
            )
        return hidden_states


# ---------------------------------------------------------------------------
# Multimodal embedder (vision-side projection into LM space)
# ---------------------------------------------------------------------------


class Gemma4MultimodalEmbedder(nn.Module):
    """HF ``Gemma4MultimodalEmbedder`` — pre-projection RMSNorm (no scale)
    followed by a single linear projection from vision hidden_size to text
    hidden_size."""

    def __init__(self, vision_config: Gemma4VisionConfig, text_hidden_size: int):
        super().__init__()
        self.embedding_pre_projection_norm = nn.RMSNorm(
            vision_config.hidden_size,
            eps=vision_config.rms_norm_eps,
            elementwise_affine=False,
        )
        self.embedding_projection = nn.Linear(
            vision_config.hidden_size, text_hidden_size, bias=False
        )

    def forward(self, inputs_embeds: torch.Tensor) -> torch.Tensor:
        return self.embedding_projection(
            self.embedding_pre_projection_norm(inputs_embeds)
        )


# ---------------------------------------------------------------------------
# Vision tower (sub-components container, no top-level forward needed)
# ---------------------------------------------------------------------------


class Gemma4VisionTower(nn.Module):
    """Container matching the HF ``Gemma4VisionModel`` structure:
    patch_embedder + encoder + pooler + (std_bias, std_scale) when standardize.

    HF state-dict keys map to this module under the prefix
    ``model.vision_tower.*`` → ``vision_tower.*``.
    """

    std_bias: torch.Tensor
    std_scale: torch.Tensor

    def __init__(self, config: Gemma4VisionConfig):
        super().__init__()
        self.config = config
        self.patch_embedder = Gemma4VisionPatchEmbedder(config)
        self.encoder = Gemma4VisionEncoder(config)
        self.pooler = Gemma4VisionPooler(config)

        if config.standardize:
            # HF stores these as buffers, not parameters.
            self.register_buffer("std_bias", torch.zeros(config.hidden_size))
            self.register_buffer("std_scale", torch.ones(config.hidden_size))


# ---------------------------------------------------------------------------
# Top-level: vision tower + embed_vision wrapper
# ---------------------------------------------------------------------------


class Gemma4_31BVisionTower(nn.Module):
    """End-to-end vision tower: pixels (pre-patchified) → text-space embeddings.

    Combines ``Gemma4VisionTower`` and ``Gemma4MultimodalEmbedder`` and replicates
    the forward path of HF's ``Gemma4VisionModel`` followed by the LM-side
    ``embed_vision``. Returns ``(embeddings, pooler_mask)`` with padding rows
    zeroed (rather than stripped) so the output shape is fixed and exportable.

    Args (forward):
        pixel_values:        ``[B, P, patch_dim]`` (pre-patchified, range [0,1]).
        pixel_position_ids:  ``[B, P, 2]`` (x, y); ``-1`` marks padding.

    Returns:
        embeddings:          ``[B, output_length, text_hidden_size]``
        pooler_mask:         ``[B, output_length]`` (True = valid soft token)
    """

    def __init__(
        self,
        vision_config: Gemma4VisionConfig,
        text_hidden_size: int,
    ):
        super().__init__()
        self.vision_config = vision_config
        self.text_hidden_size = text_hidden_size
        # Flat names match the HF state-dict prefixes "model.vision_tower.*" and
        # "model.embed_vision.*" after the LM-side rename done in model.py.
        self.vision_tower = Gemma4VisionTower(vision_config)
        self.embed_vision = Gemma4MultimodalEmbedder(vision_config, text_hidden_size)

    def forward(
        self,
        pixel_values: torch.Tensor,  # [B, P, patch_dim], dtype matches weights
        pixel_position_ids: torch.Tensor,  # [B, P, 2]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        cfg = self.vision_config
        pks = cfg.pooling_kernel_size
        output_length = pixel_values.shape[1] // (pks * pks)

        padding_positions = (pixel_position_ids == -1).all(dim=-1)  # [B, P]
        valid_mask = ~padding_positions

        inputs_embeds = self.vision_tower.patch_embedder(
            pixel_values, pixel_position_ids, padding_positions
        )
        encoder_out = self.vision_tower.encoder(
            inputs_embeds=inputs_embeds,
            valid_mask=valid_mask,
            pixel_position_ids=pixel_position_ids,
        )
        hidden_states, pooler_mask = self.vision_tower.pooler(
            hidden_states=encoder_out,
            pixel_position_ids=pixel_position_ids,
            padding_positions=padding_positions,
            output_length=output_length,
        )

        if cfg.standardize:
            hidden_states = (
                hidden_states - self.vision_tower.std_bias
            ) * self.vision_tower.std_scale
            # Re-zero padding rows — HF strips them; we keep the shape but mask
            # so embed_vision produces zero rows there (RMSNorm of 0 → 0).
            hidden_states = hidden_states.masked_fill(~pooler_mask.unsqueeze(-1), 0.0)

        embeddings = self.embed_vision(hidden_states)
        return embeddings, pooler_mask


# ---------------------------------------------------------------------------
# HF→our key map for vision_tower + embed_vision
# ---------------------------------------------------------------------------

# Mapping of HF state-dict keys to ours. The 31B checkpoint stores its vision
# linears as ``...<proj>.linear.weight`` because HF wraps each linear in
# ``Gemma4ClippableLinear`` (which exposes ``.linear``). Since the 31B vision
# config has ``use_clipped_linears=False``, the wrapper has no extra params and
# we drop the ``.linear`` segment when remapping into our flat layout.
_HF_VISION_KEY_MAP_FIXED = {
    # vision_tower (top-level constants)
    "model.vision_tower.std_bias": "vision_tower.std_bias",
    "model.vision_tower.std_scale": "vision_tower.std_scale",
    # patch_embedder
    "model.vision_tower.patch_embedder.input_proj.weight": "vision_tower.patch_embedder.input_proj.weight",
    "model.vision_tower.patch_embedder.position_embedding_table": "vision_tower.patch_embedder.position_embedding_table",
    # embed_vision projector
    "model.embed_vision.embedding_projection.weight": "embed_vision.embedding_projection.weight",
}

_HF_VISION_KEY_MAP_PER_LAYER = {
    # norms
    "model.vision_tower.encoder.layers.{}.input_layernorm.weight": "vision_tower.encoder.layers.{}.input_layernorm.weight",
    "model.vision_tower.encoder.layers.{}.post_attention_layernorm.weight": "vision_tower.encoder.layers.{}.post_attention_layernorm.weight",
    "model.vision_tower.encoder.layers.{}.pre_feedforward_layernorm.weight": "vision_tower.encoder.layers.{}.pre_feedforward_layernorm.weight",
    "model.vision_tower.encoder.layers.{}.post_feedforward_layernorm.weight": "vision_tower.encoder.layers.{}.post_feedforward_layernorm.weight",
    # attention projections (.linear segment dropped)
    "model.vision_tower.encoder.layers.{}.self_attn.q_proj.linear.weight": "vision_tower.encoder.layers.{}.self_attn.q_proj.weight",
    "model.vision_tower.encoder.layers.{}.self_attn.k_proj.linear.weight": "vision_tower.encoder.layers.{}.self_attn.k_proj.weight",
    "model.vision_tower.encoder.layers.{}.self_attn.v_proj.linear.weight": "vision_tower.encoder.layers.{}.self_attn.v_proj.weight",
    "model.vision_tower.encoder.layers.{}.self_attn.o_proj.linear.weight": "vision_tower.encoder.layers.{}.self_attn.o_proj.weight",
    # qk-norm
    "model.vision_tower.encoder.layers.{}.self_attn.q_norm.weight": "vision_tower.encoder.layers.{}.self_attn.q_norm.weight",
    "model.vision_tower.encoder.layers.{}.self_attn.k_norm.weight": "vision_tower.encoder.layers.{}.self_attn.k_norm.weight",
    # mlp (.linear segment dropped)
    "model.vision_tower.encoder.layers.{}.mlp.gate_proj.linear.weight": "vision_tower.encoder.layers.{}.mlp.gate_proj.weight",
    "model.vision_tower.encoder.layers.{}.mlp.up_proj.linear.weight": "vision_tower.encoder.layers.{}.mlp.up_proj.weight",
    "model.vision_tower.encoder.layers.{}.mlp.down_proj.linear.weight": "vision_tower.encoder.layers.{}.mlp.down_proj.weight",
}


def hf_vision_key_map() -> dict[str, str]:
    """Return the fixed (non-per-layer) part of the HF→our key map.

    Per-layer patterns are returned via ``hf_vision_per_layer_key_map()``;
    callers expand the ``{}`` placeholder with the layer index.
    """
    return dict(_HF_VISION_KEY_MAP_FIXED)


def hf_vision_per_layer_key_map() -> dict[str, str]:
    return dict(_HF_VISION_KEY_MAP_PER_LAYER)


__all__ = [
    "Gemma4VisionConfig",
    "Gemma4VisionPatchEmbedder",
    "Gemma4VisionPooler",
    "Gemma4VisionAttention",
    "Gemma4VisionRotaryEmbedding",
    "Gemma4VisionEncoderLayer",
    "Gemma4VisionEncoder",
    "Gemma4MultimodalEmbedder",
    "Gemma4VisionTower",
    "Gemma4_31BVisionTower",
    "hf_vision_key_map",
    "hf_vision_per_layer_key_map",
]
