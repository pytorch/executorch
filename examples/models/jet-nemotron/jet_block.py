# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# Based on NVIDIA's Jet-Nemotron JetBlock implementation
# Original: https://github.com/NVIDIA/Jet-Nemotron
# Licensed under Apache License 2.0

"""
JetBlock: A Gated Delta Rule based attention mechanism.

JetBlock implements efficient linear attention using the Gated Delta Rule,
which maintains a recurrent state matrix that enables O(1) memory inference.
This is particularly useful for long-context scenarios on edge devices.

Key features:
- Gated delta rule for efficient recurrent attention
- Dynamic short convolution for local context
- No quadratic attention complexity
- O(1) memory during inference (when using KV cache)
"""

from __future__ import annotations

import math
from collections import OrderedDict
from dataclasses import dataclass
from typing import Callable, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


# ==============================================================================
# Configuration
# ==============================================================================


@dataclass
class JetBlockConfig:
    """Configuration for JetBlock."""

    mode: str = "chunk"
    expand_v: float = 2.0
    num_heads: int = 6
    head_dim: int = 256
    norm_eps: float = 1e-5
    conv_size: int = 4
    dconv_generator_reduction: int = 8


# ==============================================================================
# Utility Functions
# ==============================================================================


def l2_normalize(x: torch.Tensor, dim: int = -1, eps: float = 1e-12) -> torch.Tensor:
    """L2 normalize a tensor along a dimension."""
    return F.normalize(x, p=2, dim=dim, eps=eps)


# ==============================================================================
# FusedRMSNormGated
# ==============================================================================


class FusedRMSNormGated(nn.Module):
    """
    Fused RMSNorm with gating mechanism.
    Pure PyTorch implementation.

    This applies RMSNorm to the input and then multiplies by a gating signal.
    """

    def __init__(
        self,
        hidden_size: int,
        eps: float = 1e-6,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(hidden_size))

    def forward(self, x: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor [..., hidden_size]
            g: Gate tensor [..., hidden_size]

        Returns:
            Output tensor [..., hidden_size]
        """
        input_dtype = x.dtype
        x = x.float()
        variance = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.eps)
        x = x.to(input_dtype)
        return self.weight * x * F.silu(g)


# ==============================================================================
# Dynamic Short Convolution
# ==============================================================================


class DynamicShortConvolution(nn.Module):
    """
    Dynamic short convolution with learned kernels.
    Generates position-dependent convolution kernels for local context mixing.
    """

    def __init__(
        self,
        hidden_size: int,
        kernel_size: int,
        generator_input_size: Optional[int] = None,
        generator_reduction: Optional[int] = None,
        activation: Optional[str] = "silu",
        static_conv_init: Optional[Callable] = None,
    ) -> None:
        super().__init__()

        self.hidden_size = hidden_size
        self.generator_input_size = (
            hidden_size if generator_input_size is None else generator_input_size
        )
        self.generator_hidden_size = (
            hidden_size
            if generator_reduction is None
            else (hidden_size // generator_reduction)
        )
        self.kernel_size = kernel_size
        self.activation = activation
        self.static_conv_init = static_conv_init

        self.kernel_generator = nn.Sequential(
            OrderedDict(
                [
                    (
                        "w1",
                        nn.Linear(
                            self.generator_input_size,
                            self.generator_hidden_size,
                            bias=False,
                        ),
                    ),
                    ("act", nn.SiLU()),
                    (
                        "w2",
                        nn.Linear(
                            self.generator_hidden_size,
                            self.hidden_size * self.kernel_size,
                            bias=True,
                        ),
                    ),
                ]
            )
        )
        self._init_kernel_generator()

    def _init_kernel_generator(self):
        """Initialize the kernel generator."""
        for layer in self.kernel_generator:
            if isinstance(layer, nn.Linear):
                layer.weight.data.zero_()
                if layer.bias is not None:
                    layer.bias.data.zero_()

        if self.static_conv_init is not None:
            self.static_conv_init(self.kernel_generator.w2.bias)

    def get_kernel(self, x: torch.Tensor) -> torch.Tensor:
        flat_kernels = self.kernel_generator(x)
        if flat_kernels.dim() == 3:
            kernels = rearrange(
                flat_kernels, "b t (d w) -> b t d w", w=self.kernel_size
            )
        elif flat_kernels.dim() == 2:
            kernels = rearrange(flat_kernels, "b (d w) -> b d w", w=self.kernel_size)
        else:
            raise ValueError(f"Invalid kernel shape: {flat_kernels.shape}")
        return kernels

    def forward(
        self,
        x: torch.Tensor,
        generator_input: Optional[torch.Tensor] = None,
        cache: Optional[torch.Tensor] = None,
        output_final_state: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Args:
            x: Input tensor [B, T, D]
            generator_input: Optional input for kernel generation
            cache: Optional cache tensor [N, D, W]
            output_final_state: Whether to output final state

        Returns:
            Output tensor [B, T, D] and optional cache
        """
        B, T, D, W = *x.shape, self.kernel_size
        N = B
        input_dtype = x.dtype

        # During decoding phase with cache
        if cache is not None and B * T == N:
            assert T == 1
            x, cache = self._step_naive(x, cache, generator_input=generator_input)
            return x, cache

        if output_final_state:
            new_cache = rearrange(x[..., -min(W, T) :, :], "n w d -> n d w")
        else:
            new_cache = None

        x = self._forward_naive(x, generator_input=generator_input)

        if self.activation is not None:
            x = F.silu(x)

        x = x.to(input_dtype)
        if output_final_state:
            if cache is None:
                cache = x.new_zeros(N, D, W)
            cache[:, :, -min(W, T) :].copy_(new_cache)

        return x, cache

    def _forward_naive(
        self, x: torch.Tensor, generator_input: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        W = self.kernel_size
        generator_input = x if generator_input is None else generator_input
        kernels = self.get_kernel(generator_input)
        x = F.pad(x.transpose(1, 2), (W - 1, 0))  # [B, D, T+W-1]
        x = x.unfold(dimension=2, size=W, step=1)  # [B, D, T, W]
        x = x.permute(0, 2, 1, 3)  # [B, T, D, W]
        x = (x * kernels).sum(dim=-1)  # [B, T, D]
        return x

    def _step_naive(
        self,
        x: torch.Tensor,
        cache: torch.Tensor,
        generator_input: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        assert x.shape[1] == 1, "x must be of shape [B, 1, D]"
        shape = x.shape
        generator_input = x if generator_input is None else generator_input
        x = x.squeeze(1)
        generator_input = generator_input.squeeze(1)
        B, D, W = *x.shape, self.kernel_size

        cache.copy_(cache.roll(shifts=-1, dims=-1))
        cache[:, :, -1] = x

        kernels = self.get_kernel(generator_input)
        x = torch.sum(cache * kernels, dim=-1)

        if self.activation is not None:
            x = F.silu(x)

        return x.view(shape), cache


# ==============================================================================
# Gated Delta Rule Core
# ==============================================================================


def _gated_delta_rule_step(
    q_t: torch.Tensor,
    k_t: torch.Tensor,
    v_t: torch.Tensor,
    g_t: torch.Tensor,
    beta_t: torch.Tensor,
    state: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Single step of gated delta rule.

    Args:
        q_t: Query at timestep t [batch, num_heads, head_k_dim]
        k_t: Key at timestep t [batch, num_heads, head_k_dim]
        v_t: Value at timestep t [batch, num_heads, head_v_dim]
        g_t: Decay gate at timestep t [batch, num_heads]
        beta_t: Beta at timestep t [batch, num_heads]
        state: Current state [batch, num_heads, head_k_dim, head_v_dim]

    Returns:
        o_t: Output at timestep t [batch, num_heads, head_v_dim]
        new_state: Updated state [batch, num_heads, head_k_dim, head_v_dim]
    """
    # Compute output: o_t = S^T @ q_t
    o_t = torch.einsum("bnkv,bnk->bnv", state, q_t)

    # Compute delta: delta_t = v_t - S^T @ k_t
    Sk = torch.einsum("bnkv,bnk->bnv", state, k_t)
    delta_t = v_t - Sk

    # Update state: S_t = g_t * S_{t-1} + beta_t * outer(k_t, delta_t)
    g_t_expanded = g_t.unsqueeze(-1).unsqueeze(-1)
    beta_t_expanded = beta_t.unsqueeze(-1).unsqueeze(-1)
    outer_kd = torch.einsum("bnk,bnv->bnkv", k_t, delta_t)
    new_state = g_t_expanded * state + beta_t_expanded * outer_kd

    return o_t, new_state


def _gated_delta_rule_sequential(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    decay: torch.Tensor,
    beta: torch.Tensor,
    state: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Sequential implementation using unbind."""
    q_steps = torch.unbind(q, dim=1)
    k_steps = torch.unbind(k, dim=1)
    v_steps = torch.unbind(v, dim=1)
    decay_steps = torch.unbind(decay, dim=1)
    beta_steps = torch.unbind(beta, dim=1)

    outputs = []
    for q_t, k_t, v_t, g_t, beta_t in zip(
        q_steps, k_steps, v_steps, decay_steps, beta_steps
    ):
        o_t, state = _gated_delta_rule_step(q_t, k_t, v_t, g_t, beta_t, state)
        outputs.append(o_t)

    output = torch.stack(outputs, dim=1)
    return output, state


def gated_delta_rule(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    initial_state: Optional[torch.Tensor] = None,
    output_final_state: bool = False,
    use_qk_l2norm: bool = False,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """
    Gated delta rule attention.

    The delta rule updates a state matrix S with:
        S_t = g_t * S_{t-1} + beta_t * k_t^T * (v_t - S_{t-1}^T * k_t)
        o_t = S_t^T * q_t

    Args:
        q: Query tensor [batch, seq_len, num_heads, head_dim]
        k: Key tensor [batch, seq_len, num_heads, head_dim]
        v: Value tensor [batch, seq_len, num_heads, head_v_dim]
        g: Gate tensor [batch, seq_len, num_heads] (log-space decay rates)
        beta: Beta tensor [batch, seq_len, num_heads]
        initial_state: Initial recurrent state [batch, num_heads, head_dim, head_v_dim]
        output_final_state: Whether to return final state
        use_qk_l2norm: Whether to L2 normalize q and k

    Returns:
        output: Output tensor [batch, seq_len, num_heads, head_v_dim]
        final_state: Final recurrent state (if output_final_state=True)
    """
    orig_shape = q.shape
    if q.dim() == 4:
        batch_size, seq_len, num_heads, head_k_dim = q.shape
        head_v_dim = v.shape[-1]
    else:
        batch_size = 1
        seq_len, num_heads, head_k_dim = q.shape[-3:]
        head_v_dim = v.shape[-1]
        q = q.view(batch_size, seq_len, num_heads, head_k_dim)
        k = k.view(batch_size, seq_len, num_heads, head_k_dim)
        v = v.view(batch_size, seq_len, num_heads, head_v_dim)
        g = g.view(batch_size, seq_len, num_heads)
        beta = beta.view(batch_size, seq_len, num_heads)

    # L2 normalize q and k if requested
    if use_qk_l2norm:
        q = l2_normalize(q, dim=-1)
        k = l2_normalize(k, dim=-1)

    # Compute decay from log-space gate
    decay = torch.exp(g)

    # Initialize state
    if initial_state is not None:
        state = initial_state.clone()
    else:
        state = q.new_zeros(batch_size, num_heads, head_k_dim, head_v_dim)

    # Sequential processing
    output, state = _gated_delta_rule_sequential(
        q, k, v, decay, beta, state
    )

    # Reshape output to match input shape
    if len(orig_shape) == 4:
        output = output.view(orig_shape[0], orig_shape[1], num_heads, head_v_dim)
    else:
        output = output.view(*orig_shape[:-1], head_v_dim)

    final_state = state if output_final_state else None
    return output, final_state


# ==============================================================================
# JetBlock Module
# ==============================================================================


def _init_linear_conv1d(
    weight: torch.Tensor, std: float, bias: Optional[torch.Tensor] = None
) -> None:
    weight.data.normal_(mean=0.0, std=std)
    if bias is not None:
        if not getattr(bias, "_no_reinit", False):
            nn.init.zeros_(bias)


class JetBlock(nn.Module):
    """
    JetBlock: Gated Delta Rule based attention mechanism.

    This implements efficient linear attention using the Gated Delta Rule,
    which maintains a recurrent state matrix that enables O(1) memory inference.

    Args:
        hidden_size: Input/output dimension
        config: JetBlockConfig with hyperparameters
        layer_idx: Optional layer index for caching
        initializer_range: Standard deviation for initialization
    """

    def __init__(
        self,
        hidden_size: int,
        config: Optional[JetBlockConfig] = None,
        layer_idx: Optional[int] = None,
        initializer_range: float = 0.02,
    ) -> None:
        super().__init__()

        if config is None:
            config = JetBlockConfig()

        self.mode = config.mode
        self.hidden_size = hidden_size
        self.expand_v = config.expand_v
        self.conv_size = config.conv_size
        self.head_dim = config.head_dim
        self.num_heads = config.num_heads
        self.layer_idx = layer_idx

        self.key_dim = int(self.num_heads * self.head_dim)
        self.value_dim = int(self.key_dim * self.expand_v)
        self.head_k_dim = config.head_dim
        self.head_v_dim = int(config.head_dim * self.expand_v)

        # Consistency check
        if not math.isclose(self.key_dim * self.expand_v, self.value_dim, rel_tol=1e-5):
            raise ValueError(
                f"expand_v={self.expand_v} does not produce an integer value "
                f"when multiplied by key_dim={self.key_dim}."
            )

        assert self.mode in ["chunk", "fused_recurrent"], (
            f"Unsupported mode: {config.mode}"
        )

        # Projections
        self.q_proj = nn.Linear(hidden_size, self.key_dim, bias=False)
        self.k_proj = nn.Linear(hidden_size, self.key_dim, bias=False)
        self.v_proj = nn.Linear(hidden_size, self.value_dim, bias=False)
        self.b_proj = nn.Linear(hidden_size, self.num_heads, bias=False)
        self.a_proj = nn.Linear(hidden_size, self.num_heads, bias=False)

        # Decay parameters
        A = torch.empty(self.num_heads, dtype=torch.float32).uniform_(0, 16)
        self.A_log = nn.Parameter(torch.log(A))
        self.A_log._no_weight_decay = True

        # dt initialization
        dt_min = 0.001
        dt_max = 0.1
        dt_init_floor = 1e-4
        dt = torch.exp(
            torch.rand(self.num_heads) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        )
        dt = torch.clamp(dt, min=dt_init_floor)
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        self.dt_bias = nn.Parameter(inv_dt)
        self.dt_bias._no_weight_decay = True

        # Dynamic convolution
        self.dynamic_conv1d = DynamicShortConvolution(
            hidden_size=self.value_dim,
            kernel_size=self.conv_size,
            generator_input_size=self.hidden_size,
            generator_reduction=config.dconv_generator_reduction,
            static_conv_init=lambda x: _init_linear_conv1d(x, std=initializer_range),
        )

        # Output projections
        self.g_proj = nn.Linear(hidden_size, self.value_dim, bias=False)
        self.o_norm = FusedRMSNormGated(
            self.head_v_dim,
            eps=float(config.norm_eps),
        )
        self.o_proj = nn.Linear(self.value_dim, hidden_size, bias=False)

    def forward(
        self,
        hidden_states: torch.Tensor,
        recurrent_state: Optional[torch.Tensor] = None,
        conv_state: Optional[torch.Tensor] = None,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Forward pass of JetBlock.

        Args:
            hidden_states: Input tensor [batch, seq_len, hidden_size]
            recurrent_state: Optional recurrent state [batch, num_heads, head_k_dim, head_v_dim]
            conv_state: Optional convolution cache [batch, value_dim, kernel_size]
            use_cache: Whether to return updated states

        Returns:
            output: Output tensor [batch, seq_len, hidden_size]
            recurrent_state: Updated recurrent state (if use_cache=True)
            conv_state: Updated convolution state (if use_cache=True)
        """
        batch_size, q_len, _ = hidden_states.shape

        # Use fused_recurrent for short sequences
        mode = "fused_recurrent" if q_len <= 64 else self.mode

        # Compute projections
        q = F.silu(self.q_proj(hidden_states))
        k = F.silu(self.k_proj(hidden_states))

        # Dynamic convolution on values
        v, conv_state = self.dynamic_conv1d(
            x=self.v_proj(hidden_states),
            generator_input=hidden_states,
            cache=conv_state,
            output_final_state=use_cache,
        )

        # Reshape for multi-head attention
        q, k = map(
            lambda x: rearrange(x, "... (h d) -> ... h d", d=self.head_k_dim), (q, k)
        )
        v = rearrange(v, "... (h d) -> ... h d", d=self.head_v_dim)

        # Compute beta and gating
        beta = self.b_proj(hidden_states).sigmoid()
        g = -self.A_log.float().exp() * F.softplus(
            self.a_proj(hidden_states).float() + self.dt_bias
        )

        # Apply gated delta rule
        o, recurrent_state = gated_delta_rule(
            q=q,
            k=k,
            v=v,
            g=g,
            beta=beta,
            initial_state=recurrent_state,
            output_final_state=use_cache,
            use_qk_l2norm=True,
        )

        # Output projection with gated norm
        g = rearrange(
            self.g_proj(hidden_states), "... (h d) -> ... h d", d=self.head_v_dim
        )
        o = self.o_norm(o, g)
        o = rearrange(o, "b t h d -> b t (h d)")
        o = self.o_proj(o)

        if use_cache:
            return o, recurrent_state, conv_state
        return o, None, None
