# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Gated DeltaNet decode cases, from transformers' own implementation: the
recurrent rule with the state as an explicit input and output, and the whole
Qwen3.5 GDN layer over a linear-attention cache whose conv and recurrent states
are mutable buffers carried across a chain's calls."""

import torch
import torch.nn.functional as F
from executorch.backends.native.test.samples.cases import Case, draws
from transformers.cache_utils import Cache, LinearAttentionLayer
from transformers.models.qwen3_5 import modeling_qwen3_5 as qwen3_5
from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig


class _Rule(torch.nn.Module):
    """The recurrent rule as Qwen3_5GatedDeltaNet calls it: decay and beta from
    the layer's parameters, q and k L2-normalized in the kernel."""

    def __init__(self, heads: int) -> None:
        super().__init__()
        self.A_log = torch.nn.Parameter(torch.empty(heads).uniform_(0.01, 16).log())
        self.dt_bias = torch.nn.Parameter(torch.rand(heads))

    def forward(self, q, k, v, a, b, state):
        g = -self.A_log.float().exp() * F.softplus(a.float() + self.dt_bias)
        out, state = qwen3_5.torch_recurrent_gated_delta_rule(
            q,
            k,
            v,
            g=g,
            beta=b.sigmoid(),
            initial_state=state,
            output_final_state=True,
            use_qk_l2norm_in_kernel=True,
        )
        # The graph produces the state first; to_native wants that order.
        return state, out


def _rule_case(name: str, heads: int, dim: int) -> Case:
    qkv = (1, 1, heads, dim)
    gates = (1, 1, heads)
    return Case(
        name, _Rule(heads), draws(2, qkv, qkv, qkv, gates, gates, (1, heads, dim, dim))
    )


def _config(hidden: int, k_heads: int, v_heads: int, dim: int) -> Qwen3_5TextConfig:
    return Qwen3_5TextConfig(
        hidden_size=hidden,
        num_hidden_layers=1,
        layer_types=["linear_attention"],
        linear_num_key_heads=k_heads,
        linear_num_value_heads=v_heads,
        linear_key_head_dim=dim,
        linear_value_head_dim=dim,
        linear_conv_kernel_dim=4,
    )


class _Layer(torch.nn.Module):
    """transformers' Qwen3_5GatedDeltaNet over a LinearAttentionLayer cache
    whose states are this module's buffers, continued by every call and zeroed
    in the graph by a call at position 0, as the LLM exporter does."""

    def __init__(self, config: Qwen3_5TextConfig, persistent: bool = False) -> None:
        super().__init__()
        self.gdn = qwen3_5.Qwen3_5GatedDeltaNet(config, layer_idx=0)
        torch.nn.init.normal_(self.gdn.norm.weight, 1.0, 0.1)
        self.cache = Cache(layers=[LinearAttentionLayer()])
        conv = torch.zeros(1, self.gdn.conv_dim, self.gdn.conv_kernel_size)
        recurrent = torch.zeros(
            1, self.gdn.num_v_heads, self.gdn.head_k_dim, self.gdn.head_v_dim
        )
        self.register_buffer("conv_state", conv, persistent=persistent)
        self.register_buffer("recurrent_state", recurrent, persistent=persistent)

    def forward(self, x, position):
        keep = (position[:1] != 0).to(self.conv_state.dtype)
        self.conv_state.mul_(keep)
        self.recurrent_state.mul_(keep)
        layer = self.cache.layers[0]
        layer.lazy_initialization(
            conv_states=self.conv_state, recurrent_states=self.recurrent_state
        )
        # Bind the buffers themselves, which lazy_initialization copies.
        layer.conv_states[0] = self.conv_state
        layer.recurrent_states[0] = self.recurrent_state
        layer.has_previous_state[0] = True
        return self.gdn(x, cache_params=self.cache)


def _layer_case(
    name: str,
    config: Qwen3_5TextConfig,
    calls: int = 5,
    restart: bool = False,
    persistent: bool = False,
) -> Case:
    """A chain of decode calls, one token each, carrying the state; with
    `restart`, the last one starts a new sequence at position 0."""
    xs = draws(calls, (1, 1, config.hidden_size))
    starts = list(range(calls))
    if restart:
        starts[-1] = 0
    inputs = [(x, torch.tensor([s])) for (x,), s in zip(xs, starts)]
    return Case(name, _Layer(config, persistent), inputs, dynamic_shapes=({}, {}))


def cases() -> list[Case]:
    torch.manual_seed(0)
    small = _config(hidden=64, k_heads=2, v_heads=4, dim=16)
    # Every linear's in_features is a multiple of 256, so GGUF's K-quants fit.
    wide = _config(hidden=256, k_heads=2, v_heads=4, dim=64)
    return [
        _rule_case("recurrent_small", heads=4, dim=16),
        # Qwen3.5's 32 value heads of 128 after the key heads' repeat.
        _rule_case("recurrent_qwen35", heads=32, dim=128),
        # The layer's cases hold its nn.Linears, so they are named quantized_*.
        _layer_case("quantized_layer_decode", small),
        _layer_case("quantized_layer_decode_restart", small, restart=True),
        # State as saved (data-backed) buffers rather than zero-initialized ones.
        _layer_case("quantized_layer_decode_saved_state", small, persistent=True),
        # Enough calls for the activation quantization's error biases to fit.
        _layer_case("quantized_layer_decode_wide", wide, calls=16),
    ]
