# Copyright (c) Qualcomm Innovation Center, Inc.
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# TODO: reenable pyre after fixing the issues
# pyre-ignore-all-errors

import math
from typing import List

import scipy
import torch
import torch.nn as nn

from executorch.examples.models.llama.model_args import ModelArgs

from executorch.examples.models.llama.rope import (
    hf_precompute_freqs_cis,
    precompute_freqs_cis,
)

from .rope import ROPE_REGISTRY


class AttentionSinkRope(nn.Module):
    def __init__(
        self,
        config: ModelArgs,
        sink_size: int,
        eviction_batch_size: int,
        ar_len: int,
        **kwargs,
    ):
        super().__init__()
        self.config = config
        self.sink_size = sink_size
        self.eviction_batch_size = eviction_batch_size
        self.n_layers = config.n_layers
        self.original_position = eviction_batch_size + sink_size
        self.new_position = sink_size
        self.num_to_keep = (
            config.max_context_len - sink_size - eviction_batch_size - ar_len
        )
        self.evict_k_cache_shape = [
            config.max_batch_size,
            config.n_kv_heads,
            config.head_dim,
            eviction_batch_size,
        ]
        self.evict_v_cache_shape = [
            config.max_batch_size,
            config.n_kv_heads,
            eviction_batch_size,
            config.head_dim,
        ]
        self.kv_cache_shape = {
            # single head, k input
            "k": (
                config.max_batch_size,
                config.n_kv_heads,
                config.head_dim,
                config.max_context_len - ar_len,
            ),
            # single head, v input
            "v": (
                config.max_batch_size,
                config.n_kv_heads,
                config.max_context_len - ar_len,
                config.head_dim,
            ),
        }

        if getattr(config, "enable_r3", False):
            self.register_buffer(
                "r3_weight",
                torch.tensor(
                    scipy.linalg.hadamard(config.head_dim, dtype=float)
                    / math.sqrt(config.head_dim),
                    dtype=torch.float32,
                    device="cpu",
                ),
                persistent=False,
            )

        if config.partial_rotary_factor < 1:
            self.apply_rope_emb = ROPE_REGISTRY["partial"]
        else:
            self.apply_rope_emb = ROPE_REGISTRY["default"]

        if config.use_hf_rope:
            freqs_cos, freqs_sin = hf_precompute_freqs_cis(
                config.head_dim,
                config.max_context_len,
                config.rope_freq_base,
                config.partial_rotary_factor,
            )
            freqs_cos = freqs_cos[:, : freqs_cos.shape[-1] // 2]
            freqs_sin = freqs_sin[:, : freqs_sin.shape[-1] // 2]
        else:
            freqs_cos, freqs_sin = precompute_freqs_cis(
                config.head_dim,
                config.max_context_len,
                config.rope_freq_base,
                config.use_scaled_rope,
                config.rope_scale_factor,
            )
        original_freqs_cos = freqs_cos.narrow(
            0, self.original_position, self.num_to_keep
        )
        original_freqs_sin = freqs_sin.narrow(
            0, self.original_position, self.num_to_keep
        )
        new_freqs_cos = freqs_cos.narrow(0, self.new_position, self.num_to_keep)
        new_freqs_sin = freqs_sin.narrow(0, self.new_position, self.num_to_keep)
        rerotation_cos = (
            new_freqs_cos * original_freqs_cos + new_freqs_sin * original_freqs_sin
        )
        rerotation_sin = (
            new_freqs_sin * original_freqs_cos - new_freqs_cos * original_freqs_sin
        )
        self.register_buffer("rerotation_cos", rerotation_cos, persistent=False)
        self.register_buffer("rerotation_sin", rerotation_sin, persistent=False)

        self.sliding_window = kwargs.get("sliding_window", False)
        if self.sliding_window:
            # Get attention type for each layer
            self.layer_types = kwargs["layer_types"]
            # Get local freq base for sliding attention
            rope_freq_base = kwargs["rope_local_base_freq"]
            local_freqs_cos, local_freqs_sin = hf_precompute_freqs_cis(
                config.head_dim,
                config.max_context_len,
                rope_freq_base,
                config.partial_rotary_factor,
            )
            local_freqs_cos = local_freqs_cos[:, : local_freqs_cos.shape[-1] // 2]
            local_freqs_sin = local_freqs_sin[:, : local_freqs_sin.shape[-1] // 2]
            local_original_freqs_cos = local_freqs_cos.narrow(
                0, self.original_position, self.num_to_keep
            )
            local_original_freqs_sin = local_freqs_sin.narrow(
                0, self.original_position, self.num_to_keep
            )
            local_new_freqs_cos = local_freqs_cos.narrow(
                0, self.new_position, self.num_to_keep
            )
            local_new_freqs_sin = local_freqs_sin.narrow(
                0, self.new_position, self.num_to_keep
            )
            local_rerotation_cos = (
                local_new_freqs_cos * local_original_freqs_cos
                + local_new_freqs_sin * local_original_freqs_sin
            )
            local_rerotation_sin = (
                local_new_freqs_sin * local_original_freqs_cos
                - local_new_freqs_cos * local_original_freqs_sin
            )
            self.register_buffer(
                "local_rerotation_cos", local_rerotation_cos, persistent=False
            )
            self.register_buffer(
                "local_rerotation_sin", local_rerotation_sin, persistent=False
            )

    def forward(self, k_caches: List[torch.Tensor], v_caches: List[torch.Tensor]):
        """
        Rerotate k_cache from original_position to new_position, and return the kv cache after eviction. This is done by rerotating
        k_cache with (new_position * theta - original_position * theta) with the following matrix:
        (cos(delta), -sin(delta)
         sin(delta), cos(delta))
         where delta = new_position * theta - original_position * theta

         Based on https://github.com/huggingface/transformers/pull/26681
        """

        output_k_caches, output_v_caches = [], []
        for ind, (k_cache, v_cache) in enumerate(zip(k_caches, v_caches)):
            # k_cache: (batch_size, n_kv_heads, head_dim, seq_len)
            # v_cache: (batch_size, n_kv_heads, seq_len, head_dim)
            k_dim_to_slice = 3
            v_dim_to_slice = 2

            k_to_keep = k_cache.narrow(
                k_dim_to_slice,
                self.original_position,
                self.num_to_keep,
            )
            k_to_keep = k_to_keep.transpose(2, 3)
            if getattr(self.config, "enable_r3", False):
                # We need to revert the key from spin quant before applying RoPE
                k_to_keep = torch.matmul(k_to_keep, self.r3_weight.T)

            if self.sliding_window and self.layer_types[ind] == "sliding_attention":
                k_to_keep = self.apply_rope_emb(
                    k_to_keep, self.local_rerotation_cos, self.local_rerotation_sin
                )
            else:
                k_to_keep = self.apply_rope_emb(
                    k_to_keep, self.rerotation_cos, self.rerotation_sin
                )
            if getattr(self.config, "enable_r3", False):
                k_to_keep = torch.matmul(k_to_keep, self.r3_weight)
            k_to_keep = k_to_keep.transpose(2, 3)
            new_k_cache = torch.cat(
                [
                    k_cache.narrow(k_dim_to_slice, 0, self.sink_size),
                    k_to_keep,
                    torch.zeros(self.evict_k_cache_shape),
                ],
                dim=k_dim_to_slice,
            )

            new_v_cache = torch.cat(
                [
                    v_cache.narrow(v_dim_to_slice, 0, self.sink_size),
                    v_cache.narrow(
                        v_dim_to_slice,
                        self.original_position,
                        self.num_to_keep,
                    ),
                    torch.zeros(self.evict_v_cache_shape),
                ],
                dim=v_dim_to_slice,
            )

            output_k_caches.append(new_k_cache)
            output_v_caches.append(new_v_cache)

        return output_k_caches, output_v_caches

    def get_example_inputs(self):
        k_cache, v_cache = [], []

        for _ in range(self.n_layers):
            k_cache.append(torch.zeros(self.kv_cache_shape["k"]))
            v_cache.append(torch.zeros(self.kv_cache_shape["v"]))
        return k_cache, v_cache

    def get_metadata(self):
        return {
            "get_eviction_batch_size": self.eviction_batch_size,
            "get_max_context_len": self.config.max_context_len,
            "get_sink_size": self.sink_size,
        }
