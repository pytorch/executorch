# Copyright (c) Qualcomm Innovation Center, Inc.
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import List, Optional, Tuple

import torch

from executorch.examples.models.gemma4.text_decoder.gemma4_config import Gemma4Config
from executorch.examples.qualcomm.oss_scripts.gemma4.text_decoder.text_model import (
    Gemma4TextModel,
)
from executorch.examples.qualcomm.oss_scripts.llama.masking_utils import (
    AttentionMask,
    CausalAttentionMask,
    SlidingWindowAttentionMask,
)
from torch import nn


def _precompute_freqs(
    head_dim: int,
    max_context_len: int,
    rope_theta: float,
    partial_rotary_factor: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """HuggingFace-style RoPE cos/sin with zero-padding for partial RoPE.

    Returns (freqs_cos, freqs_sin) each of shape [max_context_len, head_dim // 2].
    Non-rotated dims are zero-padded so ROTARY_EMB_REGISTRY["partial"] handles
    the half-and-half split correctly.
    """
    rotary_dim = int(head_dim * partial_rotary_factor)
    half_rot = rotary_dim // 2
    inv_freq_rot = 1.0 / (
        rope_theta ** (torch.arange(0, rotary_dim, 2).float() / head_dim)
    )
    total_half = head_dim // 2
    nope = total_half - half_rot
    inv_freq = (
        torch.cat([inv_freq_rot, torch.zeros(nope)]) if nope > 0 else inv_freq_rot
    )
    t = torch.arange(max_context_len, dtype=torch.float32)
    freqs = torch.outer(t, inv_freq)
    return torch.cos(freqs), torch.sin(freqs)


class Gemma4TextModelWrapper(nn.Module):
    """QNN-export-ready Gemma 4 text decoder (static_llama.py convention).

    KV-cache I/O covers only the 15 self-decoder layers.  Each of these layers
    has its own head_dim (full-attention=512, sliding=256), so cache shapes
    are heterogeneous.

    forward(tokens, atten_mask, window_atten_mask, input_pos,
            *k_caches[0..14], *v_caches[0..14])
        -> (logits, out_k_caches, out_v_caches)
    """

    def __init__(
        self,
        config: Gemma4Config,
        ar_len: int = 1,
        output_new_cache_only: bool = True,
        output_cache: bool = True,
        use_i64_token: bool = False,
        **kwargs,
    ):
        super().__init__()
        self.config = config
        self.ar_len = ar_len
        self.use_kv_cache = config.use_kv_cache
        self.output_cache = output_cache
        self.use_i64_token = use_i64_token
        self.n_self_layers = config.num_self_decoder_layers
        self.max_batch_size = config.max_batch_size
        self.vocab_size = config.vocab_size
        self.kv_io_bit_width = kwargs.get("kv_io_bit_width", 32)

        self.model = Gemma4TextModel(
            config,
            output_new_cache_only,
            kwargs.get("enable_masked_softmax", False),
        )

        # Precompute RoPE buffers.
        # Global: full-attention layers, partial_rotary_factor=0.25, head_dim=512.
        # Local:  sliding-attention layers, full rotation, head_dim=256.
        global_cos, global_sin = _precompute_freqs(
            config.global_head_dim,
            config.max_context_len,
            config.rope_theta,
            config.partial_rotary_factor,
        )
        local_cos, local_sin = _precompute_freqs(
            config.head_dim,
            config.max_context_len,
            config.rope_local_base_freq,
            1.0,
        )
        self.register_buffer("global_freqs_cos", global_cos, persistent=False)
        self.register_buffer("global_freqs_sin", global_sin, persistent=False)
        self.register_buffer("local_freqs_cos", local_cos, persistent=False)
        self.register_buffer("local_freqs_sin", local_sin, persistent=False)

    @property
    def layers(self) -> List[nn.Module]:
        """Flat view of all 35 decoder layers (self + cross).

        Satisfies callers that expect decoder.layers to iterate all layers,
        matching the LlamaModel convention. The actual nn.ModuleLists live in
        self.model.self_decoder.layers / self.model.cross_decoder.layers
        to preserve weight-path naming for checkpoint loading.
        """
        return list(self.model.self_decoder.layers) + list(
            self.model.cross_decoder.layers
        )

    def forward(
        self,
        tokens: torch.Tensor,
        atten_mask: torch.Tensor,
        window_atten_mask: torch.Tensor,
        input_pos: Optional[torch.Tensor] = None,
        *args,
    ) -> Tuple[
        torch.Tensor, List[Optional[torch.Tensor]], List[Optional[torch.Tensor]]
    ]:
        if self.use_kv_cache and input_pos is not None:
            global_cos = self.global_freqs_cos[input_pos][0]
            global_sin = self.global_freqs_sin[input_pos][0]
            local_cos = self.local_freqs_cos[input_pos][0]
            local_sin = self.local_freqs_sin[input_pos][0]
        else:
            global_cos = self.global_freqs_cos
            global_sin = self.global_freqs_sin
            local_cos = self.local_freqs_cos
            local_sin = self.local_freqs_sin

        k_caches = list(args[: self.n_self_layers]) if self.use_kv_cache else []
        v_caches = (
            list(args[self.n_self_layers : 2 * self.n_self_layers])
            if self.use_kv_cache
            else []
        )

        logits, output_k_cache, output_v_cache = self.model(
            tokens,
            atten_mask,
            window_atten_mask,
            input_pos,
            global_cos,
            global_sin,
            local_cos,
            local_sin,
            k_caches,
            v_caches,
        )

        if self.output_cache:
            return logits, output_k_cache, output_v_cache
        return logits

    def get_example_inputs(self):
        dtype = torch.int64 if self.use_i64_token else torch.int32
        tokens = torch.randint(
            self.config.vocab_size,
            (self.config.max_batch_size, self.ar_len),
            dtype=dtype,
        )
        causal_mask = CausalAttentionMask(
            self.config.max_batch_size, self.ar_len, self.config.max_context_len
        )
        sliding_mask = SlidingWindowAttentionMask(
            self.config.max_batch_size,
            self.ar_len,
            self.config.max_context_len,
            sliding_window=self.config.sliding_window,
        )
        attn_mask = AttentionMask([causal_mask, sliding_mask])

        if self.use_kv_cache:
            pos_ids = torch.zeros(
                (self.config.max_batch_size, self.ar_len), dtype=torch.int32
            )
            k_caches, v_caches = [], []
            for i in range(self.n_self_layers):
                hd = self.config.get_head_dim(i)
                k_caches.append(
                    torch.zeros(
                        self.config.max_batch_size,
                        self.config.num_key_value_heads,
                        hd,
                        self.config.max_context_len - self.ar_len,
                    )
                )
                v_caches.append(
                    torch.zeros(
                        self.config.max_batch_size,
                        self.config.num_key_value_heads,
                        self.config.max_context_len - self.ar_len,
                        hd,
                    )
                )
            return (tokens, attn_mask, pos_ids, k_caches, v_caches)

        return (tokens, attn_mask)

    def get_metadata(self):
        return {
            "get_ar_len": self.ar_len,
            "get_bos_id": 2,
            "get_eos_id": 1,
            "get_dim": self.config.hidden_size,
            "get_head_dim": self.config.head_dim,
            "get_global_head_dim": self.config.global_head_dim,
            "get_max_batch_size": self.config.max_batch_size,
            "get_max_seq_len": self.config.max_seq_len,
            "get_max_context_len": self.config.max_context_len,
            "get_n_bos": 1,
            "get_n_eos": 1,
            "get_n_kv_heads": self.config.num_key_value_heads,
            "get_n_layers": self.config.num_hidden_layers,
            "get_n_self_layers": self.n_self_layers,
            "get_vocab_size": self.config.vocab_size,
            "get_use_kv_cache": self.use_kv_cache,
            "get_sliding_window": self.config.sliding_window,
            "get_num_kv_shared_layers": self.config.num_kv_shared_layers,
            "get_kv_io_bit_width": self.kv_io_bit_width,
        }
