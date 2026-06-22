# Copyright (c) Qualcomm Innovation Center, Inc.
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# TODO: reenable pyre after fixing the issues
# pyre-ignore-all-errors

import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from executorch.examples.models.llama.model_args import ModelArgs

from executorch.examples.qualcomm.oss_scripts.llama.masking_utils import (
    AttentionMask,
    CausalAttentionMask,
    SlidingWindowAttentionMask,
)

from .blocks.decoder_layer import DECODER_LAYER_REGISTRY, LlamaDecoderLayer
from .blocks.norm import NORM_REGISTRY

from .rope import RopeFreqs


class LlamaModel(nn.Module):
    def __init__(
        self,
        config: ModelArgs,
        ar_len=1,
        output_new_cache_only=True,
        output_cache=True,
        use_i64_token=False,
        **kwargs,
    ):
        super().__init__()
        self.dim = config.dim
        self.head_dim = config.head_dim
        self.max_batch_size = config.max_batch_size
        self.max_seq_len = config.max_seq_len
        self.max_context_len = config.max_context_len
        self.n_heads = config.n_heads
        self.n_kv_heads = config.n_kv_heads
        self.n_layers = config.n_layers
        self.n_self_layers = config.n_layers - config.num_kv_shared_layers  # for YOCO
        self.vocab_size = config.vocab_size
        self.rope_freq_base = config.rope_freq_base
        self.use_kv_cache = config.use_kv_cache
        self.embedding_scale_factor = config.embedding_scale_factor
        self.ar_len = ar_len
        self.output_new_cache_only = output_new_cache_only
        self.use_i64_token = use_i64_token
        self.output_cache = output_cache
        self.kv_io_bit_width = config.kv_io_bit_width
        self.logits_scaling = config.logits_scaling
        self.config = config

        self.layers = nn.ModuleList(
            [
                DECODER_LAYER_REGISTRY.get(
                    config.is_kv_shared_layer(layer_idx), LlamaDecoderLayer
                )(
                    config=config,
                    layer_idx=layer_idx,
                    output_new_cache_only=self.output_new_cache_only,
                )
                for layer_idx in range(config.n_layers)
            ]
        )
        self.norm = NORM_REGISTRY[config.norm_type](config.dim, eps=config.norm_eps)
        self.output = nn.Linear(config.dim, config.vocab_size, bias=config.output_bias)

        self.tok_embeddings = nn.Embedding(config.vocab_size, config.dim)
        for buffer_name, buffer in RopeFreqs(config).compute().named_buffers():
            self.register_buffer(buffer_name, buffer, persistent=False)

    def prepare_output_conv(self):
        def forward_output_conv(x):
            bsz, _, _ = x.size()
            x = torch.reshape(x, (bsz, -1, 1, self.dim))
            x = x.transpose(1, 3)  # Transpose right before and after Conv
            x = self.output_conv(x)
            x = x.transpose(1, 3)
            x = torch.reshape(x, (bsz, -1, self.vocab_size))
            return x

        self.output_conv = nn.Conv2d(self.dim, self.vocab_size, 1, bias=False)
        self.output_conv.weight.data.copy_(self.output.weight[:, :, None, None])

        del self.output
        self.output = forward_output_conv

    def forward(
        self,
        tokens: torch.Tensor,
        atten_mask: torch.Tensor,
        input_pos: Optional[torch.Tensor] = None,
        *args,
    ) -> Tuple[torch.Tensor, List[torch.Tensor], List[torch.Tensor]]:
        output_k_cache = []
        output_v_cache = []
        # following tensors should be invariant across batches
        freqs_cos = (
            self.freqs_cos[input_pos][0] if self.use_kv_cache else self.freqs_cos
        )
        freqs_sin = (
            self.freqs_sin[input_pos][0] if self.use_kv_cache else self.freqs_sin
        )

        hidden_states = self.embedding_scale_factor * self.tok_embeddings(tokens)

        for ind, decoder_layer in enumerate(self.layers):
            k_caches = None
            v_caches = None
            if self.use_kv_cache:
                offset_k = ind
                offset_v = self.n_layers + offset_k
                k_caches = args[offset_k]
                v_caches = args[offset_v]

            hidden_states, k, v = decoder_layer(
                hidden_states,
                freqs_cos=freqs_cos,
                freqs_sin=freqs_sin,
                atten_mask=atten_mask,
                k_caches=k_caches,
                v_caches=v_caches,
            )
            output_k_cache.extend(k)
            output_v_cache.extend(v)

        hidden_states = self.norm(hidden_states)
        logits = self.output(hidden_states)

        if self.logits_scaling:
            logits = logits / self.logits_scaling

        if self.output_cache:
            return logits, output_k_cache, output_v_cache
        return logits

    def get_example_inputs(self):
        dtype = torch.int64 if self.use_i64_token else torch.int32
        tokens = torch.randint(
            self.vocab_size, (self.max_batch_size, self.ar_len), dtype=dtype
        )
        atten_mask = AttentionMask(
            CausalAttentionMask(self.max_batch_size, self.ar_len, self.max_context_len)
        )
        if self.use_kv_cache:
            pos_ids = torch.zeros((self.max_batch_size, self.ar_len), dtype=torch.int32)
            k_cache, v_cache = [], []
            for ind in range(self.n_self_layers):
                head_dim = self.config.get_head_dim(ind)
                # transpose first to decrease the runtime efforts
                k_cache.append(
                    torch.zeros(
                        self.max_batch_size,
                        self.n_kv_heads,
                        head_dim,
                        self.max_context_len - self.ar_len,
                    )
                )
                v_cache.append(
                    torch.zeros(
                        self.max_batch_size,
                        self.n_kv_heads,
                        self.max_context_len - self.ar_len,
                        head_dim,
                    )
                )
            return (
                tokens,
                atten_mask,
                pos_ids,
                k_cache,
                v_cache,
            )

        return (
            tokens,
            atten_mask,
        )

    def get_metadata(self):
        return {
            "get_ar_len": self.ar_len,
            "get_bos_id": 1,
            "get_eos_id": 2,
            "get_dim": self.dim,
            "get_head_dim": self.head_dim,
            "get_max_batch_size": self.max_batch_size,
            "get_max_seq_len": self.max_seq_len,
            "get_max_context_len": self.max_context_len,
            "get_n_bos": 1,
            "get_n_eos": 1,
            "get_n_kv_heads": self.n_kv_heads,
            "get_n_layers": self.n_layers,
            "get_n_self_layers": self.n_self_layers,
            "get_vocab_size": self.vocab_size,
            "get_use_kv_cache": self.use_kv_cache,
            "get_kv_io_bit_width": self.kv_io_bit_width,
        }

    def get_kv_head_dim(self):
        """Head dims that appear in K/V caches -- one for most models."""
        return {self.head_dim}

    def get_kv_cache_shapes(self):
        return set().union(
            *(
                {
                    # single head, kv input
                    (head_dim, self.max_context_len),
                    (self.max_context_len, head_dim),
                    # single head, kv output
                    (head_dim, self.ar_len),
                    (self.ar_len, head_dim),
                }
                for head_dim in self.get_kv_head_dim()
            )
        )

    def get_freq_shapes(self):
        """The rope cos/sin (ar_len, head_dim // 2) shapes this model produces."""
        return {(self.ar_len, head_dim // 2) for head_dim in self.get_kv_head_dim()}


class LlamaModelWithoutEmbedding(LlamaModel):
    def __init__(
        self,
        config: ModelArgs,
        ar_len=1,
        output_new_cache_only=True,
        output_cache=True,
        use_i64_token=False,
        **kwargs,
    ):
        super().__init__(
            config=config,
            ar_len=ar_len,
            output_new_cache_only=output_new_cache_only,
            output_cache=output_cache,
            use_i64_token=use_i64_token,
            **kwargs,
        )

        # Set the audio/image token ID from keyword arguments. It defaults to None if not provided.
        # If an ID is provided, it will be stored in the model's metadata.
        self.audio_token_id = kwargs.get("audio_token_id", None)
        self.image_token_id = kwargs.get("image_token_id", None)

    def forward(
        self,
        hidden_states: torch.Tensor,
        atten_mask: torch.Tensor,
        input_pos: Optional[torch.Tensor] = None,
        *args,
    ) -> Tuple[torch.Tensor, List[torch.Tensor], List[torch.Tensor]]:
        output_k_cache = []
        output_v_cache = []
        # following tensors should be invariant across batches
        freqs_cos = (
            self.freqs_cos[input_pos][0] if self.use_kv_cache else self.freqs_cos
        )
        freqs_sin = (
            self.freqs_sin[input_pos][0] if self.use_kv_cache else self.freqs_sin
        )

        hidden_states = self.embedding_scale_factor * hidden_states

        for ind, decoder_layer in enumerate(self.layers):
            k_caches = None
            v_caches = None
            if self.use_kv_cache:
                offset_k = ind
                offset_v = self.n_layers + offset_k
                k_caches = args[offset_k]
                v_caches = args[offset_v]

            hidden_states, k, v = decoder_layer(
                hidden_states,
                freqs_cos=freqs_cos,
                freqs_sin=freqs_sin,
                atten_mask=atten_mask,
                k_caches=k_caches,
                v_caches=v_caches,
            )
            output_k_cache.extend(k)
            output_v_cache.extend(v)

        hidden_states = self.norm(hidden_states)
        logits = self.output(hidden_states)

        if self.logits_scaling:
            logits = logits / self.logits_scaling

        if self.output_cache:
            return logits, output_k_cache, output_v_cache
        return logits

    def get_example_inputs(self):
        hidden_states = torch.randn(
            (self.max_batch_size, self.ar_len, self.dim), dtype=torch.float32
        )
        inputs = list(super().get_example_inputs())
        inputs[0] = hidden_states
        return tuple(inputs)

    def get_metadata(self):
        meta_data = super().get_metadata()
        if self.audio_token_id:
            meta_data["audio_token_id"] = self.audio_token_id
        if self.image_token_id:
            meta_data["image_token_id"] = self.image_token_id
        return meta_data


class MultiScopeAwareLlamaModel(LlamaModel):
    def __init__(
        self,
        config: ModelArgs,
        ar_len=1,
        output_new_cache_only=True,
        output_cache=True,
        use_i64_token=False,
        **kwargs,
    ):
        super().__init__(
            config=config,
            ar_len=ar_len,
            output_new_cache_only=output_new_cache_only,
            output_cache=output_cache,
            use_i64_token=use_i64_token,
            **kwargs,
        )
        # Parameter final_logit_softcapping is not necessary for all
        self.final_logit_softcapping = config.final_logit_softcapping

        # Gemma2/Gemma3 requires additional configuration parameters:
        # - layer_types: Specifies the type of each layer (e.g., full vs. sliding attention)
        # - local_rope_theta: Base frequency for local RoPE
        # - sliding_window: Size of the sliding window for local attention
        if config.layer_types is not None:
            assert len(config.layer_types) == self.n_layers, (
                f"Length of layer_types ({len(config.layer_types)}) must match "
                f"n_layers ({self.n_layers})"
            )
        assert (
            config.local_rope_theta is not None
        ), "local_rope_theta should not be None, please set it explicitly in config."

        self.sliding_window = config.sliding_window

        self.n_self_layers = config.n_layers - config.num_kv_shared_layers
        if config.num_kv_shared_layers > 0:
            self.embedding_scale_factor = math.sqrt(config.dim)

        # Per-Layer Embeddings (PLE)
        self.use_per_layer_embedding = (
            config.vocab_size_per_layer_input and config.hidden_size_per_layer_input
        )
        if self.use_per_layer_embedding:
            self.embed_tokens_per_layer = nn.Embedding(
                config.vocab_size_per_layer_input,
                config.n_layers * config.hidden_size_per_layer_input,
            )
            self.embed_scale_per_layer = math.sqrt(config.hidden_size_per_layer_input)
            self.per_layer_model_projection = nn.Linear(
                config.dim,
                config.n_layers * config.hidden_size_per_layer_input,
                bias=False,
            )
            self.per_layer_projection_norm = torch.nn.RMSNorm(
                config.hidden_size_per_layer_input, eps=config.norm_eps
            )
            self._ple_input_scale = 1.0 / math.sqrt(2.0)
            self._ple_proj_scale = 1.0 / math.sqrt(config.dim)

    def _compute_per_layer_inputs(
        self,
        tokens: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        bsz, seq_len, _ = hidden_states.shape
        n_layers = self.config.n_layers
        hpl = self.config.hidden_size_per_layer_input

        per_layer_proj = (
            self.per_layer_model_projection(hidden_states) * self._ple_proj_scale
        )
        per_layer_proj = per_layer_proj.view(bsz, seq_len, n_layers, hpl)
        per_layer_proj = self.per_layer_projection_norm(per_layer_proj)
        per_layer_embed = (
            self.embed_tokens_per_layer(tokens) * self.embed_scale_per_layer
        )
        per_layer_embed = per_layer_embed.view(bsz, seq_len, n_layers, hpl)

        combined = (per_layer_proj + per_layer_embed) * self._ple_input_scale
        return combined.permute(2, 0, 1, 3)

    def forward(  # noqa: C901
        self,
        tokens: torch.Tensor,
        atten_mask: torch.Tensor,
        window_atten_mask: torch.Tensor,
        input_pos: Optional[torch.Tensor] = None,
        *args,
    ) -> Tuple[torch.Tensor, List[torch.Tensor], List[torch.Tensor]]:

        output_k_cache = []
        output_v_cache = []
        # following tensors should be invariant across batches
        global_freqs_cos = (
            self.freqs_cos[input_pos][0] if self.use_kv_cache else self.freqs_cos
        )
        global_freqs_sin = (
            self.freqs_sin[input_pos][0] if self.use_kv_cache else self.freqs_sin
        )
        local_freqs_cos = (
            self.local_freqs_cos[input_pos][0]
            if self.use_kv_cache
            else self.local_freqs_cos
        )
        local_freqs_sin = (
            self.local_freqs_sin[input_pos][0]
            if self.use_kv_cache
            else self.local_freqs_sin
        )

        hidden_states = self.embedding_scale_factor * self.tok_embeddings(tokens)
        if self.use_per_layer_embedding:
            per_layer_inputs = self._compute_per_layer_inputs(tokens, hidden_states)

        if self.config.num_kv_shared_layers > 0:
            shared_kv: Dict[int, Tuple[torch.Tensor, torch.Tensor]] = {}
            for ind, decoder_layer in enumerate(self.layers):
                if self.config.is_sliding_attention(ind):
                    freqs_cos, freqs_sin, mask = (
                        local_freqs_cos,
                        local_freqs_sin,
                        window_atten_mask,
                    )
                else:
                    freqs_cos, freqs_sin, mask = (
                        global_freqs_cos,
                        global_freqs_sin,
                        atten_mask,
                    )
                per_layer_input = (
                    per_layer_inputs[ind] if self.use_per_layer_embedding else None
                )

                if ind < self.n_self_layers:
                    # Self-decoder layer: own KV I/O.
                    k_caches = None
                    v_caches = None
                    if self.use_kv_cache:
                        offset_k = ind
                        offset_v = self.n_self_layers + offset_k
                        k_caches = args[offset_k]
                        v_caches = args[offset_v]

                    hidden_states, k, v = decoder_layer(
                        hidden_states,
                        freqs_cos=freqs_cos,
                        freqs_sin=freqs_sin,
                        atten_mask=mask,
                        k_caches=k_caches,
                        v_caches=v_caches,
                        per_layer_input=per_layer_input,
                    )
                    output_k_cache.append(k)
                    output_v_cache.append(v)

                    # Donor layers expose their full updated K/V for YOCO sharing.
                    if self.config.is_kv_donor_layer(ind):
                        if (
                            decoder_layer.attention.output_new_cache_only
                            and k_caches is not None
                        ):
                            full_k = torch.cat([k_caches, k], dim=-1)
                            full_v = torch.cat([v_caches, v], dim=2)
                        else:
                            full_k, full_v = k, v
                        shared_kv[ind] = (full_k, full_v)
                else:
                    # Cross-decoder layer: no KV I/O, consumes donor K/V.
                    kv_src = self.config.get_kv_shared_layer_index(ind)
                    donor_k, donor_v = None, None
                    if kv_src is not None and kv_src in shared_kv:
                        donor_k, donor_v = shared_kv[kv_src]

                    hidden_states, _, _ = decoder_layer(
                        hidden_states,
                        freqs_cos=freqs_cos,
                        freqs_sin=freqs_sin,
                        atten_mask=mask,
                        donor_k=donor_k,
                        donor_v=donor_v,
                        per_layer_input=per_layer_input,
                    )
        else:
            for ind, decoder_layer in enumerate(self.layers):
                k_caches = None
                v_caches = None
                if self.use_kv_cache:
                    offset_k = ind
                    offset_v = self.n_layers + offset_k
                    k_caches = args[offset_k]
                    v_caches = args[offset_v]

                if self.config.is_sliding_attention(ind):
                    hidden_states, k, v = decoder_layer(
                        hidden_states,
                        freqs_cos=local_freqs_cos,
                        freqs_sin=local_freqs_sin,
                        atten_mask=window_atten_mask,
                        k_caches=k_caches,
                        v_caches=v_caches,
                    )
                else:
                    hidden_states, k, v = decoder_layer(
                        hidden_states,
                        freqs_cos=global_freqs_cos,
                        freqs_sin=global_freqs_sin,
                        atten_mask=atten_mask,
                        k_caches=k_caches,
                        v_caches=v_caches,
                    )

                output_k_cache.extend(k)
                output_v_cache.extend(v)

        hidden_states = self.norm(hidden_states)
        logits = self.output(hidden_states)
        if self.final_logit_softcapping:
            logits = logits / self.final_logit_softcapping
            logits = torch.tanh(logits)
            logits = logits * self.final_logit_softcapping

        if self.output_cache:
            return logits, output_k_cache, output_v_cache
        return logits

    def get_example_inputs(self):
        inputs = list(super().get_example_inputs())
        causal_mask = CausalAttentionMask(
            self.max_batch_size, self.ar_len, self.max_context_len
        )
        sliding_window_mask = SlidingWindowAttentionMask(
            self.max_batch_size,
            self.ar_len,
            self.max_context_len,
            sliding_window=self.sliding_window,
        )
        # Don't reverse the order of attention mask
        inputs[1] = AttentionMask([causal_mask, sliding_window_mask])
        return tuple(inputs)

    def get_metadata(self):
        meta_data = super().get_metadata()
        meta_data["get_sliding_window"] = self.sliding_window
        if self.config.global_head_dim:
            meta_data["get_global_head_dim"] = self.config.global_head_dim
        return meta_data

    def get_kv_head_dim(self):
        return {
            self.head_dim,
            *filter(None, [self.config.global_head_dim]),
        }
