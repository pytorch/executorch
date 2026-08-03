# Copyright (c) Intel Corporation
#
# Licensed under the BSD License (the "License"); you may not use this file
# except in compliance with the License. See the license file found in the
# LICENSE file in the root directory of this source tree.

"""
Whisper module split for the OpenVINO backend.

Three exportable modules:
  * WhisperEncoderModule      : mel -> encoder_hidden_states
  * WhisperCrossKVProjection  : encoder_hidden_states -> (cross_k_tuple, cross_v_tuple)
  * WhisperDecoderWithCache   : (tokens, cache_position, attn_mask,
                                 cross_k_tuple, cross_v_tuple) -> logits

The cross-attention K/V projections live in their own graph and run once per
utterance. The decoder graph has no k_proj/v_proj on the encoder outputs; the
cross K/V flow in as plain input tensors every step.

"""

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from executorch.exir.dialects._ops import ops as exir_ops
from executorch.exir.pass_base import ExportPass, PassResult


class RemovePaddingIdxEmbeddingPass(ExportPass):
    """Drop the padding_idx argument from edge `aten.embedding` calls """

    def call(self, graph_module: torch.fx.GraphModule) -> PassResult:
        for node in graph_module.graph.nodes:
            if (
                node.op == "call_function"
                and node.target == exir_ops.edge.aten.embedding.default
                and len(node.args) == 3
            ):
                node.args = node.args[:2]
        graph_module.recompile()
        return PassResult(graph_module, True)


class WhisperEncoderModule(nn.Module):
    """mel features -> encoder hidden states """

    def __init__(self, encoder: nn.Module, feature_size: int, num_frames: int):
        super().__init__()
        self.encoder = encoder
        self.feature_size = feature_size
        self.num_frames = num_frames

    def forward(self, input_features: torch.Tensor) -> torch.Tensor:
        return self.encoder(input_features=input_features).last_hidden_state

    def get_example_inputs(self, dtype: torch.dtype = torch.float32):
        return (torch.rand(1, self.feature_size, self.num_frames, dtype=dtype),)


class WhisperCrossKVProjection(nn.Module):
    """Runs once per utterance: encoder hidden states -> per-layer cross-attention K/V"""

    def __init__(self, whisper_model: nn.Module):
        super().__init__()
        decoder = whisper_model.get_decoder()
        self.k_projs = nn.ModuleList()
        self.v_projs = nn.ModuleList()
        self.num_heads_list = []
        self.head_dim_list = []
        for layer in decoder.layers:
            self.k_projs.append(layer.encoder_attn.k_proj)
            self.v_projs.append(layer.encoder_attn.v_proj)
            self.num_heads_list.append(layer.encoder_attn.num_heads)
            self.head_dim_list.append(layer.encoder_attn.head_dim)

    def forward(
        self, encoder_hidden_states: torch.Tensor
    ) -> Tuple[Tuple[torch.Tensor, ...], Tuple[torch.Tensor, ...]]:
        B, T_enc, _ = encoder_hidden_states.shape
        k_list, v_list = [], []
        for i, (k_proj, v_proj) in enumerate(zip(self.k_projs, self.v_projs)):
            H, D = self.num_heads_list[i], self.head_dim_list[i]
            k = k_proj(encoder_hidden_states).view(B, T_enc, H, D).transpose(1, 2)
            v = v_proj(encoder_hidden_states).view(B, T_enc, H, D).transpose(1, 2)
            k_list.append(k.contiguous())
            v_list.append(v.contiguous())
        return tuple(k_list), tuple(v_list)


class StaticKVCache(nn.Module):
    """Fixed-size self-attention K/V cache; index_copy_ writes at the absolute position."""

    def __init__(self, max_context_length, n_heads, head_dim, dtype):
        super().__init__()
        self.max_context_length = max_context_length
        shape = (1, n_heads, max_context_length, head_dim)
        self.register_buffer("k_cache", torch.zeros(shape, dtype=dtype))
        self.register_buffer("v_cache", torch.zeros(shape, dtype=dtype))

    def update(self, cache_position, k_val, v_val):
        # cache_position: [S] (long), k_val/v_val: [B, H, S, D].
        self.k_cache.index_copy_(2, cache_position, k_val)
        self.v_cache.index_copy_(2, cache_position, v_val)
        return self.k_cache, self.v_cache


class WhisperSelfAttentionWithCache(nn.Module):
    """Self-attention over a static cache; the additive mask comes from the host."""

    def __init__(self, attn, max_cache_len, dtype):
        super().__init__()
        self.q_proj = attn.q_proj
        self.k_proj = attn.k_proj
        self.v_proj = attn.v_proj
        self.out_proj = attn.out_proj
        self.num_heads = attn.num_heads
        self.head_dim = attn.head_dim
        self.scale = self.head_dim**-0.5
        self.cache = StaticKVCache(max_cache_len, self.num_heads, self.head_dim, dtype)

    def forward(self, hidden_states, cache_position, attn_mask):
        B, T, _ = hidden_states.shape
        H, D = self.num_heads, self.head_dim
        q = self.q_proj(hidden_states).view(B, T, H, D).transpose(1, 2)
        k = self.k_proj(hidden_states).view(B, T, H, D).transpose(1, 2)
        v = self.v_proj(hidden_states).view(B, T, H, D).transpose(1, 2)
        k_cache, v_cache = self.cache.update(cache_position, k, v)
        attn_out = F.scaled_dot_product_attention(
            q, k_cache, v_cache, attn_mask=attn_mask, is_causal=False, scale=self.scale
        )
        return self.out_proj(attn_out.transpose(1, 2).contiguous().view(B, T, H * D))


class WhisperCrossAttentionPrecomputed(nn.Module):
    """Cross-attention where K/V are pre-projected and passed in as inputs."""

    def __init__(self, attn):
        super().__init__()
        self.q_proj = attn.q_proj
        self.out_proj = attn.out_proj
        self.num_heads = attn.num_heads
        self.head_dim = attn.head_dim
        self.scale = self.head_dim**-0.5

    def forward(self, hidden_states, cross_k, cross_v):
        B, T, _ = hidden_states.shape
        H, D = self.num_heads, self.head_dim
        q = self.q_proj(hidden_states).view(B, T, H, D).transpose(1, 2)
        attn_out = F.scaled_dot_product_attention(
            q, cross_k, cross_v, attn_mask=None, is_causal=False, scale=self.scale
        )
        return self.out_proj(attn_out.transpose(1, 2).contiguous().view(B, T, H * D))


class WhisperDecoderLayerWithCache(nn.Module):
    """One decoder layer: cached self-attention, pre-computed cross-attention, then FFN."""

    def __init__(self, layer, max_cache_len, dtype):
        super().__init__()
        self.self_attn = WhisperSelfAttentionWithCache(
            layer.self_attn, max_cache_len, dtype
        )
        self.self_attn_layer_norm = layer.self_attn_layer_norm
        self.encoder_attn = WhisperCrossAttentionPrecomputed(layer.encoder_attn)
        self.encoder_attn_layer_norm = layer.encoder_attn_layer_norm
        self.fc1 = layer.fc1
        self.fc2 = layer.fc2
        self.final_layer_norm = layer.final_layer_norm
        self.activation_fn = layer.activation_fn

    def forward(self, hidden_states, cache_position, attn_mask, cross_k, cross_v):
        residual = hidden_states
        hidden_states = self.self_attn_layer_norm(hidden_states)
        hidden_states = self.self_attn(hidden_states, cache_position, attn_mask)
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.encoder_attn_layer_norm(hidden_states)
        hidden_states = self.encoder_attn(hidden_states, cross_k, cross_v)
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.final_layer_norm(hidden_states)
        hidden_states = self.activation_fn(self.fc1(hidden_states))
        hidden_states = self.fc2(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states


class WhisperDecoderWithCache(nn.Module):
    """
    Takes:
      * decoder_input_ids : [B, T_dec] token IDs
      * cache_position    : [T_dec] tensor of absolute positions
      * attn_mask         : [1, 1, T_dec, max_cache_length] additive mask
      * cross_k_tuple     : num_layers x [B, H, T_enc, D] pre-computed cross K
      * cross_v_tuple     : num_layers x [B, H, T_enc, D] pre-computed cross V

    Returns logits [B, T_dec, vocab_size].
    """

    def __init__(self, whisper_model: nn.Module, max_decoder_seq_len: int):
        super().__init__()
        decoder = whisper_model.get_decoder()
        dtype = decoder.embed_tokens.weight.dtype
        self.embed_tokens = decoder.embed_tokens
        self.embed_positions = decoder.embed_positions
        self.layer_norm = decoder.layer_norm
        self.proj_out = whisper_model.proj_out
        self.layers = nn.ModuleList(
            [
                WhisperDecoderLayerWithCache(layer, max_decoder_seq_len, dtype)
                for layer in decoder.layers
            ]
        )
        self.num_layers = len(self.layers)
        self.max_decoder_seq_len = max_decoder_seq_len

    def forward(
        self,
        decoder_input_ids: torch.Tensor,
        cache_position: torch.Tensor,
        attn_mask: torch.Tensor,
        cross_k_tuple: Tuple[torch.Tensor, ...],
        cross_v_tuple: Tuple[torch.Tensor, ...],
    ) -> torch.Tensor:
        hidden_states = self.embed_tokens(decoder_input_ids)
        pos_embed = F.embedding(cache_position, self.embed_positions.weight)
        hidden_states = hidden_states + pos_embed
        for i, layer in enumerate(self.layers):
            hidden_states = layer(
                hidden_states,
                cache_position,
                attn_mask,
                cross_k_tuple[i],
                cross_v_tuple[i],
            )
        hidden_states = self.layer_norm(hidden_states)
        return self.proj_out(hidden_states)
