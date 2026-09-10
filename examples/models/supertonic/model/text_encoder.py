# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import math

import torch
from torch import nn
from torch.nn import functional as F

from .config import TTSConfig
from .layers import ConvNeXt, LayerNorm1d, LinearProjection


class TextEmbedder(nn.Module):
    def __init__(self, vocab_size: int, channels: int) -> None:
        super().__init__()
        self.char_embedder = nn.Embedding(vocab_size, channels)

    def forward(self, text_ids: torch.Tensor, text_mask: torch.Tensor) -> torch.Tensor:
        return self.char_embedder(text_ids).transpose(1, 2) * text_mask


class RelativeMultiHeadAttention(nn.Module):
    def __init__(self, channels: int, num_heads: int, window_size: int = 4) -> None:
        super().__init__()
        if channels % num_heads != 0:
            raise ValueError("channels must be divisible by num_heads")
        self.channels = channels
        self.num_heads = num_heads
        self.head_channels = channels // num_heads
        self.window_size = window_size
        self.emb_rel_k = nn.Parameter(
            torch.randn(1, 2 * window_size + 1, self.head_channels)
            * (self.head_channels**-0.5)
        )
        self.emb_rel_v = nn.Parameter(
            torch.randn(1, 2 * window_size + 1, self.head_channels)
            * (self.head_channels**-0.5)
        )
        self.conv_q = nn.Conv1d(channels, channels, 1)
        self.conv_k = nn.Conv1d(channels, channels, 1)
        self.conv_v = nn.Conv1d(channels, channels, 1)
        self.conv_o = nn.Conv1d(channels, channels, 1)

    def _relative_embeddings(
        self, embeddings: torch.Tensor, length: int
    ) -> torch.Tensor:
        pad_length = max(length - (self.window_size + 1), 0)
        slice_start = max((self.window_size + 1) - length, 0)
        slice_end = slice_start + 2 * length - 1
        padded = F.pad(embeddings, (0, 0, pad_length, pad_length))
        return padded[:, slice_start:slice_end]

    @staticmethod
    def _relative_to_absolute(inputs: torch.Tensor) -> torch.Tensor:
        batch, heads, length, _ = inputs.shape
        padded = F.pad(inputs, (0, 1))
        flattened = padded.reshape(batch, heads, length * 2 * length)
        flattened = F.pad(flattened, (0, length - 1))
        final = flattened.reshape(batch, heads, length + 1, 2 * length - 1)
        return final[:, :, :length, length - 1 :]

    @staticmethod
    def _absolute_to_relative(inputs: torch.Tensor) -> torch.Tensor:
        batch, heads, length, _ = inputs.shape
        padded = F.pad(inputs, (0, length - 1))
        flattened = padded.reshape(batch, heads, length * (2 * length - 1))
        flattened = F.pad(flattened, (length, 0))
        final = flattened.reshape(batch, heads, length, 2 * length)
        return final[:, :, :, 1:]

    def forward(
        self, inputs: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        batch, _, length = inputs.shape
        query = self.conv_q(inputs).reshape(
            batch, self.num_heads, self.head_channels, length
        )
        key = self.conv_k(inputs).reshape(
            batch, self.num_heads, self.head_channels, length
        )
        value = self.conv_v(inputs).reshape(
            batch, self.num_heads, self.head_channels, length
        )
        query = query.transpose(2, 3) / math.sqrt(self.head_channels)
        key = key.transpose(2, 3)
        value = value.transpose(2, 3)

        scores = torch.matmul(query, key.transpose(-2, -1))
        relative_key = self._relative_embeddings(self.emb_rel_k, length)
        relative_scores = torch.matmul(
            query, relative_key.unsqueeze(0).transpose(-2, -1)
        )
        scores = scores + self._relative_to_absolute(relative_scores)
        scores = scores.masked_fill(attention_mask == 0, -10000.0)
        weights = torch.softmax(scores, dim=-1)

        attended = torch.matmul(weights, value)
        relative_weights = self._absolute_to_relative(weights)
        relative_value = self._relative_embeddings(self.emb_rel_v, length)
        attended = attended + torch.matmul(
            relative_weights, relative_value.unsqueeze(0)
        )
        attended = attended.transpose(2, 3).reshape(batch, self.channels, length)
        return self.conv_o(attended)


class FeedForward(nn.Module):
    def __init__(self, channels: int, filter_channels: int) -> None:
        super().__init__()
        self.conv_1 = nn.Conv1d(channels, filter_channels, 1)
        self.conv_2 = nn.Conv1d(filter_channels, channels, 1)

    def forward(self, inputs: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        hidden = self.conv_1(inputs * mask)
        hidden = F.relu(hidden) * mask
        return self.conv_2(hidden) * mask


class AttentionEncoder(nn.Module):
    def __init__(
        self,
        channels: int,
        filter_channels: int,
        num_heads: int,
        num_layers: int,
        relative_window: int,
    ) -> None:
        super().__init__()
        self.attn_layers = nn.ModuleList(
            [
                RelativeMultiHeadAttention(channels, num_heads, relative_window)
                for _ in range(num_layers)
            ]
        )
        self.norm_layers_1 = nn.ModuleList(
            [LayerNorm1d(channels) for _ in range(num_layers)]
        )
        self.ffn_layers = nn.ModuleList(
            [FeedForward(channels, filter_channels) for _ in range(num_layers)]
        )
        self.norm_layers_2 = nn.ModuleList(
            [LayerNorm1d(channels) for _ in range(num_layers)]
        )

    def forward(self, inputs: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        hidden = inputs * mask
        attention_mask = mask.unsqueeze(2) * mask.unsqueeze(-1)
        for attention, norm_1, feed_forward, norm_2 in zip(
            self.attn_layers,
            self.norm_layers_1,
            self.ffn_layers,
            self.norm_layers_2,
        ):
            hidden = norm_1(hidden + attention(hidden, attention_mask))
            hidden = norm_2(hidden + feed_forward(hidden, mask))
        return hidden * mask


class TextBackbone(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        channels: int,
        convnext_dilations: tuple[int, ...],
        attention_layers: int,
        attention_heads: int,
        ff_channels: int,
        relative_window: int,
    ) -> None:
        super().__init__()
        self.text_embedder = TextEmbedder(vocab_size, channels)
        self.convnext = ConvNeXt(
            channels,
            len(convnext_dilations),
            kernel_size=5,
            dilations=convnext_dilations,
        )
        self.attn_encoder = AttentionEncoder(
            channels,
            ff_channels,
            attention_heads,
            attention_layers,
            relative_window,
        )

    def forward(self, text_ids: torch.Tensor, text_mask: torch.Tensor) -> torch.Tensor:
        hidden = self.convnext(
            self.text_embedder(text_ids, text_mask),
            text_mask,
        )
        return (hidden + self.attn_encoder(hidden, text_mask)) * text_mask


class StyleTokenLayer(nn.Module):
    def __init__(self, style_tokens: int, channels: int) -> None:
        super().__init__()
        self.style_key = nn.Parameter(torch.randn(1, style_tokens, channels))


class StyleEncoder(nn.Module):
    def __init__(self, style_tokens: int, channels: int) -> None:
        super().__init__()
        self.style_token_layer = StyleTokenLayer(style_tokens, channels)


class TanhKeyAttention(nn.Module):
    def __init__(self, channels: int, num_heads: int) -> None:
        super().__init__()
        if channels % num_heads != 0:
            raise ValueError("channels must be divisible by num_heads")
        self.channels = channels
        self.num_heads = num_heads
        self.head_channels = channels // num_heads
        self.W_query = LinearProjection(channels, channels)
        self.W_key = LinearProjection(channels, channels)
        self.W_value = LinearProjection(channels, channels)
        self.out_fc = LinearProjection(channels, channels)

    def _split_heads(self, inputs: torch.Tensor) -> torch.Tensor:
        # The ONNX graph stacks feature chunks before the batch axis.
        return torch.stack(torch.split(inputs, self.head_channels, dim=-1), dim=0)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        query_mask: torch.Tensor,
    ) -> torch.Tensor:
        projected_query = self._split_heads(self.W_query(query))
        projected_key = self._split_heads(self.W_key(key))
        projected_value = self._split_heads(self.W_value(value))
        # The graph scales by full width, not per-head width.
        scores = torch.matmul(
            projected_query,
            torch.tanh(projected_key.transpose(-2, -1)),
        ) / math.sqrt(self.channels)
        weights = torch.softmax(scores, dim=-1)
        weights = torch.where(
            query_mask.transpose(1, 2).unsqueeze(0) == 0,
            torch.zeros_like(weights),
            weights,
        )
        attended = torch.matmul(weights, projected_value)
        attended = torch.cat(torch.unbind(attended, dim=0), dim=-1)
        return self.out_fc(attended) * query_mask.transpose(1, 2)


class SpeechPromptedTextEncoder(nn.Module):
    def __init__(self, channels: int, num_heads: int) -> None:
        super().__init__()
        self.attention1 = TanhKeyAttention(channels, num_heads)
        self.attention2 = TanhKeyAttention(channels, num_heads)
        self.norm = LayerNorm1d(channels)

    def forward(
        self,
        text: torch.Tensor,
        style_key: torch.Tensor,
        style_value: torch.Tensor,
        text_mask: torch.Tensor,
    ) -> torch.Tensor:
        text_sequence = text.transpose(1, 2)
        first = text_sequence + self.attention1(
            text_sequence, style_key, style_value, text_mask
        )
        second = text_sequence + self.attention2(
            first, style_key, style_value, text_mask
        )
        return self.norm(second.transpose(1, 2)) * text_mask


class TextEncoder(nn.Module):
    def __init__(
        self,
        config: TTSConfig,
        *,
        vocab_size: int = 8322,
        channels: int = 256,
        convnext_dilations: tuple[int, ...] = (1, 1, 2, 2, 4, 4),
        attention_layers: int = 4,
        attention_heads: int = 4,
        ff_channels: int = 1024,
        relative_window: int = 4,
        style_tokens: int = 50,
        style_attention_heads: int = 2,
    ) -> None:
        super().__init__()
        if config.ttl.latent_dim <= 0 or config.ttl.chunk_compress_factor <= 0:
            raise ValueError("config.ttl dimensions must be positive")
        self.text_encoder = TextBackbone(
            vocab_size,
            channels,
            convnext_dilations,
            attention_layers,
            attention_heads,
            ff_channels,
            relative_window,
        )
        self.style_encoder = StyleEncoder(style_tokens, channels)
        self.speech_prompted_text_encoder = SpeechPromptedTextEncoder(
            channels, style_attention_heads
        )
        self.style_tokens = style_tokens
        self.style_channels = channels

    def _validate_inputs(
        self,
        text_ids: torch.Tensor,
        style_ttl: torch.Tensor,
        text_mask: torch.Tensor,
    ) -> None:
        if text_ids.ndim != 2:
            raise ValueError("text_ids must have shape [B, T]")
        if style_ttl.ndim != 3 or style_ttl.shape[1:] != (
            self.style_tokens,
            self.style_channels,
        ):
            raise ValueError(
                f"style_ttl must have shape [B, {self.style_tokens}, "
                f"{self.style_channels}]"
            )
        if text_mask.ndim != 3 or text_mask.shape[1] != 1:
            raise ValueError("text_mask must have shape [B, 1, T]")
        if (
            text_ids.shape[0] != style_ttl.shape[0]
            or text_ids.shape[0] != text_mask.shape[0]
        ):
            raise ValueError("input batch sizes must match")
        if text_ids.shape[1] != text_mask.shape[2]:
            raise ValueError("text_ids and text_mask text lengths must match")

    def forward(
        self,
        text_ids: torch.Tensor,
        style_ttl: torch.Tensor,
        text_mask: torch.Tensor,
    ) -> torch.Tensor:
        self._validate_inputs(text_ids, style_ttl, text_mask)
        text = self.text_encoder(text_ids, text_mask)
        style_key = self.style_encoder.style_token_layer.style_key
        return self.speech_prompted_text_encoder(text, style_key, style_ttl, text_mask)
