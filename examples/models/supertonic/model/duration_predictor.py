# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch
from torch import nn

from .config import TTSConfig
from .layers import Conv1dProjection
from .text_encoder import TextBackbone


class SentenceEncoder(TextBackbone):
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
        super().__init__(
            vocab_size,
            channels,
            convnext_dilations,
            attention_layers,
            attention_heads,
            ff_channels,
            relative_window,
        )
        self.sentence_token = nn.Parameter(torch.randn(1, channels, 1))
        self.proj_out = Conv1dProjection(channels, channels, 1, bias=False)

    def forward(self, text_ids: torch.Tensor, text_mask: torch.Tensor) -> torch.Tensor:
        text = self.text_embedder(text_ids, text_mask)
        token = self.sentence_token.expand(text.shape[0], -1, -1)
        hidden = torch.cat((token, text), dim=-1)
        token_mask = torch.ones_like(text_mask[:, :, :1])
        mask = torch.cat((token_mask, text_mask), dim=-1)
        hidden = self.convnext(hidden, mask)
        hidden = (hidden + self.attn_encoder(hidden, mask)) * mask
        sentence = hidden[:, :, :1]
        sentence_mask = mask[:, :, :1]
        return self.proj_out(sentence) * sentence_mask


class Predictor(nn.Module):
    def __init__(
        self,
        sentence_channels: int,
        style_tokens: int,
        style_dim: int,
        hidden_dim: int,
    ) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            [
                nn.Linear(sentence_channels + style_tokens * style_dim, hidden_dim),
                nn.Linear(hidden_dim, 1),
            ]
        )
        self.activation = nn.PReLU()

    def forward(self, sentence: torch.Tensor, style: torch.Tensor) -> torch.Tensor:
        combined = torch.cat(
            (
                sentence.reshape(sentence.shape[0], -1),
                style.reshape(style.shape[0], -1),
            ),
            dim=-1,
        )
        return torch.exp(
            self.layers[1](self.activation(self.layers[0](combined)))
        ).squeeze(-1)


class DurationPredictor(nn.Module):
    def __init__(
        self,
        config: TTSConfig,
        *,
        vocab_size: int = 8322,
        channels: int = 64,
        convnext_dilations: tuple[int, ...] = (1, 1, 1, 1, 1, 1),
        attention_layers: int = 2,
        attention_heads: int = 2,
        ff_channels: int = 256,
        relative_window: int = 4,
        style_tokens: int = 8,
        style_dim: int = 16,
        hidden_dim: int = 128,
    ) -> None:
        super().__init__()
        if config.dp.latent_dim <= 0 or config.dp.chunk_compress_factor <= 0:
            raise ValueError("config.dp dimensions must be positive")
        self.sentence_encoder = SentenceEncoder(
            vocab_size,
            channels,
            convnext_dilations,
            attention_layers,
            attention_heads,
            ff_channels,
            relative_window,
        )
        self.predictor = Predictor(
            channels,
            style_tokens,
            style_dim,
            hidden_dim,
        )
        self.style_tokens = style_tokens
        self.style_dim = style_dim

    def _validate_inputs(
        self,
        text_ids: torch.Tensor,
        style_dp: torch.Tensor,
        text_mask: torch.Tensor,
    ) -> None:
        if text_ids.ndim != 2:
            raise ValueError("text_ids must have shape [B, T]")
        if style_dp.ndim != 3 or style_dp.shape[1:] != (
            self.style_tokens,
            self.style_dim,
        ):
            raise ValueError(
                f"style_dp must have shape [B, {self.style_tokens}, {self.style_dim}]"
            )
        if text_mask.ndim != 3 or text_mask.shape[1] != 1:
            raise ValueError("text_mask must have shape [B, 1, T]")
        if (
            text_ids.shape[0] != style_dp.shape[0]
            or text_ids.shape[0] != text_mask.shape[0]
        ):
            raise ValueError("input batch sizes must match")
        if text_ids.shape[1] != text_mask.shape[2]:
            raise ValueError("text_ids and text_mask text lengths must match")

    def forward(
        self,
        text_ids: torch.Tensor,
        style_dp: torch.Tensor,
        text_mask: torch.Tensor,
    ) -> torch.Tensor:
        self._validate_inputs(text_ids, style_dp, text_mask)
        return self.predictor(
            self.sentence_encoder(text_ids, text_mask),
            style_dp,
        )
