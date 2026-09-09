# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import math

import torch
from torch import nn

from ..model.duration_predictor import DurationPredictor
from ..model.layers import ConvNeXtBlock
from ..model.text_encoder import RelativeMultiHeadAttention, TextEncoder
from ..model.vector_estimator import VectorEstimator
from ..model.vocoder import Vocoder


class MLXCausalPad1d(nn.Module):
    """Replicate the first frame without dynamic clamp bounds."""

    def __init__(self, padding: int) -> None:
        super().__init__()
        self.padding = padding

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if self.padding == 0:
            return inputs
        prefix = inputs[:, :, :1].expand(-1, -1, self.padding)
        return torch.cat((prefix, inputs), dim=-1)


class MLXSamePad1d(nn.Module):
    """Replicate both boundary frames without dynamic clamp bounds."""

    def __init__(self, padding: tuple[int, int]) -> None:
        super().__init__()
        self.padding = padding

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        left, right = self.padding
        pieces = []
        if left:
            pieces.append(inputs[:, :, :1].expand(-1, -1, left))
        pieces.append(inputs)
        if right:
            pieces.append(inputs[:, :, -1:].expand(-1, -1, right))
        return torch.cat(pieces, dim=-1)


class MLXRelativeMultiHeadAttention(nn.Module):
    """Relative attention using dynamic-safe gather indices."""

    @classmethod
    def from_attention(
        cls, attention: RelativeMultiHeadAttention
    ) -> "MLXRelativeMultiHeadAttention":
        transformed = cls.__new__(cls)
        nn.Module.__init__(transformed)
        transformed.channels = attention.channels
        transformed.num_heads = attention.num_heads
        transformed.head_channels = attention.head_channels
        transformed.window_size = attention.window_size
        transformed.emb_rel_k = attention.emb_rel_k
        transformed.emb_rel_v = attention.emb_rel_v
        transformed.conv_q = attention.conv_q
        transformed.conv_k = attention.conv_k
        transformed.conv_v = attention.conv_v
        transformed.conv_o = attention.conv_o
        return transformed

    def _relative_embeddings(
        self, embeddings: torch.Tensor, length: int
    ) -> torch.Tensor:
        offsets = torch.arange(2 * length - 1, device=embeddings.device) - (length - 1)
        valid = (offsets >= -self.window_size) & (offsets <= self.window_size)
        indices = torch.where(
            valid,
            offsets + self.window_size,
            torch.zeros_like(offsets),
        )
        selected = torch.index_select(embeddings, 1, indices)
        return selected * valid.reshape(1, -1, 1)

    @staticmethod
    def _relative_to_absolute(inputs: torch.Tensor) -> torch.Tensor:
        length = inputs.shape[2]
        positions = torch.arange(length, device=inputs.device)
        relative_indices = positions.unsqueeze(0) - positions.unsqueeze(1) + length - 1
        linear_indices = positions.unsqueeze(1) * (2 * length - 1) + relative_indices
        selected = torch.index_select(
            inputs.reshape(inputs.shape[0], inputs.shape[1], -1),
            -1,
            linear_indices.reshape(-1),
        )
        return selected.reshape(inputs.shape[0], inputs.shape[1], length, length)

    @staticmethod
    def _absolute_to_relative(inputs: torch.Tensor) -> torch.Tensor:
        length = inputs.shape[2]
        query_positions = torch.arange(length, device=inputs.device)
        relative_positions = torch.arange(2 * length - 1, device=inputs.device)
        key_indices = (
            relative_positions.unsqueeze(0)
            + query_positions.unsqueeze(1)
            - (length - 1)
        )
        valid = (key_indices >= 0) & (key_indices < length)
        safe_indices = torch.where(valid, key_indices, torch.zeros_like(key_indices))
        linear_indices = query_positions.unsqueeze(1) * length + safe_indices
        relative = torch.index_select(
            inputs.reshape(inputs.shape[0], inputs.shape[1], -1),
            -1,
            linear_indices.reshape(-1),
        ).reshape(
            inputs.shape[0],
            inputs.shape[1],
            length,
            2 * length - 1,
        )
        return relative * valid.reshape(1, 1, length, 2 * length - 1)

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


class ExportableVectorEstimator(nn.Module):
    """Valid-domain compute; host validation is required before every call."""

    valid_domain_only = True

    def __init__(self, model: VectorEstimator) -> None:
        super().__init__()
        self.model = model

    def forward(
        self,
        noisy_latent: torch.Tensor,
        text_emb: torch.Tensor,
        style_ttl: torch.Tensor,
        latent_mask: torch.Tensor,
        text_mask: torch.Tensor,
        current_step: torch.Tensor,
        total_step: torch.Tensor,
    ) -> torch.Tensor:
        batch = noisy_latent.shape[0]
        text_unconditional = self.model.uncond_masker.text_special_token.expand(
            batch, -1, text_emb.shape[2]
        )
        style_key = torch.cat(
            (
                self.model.style_key.expand(batch, -1, -1),
                self.model.uncond_masker.style_key_special_token.expand(batch, -1, -1),
            ),
            dim=0,
        )
        style_value = torch.cat(
            (
                style_ttl,
                self.model.uncond_masker.style_value_special_token.expand(
                    batch, -1, -1
                ),
            ),
            dim=0,
        )
        vector = self.model.vector_field(
            noisy_latent.repeat(2, 1, 1),
            (current_step / total_step).repeat(2),
            torch.cat((text_emb, text_unconditional), dim=0),
            style_key,
            style_value,
            latent_mask.repeat(2, 1, 1),
            text_mask.repeat(2, 1, 1),
        )
        conditional, unconditional = vector.chunk(2, dim=0)
        guided = 4.0 * conditional - 3.0 * unconditional
        step = torch.reciprocal(total_step).reshape(-1, 1, 1)
        return (noisy_latent + step * guided) * latent_mask


def exportable_vector_estimator(
    model: VectorEstimator,
) -> ExportableVectorEstimator:
    return ExportableVectorEstimator(model)


def replace_vocoder_causal_padding(model: Vocoder) -> Vocoder:
    model.decoder.embed_pad = MLXCausalPad1d(model.decoder.embed_pad.padding)
    for block in model.decoder.convnext:
        block.pad = MLXCausalPad1d(block.pad.padding)
    model.decoder.head.pad = MLXCausalPad1d(model.decoder.head.pad.padding)
    return model


def replace_relative_attention(
    model: DurationPredictor | TextEncoder,
) -> DurationPredictor | TextEncoder:
    if isinstance(model, DurationPredictor):
        encoder = model.sentence_encoder.attn_encoder
    else:
        encoder = model.text_encoder.attn_encoder
    encoder.attn_layers = nn.ModuleList(
        MLXRelativeMultiHeadAttention.from_attention(attention)
        for attention in encoder.attn_layers
    )
    return model


def replace_same_padding(model: nn.Module) -> nn.Module:
    for module in model.modules():
        if isinstance(module, ConvNeXtBlock):
            module.pad = MLXSamePad1d(module.pad.padding)
    return model
