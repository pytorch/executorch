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
from .layers import Conv1dProjection, ConvNeXt, LayerNorm1d, LinearProjection


class Mish(nn.Module):
    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return inputs * torch.tanh(F.softplus(inputs))


class TimeEncoder(nn.Module):
    def __init__(self, time_dim: int = 64, hidden_channels: int = 256) -> None:
        super().__init__()
        if time_dim <= 0 or time_dim % 2 != 0:
            raise ValueError("time_dim must be a positive even number")
        half_dim = time_dim // 2
        frequencies = 10000.0 ** (
            -torch.arange(half_dim, dtype=torch.float32) / max(half_dim - 1, 1)
        )
        self.register_buffer("frequencies", frequencies, persistent=False)
        self.mlp = nn.ModuleList(
            [
                LinearProjection(time_dim, hidden_channels),
                Mish(),
                LinearProjection(hidden_channels, time_dim),
            ]
        )

    def forward(self, time: torch.Tensor) -> torch.Tensor:
        angles = time.reshape(-1, 1) * 1000.0 * self.frequencies
        hidden = torch.cat((torch.sin(angles), torch.cos(angles)), dim=-1)
        for layer in self.mlp:
            hidden = layer(hidden)
        return hidden.unsqueeze(-1)


class TimeConditioning(nn.Module):
    def __init__(self, channels: int, time_dim: int) -> None:
        super().__init__()
        self.linear = LinearProjection(time_dim, channels)

    def forward(
        self,
        inputs: torch.Tensor,
        time_embedding: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        condition = self.linear(time_embedding.transpose(1, 2)).transpose(1, 2)
        return (inputs + condition) * mask


class RotaryCrossAttention(nn.Module):
    def __init__(
        self,
        channels: int,
        context_channels: int,
        num_heads: int,
        max_positions: int = 1000,
        rotary_base: float = 10000.0,
        rotary_scale: float = 10.0,
        persistent_rotary_buffers: bool = True,
    ) -> None:
        super().__init__()
        if channels % num_heads != 0:
            raise ValueError("channels must be divisible by num_heads")
        head_channels = channels // num_heads
        if head_channels % 2 != 0:
            raise ValueError("attention head width must be even")
        self.num_heads = num_heads
        self.head_channels = head_channels
        self.score_scale = math.sqrt(context_channels)
        self.W_query = LinearProjection(channels, channels)
        self.W_key = LinearProjection(context_channels, channels)
        self.W_value = LinearProjection(context_channels, channels)
        self.out_fc = LinearProjection(channels, channels)
        self.register_buffer(
            "increments",
            torch.arange(max_positions, dtype=torch.int64).reshape(1, -1, 1),
            persistent=persistent_rotary_buffers,
        )
        theta = rotary_scale * rotary_base ** (
            -torch.arange(head_channels // 2, dtype=torch.float32)
            / (head_channels // 2)
        )
        self.register_buffer(
            "theta",
            theta.reshape(1, 1, -1),
            persistent=persistent_rotary_buffers,
        )

    def _split_heads(self, inputs: torch.Tensor) -> torch.Tensor:
        batch, length, _ = inputs.shape
        return inputs.reshape(
            batch, length, self.num_heads, self.head_channels
        ).permute(2, 0, 1, 3)

    @staticmethod
    def _apply_rotary(
        inputs: torch.Tensor,
        sine: torch.Tensor,
        cosine: torch.Tensor,
    ) -> torch.Tensor:
        first, second = inputs.chunk(2, dim=-1)
        sine = sine.unsqueeze(0)
        cosine = cosine.unsqueeze(0)
        return torch.cat(
            (
                first * cosine - second * sine,
                first * sine + second * cosine,
            ),
            dim=-1,
        )

    def _angles(self, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        length = mask.shape[2]
        positions = self.increments[:, :length].to(dtype=mask.dtype)
        positions = positions / mask.sum(dim=(1, 2)).reshape(-1, 1, 1)
        angles = positions * self.theta
        return torch.sin(angles), torch.cos(angles)

    def _scaled_scores(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
    ) -> torch.Tensor:
        return torch.matmul(query, key.transpose(-2, -1)) / self.score_scale

    def forward(
        self,
        inputs: torch.Tensor,
        context: torch.Tensor,
        query_mask: torch.Tensor,
        key_mask: torch.Tensor,
    ) -> torch.Tensor:
        query = self._split_heads(self.W_query(inputs))
        key = self._split_heads(self.W_key(context))
        value = self._split_heads(self.W_value(context))
        query_sine, query_cosine = self._angles(query_mask)
        key_sine, key_cosine = self._angles(key_mask)
        query = self._apply_rotary(query, query_sine, query_cosine)
        key = self._apply_rotary(key, key_sine, key_cosine)
        scores = self._scaled_scores(query, key)
        valid_keys = key_mask[:, 0, :] != 0
        scores = scores.masked_fill(
            ~valid_keys.unsqueeze(0).unsqueeze(2),
            float("-inf"),
        )
        weights = torch.softmax(scores, dim=-1)
        valid_queries = query_mask[:, 0, :] != 0
        weights = torch.where(
            valid_queries.unsqueeze(0).unsqueeze(-1),
            weights,
            torch.zeros_like(weights),
        )
        attended = torch.matmul(weights, value)
        attended = attended.permute(1, 2, 0, 3).reshape(
            inputs.shape[0], inputs.shape[1], -1
        )
        return self.out_fc(attended) * query_mask.transpose(1, 2)


class TextConditioning(nn.Module):
    def __init__(
        self,
        channels: int,
        text_channels: int,
        num_heads: int,
        max_positions: int,
        persistent_rotary_buffers: bool,
    ) -> None:
        super().__init__()
        self.attn = RotaryCrossAttention(
            channels,
            text_channels,
            num_heads,
            max_positions=max_positions,
            persistent_rotary_buffers=persistent_rotary_buffers,
        )
        self.norm = LayerNorm1d(channels)

    def forward(
        self,
        inputs: torch.Tensor,
        text: torch.Tensor,
        latent_mask: torch.Tensor,
        text_mask: torch.Tensor,
    ) -> torch.Tensor:
        residual = inputs * latent_mask
        attended = self.attn(
            residual.transpose(1, 2),
            text.transpose(1, 2),
            latent_mask,
            text_mask,
        ).transpose(1, 2)
        return self.norm(residual + attended) * latent_mask


class StyleAttention(nn.Module):
    def __init__(
        self,
        channels: int,
        style_channels: int,
        num_heads: int,
    ) -> None:
        super().__init__()
        if style_channels % num_heads != 0:
            raise ValueError("style_channels must be divisible by num_heads")
        self.num_heads = num_heads
        self.head_channels = style_channels // num_heads
        self.score_scale = math.sqrt(style_channels)
        self.W_query = LinearProjection(channels, style_channels)
        self.W_key = LinearProjection(style_channels, style_channels)
        self.W_value = LinearProjection(style_channels, style_channels)
        self.out_fc = LinearProjection(style_channels, channels)

    def _split_heads(self, inputs: torch.Tensor) -> torch.Tensor:
        return torch.stack(torch.split(inputs, self.head_channels, dim=-1), dim=0)

    def forward(
        self,
        inputs: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        query_mask: torch.Tensor,
    ) -> torch.Tensor:
        query = self._split_heads(self.W_query(inputs))
        projected_key = self._split_heads(self.W_key(key))
        projected_value = self._split_heads(self.W_value(value))
        scores = (
            torch.matmul(
                query,
                torch.tanh(projected_key.transpose(-2, -1)),
            )
            / self.score_scale
        )
        weights = torch.softmax(scores, dim=-1)
        weights = torch.where(
            query_mask.transpose(1, 2).unsqueeze(0) != 0,
            weights,
            torch.zeros_like(weights),
        )
        attended = torch.matmul(weights, projected_value)
        attended = torch.cat(torch.unbind(attended, dim=0), dim=-1)
        return self.out_fc(attended) * query_mask.transpose(1, 2)


class StyleConditioning(nn.Module):
    def __init__(
        self,
        channels: int,
        style_channels: int,
        num_heads: int,
    ) -> None:
        super().__init__()
        self.attention = StyleAttention(channels, style_channels, num_heads)
        self.norm = LayerNorm1d(channels)

    def forward(
        self,
        inputs: torch.Tensor,
        style_key: torch.Tensor,
        style_value: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        residual = inputs * mask
        attended = self.attention(
            residual.transpose(1, 2),
            style_key,
            style_value,
            mask,
        ).transpose(1, 2)
        return self.norm(residual + attended) * mask


class UnconditionalMasker(nn.Module):
    def __init__(
        self,
        text_channels: int,
        style_tokens: int,
        style_channels: int,
    ) -> None:
        super().__init__()
        self.text_special_token = nn.Parameter(torch.randn(1, text_channels, 1))
        self.style_key_special_token = nn.Parameter(
            torch.randn(1, style_tokens, style_channels)
        )
        self.style_value_special_token = nn.Parameter(
            torch.randn(1, style_tokens, style_channels)
        )


class VectorField(nn.Module):
    def __init__(
        self,
        latent_channels: int,
        hidden_channels: int,
        time_dim: int,
        time_hidden_channels: int,
        num_main_blocks: int,
        main_convnext_dilations: tuple[int, ...],
        post_time_dilations: tuple[int, ...],
        post_text_dilations: tuple[int, ...],
        final_dilations: tuple[int, ...],
        text_channels: int,
        style_channels: int,
        attention_heads: int,
        style_attention_heads: int,
        max_positions: int,
    ) -> None:
        super().__init__()
        self.proj_in = Conv1dProjection(latent_channels, hidden_channels, 1, bias=False)
        self.time_encoder = TimeEncoder(time_dim, time_hidden_channels)
        blocks: list[nn.Module] = []
        for block_index in range(num_main_blocks):
            blocks.extend(
                (
                    ConvNeXt(
                        hidden_channels,
                        len(main_convnext_dilations),
                        kernel_size=5,
                        dilations=main_convnext_dilations,
                    ),
                    TimeConditioning(hidden_channels, time_dim),
                    ConvNeXt(
                        hidden_channels,
                        len(post_time_dilations),
                        kernel_size=5,
                        dilations=post_time_dilations,
                    ),
                    TextConditioning(
                        hidden_channels,
                        text_channels,
                        attention_heads,
                        max_positions,
                        block_index == 0,
                    ),
                    ConvNeXt(
                        hidden_channels,
                        len(post_text_dilations),
                        kernel_size=5,
                        dilations=post_text_dilations,
                    ),
                    StyleConditioning(
                        hidden_channels,
                        style_channels,
                        style_attention_heads,
                    ),
                )
            )
        self.main_blocks = nn.ModuleList(blocks)
        self.last_convnext = ConvNeXt(
            hidden_channels,
            len(final_dilations),
            kernel_size=5,
            dilations=final_dilations,
        )
        self.proj_out = Conv1dProjection(
            hidden_channels, latent_channels, 1, bias=False
        )
        self.num_main_blocks = num_main_blocks

    def forward(
        self,
        noisy_latent: torch.Tensor,
        time: torch.Tensor,
        text: torch.Tensor,
        style_key: torch.Tensor,
        style_value: torch.Tensor,
        latent_mask: torch.Tensor,
        text_mask: torch.Tensor,
    ) -> torch.Tensor:
        hidden = self.proj_in(noisy_latent) * latent_mask
        time_embedding = self.time_encoder(time)
        for block_index in range(self.num_main_blocks):
            offset = block_index * 6
            hidden = self.main_blocks[offset](hidden, latent_mask)
            hidden = self.main_blocks[offset + 1](hidden, time_embedding, latent_mask)
            hidden = self.main_blocks[offset + 2](hidden, latent_mask)
            hidden = self.main_blocks[offset + 3](hidden, text, latent_mask, text_mask)
            hidden = self.main_blocks[offset + 4](hidden, latent_mask)
            hidden = self.main_blocks[offset + 5](
                hidden, style_key, style_value, latent_mask
            )
        hidden = self.last_convnext(hidden, latent_mask)
        return self.proj_out(hidden) * latent_mask


class VectorEstimator(nn.Module):
    def __init__(
        self,
        config: TTSConfig,
        *,
        hidden_channels: int = 512,
        time_dim: int = 64,
        time_hidden_channels: int = 256,
        num_main_blocks: int = 4,
        main_convnext_dilations: tuple[int, ...] = (1, 2, 4, 8),
        post_time_dilations: tuple[int, ...] = (1,),
        post_text_dilations: tuple[int, ...] = (1,),
        final_dilations: tuple[int, ...] = (1, 1, 1, 1),
        text_channels: int = 256,
        style_tokens: int = 50,
        style_channels: int = 256,
        attention_heads: int = 8,
        style_attention_heads: int = 2,
        max_positions: int = 1000,
    ) -> None:
        super().__init__()
        if config.ttl.latent_dim <= 0 or config.ttl.chunk_compress_factor <= 0:
            raise ValueError("config.ttl dimensions must be positive")
        latent_channels = config.ttl.latent_dim * config.ttl.chunk_compress_factor
        self.uncond_masker = UnconditionalMasker(
            text_channels, style_tokens, style_channels
        )
        self.style_key = nn.Parameter(torch.randn(1, style_tokens, style_channels))
        self.vector_field = VectorField(
            latent_channels,
            hidden_channels,
            time_dim,
            time_hidden_channels,
            num_main_blocks,
            main_convnext_dilations,
            post_time_dilations,
            post_text_dilations,
            final_dilations,
            text_channels,
            style_channels,
            attention_heads,
            style_attention_heads,
            max_positions,
        )
        self.latent_channels = latent_channels
        self.text_channels = text_channels
        self.style_tokens = style_tokens
        self.style_channels = style_channels
        self.max_positions = max_positions

    def _validate_inputs(  # noqa: C901
        self,
        noisy_latent: torch.Tensor,
        text_emb: torch.Tensor,
        style_ttl: torch.Tensor,
        latent_mask: torch.Tensor,
        text_mask: torch.Tensor,
        current_step: torch.Tensor,
        total_step: torch.Tensor,
    ) -> None:
        if noisy_latent.ndim != 3 or noisy_latent.shape[1] != self.latent_channels:
            raise ValueError(
                f"noisy_latent must have shape [B, {self.latent_channels}, L]"
            )
        if text_emb.ndim != 3 or text_emb.shape[1] != self.text_channels:
            raise ValueError(f"text_emb must have shape [B, {self.text_channels}, T]")
        if style_ttl.ndim != 3 or style_ttl.shape[1:] != (
            self.style_tokens,
            self.style_channels,
        ):
            raise ValueError(
                f"style_ttl must have shape [B, {self.style_tokens}, "
                f"{self.style_channels}]"
            )
        if latent_mask.ndim != 3 or latent_mask.shape[1] != 1:
            raise ValueError("latent_mask must have shape [B, 1, L]")
        if text_mask.ndim != 3 or text_mask.shape[1] != 1:
            raise ValueError("text_mask must have shape [B, 1, T]")
        if current_step.ndim != 1:
            raise ValueError("current_step must have shape [B]")
        if total_step.ndim != 1:
            raise ValueError("total_step must have shape [B]")
        batch = noisy_latent.shape[0]
        if batch <= 0:
            raise ValueError("batch size must be positive")
        if noisy_latent.shape[2] <= 0:
            raise ValueError("latent length must be positive")
        if text_emb.shape[2] <= 0:
            raise ValueError("text length must be positive")
        if any(
            value.shape[0] != batch
            for value in (
                text_emb,
                style_ttl,
                latent_mask,
                text_mask,
                current_step,
                total_step,
            )
        ):
            raise ValueError("input batch sizes must match")
        if noisy_latent.shape[2] != latent_mask.shape[2]:
            raise ValueError("noisy_latent and latent_mask latent lengths must match")
        if text_emb.shape[2] != text_mask.shape[2]:
            raise ValueError("text_emb and text_mask text lengths must match")
        if noisy_latent.shape[2] > self.max_positions:
            raise ValueError(f"latent length must not exceed {self.max_positions}")
        if text_emb.shape[2] > self.max_positions:
            raise ValueError(f"text length must not exceed {self.max_positions}")
        latent_valid_counts = latent_mask.sum(dim=(1, 2))
        if not torch.all(
            torch.isfinite(latent_valid_counts) & (latent_valid_counts > 0)
        ).item():
            raise ValueError("latent_mask must contain a valid position per sample")
        text_valid_counts = text_mask.sum(dim=(1, 2))
        if not torch.all(
            torch.isfinite(text_valid_counts) & (text_valid_counts > 0)
        ).item():
            raise ValueError("text_mask must contain a valid position per sample")
        if not torch.all(torch.isfinite(current_step)).item():
            raise ValueError("current_step must be finite")
        if not torch.all(torch.isfinite(total_step) & (total_step > 0)).item():
            raise ValueError("total_step must be finite and positive")

    @staticmethod
    def _apply_guidance(vector: torch.Tensor) -> torch.Tensor:
        conditional, unconditional = vector.chunk(2, dim=0)
        return 4.0 * conditional - 3.0 * unconditional

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
        self._validate_inputs(
            noisy_latent,
            text_emb,
            style_ttl,
            latent_mask,
            text_mask,
            current_step,
            total_step,
        )
        batch = noisy_latent.shape[0]
        text_unconditional = self.uncond_masker.text_special_token.expand(
            batch, -1, text_emb.shape[2]
        )
        style_key = torch.cat(
            (
                self.style_key.expand(batch, -1, -1),
                self.uncond_masker.style_key_special_token.expand(batch, -1, -1),
            ),
            dim=0,
        )
        style_value = torch.cat(
            (
                style_ttl,
                self.uncond_masker.style_value_special_token.expand(batch, -1, -1),
            ),
            dim=0,
        )
        vector = self.vector_field(
            noisy_latent.repeat(2, 1, 1),
            (current_step / total_step).repeat(2),
            torch.cat((text_emb, text_unconditional), dim=0),
            style_key,
            style_value,
            latent_mask.repeat(2, 1, 1),
            text_mask.repeat(2, 1, 1),
        )
        guided = self._apply_guidance(vector)
        step = torch.reciprocal(total_step).reshape(-1, 1, 1)
        return (noisy_latent + step * guided) * latent_mask
