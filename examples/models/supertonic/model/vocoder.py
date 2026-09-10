# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch
from torch import nn
from torch.nn import functional as F

from .config import TTSConfig
from .layers import Conv1dProjection, LayerNorm1d


class Normalizer(nn.Module):
    def __init__(self, scale: float = 0.25) -> None:
        super().__init__()
        self.register_buffer("scale", torch.tensor(scale))


class CausalPad1d(nn.Module):
    def __init__(self, kernel_size: int, dilation: int = 1) -> None:
        super().__init__()
        self.padding = dilation * (kernel_size - 1)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return F.pad(inputs, (self.padding, 0), mode="replicate")


class DecoderConvNeXtBlock(nn.Module):
    def __init__(
        self,
        channels: int,
        dilation: int,
        expansion: int,
    ) -> None:
        super().__init__()
        self.pad = CausalPad1d(7, dilation)
        self.dwconv = Conv1dProjection(channels, channels, 7)
        self.dwconv.net = nn.Conv1d(
            channels,
            channels,
            kernel_size=7,
            dilation=dilation,
            groups=channels,
        )
        self.norm = LayerNorm1d(channels)
        self.pwconv1 = nn.Conv1d(channels, expansion * channels, 1)
        self.act = nn.GELU()
        self.pwconv2 = nn.Conv1d(expansion * channels, channels, 1)
        self.gamma = nn.Parameter(torch.full((1, channels, 1), 1e-6))

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        hidden = self.dwconv(self.pad(inputs))
        hidden = self.pwconv2(self.act(self.pwconv1(self.norm(hidden))))
        return inputs + self.gamma * hidden


class InferenceBatchNorm1d(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.norm = nn.Module()
        self.norm.weight = nn.Parameter(torch.ones(channels))
        self.norm.bias = nn.Parameter(torch.zeros(channels))
        self.norm.register_buffer("running_mean", torch.zeros(channels))
        self.norm.register_buffer("running_var", torch.ones(channels))

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return F.batch_norm(
            inputs,
            self.norm.running_mean,
            self.norm.running_var,
            self.norm.weight,
            self.norm.bias,
            training=False,
            momentum=0.1,
            eps=1e-5,
        )


class SharedPReLU(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.tensor([[0.25]]))

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return torch.where(inputs >= 0, inputs, inputs * self.weight)


class DecoderHead(nn.Module):
    def __init__(
        self,
        channels: int = 512,
        hidden_channels: int = 2048,
        output_channels: int = 512,
    ) -> None:
        super().__init__()
        self.pad = CausalPad1d(3)
        self.layer1 = Conv1dProjection(channels, hidden_channels, 3)
        self.act = SharedPReLU()
        self.layer2 = nn.Conv1d(hidden_channels, output_channels, 1, bias=False)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        hidden = self.layer1(self.pad(inputs))
        hidden = self.layer2(self.act(hidden))
        return hidden.transpose(1, 2).reshape(hidden.shape[0], -1)


class Decoder(nn.Module):
    def __init__(
        self,
        latent_dim: int,
        channels: int,
        dilations: tuple[int, ...],
        expansion: int,
        head_hidden_channels: int,
        output_channels: int,
    ) -> None:
        super().__init__()
        self.embed_pad = CausalPad1d(7)
        self.embed = Conv1dProjection(latent_dim, channels, 7)
        self.convnext = nn.ModuleList(
            [
                DecoderConvNeXtBlock(channels, dilation, expansion)
                for dilation in dilations
            ]
        )
        self.final_norm = InferenceBatchNorm1d(channels)
        self.head = DecoderHead(channels, head_hidden_channels, output_channels)

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        hidden = self.embed(self.embed_pad(latent))
        for block in self.convnext:
            hidden = block(hidden)
        return self.head(self.final_norm(hidden))


class Vocoder(nn.Module):
    def __init__(
        self,
        config: TTSConfig,
        *,
        decoder_channels: int = 512,
        decoder_dilations: tuple[int, ...] = (1, 2, 4, 1, 2, 4, 1, 1, 1, 1),
        decoder_expansion: int = 4,
        head_hidden_channels: int = 2048,
    ) -> None:
        super().__init__()
        if config.ttl.latent_dim <= 0 or config.ttl.chunk_compress_factor <= 0:
            raise ValueError("config.ttl dimensions must be positive")
        if config.ae.latent_dim != config.ttl.latent_dim:
            raise ValueError("config ttl and autoencoder latent dimensions must match")
        if config.ae.base_chunk_size <= 0:
            raise ValueError("config.ae.base_chunk_size must be positive")
        self.normalizer = Normalizer()
        self.register_buffer(
            "latent_mean",
            torch.zeros(1, config.ae.latent_dim, 1),
        )
        self.register_buffer(
            "latent_std",
            torch.ones(1, config.ae.latent_dim, 1),
        )
        self.decoder = Decoder(
            config.ae.latent_dim,
            decoder_channels,
            decoder_dilations,
            decoder_expansion,
            head_hidden_channels,
            config.ae.base_chunk_size,
        )
        self.latent_dim = config.ttl.latent_dim
        self.compress_factor = config.ttl.chunk_compress_factor
        self.latent_channels = self.latent_dim * self.compress_factor

    @staticmethod
    def _unpack_latent(
        latent: torch.Tensor,
        *,
        latent_dim: int,
        compress_factor: int,
    ) -> torch.Tensor:
        batch, _, length = latent.shape
        return (
            latent.reshape(batch, latent_dim, compress_factor, length)
            .transpose(2, 3)
            .reshape(batch, latent_dim, length * compress_factor)
        )

    def _validate_input(self, latent: torch.Tensor) -> None:
        if latent.ndim != 3 or latent.shape[1] != self.latent_channels:
            raise ValueError(f"latent must have shape [B, {self.latent_channels}, L]")
        if latent.shape[2] <= 0:
            raise ValueError("latent length must be positive")

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        self._validate_input(latent)
        latent = self._unpack_latent(
            latent / self.normalizer.scale,
            latent_dim=self.latent_dim,
            compress_factor=self.compress_factor,
        )
        latent = latent * self.latent_std + self.latent_mean
        return self.decoder(latent)
