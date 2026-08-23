# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch
from torch import nn
from torch.nn import functional as F


class SamePad1d(nn.Module):
    def __init__(self, kernel_size: int, dilation: int = 1) -> None:
        super().__init__()
        total_padding = dilation * (kernel_size - 1)
        self.padding = (total_padding // 2, total_padding - total_padding // 2)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return F.pad(inputs, self.padding, mode="replicate")


class LayerNorm1d(nn.Module):
    def __init__(self, channels: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(channels, eps=eps)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.norm(inputs.transpose(1, 2)).transpose(1, 2)


class LinearProjection(nn.Module):
    def __init__(self, in_features: int, out_features: int, bias: bool = True) -> None:
        super().__init__()
        self.linear = nn.Linear(in_features, out_features, bias=bias)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.linear(inputs)


class Conv1dProjection(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        bias: bool = True,
    ) -> None:
        super().__init__()
        self.net = nn.Conv1d(
            in_channels, out_channels, kernel_size=kernel_size, bias=bias
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.net(inputs)


class ConvNeXtBlock(nn.Module):
    def __init__(
        self,
        channels: int,
        kernel_size: int,
        dilation: int = 1,
        expansion: int = 4,
        layer_scale_init_value: float = 1e-6,
    ) -> None:
        super().__init__()
        self.pad = SamePad1d(kernel_size, dilation)
        self.dwconv = nn.Conv1d(
            channels,
            channels,
            kernel_size=kernel_size,
            dilation=dilation,
            groups=channels,
        )
        self.norm = LayerNorm1d(channels)
        self.pwconv1 = nn.Conv1d(channels, expansion * channels, kernel_size=1)
        self.act = nn.GELU()
        self.pwconv2 = nn.Conv1d(expansion * channels, channels, kernel_size=1)
        self.gamma = nn.Parameter(torch.full((1, channels, 1), layer_scale_init_value))

    def forward(
        self, inputs: torch.Tensor, mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        residual = inputs if mask is None else inputs * mask
        hidden = self.dwconv(self.pad(residual))
        if mask is not None:
            hidden = hidden * mask
        hidden = self.pwconv2(self.act(self.pwconv1(self.norm(hidden))))
        output = residual + self.gamma * hidden
        return output if mask is None else output * mask


class ConvNeXt(nn.Module):
    def __init__(
        self,
        channels: int,
        num_layers: int,
        kernel_size: int,
        dilations: tuple[int, ...] | None = None,
        expansion: int = 4,
        layer_scale_init_value: float = 1e-6,
    ) -> None:
        super().__init__()
        if dilations is None:
            dilations = (1,) * num_layers
        if len(dilations) != num_layers:
            raise ValueError("dilations must contain one value per layer")
        self.convnext = nn.ModuleList(
            [
                ConvNeXtBlock(
                    channels,
                    kernel_size,
                    dilation=dilation,
                    expansion=expansion,
                    layer_scale_init_value=layer_scale_init_value,
                )
                for dilation in dilations
            ]
        )

    def forward(
        self, inputs: torch.Tensor, mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        output = inputs
        for block in self.convnext:
            output = block(output, mask)
        return output


class MultiHeadAttention(nn.Module):
    def __init__(
        self,
        channels: int,
        num_heads: int,
        context_channels: int | None = None,
        attention_channels: int | None = None,
        bias: bool = True,
    ) -> None:
        super().__init__()
        context_channels = channels if context_channels is None else context_channels
        attention_channels = (
            channels if attention_channels is None else attention_channels
        )
        if attention_channels % num_heads != 0:
            raise ValueError("attention_channels must be divisible by num_heads")
        self.num_heads = num_heads
        self.head_channels = attention_channels // num_heads
        self.W_query = LinearProjection(channels, attention_channels, bias=bias)
        self.W_key = LinearProjection(context_channels, attention_channels, bias=bias)
        self.W_value = LinearProjection(context_channels, attention_channels, bias=bias)
        self.out_fc = LinearProjection(attention_channels, channels, bias=bias)

    def _split_heads(self, inputs: torch.Tensor) -> torch.Tensor:
        batch, length, _ = inputs.shape
        return inputs.reshape(
            batch, length, self.num_heads, self.head_channels
        ).transpose(1, 2)

    def forward(
        self,
        inputs: torch.Tensor,
        context: torch.Tensor | None = None,
        *,
        query_mask: torch.Tensor | None = None,
        key_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        context = inputs if context is None else context
        query = self._split_heads(self.W_query(inputs))
        key = self._split_heads(self.W_key(context))
        value = self._split_heads(self.W_value(context))
        scores = torch.matmul(query, key.transpose(-1, -2)) / (self.head_channels**0.5)
        valid_keys = None
        if key_mask is not None:
            valid_keys = key_mask != 0
            scores = scores.masked_fill(
                ~valid_keys[:, None, None, :],
                torch.finfo(scores.dtype).min,
            )
        weights = torch.softmax(scores, dim=-1)
        if valid_keys is not None:
            weights = weights * valid_keys[:, None, None, :]
        attended = torch.matmul(weights, value)
        attended = attended.transpose(1, 2).reshape(
            inputs.shape[0], inputs.shape[1], -1
        )
        output = self.out_fc(attended)
        if valid_keys is not None:
            output = output * valid_keys.any(dim=-1)[:, None, None]
        if query_mask is not None:
            output = output * query_mask[:, :, None]
        return output


class AddConditioning(nn.Module):
    def __init__(self, condition_features: int, channels: int) -> None:
        super().__init__()
        self.linear = LinearProjection(condition_features, channels)

    def forward(self, inputs: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        return inputs + self.linear(condition).unsqueeze(-1)
