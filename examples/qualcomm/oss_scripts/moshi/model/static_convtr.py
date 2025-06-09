# Copyright (c) Kyutai, all rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# Copyright (c) Qualcomm Innovation Center, Inc.
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import typing as tp

import torch
from moshi.modules.conv import (
    apply_parametrization_norm,
    NormConvTranspose1d,
    StreamingConvTranspose1d,
)
from moshi.modules.resample import ConvTrUpsample1d
from moshi.modules.streaming import RawStreamingConvTranspose1d


class StaticRawStreamingConvTranspose1d(RawStreamingConvTranspose1d):
    def __init__(self, *args, **kwargs):
        super(RawStreamingConvTranspose1d, self).__init__(*args, **kwargs)
        assert self.padding[0] == 0, "Padding should be handled outside."
        assert self.dilation[0] == 1, "No dilation for now"
        assert (
            self.stride[0] <= self.kernel_size[0]
        ), "stride must be less than kernel_size."
        assert self.output_padding[0] == 0, "Output padding not supported."

    # Static Mimi Changes:
    # 1) Remove non streaming mode logic
    # 2) Remove all states related variables
    # 3) Create partial tensor ahead of time, shape should be constant throughout execution.
    def forward(self, x: torch.Tensor, partial: torch.Tensor):  # type: ignore
        # Batch, Channel, Temp_Dimension
        B, C, T = x.shape
        stride = self.stride[0]
        kernel = self.kernel_size[0]
        if T == 0:
            return torch.empty(B, self.out_channels, 0, device=x.device, dtype=x.dtype)
        out = super(RawStreamingConvTranspose1d, self).forward(x)

        OT = out.shape[-1]
        # Due to the potential overlap, the rightmost output of the conv transpose is not
        # ready to be output, as it will receive contributions from the next input frames.
        # Here we recover those `partial` output frames. We know that the first time step
        # of the `partial` tensor corresponds to the first time step of `out` as anything
        # coming before the first time step of `out` would have been already flushed.
        PT = partial.shape[-1]
        if self.bias is not None:
            condition = torch.all(partial != 0)
            updated_part = torch.where(
                condition, (partial - self.bias[:, None]), partial
            )
            out = torch.cat((out[..., :PT] + updated_part, out[..., PT:]), dim=-1)
        else:
            updated_part = out[..., :PT] + partial
            out = torch.cat((updated_part, out[..., PT:]), dim=-1)

        # The input is T, the output is S * (T - 1) + K.
        # The offset of the left of the next frame will be S * T
        # so everything between 0 and S * T is ready to be output, and we need
        # to keep in the internal state everything beyond that, i.e. S (T - 1) + K - S T = K - S
        invalid_steps = kernel - stride

        start_idx = OT - invalid_steps

        partial = out[..., start_idx:]
        out = out[..., :start_idx]
        return out, partial


class StaticNormConvTranspose1d(NormConvTranspose1d):
    """Wrapper around ConvTranspose1d and normalization applied to this conv
    to provide a uniform interface across normalization approaches.
    """

    def __init__(
        self,
        *args,
        causal: bool = False,
        norm: str = "none",
        norm_kwargs: tp.Dict[str, tp.Any] = {},  # noqa: B006
        **kwargs,
    ):
        super(NormConvTranspose1d, self).__init__()
        self.convtr = apply_parametrization_norm(
            StaticRawStreamingConvTranspose1d(*args, **kwargs), norm
        )
        self.norm_type = norm

    def forward(self, x, partial):
        return self.convtr(x, partial)


class StaticStreamingConvTranspose1d(StreamingConvTranspose1d):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        groups: int = 1,
        bias: bool = True,
        causal: bool = False,
        norm: str = "none",
        trim_right_ratio: float = 1.0,
        norm_kwargs: tp.Dict[str, tp.Any] = {},  # noqa: B006
    ):
        super(StreamingConvTranspose1d, self).__init__()
        self.convtr = StaticNormConvTranspose1d(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            groups=groups,
            bias=bias,
            causal=causal,
            norm=norm,
            norm_kwargs=norm_kwargs,
        )
        self.causal = causal
        self.trim_right_ratio = trim_right_ratio
        assert (
            self.causal or self.trim_right_ratio == 1.0
        ), "`trim_right_ratio` != 1.0 only makes sense for causal convolutions"
        assert self.trim_right_ratio >= 0.0 and self.trim_right_ratio <= 1.0

    # Static Mimi Changes:
    # 1) Remove non streaming mode logic
    def forward(self, x, partial):
        return self.convtr(x, partial)


class StaticConvTrUpsample1d(ConvTrUpsample1d):
    """
    Upsample by some integer amount `stride` using transposed convolutions.
    """

    def __init__(
        self,
        stride: int,
        dimension: tp.Optional[int] = None,
        causal: bool = False,
        learnt: bool = False,
        channel_wise: bool = False,
    ):
        super(ConvTrUpsample1d, self).__init__()
        self.learnt = learnt
        self.channel_wise = channel_wise
        groups = 1

        assert dimension is not None, "Dimension required for learnt convolutions."
        in_channels = dimension
        out_channels = dimension
        if channel_wise:
            groups = dimension

        self.convtr = StaticStreamingConvTranspose1d(
            in_channels,
            out_channels,
            kernel_size=2 * stride,
            stride=stride,
            causal=causal,
            groups=groups,
            bias=False,
        )

    # Static Mimi Changes:
    # 1) Remove not self.learnt logic since it doesn't go in.
    def forward(self, x: torch.Tensor, partial: torch.Tensor):
        return self.convtr(x, partial)
