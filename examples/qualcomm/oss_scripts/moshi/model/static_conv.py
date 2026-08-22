# Copyright (c) Kyutai, all rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# Copyright (c) Qualcomm Innovation Center, Inc.
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
import math
import typing as tp
import warnings

import torch
from moshi.modules.conv import apply_parametrization_norm, NormConv1d, StreamingConv1d
from moshi.modules.streaming import RawStreamingConv1d


class StaticRawStreamingConv1d(RawStreamingConv1d):
    def __init__(self, ignore_previous: bool = False, *args, **kwargs):
        super(RawStreamingConv1d, self).__init__(*args, **kwargs)
        assert self.padding[0] == 0, "Padding should be handled outside."
        assert (
            self.stride[0] <= self.kernel_size[0]
        ), "stride must be less than kernel_size."

        self.ignore_previous = ignore_previous

    # Static Mimi Changes
    # 1) If ignore_previous, return only output but not previous since it is an empty tensor.
    #    Refer to StaticStreamingConv1d's forward() comments for more detail
    # 2) Remove all states related variables
    # 3) Create previous tensor ahead of time, shape should be constant throughout execution.
    def forward(self, input: torch.Tensor, previous: torch.Tensor = None):
        stride = self.stride[0]
        # Effective kernel size accounting for dilation.
        kernel = (self.kernel_size[0] - 1) * self.dilation[0] + 1

        if not self.ignore_previous:
            input = torch.cat([previous, input], dim=-1)
        B, C, T = input.shape
        # We now compute the number of full convolution frames, i.e. the frames
        # that are ready to be computed.
        num_frames = max(0, int(math.floor((T - kernel) / stride) + 1))
        offset = num_frames * stride
        # We will compute `num_frames` outputs, and we are advancing by `stride`
        # for each of the frame, so we know the data before `stride * num_frames`
        # will never be used again.
        if num_frames > 0:
            out = super(RawStreamingConv1d, self).forward(input)
        else:
            # Not enough data as this point to output some new frames.
            out = torch.empty(
                B, self.out_channels, 0, device=input.device, dtype=input.dtype
            )
        if self.ignore_previous:
            return out
        else:
            previous = input[..., offset:]
            return out, previous


class StaticNormConv1d(NormConv1d):
    """Wrapper around Conv1d and normalization applied to this conv
    to provide a uniform interface across normalization approaches.
    """

    def __init__(
        self,
        *args,
        causal: bool = False,
        norm: str = "none",
        norm_kwargs: tp.Dict[str, tp.Any] = {},  # noqa: B006
        ignore_previous: bool = False,
        **kwargs,
    ):
        super(NormConv1d, self).__init__()
        self.conv = apply_parametrization_norm(
            StaticRawStreamingConv1d(ignore_previous, *args, **kwargs), norm
        )
        self.norm_type = norm
        self.ignore_previous = ignore_previous

    def forward(self, x, previous=None):
        if self.ignore_previous:
            return self.conv(x)
        else:
            return self.conv(x, previous)


class StaticStreamingConv1d(StreamingConv1d):
    """Conv1d with some builtin handling of asymmetric or causal padding
    and normalization.
    """

    # Static Mimi Change
    # 1) Add ignore_previous variable. Some previous tensor has shape such as (1, 512, 0), which is an empty tensor.
    #    These shapes does not really makes sense in QNN, so just ignore for conv_layers that has empty tensor.
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        dilation: int = 1,
        groups: int = 1,
        bias: bool = True,
        causal: bool = False,
        norm: str = "none",
        norm_kwargs: tp.Dict[str, tp.Any] = {},  # noqa: B006
        pad_mode: str = "reflect",
        ignore_previous: bool = False,
    ):
        super(StreamingConv1d, self).__init__()
        # warn user on unusual setup between dilation and stride
        if stride > 1 and dilation > 1:
            warnings.warn(  # noqa: B028
                "StreamingConv1d has been initialized with stride > 1 and dilation > 1"
                f" (kernel_size={kernel_size} stride={stride}, dilation={dilation})."
            )
        self.conv = StaticNormConv1d(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            dilation=dilation,
            groups=groups,
            bias=bias,
            causal=causal,
            norm=norm,
            norm_kwargs=norm_kwargs,
            ignore_previous=ignore_previous,
        )
        self.causal = causal
        self.pad_mode = pad_mode
        self.ignore_previous = ignore_previous

    # Static Mimi Changes
    # 1) Remove all non streaming mode logic.
    # 2) Always perform padding since we need constant output shape
    def forward(self, x, previous=None):
        if self.ignore_previous:
            return self.conv(x)
        else:
            return self.conv(x, previous)
