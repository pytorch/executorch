# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch


# Get the output size of a 1D convolution given the input size and parameters
def get_conv1d_output_size(
    in_size: torch.Size,
    out_channels: int,
    stride: int,
    padding: int,
    dilation: int,
    kernel_size: int,
) -> torch.Size:
    assert len(in_size) == 3
    N, C, L = in_size

    # Reference: https://pytorch.org/docs/stable/generated/torch.nn.Conv1d.html
    lout = (L + 2 * padding - dilation * (kernel_size - 1) - 1) // stride + 1

    return torch.Size((in_size[0], out_channels, lout))
