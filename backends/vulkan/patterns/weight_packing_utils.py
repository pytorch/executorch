# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Optional

import torch
import torch.nn.functional as F


def pack_4bit_weight_tensor(
    weight_tensor: torch.Tensor,
    *,
    inner_dim_padding: Optional[int] = 8,
) -> torch.Tensor:
    """Pack signed 4-bit values stored in int8 into uint8 byte pairs.

    The input tensor stores one quantized value per byte in the range [-8, 7].
    The returned tensor stores two 4-bit values per byte along the innermost dim.

    This preserves the legacy linear q4gsw packing convention:
      packed_byte = (odd_val + 8) << 4 | (even_val + 8)
    """
    min_val, max_val = weight_tensor.min().item(), weight_tensor.max().item()
    assert (
        max_val <= 7 and min_val >= -8
    ), f"pack_4bit_weight_tensor: [min_val,max_val] out of [-8, 7] range, got [{min_val}, {max_val}]"

    if weight_tensor.ndim != 2:
        weight_tensor = weight_tensor.squeeze()
    assert (
        weight_tensor.ndim == 2
    ), f"pack_4bit_weight_tensor: expecting input tensor to be 2d, got {weight_tensor.ndim}"

    if (
        inner_dim_padding is not None
        and weight_tensor.shape[-1] % inner_dim_padding != 0
    ):
        num_pad = inner_dim_padding - (weight_tensor.shape[-1] % inner_dim_padding)
        weight_tensor = F.pad(input=weight_tensor, pad=(0, num_pad))

    assert (
        weight_tensor.shape[-1] % 2 == 0
    ), "pack_4bit_weight_tensor: expecting innermost dim to be divisible by 2"

    shifted_weight_tensor = weight_tensor.to(dtype=torch.uint8) + 8
    even_values = shifted_weight_tensor[:, ::2]
    odd_values = shifted_weight_tensor[:, 1::2]
    return odd_values << 4 | even_values
