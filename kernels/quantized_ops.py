# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# this contains a defineLib that we want
import executorch.exir.passes._quant_patterns_and_replacements  # noqa

import torch
from torch.library import impl_abstract


@impl_abstract("quantized_decomposed::embedding_byte")
def embedding_byte_meta(
    weight,
    weight_scales,
    weight_zero_points,
    weight_quant_min,
    weight_quant_max,
    indices,
):
    assert weight.dtype in [
        torch.int8,
        torch.uint8,
    ], f"Expecting weights to be of dtype in [torch.int8, torch.uint8], but got {weight.dtype}"
    assert (
        weight.dim() == 2
    ), f"Expecting weight tensor to have dim()==2, but found {weight.dim()}"

    assert weight_scales.dtype in [
        torch.float16,
        torch.float32,
    ], f"Expecting weight_scales to be of dtype in [torch.float16, torch.float32], but got {weight_scales.dtype}"
    assert (
        weight_scales.dim() == 1
    ), f"Expecting weight_scales tensor to have dim()==1, but found {weight_scales.dim()}"
    assert weight_scales.size(0) == weight.size(
        0
    ), f"Expecting weight and scale tensor to have same number of rows, but found {weight.size()} and {weight_scales.size()}"

    assert (
        weight_zero_points is None or weight_zero_points.dtype == weight_scales.dtype
    ), "Expecting weight_zero_points to be None or have same dtype as weight_scales"
    assert (
        weight_zero_points is None or weight_zero_points.dim() == 1
    ), f"Expecting weight_zero_points tensor to be None or have dim()==1, but found {weight_zero_points.dim()}"
    assert weight_zero_points is None or weight_zero_points.size(0) == weight.size(
        0
    ), f"Expecting weight_zero_points tensor to be None or have same number of rows as weights, but found {weight.size()} and {weight_zero_points.size()}"

    output_shape = list(indices.size()) + [weight.size(1)]
    return torch.empty(output_shape, device="meta", dtype=weight_scales.dtype)
