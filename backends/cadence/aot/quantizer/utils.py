# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from math import frexp, isclose, trunc
from typing import List, Tuple

import torch
from torch import fx
from torch.ao.quantization import ObserverOrFakeQuantize

from torch.fx import GraphModule


def quantize_tensor_multiplier(
    requantize_scale_tensor: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Given requantize_scale_tensor with values in the interval (0, 1),
    produce a pair of tensors (out_multiplier, right_shift) where out_multiplier
    is an int32 tensor representing fixed-point values in the interval [-1, 1),
    and right_shift is an amount to shift right by, so that the floating-point
    multiplication of some int32 input with each value of requantize_scale_tensor:
        result = int32_value * requantize_scale_tensors[i]
    is best approximated by the integer-arithmetic-only code:
        result = RoundingRightShift(FixedPointMultiplication(int32_value,
                                    out_multiplier[i]), right_shift[i])
    """

    # This is identical to C++11 std::round(). The general python round rounds
    # down, and C++ rounds away from zero.
    def round_away_zero(f) -> int:
        r = -0.5 if (f < 0) else 0.5
        return trunc(f + r)

    def quantize_scalar_multiplier(requantize_scale: float) -> Tuple[int, int]:
        significand, exponent = frexp(requantize_scale)
        significand_q31 = int(round_away_zero(significand * (1 << 31)))
        # Handle the special case when the real multiplier was so close to 1
        # that its fixed-point approximation was indistinguishable from 1.
        # We handle this by dividing it by two, incrementing exponent by 1.
        # the right shift amount.
        if significand_q31 == (1 << 31):
            significand_q31 //= 2
            exponent += 1

        # Verify that the decomposition of requantize_scale into significand
        # and exponent is correct.
        reconstructed = significand_q31 / (1 << 31) * pow(2, exponent)
        assert isclose(
            requantize_scale, reconstructed, rel_tol=1e-4, abs_tol=1e-4
        ), "computation of significand and exponent from requantize_scale is not accurate"

        return (significand_q31, exponent)

    # Flatten the input scale tensor so that we can operate on individual values
    orig_shape = requantize_scale_tensor.shape
    flattened_tensor = requantize_scale_tensor.flatten().to(torch.float32)
    out_multiplier = torch.zeros(flattened_tensor.shape, dtype=torch.int32)
    right_shift = torch.zeros(flattened_tensor.shape, dtype=torch.int32)

    # Iterate over the flattened scale tensor and compute the decomposition of
    # each value in scale tensor into significand(out_multiplier) and
    # exponent(right_shift)
    for idx, scale in enumerate(flattened_tensor):
        (si, ex) = quantize_scalar_multiplier(scale)
        out_multiplier[idx], right_shift[idx] = si, ex

    # Reshape the tensors back to the original shape
    out_multiplier = out_multiplier.reshape(orig_shape)
    right_shift = right_shift.reshape(orig_shape)

    return (out_multiplier, right_shift)


def is_annotated(nodes: List[fx.Node]) -> bool:
    annotated = False
    for node in nodes:
        annotated = annotated or (
            "quantization_annotation" in node.meta
            and node.meta["quantization_annotation"]._annotated
        )
    return annotated


def no_outside_users(fused_partition) -> bool:
    """
    Checks if each partition other than the last does not have any outside users.
    """
    for source_partition in fused_partition[:-1]:
        if len(source_partition.output_nodes) != 1:
            return False
        if len(source_partition.output_nodes[0].users) != 1:
            return False
    return True


def create_zero_bias_int32(
    graph_module: GraphModule,
    weight_node: fx.Node,
    bias_scale: float,
) -> fx.Node:
    """
    Creates a zero bias tensor with the shape of weight[0]
    """
    attr_node = getattr(graph_module, weight_node.target)
    weight_shape = list(attr_node.shape)
    bias_shape = weight_shape[0]
    return graph_module.graph.call_function(
        torch.ops.aten.full.default,
        ([bias_shape], 0.0),
        {"dtype": torch.int32},
    )


def get_bias_qparams(
    obs_or_fqs: List[ObserverOrFakeQuantize],
) -> Tuple[torch.Tensor, torch.Tensor]:
    act_scale, _ = obs_or_fqs[0].calculate_qparams()
    weight_scale, _ = obs_or_fqs[1].calculate_qparams()
    bias_scale = act_scale * weight_scale
    bias_zero_point = torch.zeros_like(bias_scale, dtype=torch.int32)
    return bias_scale, bias_zero_point


def get_conv_args(arg, first_val: int) -> List[fx.Node]:
    return arg if len(arg) == 2 else [first_val, arg[0]]
