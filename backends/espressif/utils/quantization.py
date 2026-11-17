# Copyright 2026 Mwaniki Njagi.
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from math import frexp, isfinite

import torch


def quantize_multiplier(real_multiplier: float) -> tuple[int, int]:
    if not isfinite(real_multiplier) or real_multiplier <= 0.0:
        raise ValueError(
            f"Quantized multiplier must be finite and positive, got {real_multiplier}"
        )
    significand, exponent = frexp(real_multiplier)
    multiplier = round(significand * (1 << 31))
    if multiplier == (1 << 31):
        multiplier >>= 1
        exponent += 1
    if exponent < -31 or exponent > 30:
        raise ValueError(
            f"Requantization multiplier {real_multiplier} is outside the ESP-NN shift range"
        )
    return int(multiplier), int(exponent)


def requantize(value: torch.Tensor, multiplier: int, shift: int) -> torch.Tensor:
    """ESP-NN's two-rounding Q31 rescale, for values whose left shift fits int32."""
    value = value.to(torch.int64) << max(shift, 0)
    # High multiplication rounds ties upward; division rounds ties away from zero.
    result = (value * multiplier + (1 << 30)) >> 31
    if shift < 0:
        mask = (1 << -shift) - 1
        remainder = result & mask
        result = result >> -shift
        threshold = (mask >> 1) + (result < 0).to(torch.int64)
        result = result + (remainder > threshold)
    return result.to(torch.int32)
