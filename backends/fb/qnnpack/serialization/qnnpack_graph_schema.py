# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

# pyre-strict

from dataclasses import dataclass
from typing import List


@dataclass
class ConstTensor:
    shape: List[int]
    buffer: bytes


@dataclass
class QNNDynamicLinear:
    input_shape: List[int]
    bias: ConstTensor
    weights: ConstTensor
    weights_zero_point: ConstTensor
    weights_scale: ConstTensor
