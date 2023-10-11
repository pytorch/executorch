# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

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
