# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe

from dataclasses import dataclass
from typing import List

from executorch.exir.scalar_type import ScalarType


# Note: keep this in sync with the TensorLayout definition in
# executorch/extension/flat_tensor/serialize/flat_tensor.fbs
@dataclass
class TensorLayout:
    scalar_type: ScalarType
    sizes: List[int]
    dim_order: List[int]
