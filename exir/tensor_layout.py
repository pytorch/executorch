# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe

from dataclasses import dataclass
from typing import List

import torch

from executorch.exir.scalar_type import ScalarType
from executorch.exir.tensor import dim_order_from_stride, scalar_type_enum


# Note: keep this in sync with the TensorLayout definition in
# executorch/extension/flat_tensor/serialize/flat_tensor.fbs
@dataclass
class TensorLayout:
    scalar_type: ScalarType
    sizes: List[int]
    dim_order: List[int]

    @classmethod
    def from_tensor(cls, tensor: torch.Tensor) -> "TensorLayout":
        if not (
            tensor.is_contiguous(memory_format=torch.contiguous_format)
            or tensor.is_contiguous(memory_format=torch.channels_last)
        ):
            raise ValueError(
                "Tensor is not contiguous. Please call .contiguous() before creating the TensorLayout."
            )
        return TensorLayout(
            scalar_type=scalar_type_enum(tensor.dtype),
            sizes=list(tensor.shape),
            dim_order=list(dim_order_from_stride(tensor.stride())),
        )
