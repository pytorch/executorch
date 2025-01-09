# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


from dataclasses import dataclass
from typing import List

from executorch.exir.scalar_type import ScalarType

# Note: check executorch/extension/data_format/flat_tensor.fbs for explanations of these fields.


@dataclass
class TensorMetadata:
    fully_qualified_name: str
    scalar_type: ScalarType
    sizes: List[int]
    dim_order: List[bytes]

    segment_index: int
    offset: int


@dataclass
class DataSegment:
    offset: int
    size: int


@dataclass
class FlatTensor:
    version: int
    tensor_alignment: int
    tensors: List[TensorMetadata]
    segments: List[DataSegment]
