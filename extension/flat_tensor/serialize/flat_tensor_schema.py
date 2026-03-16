# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from dataclasses import dataclass
from typing import List, Optional

from executorch.exir.tensor_layout import TensorLayout

# Note: check executorch/extension/data_format/flat_tensor.fbs for explanations of these fields.


@dataclass
class DataSegment:
    offset: int
    size: int


@dataclass
class NamedData:
    key: str
    segment_index: int
    tensor_layout: Optional[TensorLayout] = None


@dataclass
class FlatTensor:
    version: int
    segments: List[DataSegment]
    named_data: List[NamedData]
