# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

"""
Please refer to fbcode/caffe2/executorch/backends/vulkan/serialization/schema/schema.fbs for the schema definitions
"""

from dataclasses import dataclass
from enum import IntEnum
from typing import List


@dataclass
class OperatorCall:
    name: str
    args: List[int]


class VkDataType(IntEnum):
    fp32 = 0


@dataclass
class VkTensor:
    datatype: VkDataType
    dims: List[int]
    constant_id: int
    mem_obj_id: int


@dataclass
class VkScalar:
    pass


@dataclass
class VkValue:
    value: VkTensor


@dataclass
class VkBytes:
    offset: int
    length: int


@dataclass
class VkGraph:
    version: str

    chain: List[OperatorCall]
    values: List[VkValue]

    input_ids: List[int]
    output_ids: List[int]

    constants: List[VkBytes]
    shaders: List[VkBytes]
