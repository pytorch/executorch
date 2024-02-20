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


class VkDatatype(IntEnum):
    vk_datatype_fp32 = 0


@dataclass
class VkBytes:
    offset: int
    length: int


@dataclass
class VkTensor:
    datatype: VkDatatype
    dims: List[int]
    constant_buffer_idx: int
    mem_obj_id: int


@dataclass
class VkScalar:
    pass


@dataclass
class VkValue:
    value: VkTensor


class VkArithmeticOpType(IntEnum):
    vk_arithmetic_op_type_add = 0
    vk_arithmetic_op_type_sub = 1
    vk_arithmetic_op_type_mul = 2
    vk_arithmetic_op_type_div = 3
    vk_arithmetic_op_type_floor_div = 4
    vk_arithmetic_op_type_pow = 5


@dataclass
class VkArithmeticNode:
    input1_id: int
    input2_id: int
    output_id: int
    op_type: VkArithmeticOpType
    flags: int


@dataclass
class VkNode:
    node: VkArithmeticNode
    debug_handle: int


@dataclass
class VkGraph:
    version: str
    chain: List[VkNode]
    values: List[VkValue]

    input_ids: List[int]
    output_ids: List[int]

    constants: List[VkBytes]
    shaders: List[VkBytes]
