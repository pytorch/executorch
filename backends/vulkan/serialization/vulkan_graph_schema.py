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
from typing import List, Union


@dataclass
class OperatorCall:
    name: str
    args: List[int]


class VkDataType(IntEnum):
    BOOL = 0
    UINT8 = 1
    INT8 = 2
    INT32 = 3
    FLOAT16 = 4
    FLOAT32 = 5


@dataclass
class VkTensor:
    datatype: VkDataType
    dims: List[int]
    constant_id: int
    mem_obj_id: int


@dataclass
class Null:
    pass


@dataclass
class Int:
    int_val: int


@dataclass
class Bool:
    bool_val: bool


@dataclass
class Double:
    double_val: float


@dataclass
class IntList:
    items: List[int]


@dataclass
class DoubleList:
    items: List[float]


@dataclass
class BoolList:
    items: List[bool]


@dataclass
class ValueList:
    items: List[int]


@dataclass
class String:
    string_val: str


GraphTypes = Union[
    Null,
    Int,
    Double,
    Bool,
    VkTensor,
    IntList,
    BoolList,
    DoubleList,
    ValueList,
    String,
]


@dataclass
class VkValue:
    value: "GraphTypes"


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
