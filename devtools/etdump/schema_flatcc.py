# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict
"""
This file is the python representation of the schema contained in
executorch/devtools/etdump/etdump_schema.fbs. Any changes made to that
flatbuffer schema should accordingly be reflected here also.
"""

from dataclasses import dataclass
from enum import Enum
from typing import List, Optional

from executorch.exir.scalar_type import ScalarType


@dataclass
class Tensor:
    scalar_type: ScalarType
    sizes: List[int]
    strides: List[int]
    offset: Optional[int]


@dataclass
class TensorList:
    tensors: List[Tensor]


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
class Float:
    float_val: float


@dataclass
class String:
    string_val: str


@dataclass
class ContainerMetadata:
    encoded_inp_str: str
    encoded_out_str: str


@dataclass
class ValueType(Enum):
    NULL = "Null"
    INT = "Int"
    BOOL = "Bool"
    FLOAT = "Float"
    DOUBLE = "Double"
    TENSOR = "Tensor"
    TENSOR_LIST = "TensorList"
    STRING = "String"


@dataclass
class Value:
    val: str  # Member of ValueType
    tensor: Optional[Tensor]
    tensor_list: Optional[TensorList]
    int_value: Optional[Int]
    float_value: Optional[Float]
    double_value: Optional[Double]
    bool_value: Optional[Bool]
    output: Optional[Bool]


@dataclass
class DebugEvent:
    name: Optional[str]
    chain_index: int
    instruction_id: int
    delegate_debug_id_int: Optional[int]
    delegate_debug_id_str: Optional[str]
    debug_entry: Value


# Note the differing value style is a result of ETDump string
class PROFILE_EVENT_ENUM(Enum):
    RUN_MODEL = "Method::execute"
    OPERATOR_CALL = "OPERATOR_CALL"
    DELEGATE_CALL = "DELEGATE_CALL"
    LOAD_MODEL = "Program::load_method"


@dataclass
class ProfileEvent:
    name: Optional[str]
    chain_index: int
    instruction_id: int
    delegate_debug_id_int: Optional[int]
    delegate_debug_id_str: Optional[str]
    delegate_debug_metadata: Optional[bytes]
    start_time: int
    end_time: int


@dataclass
class AllocationEvent:
    allocator_id: int
    allocation_size: int


@dataclass
class Allocator:
    name: str


# Must have one of profile_event, allocation_event, or debug_event
@dataclass
class Event:
    profile_event: Optional[ProfileEvent]
    allocation_event: Optional[AllocationEvent]
    debug_event: Optional[DebugEvent]


@dataclass
class RunData:
    name: str
    bundled_input_index: Optional[int]
    allocators: Optional[List[Allocator]]
    events: Optional[List[Event]]


@dataclass
class ETDumpFlatCC:
    version: int
    run_data: List[RunData]
