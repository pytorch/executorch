# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict
"""
This file is the python representation of the schema contained in
executorch/sdk/etdump/etdump_schema.fbs. Any changes made to that
flatbuffer schema should accordingly be reflected here also.
"""

from dataclasses import dataclass
from enum import Enum
from typing import List, Union

from executorch.exir.scalar_type import ScalarType


@dataclass
class Tensor:
    scalar_type: ScalarType
    sizes: List[int]
    strides: List[int]
    data: bytes


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
class String:
    string_val: str


@dataclass
class ContainerMetadata:
    encoded_inp_str: str
    encoded_out_str: str


ValueUnion = Union[
    Int,
    Double,
    Bool,
    String,
    Tensor,
    Null,
]


@dataclass
class Value:
    val: "ValueUnion"


@dataclass
class DebugEvent:
    debug_handle: int
    debug_entries: List[Value]


@dataclass
class DebugBlock:
    name: str
    debug_events: List[DebugEvent]


# Note the differing value style is a result of ETDump string
class PROFILE_EVENT_ENUM(Enum):
    RUN_MODEL = "Method::execute"
    OPERATOR_CALL = "OPERATOR_CALL"
    DELEGATE_CALL = "DELEGATE_CALL"
    LOAD_MODEL = "Program::load_method"


@dataclass
class ProfileEvent:
    name: str
    debug_handle: int
    start_time: int
    end_time: int


@dataclass
class AllocationEvent:
    allocator_id: int
    allocation_size: int


@dataclass
class Allocator:
    name: str


@dataclass
class ProfileBlock:
    name: str
    allocators: List[Allocator]
    profile_events: List[ProfileEvent]
    allocation_events: List[AllocationEvent]


@dataclass
class RunData:
    debug_blocks: List[DebugBlock]
    profile_blocks: List[ProfileBlock]


@dataclass
class ETDump:
    version: int
    run_data: List[RunData]
