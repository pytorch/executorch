# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from dataclasses import dataclass
from enum import IntEnum
from typing import List, Optional, Union

from executorch.exir.backend.compile_spec_schema import CompileSpec

from executorch.exir.scalar_type import ScalarType


@dataclass
class AllocationDetails:
    memory_id: int
    # Low 32 bits
    memory_offset_low: int
    # High 32 bits (typically zero)
    memory_offset_high: int

    @property
    def memory_offset(self) -> int:
        return self.memory_offset_low | (self.memory_offset_high << 32)


@dataclass
class OptionalTensorList:
    items: List[int]


class TensorShapeDynamism(IntEnum):
    """
    Check schema.fbs for explanations of this enum.
    """

    STATIC = 0
    DYNAMIC_BOUND = 1
    DYNAMIC_UNBOUND = 2


@dataclass
class Tensor:
    scalar_type: ScalarType
    storage_offset: int
    sizes: List[int]
    dim_order: List[bytes]
    requires_grad: bool
    layout: int
    data_buffer_idx: int
    allocation_info: Optional[AllocationDetails]

    # check schema.fbs for explanations
    shape_dynamism: TensorShapeDynamism


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
    double_val: Union[float, str]

    def __init__(self, double_val: float) -> None:
        if double_val == float("inf"):
            self.double_val = "inf"
        elif double_val == float("-inf"):
            self.double_val = "-inf"
        else:
            self.double_val = double_val

    def __post_init__(self) -> None:
        if isinstance(self.double_val, str):
            assert self.double_val in ["inf", "-inf"]
        else:
            assert isinstance(self.double_val, float)
            assert not self.double_val == float("inf")
            assert not self.double_val == float("-inf")


@dataclass
class String:
    string_val: str


@dataclass
class ContainerMetadata:
    encoded_inp_str: str
    encoded_out_str: str


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
class TensorList:
    items: List[int]


KernelTypes = Union[
    Int,
    Double,
    Bool,
    String,
    Tensor,
    IntList,
    BoolList,
    DoubleList,
    TensorList,
    Null,
    OptionalTensorList,
]


@dataclass
class EValue:
    # Union types must be specified as strings so DataclassEncoder can see them.
    val: "KernelTypes"


@dataclass
class Buffer:
    storage: bytes


@dataclass
class BackendDelegateInlineData:
    data: bytes


@dataclass
class KernelCall:
    op_index: int
    args: List[int]


@dataclass
class DelegateCall:
    delegate_index: int
    args: List[int]


@dataclass
class MoveCall:
    move_from: int
    move_to: int


@dataclass
class JumpFalseCall:
    cond_value_index: int
    destination_instruction: int


@dataclass
class FreeCall:
    value_index: int


InstructionArguments = Union[
    KernelCall,
    DelegateCall,
    MoveCall,
    JumpFalseCall,
    FreeCall,
]


@dataclass
class Instruction:
    instr_args: "InstructionArguments"


@dataclass
class Frame:
    filename: str
    lineno: int
    name: str
    context: str


@dataclass
class FrameList:
    items: List[Frame]


class DataLocation(IntEnum):
    INLINE = 0
    SEGMENT = 1


@dataclass
class BackendDelegateDataReference:
    location: DataLocation
    index: int


@dataclass
class BackendDelegate:
    id: str
    processed: BackendDelegateDataReference
    compile_specs: List[CompileSpec]


@dataclass
class Chain:
    inputs: List[int]
    outputs: List[int]
    instructions: List[Instruction]
    stacktrace: Optional[List[FrameList]]


@dataclass
class Operator:
    name: str
    overload: str


@dataclass
class ExecutionPlan:
    name: str
    container_meta_type: ContainerMetadata
    values: List[EValue]
    inputs: List[int]
    outputs: List[int]
    chains: List[Chain]
    operators: List[Operator]
    delegates: List[BackendDelegate]
    # the list index is memory buffer id, the value is the memory buffer size.
    # memory_buffer_id == 0 is special and is for constant memory buffer.
    # Runtime should use the len(constant_buffer) as the ground truch of
    # constant memory buffer size, and ignore non_const_buffer_sizes[0].
    non_const_buffer_sizes: List[int]


@dataclass
class DataSegment:
    offset: int
    size: int


@dataclass
class SubsegmentOffsets:
    segment_index: int
    offsets: List[int]


@dataclass
class Program:
    version: int
    execution_plan: List[ExecutionPlan]
    constant_buffer: List[Buffer]
    backend_delegate_data: List[BackendDelegateInlineData]
    segments: List[DataSegment]
    constant_segment: SubsegmentOffsets
    mutable_data_segments: Optional[List[SubsegmentOffsets]] = None
