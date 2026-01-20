#
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
#
# ============================================================================
# AUTO-GENERATED FILE - DO NOT EDIT MANUALLY
# ============================================================================
#
# This file was generated from schema.fbs by the MLX delegate code generator.
#
# Source:    backends/apple/mlx/serialization/schema.fbs
# Generator: backends/apple/mlx/serialization/generate.py
#
# To regenerate, run from the executorch root:
#     python backends/apple/mlx/serialization/generate.py
#
# ============================================================================

from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum
from typing import List, Optional, Union


# =============================================================================
# Enums
# =============================================================================

class DTypeId(IntEnum):
    f16 = 0
    f32 = 1
    bf16 = 2
    i32 = 3
    i64 = 4
    u32 = 5
    u8 = 6
    boolean = 7
    i8 = 8


class SlotType(IntEnum):
    TensorSlot = 0
    IntValueSlot = 1
    FloatValueSlot = 2
    BoolValueSlot = 3


# =============================================================================
# Core types
# =============================================================================

@dataclass
class Tid:
    idx: int


@dataclass
class Vid:
    idx: int


@dataclass
class IntOrVid:
    """Represents either a literal int or a runtime Vid reference."""
    literal: int = 0
    vid: Optional[Vid] = None
    is_vid: bool = False

    @classmethod
    def from_literal(cls, value: int) -> "IntOrVid":
        """Create an IntOrVid from a literal integer."""
        return cls(literal=value, is_vid=False)

    @classmethod
    def from_vid(cls, vid: Vid) -> "IntOrVid":
        """Create an IntOrVid from a Vid reference."""
        return cls(vid=vid, is_vid=True)


@dataclass
class FloatOrVid:
    """Represents either a literal float or a runtime Vid reference."""
    literal: float = 0.0
    vid: Optional[Vid] = None
    is_vid: bool = False

    @classmethod
    def from_literal(cls, value: float) -> "FloatOrVid":
        """Create a FloatOrVid from a literal float."""
        return cls(literal=value, is_vid=False)

    @classmethod
    def from_vid(cls, vid: Vid) -> "FloatOrVid":
        """Create a FloatOrVid from a Vid reference."""
        return cls(vid=vid, is_vid=True)


@dataclass
class SlotVariant:
    idx: int = None
    slot_type: Optional[SlotType] = SlotType.TensorSlot


@dataclass
class NamedSlot:
    name: str
    slot: SlotVariant


@dataclass
class DataSegment:
    offset: int = None
    size: int = None


@dataclass
class TensorMeta:
    shape: List[IntOrVid]
    dtype: Optional[DTypeId] = None
    strides: List[int] = None


# =============================================================================
# Op nodes
# =============================================================================

@dataclass
class NoopNode:
    pass


@dataclass
class LinearNode:
    x: Tid
    weight: Tid
    out: Tid
    bias: Optional[Tid] = None


@dataclass
class ItemIntNode:
    x: Tid
    out: Vid


@dataclass
class ExpandDimsNode:
    x: Tid
    out: Tid
    axis: int = None


@dataclass
class TileNode:
    x: Tid
    out: Tid
    reps: List[int]


@dataclass
class TakeAlongAxisNode:
    x: Tid
    indices: Tid
    out: Tid
    axis: int = None


@dataclass
class RMSNormNode:
    x: Tid
    weight: Tid
    out: Tid
    eps: float = None


@dataclass
class LayerNormNode:
    x: Tid
    out: Tid
    weight: Optional[Tid] = None
    bias: Optional[Tid] = None
    eps: float = None


@dataclass
class RopeNode:
    q_in: Tid
    k_in: Tid
    q_out: Tid
    k_out: Tid
    pos: Vid
    traditional: bool = False
    scale: float = 1.0
    head_dim: int = None
    freqs: Optional[Tid] = None
    base: float = None


@dataclass
class SdpaNode:
    q: Tid
    k: Tid
    v: Tid
    out: Tid
    causal: bool = False
    scale: float = None
    mask: Optional[Tid] = None


@dataclass
class AddNode:
    a: Tid
    b: Tid
    out: Tid


@dataclass
class AddScalarNode:
    a: IntOrVid
    b: IntOrVid
    out: Vid


@dataclass
class SymSizeNode:
    a: Tid
    out: Vid
    dim: int = None


@dataclass
class MulNode:
    a: Tid
    b: Tid
    out: Tid


@dataclass
class Conv1DNode:
    x: Tid
    w: Tid
    out: Tid
    stride: int = 1
    padding: int = 0
    dilation: int = 1
    groups: int = 1


@dataclass
class GeluNode:
    x: Tid
    out: Tid


@dataclass
class ARangeNode:
    out: Tid
    step: int = 1
    start: int = None
    stop: int = None
    dtype: Optional[DTypeId] = None


@dataclass
class SiluNode:
    x: Tid
    out: Tid


@dataclass
class ReshapeNode:
    x: Tid
    out: Tid
    shape: List[IntOrVid]


@dataclass
class TransposeNode:
    x: Tid
    out: Tid
    perm: List[int]


@dataclass
class ContiguousNode:
    x: Tid
    out: Tid


@dataclass
class IdCopyNode:
    x: Tid
    out: Tid


@dataclass
class GatherNode:
    table_: Tid
    ids: Tid
    out: Tid


@dataclass
class SliceNode:
    x: Tid
    out: Tid
    axis: IntOrVid
    start: IntOrVid
    end: IntOrVid


@dataclass
class CastNode:
    x: Tid
    out: Tid
    dtype: Optional[DTypeId] = None


@dataclass
class QuantizedLinearNode:
    x: Tid
    w: Tid
    scales: Tid
    out: Tid
    mode: str
    biases: Optional[Tid] = None
    bias: Optional[Tid] = None
    group_size: int = None
    bits: int = None
    out_dtype: Optional[DTypeId] = None


@dataclass
class ConcatNode:
    a: Tid
    b: Tid
    out: Tid
    axis: int = None


@dataclass
class FullNode:
    out: Tid
    shape: List[int]
    v: float = None
    dtype: Optional[DTypeId] = None


@dataclass
class ZerosNode:
    out: Tid
    shape: List[int]
    dtype: Optional[DTypeId] = None


@dataclass
class OnesNode:
    out: Tid
    shape: List[int]
    dtype: Optional[DTypeId] = None


@dataclass
class ArgmaxNode:
    x: Tid
    out: Tid
    axis: int = None


@dataclass
class SliceUpdateNode:
    dst: Tid
    update: Tid
    axis: IntOrVid
    start: IntOrVid
    stop: IntOrVid


@dataclass
class QuantizedGatherNode:
    table_q: Tid
    scales: Tid
    ids: Tid
    out: Tid
    mode: str
    biases: Optional[Tid] = None
    group_size: int = None
    bits: int = None
    out_dtype: Optional[DTypeId] = None


# Union of all op types
OpNodeUnion = Union[
    NoopNode,
    LinearNode,
    ItemIntNode,
    ExpandDimsNode,
    TileNode,
    TakeAlongAxisNode,
    RMSNormNode,
    LayerNormNode,
    RopeNode,
    SdpaNode,
    AddNode,
    AddScalarNode,
    SymSizeNode,
    MulNode,
    Conv1DNode,
    GeluNode,
    ARangeNode,
    SiluNode,
    ReshapeNode,
    TransposeNode,
    ContiguousNode,
    IdCopyNode,
    GatherNode,
    SliceNode,
    CastNode,
    QuantizedLinearNode,
    ConcatNode,
    FullNode,
    ZerosNode,
    OnesNode,
    ArgmaxNode,
    SliceUpdateNode,
    QuantizedGatherNode,
]

# =============================================================================
# Container types (reference OpNodeUnion)
# =============================================================================

@dataclass
class Instruction:
    op: OpNodeUnion


@dataclass
class MLXGraph:
    instructions: List[Instruction]
    version: Optional[str] = None
    num_constant_tensors: int = 0
    num_non_constant_tensors: int = 0
    num_non_constant_values: int = 0
    input_map: Optional[List[SlotVariant]] = None
    output_map: Optional[List[SlotVariant]] = None
    mutable_buffer_map: Optional[List[SlotVariant]] = None
    named_slots: Optional[List[NamedSlot]] = None
    tensor_meta: Optional[List[TensorMeta]] = None
    constant_segment: Optional[DataSegment] = None
