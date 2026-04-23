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
# Source:    backends/mlx/serialization/schema.fbs
# Generator: backends/mlx/serialization/generate.py
#
# To regenerate, run from the executorch root:
#     python backends/mlx/serialization/generate.py
#
# ============================================================================

from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum
from typing import List, Optional, Union


# ============================================================================
# Enums
# ============================================================================

class SlotType(IntEnum):
    TensorSlot = 0
    IntValueSlot = 1
    FloatValueSlot = 2
    BoolValueSlot = 3


# ============================================================================
# Core types
# ============================================================================

@dataclass
class Tid:
    idx: Optional[int]


@dataclass
class Vid:
    idx: Optional[int]


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
class IntOrVid:
    """Represents either a literal integer or a runtime Vid reference."""
    literal: int = 0
    vid: Optional[Vid] = None
    is_vid: bool = False

    @classmethod
    def from_literal(cls, value: int) -> "IntOrVid":
        """Create a IntOrVid from a literal integer."""
        return cls(literal=value, is_vid=False)

    @classmethod
    def from_vid(cls, vid: Vid) -> "IntOrVid":
        """Create a IntOrVid from a Vid reference."""
        return cls(vid=vid, is_vid=True)


@dataclass
class IntOrVidOrTid:
    """Represents either a literal integer or a runtime Vid reference."""
    literal: int = 0
    vid: Optional[Vid] = None
    tid: Optional[Tid] = None
    kind: int = 0

    @classmethod
    def from_literal(cls, value: int) -> "IntOrVidOrTid":
        """Create a IntOrVidOrTid from a literal integer."""
        return cls(literal=value, kind=0)

    @classmethod
    def from_vid(cls, vid: Vid) -> "IntOrVidOrTid":
        """Create a IntOrVidOrTid from a Vid reference."""
        return cls(vid=vid, kind=1)

    @classmethod
    def from_tid(cls, tid: Tid) -> "IntOrVidOrTid":
        """Create a IntOrVidOrTid from a Tid tensor reference."""
        return cls(tid=tid, kind=2)


@dataclass
class VidOrTid:
    """Represents either a tensor reference or a runtime Vid reference."""
    vid: Optional[Vid] = None
    tid: Optional[Tid] = None
    is_vid: bool = False

    @classmethod
    def from_tid(cls, value: Tid) -> "VidOrTid":
        """Create a VidOrTid from a tensor reference."""
        return cls(tid=value, is_vid=False)

    @classmethod
    def from_vid(cls, vid: Vid) -> "VidOrTid":
        """Create a VidOrTid from a Vid reference."""
        return cls(vid=vid, is_vid=True)

    @classmethod
    def from_tid(cls, tid: Tid) -> "VidOrTid":
        """Create a VidOrTid from a Tid tensor reference."""
        return cls(tid=tid, is_vid=False)


@dataclass
class ShapeDim:
    value: int = -1
    min_value: int = 0
    max_value: int = -1


@dataclass
class SlotVariant:
    slot_type: SlotType = SlotType.TensorSlot
    idx: Optional[int] = None


@dataclass
class NamedSlot:
    name: str
    slot: SlotVariant


@dataclass
class TensorMeta:
    shape: List[ShapeDim]
    scalar_type: Optional[int] = None
    dim_order: Optional[List[int]] = None


# ============================================================================
# Op nodes
# ============================================================================

@dataclass
class NoopNode:
    pass


@dataclass
class IdCopyNode:
    x: Tid
    out: Tid


@dataclass
class AddmmNode:
    mat1: Tid
    mat2: Tid
    out: Tid
    alpha: float = 1.0
    beta: float = 1.0
    bias: Optional[Tid] = None


@dataclass
class ItemIntNode:
    x: Tid
    out: Vid


@dataclass
class ExpandDimsNode:
    x: Tid
    out: Tid
    axis: Optional[int] = None


@dataclass
class TileNode:
    x: Tid
    out: Tid
    reps: List[IntOrVid]


@dataclass
class TakeAlongAxisNode:
    x: Tid
    indices: Tid
    out: Tid
    axis: Optional[int] = None


@dataclass
class TakeNode:
    x: Tid
    out: Tid
    index: IntOrVidOrTid
    axis: Optional[int] = None


@dataclass
class RMSNormNode:
    x: Tid
    out: Tid
    weight: Optional[Tid] = None
    eps: Optional[float] = None


@dataclass
class LayerNormNode:
    x: Tid
    out: Tid
    weight: Optional[Tid] = None
    bias: Optional[Tid] = None
    eps: Optional[float] = None


@dataclass
class RopeNode:
    x: Tid
    out: Tid
    offset: VidOrTid
    traditional: bool = False
    base: float = 500000.0
    scale: float = 1.0
    dims: Optional[int] = None
    freqs: Optional[Tid] = None


@dataclass
class SdpaNode:
    q: Tid
    k: Tid
    v: Tid
    out: Tid
    causal: bool = False
    scale: Optional[float] = None
    mask: Optional[Tid] = None


@dataclass
class AddNode:
    a: Tid
    b: Tid
    out: Tid


@dataclass
class AddIntNode:
    a: IntOrVid
    b: IntOrVid
    out: Vid


@dataclass
class SubtractIntNode:
    a: IntOrVid
    b: IntOrVid
    out: Vid


@dataclass
class MultiplyIntNode:
    a: IntOrVid
    b: IntOrVid
    out: Vid


@dataclass
class FloorDivideIntNode:
    a: IntOrVid
    b: IntOrVid
    out: Vid


@dataclass
class ModIntNode:
    a: IntOrVid
    b: IntOrVid
    out: Vid


@dataclass
class SymSizeNode:
    a: Tid
    out: Vid
    dim: Optional[int] = None


@dataclass
class MultiplyNode:
    a: Tid
    b: Tid
    out: Tid


@dataclass
class DivideNode:
    a: Tid
    b: Tid
    out: Tid


@dataclass
class SubtractNode:
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
class Conv2DNode:
    x: Tid
    w: Tid
    out: Tid
    stride_h: int = 1
    stride_w: int = 1
    padding_h: int = 0
    padding_w: int = 0
    dilation_h: int = 1
    dilation_w: int = 1
    groups: int = 1


@dataclass
class Conv3DNode:
    x: Tid
    w: Tid
    out: Tid
    stride_d: int = 1
    stride_h: int = 1
    stride_w: int = 1
    padding_d: int = 0
    padding_h: int = 0
    padding_w: int = 0
    dilation_d: int = 1
    dilation_h: int = 1
    dilation_w: int = 1
    groups: int = 1


@dataclass
class ConvTranspose1DNode:
    x: Tid
    w: Tid
    out: Tid
    stride: int = 1
    padding: int = 0
    dilation: int = 1
    output_padding: int = 0
    groups: int = 1


@dataclass
class ConvTranspose2DNode:
    x: Tid
    w: Tid
    out: Tid
    stride_h: int = 1
    stride_w: int = 1
    padding_h: int = 0
    padding_w: int = 0
    dilation_h: int = 1
    dilation_w: int = 1
    output_padding_h: int = 0
    output_padding_w: int = 0
    groups: int = 1


@dataclass
class ConvTranspose3DNode:
    x: Tid
    w: Tid
    out: Tid
    stride_d: int = 1
    stride_h: int = 1
    stride_w: int = 1
    padding_d: int = 0
    padding_h: int = 0
    padding_w: int = 0
    dilation_d: int = 1
    dilation_h: int = 1
    dilation_w: int = 1
    output_padding_d: int = 0
    output_padding_h: int = 0
    output_padding_w: int = 0
    groups: int = 1


@dataclass
class GeluNode:
    x: Tid
    out: Tid
    approximate: str


@dataclass
class ARangeNode:
    out: Tid
    start: IntOrVid
    stop: IntOrVid
    step: IntOrVid
    scalar_type: int = None


@dataclass
class SiluNode:
    x: Tid
    out: Tid


@dataclass
class SigmoidNode:
    x: Tid
    out: Tid


@dataclass
class TanhNode:
    x: Tid
    out: Tid


@dataclass
class SqueezeNode:
    x: Tid
    out: Tid
    dims: Optional[List[int]] = None


@dataclass
class SplitNode:
    x: Tid
    outs: List[Tid]
    sizes: List[IntOrVid]
    axis: Optional[int] = None


@dataclass
class RsqrtNode:
    x: Tid
    out: Tid


@dataclass
class MaximumNode:
    a: Tid
    b: Tid
    out: Tid


@dataclass
class MinimumNode:
    a: Tid
    b: Tid
    out: Tid


@dataclass
class LogNode:
    x: Tid
    out: Tid


@dataclass
class SoftmaxNode:
    x: Tid
    out: Tid
    precise: bool = False
    axis: Optional[int] = None


@dataclass
class BroadcastToNode:
    x: Tid
    out: Tid
    shape: List[IntOrVid]


@dataclass
class PadNode:
    x: Tid
    out: Tid
    pad_width: List[IntOrVid]
    mode: str
    constant_value: float = 0.0


@dataclass
class WhereNode:
    condition: Tid
    x: Tid
    y: Tid
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
class AsStridedNode:
    x: Tid
    out: Tid
    shape: List[IntOrVid]
    strides: List[IntOrVid]
    offset: int = 0


@dataclass
class ContiguousNode:
    x: Tid
    out: Tid


@dataclass
class GatherNode:
    x: Tid
    indices: List[Tid]
    out: Tid
    axes: List[int]
    slice_sizes: List[int]


@dataclass
class SliceNode:
    x: Tid
    out: Tid
    axis: IntOrVid
    start: IntOrVid
    stop: IntOrVid
    step: int = 1


@dataclass
class AsTypeNode:
    x: Tid
    out: Tid
    scalar_type: Optional[int] = None


@dataclass
class QuantizedMatmulNode:
    x: Tid
    w: Tid
    scales: Tid
    out: Tid
    mode: str
    transpose: bool = True
    biases: Optional[Tid] = None
    group_size: Optional[int] = None
    bits: Optional[int] = None


@dataclass
class ScatterAddNode:
    x: Tid
    indices: Tid
    updates: Tid
    out: Tid
    axis: Optional[int] = None


@dataclass
class ConcatenateNode:
    tensors: List[Tid]
    out: Tid
    axis: Optional[int] = None


@dataclass
class FullNode:
    out: Tid
    shape: List[IntOrVid]
    v: FloatOrVid
    scalar_type: Optional[int] = None


@dataclass
class FullLikeNode:
    x: Tid
    out: Tid
    v: FloatOrVid
    scalar_type: int = None


@dataclass
class ArgmaxNode:
    x: Tid
    out: Tid
    keepdims: bool = False
    axis: Optional[int] = None


@dataclass
class SliceUpdateNode:
    dst: Tid
    update: Tid
    out: Tid
    axis: IntOrVid
    start: IntOrVid
    stop: IntOrVid
    step: int = 1


@dataclass
class IndexCopyNode:
    dst: Tid
    update: Tid
    indices: Tid
    out: Tid
    axis: Optional[int] = None


@dataclass
class DequantizeNode:
    w: Tid
    scales: Tid
    out: Tid
    mode: str
    dtype: int = None
    biases: Optional[Tid] = None
    group_size: Optional[int] = None
    bits: Optional[int] = None
    global_scale: Optional[Tid] = None


@dataclass
class LessNode:
    a: Tid
    b: Tid
    out: Tid


@dataclass
class LessEqualNode:
    a: Tid
    b: Tid
    out: Tid


@dataclass
class GreaterNode:
    a: Tid
    b: Tid
    out: Tid


@dataclass
class GreaterEqualNode:
    a: Tid
    b: Tid
    out: Tid


@dataclass
class EqualNode:
    a: Tid
    b: Tid
    out: Tid


@dataclass
class NotEqualNode:
    a: Tid
    b: Tid
    out: Tid


@dataclass
class LogicalNotNode:
    x: Tid
    out: Tid


@dataclass
class LogicalAndNode:
    a: Tid
    b: Tid
    out: Tid


@dataclass
class LogicalOrNode:
    a: Tid
    b: Tid
    out: Tid


@dataclass
class TriNode:
    out: Tid
    n: IntOrVid
    m: IntOrVid
    k: int = 0
    scalar_type: Optional[int] = None


@dataclass
class TrilNode:
    x: Tid
    out: Tid
    k: int = 0


@dataclass
class TriuNode:
    x: Tid
    out: Tid
    k: int = 0


@dataclass
class ClipNode:
    x: Tid
    out: Tid
    a_min: Optional[Tid] = None
    a_max: Optional[Tid] = None


@dataclass
class CumsumNode:
    x: Tid
    out: Tid
    reverse: bool = False
    inclusive: bool = True
    axis: Optional[int] = None


@dataclass
class StackNode:
    tensors: List[Tid]
    out: Tid
    axis: int = 0


@dataclass
class SignNode:
    x: Tid
    out: Tid


@dataclass
class AnyNode:
    x: Tid
    out: Tid
    keepdims: bool = False
    axes: Optional[List[int]] = None


@dataclass
class AllNode:
    x: Tid
    out: Tid
    keepdims: bool = False
    axes: Optional[List[int]] = None


@dataclass
class RepeatNode:
    x: Tid
    out: Tid
    repeats: IntOrVid
    axis: Optional[int] = None


@dataclass
class SortNode:
    x: Tid
    out: Tid
    axis: Optional[int] = None


@dataclass
class ArgsortNode:
    x: Tid
    out: Tid
    axis: Optional[int] = None


@dataclass
class PartitionNode:
    x: Tid
    out: Tid
    kth: IntOrVid
    axis: Optional[int] = None


@dataclass
class ArgPartitionNode:
    x: Tid
    out: Tid
    kth: IntOrVid
    axis: Optional[int] = None


@dataclass
class FloorNode:
    x: Tid
    out: Tid


@dataclass
class CeilNode:
    x: Tid
    out: Tid


@dataclass
class SquareNode:
    x: Tid
    out: Tid


@dataclass
class ExpNode:
    x: Tid
    out: Tid


@dataclass
class SinNode:
    x: Tid
    out: Tid


@dataclass
class CosNode:
    x: Tid
    out: Tid


@dataclass
class TanNode:
    x: Tid
    out: Tid


@dataclass
class ArcsinNode:
    x: Tid
    out: Tid


@dataclass
class ArccosNode:
    x: Tid
    out: Tid


@dataclass
class ArctanNode:
    x: Tid
    out: Tid


@dataclass
class SinhNode:
    x: Tid
    out: Tid


@dataclass
class CoshNode:
    x: Tid
    out: Tid


@dataclass
class ArcsinhNode:
    x: Tid
    out: Tid


@dataclass
class ArccoshNode:
    x: Tid
    out: Tid


@dataclass
class ArctanhNode:
    x: Tid
    out: Tid


@dataclass
class Log2Node:
    x: Tid
    out: Tid


@dataclass
class Log10Node:
    x: Tid
    out: Tid


@dataclass
class Log1pNode:
    x: Tid
    out: Tid


@dataclass
class ErfNode:
    x: Tid
    out: Tid


@dataclass
class Expm1Node:
    x: Tid
    out: Tid


@dataclass
class RoundNode:
    x: Tid
    out: Tid
    decimals: int = 0


@dataclass
class ReciprocalNode:
    x: Tid
    out: Tid


@dataclass
class SqrtNode:
    x: Tid
    out: Tid


@dataclass
class AbsNode:
    x: Tid
    out: Tid


@dataclass
class NegNode:
    x: Tid
    out: Tid


@dataclass
class Atan2Node:
    a: Tid
    b: Tid
    out: Tid


@dataclass
class LogAddExpNode:
    a: Tid
    b: Tid
    out: Tid


@dataclass
class FloorDivideNode:
    a: Tid
    b: Tid
    out: Tid


@dataclass
class RemainderNode:
    a: Tid
    b: Tid
    out: Tid


@dataclass
class PowerNode:
    a: Tid
    b: Tid
    out: Tid


@dataclass
class LogSumExpNode:
    x: Tid
    out: Tid
    keepdims: bool = False
    axes: Optional[List[int]] = None


@dataclass
class SumNode:
    x: Tid
    out: Tid
    keepdims: bool = False
    axes: Optional[List[int]] = None


@dataclass
class MeanNode:
    x: Tid
    out: Tid
    keepdims: bool = False
    axes: Optional[List[int]] = None


@dataclass
class VarNode:
    x: Tid
    out: Tid
    keepdims: bool = False
    ddof: int = 0
    axes: Optional[List[int]] = None


@dataclass
class StdNode:
    x: Tid
    out: Tid
    keepdims: bool = False
    ddof: int = 0
    axes: Optional[List[int]] = None


@dataclass
class ProdNode:
    x: Tid
    out: Tid
    keepdims: bool = False
    axes: Optional[List[int]] = None


@dataclass
class MaxNode:
    x: Tid
    out: Tid
    keepdims: bool = False
    axes: Optional[List[int]] = None


@dataclass
class MinNode:
    x: Tid
    out: Tid
    keepdims: bool = False
    axes: Optional[List[int]] = None


@dataclass
class ArgminNode:
    x: Tid
    out: Tid
    keepdims: bool = False
    axis: Optional[int] = None


@dataclass
class MedianNode:
    x: Tid
    out: Tid
    keepdims: bool = False
    axes: Optional[List[int]] = None


@dataclass
class GatherMmNode:
    a: Tid
    b: Tid
    out: Tid
    sorted_indices: bool = False
    lhs_indices: Optional[Tid] = None
    rhs_indices: Optional[Tid] = None


@dataclass
class GatherQmmNode:
    x: Tid
    w: Tid
    scales: Tid
    out: Tid
    mode: str
    transpose: bool = True
    sorted_indices: bool = False
    biases: Optional[Tid] = None
    lhs_indices: Optional[Tid] = None
    rhs_indices: Optional[Tid] = None
    group_size: Optional[int] = None
    bits: Optional[int] = None


@dataclass
class ScanNode:
    originals: List[Tid]
    sliced: List[Tid]
    outputs: List[Tid]
    carry: List[Tid]
    scan_axis: int = 1
    body_chain_idx: Optional[int] = None


# Union of all op types
OpNodeUnion = Union[
    NoopNode,
    IdCopyNode,
    AddmmNode,
    ItemIntNode,
    ExpandDimsNode,
    TileNode,
    TakeAlongAxisNode,
    TakeNode,
    RMSNormNode,
    LayerNormNode,
    RopeNode,
    SdpaNode,
    AddNode,
    AddIntNode,
    SubtractIntNode,
    MultiplyIntNode,
    FloorDivideIntNode,
    ModIntNode,
    SymSizeNode,
    MultiplyNode,
    DivideNode,
    SubtractNode,
    Conv1DNode,
    Conv2DNode,
    Conv3DNode,
    ConvTranspose1DNode,
    ConvTranspose2DNode,
    ConvTranspose3DNode,
    GeluNode,
    ARangeNode,
    SiluNode,
    SigmoidNode,
    TanhNode,
    SqueezeNode,
    SplitNode,
    RsqrtNode,
    MaximumNode,
    MinimumNode,
    LogNode,
    SoftmaxNode,
    BroadcastToNode,
    PadNode,
    WhereNode,
    ReshapeNode,
    TransposeNode,
    AsStridedNode,
    ContiguousNode,
    GatherNode,
    SliceNode,
    AsTypeNode,
    QuantizedMatmulNode,
    ScatterAddNode,
    ConcatenateNode,
    FullNode,
    FullLikeNode,
    ArgmaxNode,
    SliceUpdateNode,
    IndexCopyNode,
    DequantizeNode,
    LessNode,
    LessEqualNode,
    GreaterNode,
    GreaterEqualNode,
    EqualNode,
    NotEqualNode,
    LogicalNotNode,
    LogicalAndNode,
    LogicalOrNode,
    TriNode,
    TrilNode,
    TriuNode,
    ClipNode,
    CumsumNode,
    StackNode,
    SignNode,
    AnyNode,
    AllNode,
    RepeatNode,
    SortNode,
    ArgsortNode,
    PartitionNode,
    ArgPartitionNode,
    FloorNode,
    CeilNode,
    SquareNode,
    ExpNode,
    SinNode,
    CosNode,
    TanNode,
    ArcsinNode,
    ArccosNode,
    ArctanNode,
    SinhNode,
    CoshNode,
    ArcsinhNode,
    ArccoshNode,
    ArctanhNode,
    Log2Node,
    Log10Node,
    Log1pNode,
    ErfNode,
    Expm1Node,
    RoundNode,
    ReciprocalNode,
    SqrtNode,
    AbsNode,
    NegNode,
    Atan2Node,
    LogAddExpNode,
    FloorDivideNode,
    RemainderNode,
    PowerNode,
    LogSumExpNode,
    SumNode,
    MeanNode,
    VarNode,
    StdNode,
    ProdNode,
    MaxNode,
    MinNode,
    ArgminNode,
    MedianNode,
    GatherMmNode,
    GatherQmmNode,
    ScanNode,
]

# ============================================================================
# Container types (reference OpNodeUnion)
# ============================================================================

@dataclass
class Instruction:
    op: OpNodeUnion


@dataclass
class InstructionChain:
    instructions: List[Instruction]


@dataclass
class MLXGraph:
    instruction_chains: List[InstructionChain]
    version: Optional[str] = None
    num_constant_tensors: int = 0
    num_input_tensors: int = 0
    num_output_tensors: int = 0
    num_mutable_buffer_tensors: int = 0
    num_temp_tensors: int = 0
    num_values: int = 0
    main_chain_idx: int = 0
    init_chain_idx: int = -1
    input_map: Optional[List[SlotVariant]] = None
    output_map: Optional[List[SlotVariant]] = None
    mutable_buffer_map: Optional[List[SlotVariant]] = None
    named_slots: Optional[List[NamedSlot]] = None
    tensor_meta: Optional[List[TensorMeta]] = None
