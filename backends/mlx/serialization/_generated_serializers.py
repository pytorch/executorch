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
#
# This file contains auto-generated serializer methods for all op types.

from __future__ import annotations

from typing import List, Tuple, Dict

import flatbuffers

# FlatBuffer union indices: 0 = NONE, then 1-indexed from union order
MLX_OP_TYPE_NAMES = {
    0: "NONE",
    1: "NoopNode",
    2: "IdCopyNode",
    3: "AddmmNode",
    4: "ItemIntNode",
    5: "ExpandDimsNode",
    6: "TileNode",
    7: "TakeAlongAxisNode",
    8: "TakeNode",
    9: "RMSNormNode",
    10: "LayerNormNode",
    11: "RopeNode",
    12: "SdpaNode",
    13: "AddNode",
    14: "AddIntNode",
    15: "SubtractIntNode",
    16: "MultiplyIntNode",
    17: "FloorDivideIntNode",
    18: "SymSizeNode",
    19: "MultiplyNode",
    20: "DivideNode",
    21: "SubtractNode",
    22: "Conv1DNode",
    23: "Conv2DNode",
    24: "Conv3DNode",
    25: "GeluNode",
    26: "ARangeNode",
    27: "SiluNode",
    28: "SigmoidNode",
    29: "TanhNode",
    30: "SqueezeNode",
    31: "SplitNode",
    32: "RsqrtNode",
    33: "MaximumNode",
    34: "MinimumNode",
    35: "LogNode",
    36: "SoftmaxNode",
    37: "BroadcastToNode",
    38: "PadNode",
    39: "WhereNode",
    40: "ReshapeNode",
    41: "TransposeNode",
    42: "AsStridedNode",
    43: "ContiguousNode",
    44: "GatherNode",
    45: "SliceNode",
    46: "AsTypeNode",
    47: "ConcatenateNode",
    48: "FullNode",
    49: "FullLikeNode",
    50: "ArgmaxNode",
    51: "SliceUpdateNode",
    52: "IndexCopyNode",
    53: "DequantizeNode",
    54: "LessNode",
    55: "LessEqualNode",
    56: "GreaterNode",
    57: "GreaterEqualNode",
    58: "EqualNode",
    59: "NotEqualNode",
    60: "LogicalNotNode",
    61: "LogicalAndNode",
    62: "LogicalOrNode",
    63: "TriNode",
    64: "TrilNode",
    65: "TriuNode",
    66: "FloorNode",
    67: "CeilNode",
    68: "SquareNode",
    69: "ExpNode",
    70: "SinNode",
    71: "CosNode",
    72: "TanNode",
    73: "ArcsinNode",
    74: "ArccosNode",
    75: "ArctanNode",
    76: "SinhNode",
    77: "CoshNode",
    78: "ArcsinhNode",
    79: "ArccoshNode",
    80: "ArctanhNode",
    81: "Log2Node",
    82: "Log10Node",
    83: "Log1pNode",
    84: "ErfNode",
    85: "Expm1Node",
    86: "RoundNode",
    87: "ReciprocalNode",
    88: "SqrtNode",
    89: "AbsNode",
    90: "NegNode",
    91: "Atan2Node",
    92: "LogAddExpNode",
    93: "FloorDivideNode",
    94: "PowerNode",
    95: "LogSumExpNode",
    96: "SumNode",
    97: "MeanNode",
    98: "VarNode",
    99: "StdNode",
    100: "ProdNode",
    101: "MaxNode",
    102: "MinNode",
    103: "ArgminNode",
    104: "MedianNode",
    105: "ModIntNode",
    106: "RemainderNode",
    107: "ConvTranspose1DNode",
    108: "ConvTranspose2DNode",
    109: "ConvTranspose3DNode",
    110: "ClipNode",
    111: "CumsumNode",
    112: "StackNode",
    113: "SignNode",
    114: "AnyNode",
    115: "AllNode",
    116: "RepeatNode",
    117: "SortNode",
    118: "ArgsortNode",
    119: "PartitionNode",
    120: "ArgPartitionNode",
    121: "QuantizedMatmulNode",
    122: "ScatterAddNode",
    123: "GatherMmNode",
    124: "GatherQmmNode",
    125: "ScanNode",
    126: "MetalKernelNode",
    127: "BitwiseXorNode",
}

from executorch.backends.mlx.serialization.mlx_graph_schema import (
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
    MetalKernelNode,
    BitwiseXorNode,
    IntOrVid,
    FloatOrVid,
    VidOrTid,
    IntOrVidOrTid,
    Tid,
    Vid,
)


def _build_int_vector(builder: flatbuffers.Builder, vec: List[int]) -> int:
    """Pre-build a vector of int32 values (must be called before table Start)."""
    builder.StartVector(4, len(vec), 4)
    for v in reversed(vec):
        builder.PrependInt32(v)
    return builder.EndVector()


def _build_int8_vector(builder: flatbuffers.Builder, vec: List[int]) -> int:
    """Pre-build a vector of int8 values (must be called before table Start)."""
    builder.StartVector(1, len(vec), 1)
    for v in reversed(vec):
        builder.PrependInt8(v)
    return builder.EndVector()


def _build_uint8_vector(builder: flatbuffers.Builder, vec: List[int]) -> int:
    """Pre-build a vector of uint8 values (must be called before table Start)."""
    builder.StartVector(1, len(vec), 1)
    for v in reversed(vec):
        builder.PrependUint8(v)
    return builder.EndVector()


class GeneratedOpBuilders:
    """Mixin class with auto-generated op builder methods."""

    def _build_int_or_vid(self, builder: flatbuffers.Builder, iov: IntOrVid) -> int:
        """Build an IntOrVid table."""
        from executorch.backends.mlx.serialization._generated.mlx_delegate import IntOrVid as FBIntOrVidModule
        from executorch.backends.mlx.serialization._generated.mlx_delegate.Vid import CreateVid

        FBIntOrVidModule.Start(builder)
        FBIntOrVidModule.AddLiteral(builder, iov.literal)
        FBIntOrVidModule.AddIsVid(builder, iov.is_vid)
        if iov.vid is not None:
            # Vid is an inline struct - must be added last for proper FlatBuffer layout
            FBIntOrVidModule.AddVid(builder, CreateVid(builder, iov.vid.idx))
        return FBIntOrVidModule.End(builder)

    def _build_float_or_vid(self, builder: flatbuffers.Builder, fov: FloatOrVid) -> int:
        """Build a FloatOrVid table."""
        from executorch.backends.mlx.serialization._generated.mlx_delegate import FloatOrVid as FBFloatOrVidModule
        from executorch.backends.mlx.serialization._generated.mlx_delegate.Vid import CreateVid

        FBFloatOrVidModule.Start(builder)
        FBFloatOrVidModule.AddLiteral(builder, fov.literal)
        FBFloatOrVidModule.AddIsVid(builder, fov.is_vid)
        if fov.vid is not None:
            FBFloatOrVidModule.AddVid(builder, CreateVid(builder, fov.vid.idx))
        return FBFloatOrVidModule.End(builder)

    def _build_vid_or_tid(self, builder: flatbuffers.Builder, vot: VidOrTid) -> int:
        """Build a TidOrVid table."""
        from executorch.backends.mlx.serialization._generated.mlx_delegate import VidOrTid as FBVidOrTidModule
        from executorch.backends.mlx.serialization._generated.mlx_delegate.Tid import CreateTid
        from executorch.backends.mlx.serialization._generated.mlx_delegate.Vid import CreateVid

        FBVidOrTidModule.Start(builder)
        FBVidOrTidModule.AddIsVid(builder, vot.is_vid)
        if vot.tid is not None:
            FBVidOrTidModule.AddTid(builder, CreateTid(builder, vot.tid.idx))
        if vot.vid is not None:
            FBVidOrTidModule.AddVid(builder, CreateVid(builder, vot.vid.idx))
        return FBVidOrTidModule.End(builder)

    def _build_int_or_vid_or_tid(self, builder: flatbuffers.Builder, ivt: IntOrVidOrTid) -> int:
        """Build an IntOrVidOrTid table."""
        from executorch.backends.mlx.serialization._generated.mlx_delegate import IntOrVidOrTid as FBIntOrVidOrTidModule
        from executorch.backends.mlx.serialization._generated.mlx_delegate.Tid import CreateTid
        from executorch.backends.mlx.serialization._generated.mlx_delegate.Vid import CreateVid

        FBIntOrVidOrTidModule.Start(builder)
        FBIntOrVidOrTidModule.AddLiteral(builder, ivt.literal)
        FBIntOrVidOrTidModule.AddKind(builder, ivt.kind)
        if ivt.tid is not None:
            FBIntOrVidOrTidModule.AddTid(builder, CreateTid(builder, ivt.tid.idx))
        if ivt.vid is not None:
            FBIntOrVidOrTidModule.AddVid(builder, CreateVid(builder, ivt.vid.idx))
        return FBIntOrVidOrTidModule.End(builder)

    def _build_int_or_vid_vector(
        self, builder: flatbuffers.Builder, vec: List[IntOrVid]
    ) -> int:
        """Build a vector of IntOrVid tables."""
        offsets = []
        for iov in vec:
            offsets.append(self._build_int_or_vid(builder, iov))
        builder.StartVector(4, len(offsets), 4)
        for off in reversed(offsets):
            builder.PrependUOffsetTRelative(off)
        return builder.EndVector()

    def _build_tid_vector(
        self, builder: flatbuffers.Builder, vec: List[Tid]
    ) -> int:
        """Build a vector of Tid structs."""
        from executorch.backends.mlx.serialization._generated.mlx_delegate.Tid import CreateTid

        # For vectors of structs, we need to build the vector differently
        # Each Tid struct is 4 bytes (uint32), so we manually write them
        builder.StartVector(4, len(vec), 4)
        for tid in reversed(vec):
            builder.Prep(4, 0)  # Align for struct
            builder.PrependUint32(tid.idx)
        return builder.EndVector()

    def _build_string_vector(
        self, builder: flatbuffers.Builder, vec: List[str]
    ) -> int:
        """Pre-build a vector of strings (offsets must be created before table Start)."""
        offsets = [builder.CreateString(s) for s in vec]
        builder.StartVector(4, len(offsets), 4)
        for off in reversed(offsets):
            builder.PrependUOffsetTRelative(off)
        return builder.EndVector()

    def _build_NoopNode(
        self, builder: flatbuffers.Builder, op: NoopNode
    ) -> Tuple[int, int]:
        """Auto-generated builder for NoopNode."""
        # Import the MODULE (not class) to access builder functions like Start(), Add*(), End()
        from executorch.backends.mlx.serialization._generated.mlx_delegate import NoopNode as FBNoopNodeModule
        from executorch.backends.mlx.serialization._generated.mlx_delegate import OpNode as FBOpNodeModule
        from executorch.backends.mlx.serialization._generated.mlx_delegate.Tid import CreateTid
        from executorch.backends.mlx.serialization._generated.mlx_delegate.Vid import CreateVid

        FBNoopNodeModule.Start(builder)
        offset = FBNoopNodeModule.End(builder)
        return offset, FBOpNodeModule.OpNode.NoopNode

    def _build_IdCopyNode(
        self, builder: flatbuffers.Builder, op: IdCopyNode
    ) -> Tuple[int, int]:
        """Auto-generated builder for IdCopyNode."""
        # Import the MODULE (not class) to access builder functions like Start(), Add*(), End()
        from executorch.backends.mlx.serialization._generated.mlx_delegate import IdCopyNode as FBIdCopyNodeModule
        from executorch.backends.mlx.serialization._generated.mlx_delegate import OpNode as FBOpNodeModule
        from executorch.backends.mlx.serialization._generated.mlx_delegate.Tid import CreateTid
        from executorch.backends.mlx.serialization._generated.mlx_delegate.Vid import CreateVid

        FBIdCopyNodeModule.Start(builder)
        FBIdCopyNodeModule.AddX(builder, CreateTid(builder, op.x.idx))
        FBIdCopyNodeModule.AddOut(builder, CreateTid(builder, op.out.idx))
        offset = FBIdCopyNodeModule.End(builder)
        return offset, FBOpNodeModule.OpNode.IdCopyNode

    def _build_AddmmNode(
        self, builder: flatbuffers.Builder, op: AddmmNode
    ) -> Tuple[int, int]:
        """Auto-generated builder for AddmmNode."""
        # Import the MODULE (not class) to access builder functions like Start(), Add*(), End()
        from executorch.backends.mlx.serialization._generated.mlx_delegate import AddmmNode as FBAddmmNodeModule
        from executorch.backends.mlx.serialization._generated.mlx_delegate import OpNode as FBOpNodeModule
        from executorch.backends.mlx.serialization._generated.mlx_delegate.Tid import CreateTid
        from executorch.backends.mlx.serialization._generated.mlx_delegate.Vid import CreateVid

        FBAddmmNodeModule.Start(builder)
        FBAddmmNodeModule.AddMat1(builder, CreateTid(builder, op.mat1.idx))
        FBAddmmNodeModule.AddMat2(builder, CreateTid(builder, op.mat2.idx))
        FBAddmmNodeModule.AddOut(builder, CreateTid(builder, op.out.idx))
        if op.bias is not None:
            FBAddmmNodeModule.AddBias(builder, CreateTid(builder, op.bias.idx))
        FBAddmmNodeModule.AddAlpha(builder, op.alpha)
        FBAddmmNodeModule.AddBeta(builder, op.beta)
        offset = FBAddmmNodeModule.End(builder)
        return offset, FBOpNodeModule.OpNode.AddmmNode

    def _build_ItemIntNode(
        self, builder: flatbuffers.Builder, op: ItemIntNode
    ) -> Tuple[int, int]:
        """Auto-generated builder for ItemIntNode."""
        # Import the MODULE (not class) to access builder functions like Start(), Add*(), End()
        from executorch.backends.mlx.serialization._generated.mlx_delegate import ItemIntNode as FBItemIntNodeModule
        from executorch.backends.mlx.serialization._generated.mlx_delegate import OpNode as FBOpNodeModule
        from executorch.backends.mlx.serialization._generated.mlx_delegate.Tid import CreateTid
        from executorch.backends.mlx.serialization._generated.mlx_delegate.Vid import CreateVid

        FBItemIntNodeModule.Start(builder)
        FBItemIntNodeModule.AddX(builder, CreateTid(builder, op.x.idx))
        FBItemIntNodeModule.AddOut(builder, CreateVid(builder, op.out.idx))
        offset = FBItemIntNodeModule.End(builder)
        return offset, FBOpNodeModule.OpNode.ItemIntNode

    def _build_ExpandDimsNode(
        self, builder: flatbuffers.Builder, op: ExpandDimsNode
    ) -> Tuple[int, int]:
        """Auto-generated builder for ExpandDimsNode."""
        # Import the MODULE (not class) to access builder functions like Start(), Add*(), End()
        from executorch.backends.mlx.serialization._generated.mlx_delegate import ExpandDimsNode as FBExpandDimsNodeModule
        from executorch.backends.mlx.serialization._generated.mlx_delegate import OpNode as FBOpNodeModule
        from executorch.backends.mlx.serialization._generated.mlx_delegate.Tid import CreateTid
        from executorch.backends.mlx.serialization._generated.mlx_delegate.Vid import CreateVid

        FBExpandDimsNodeModule.Start(builder)
        FBExpandDimsNodeModule.AddX(builder, CreateTid(builder, op.x.idx))
        FBExpandDimsNodeModule.AddOut(builder, CreateTid(builder, op.out.idx))
        FBExpandDimsNodeModule.AddAxis(builder, op.axis)
        offset = FBExpandDimsNodeModule.End(builder)
        return offset, FBOpNodeModule.OpNode.ExpandDimsNode

    def _build_TileNode(
        self, builder: flatbuffers.Builder, op: TileNode
    ) -> Tuple[int, int]:
        """Auto-generated builder for TileNode."""
        # Import the MODULE (not class) to access builder functions like Start(), Add*(), End()
        from executorch.backends.mlx.serialization._generated.mlx_delegate import TileNode as FBTileNodeModule
        from executorch.backends.mlx.serialization._generated.mlx_delegate import OpNode as FBOpNodeModule
        from executorch.backends.mlx.serialization._generated.mlx_delegate.Tid import CreateTid
        from executorch.backends.mlx.serialization._generated.mlx_delegate.Vid import CreateVid

        reps_vec = self._build_int_or_vid_vector(builder, op.reps)

        FBTileNodeModule.Start(builder)
        FBTileNodeModule.AddX(builder, CreateTid(builder, op.x.idx))
        FBTileNodeModule.AddOut(builder, CreateTid(builder, op.out.idx))
        FBTileNodeModule.AddReps(builder, reps_vec)
        offset = FBTileNodeModule.End(builder)
        return offset, FBOpNodeModule.OpNode.TileNode

    def _build_TakeAlongAxisNode(
        self, builder: flatbuffers.Builder, op: TakeAlongAxisNode
    ) -> Tuple[int, int]:
        """Auto-generated builder for TakeAlongAxisNode."""
        # Import the MODULE (not class) to access builder functions like Start(), Add*(), End()
        from executorch.backends.mlx.serialization._generated.mlx_delegate import TakeAlongAxisNode as FBTakeAlongAxisNodeModule
        from executorch.backends.mlx.serialization._generated.mlx_delegate import OpNode as FBOpNodeModule
        from executorch.backends.mlx.serialization._generated.mlx_delegate.Tid import CreateTid
        from executorch.backends.mlx.serialization._generated.mlx_delegate.Vid import CreateVid

        FBTakeAlongAxisNodeModule.Start(builder)
        FBTakeAlongAxisNodeModule.AddX(builder, CreateTid(builder, op.x.idx))
        FBTakeAlongAxisNodeModule.AddIndices(builder, CreateTid(builder, op.indices.idx))
        FBTakeAlongAxisNodeModule.AddOut(builder, CreateTid(builder, op.out.idx))
        FBTakeAlongAxisNodeModule.AddAxis(builder, op.axis)
        offset = FBTakeAlongAxisNodeModule.End(builder)
        return offset, FBOpNodeModule.OpNode.TakeAlongAxisNode

    def _build_TakeNode(
        self, builder: flatbuffers.Builder, op: TakeNode
    ) -> Tuple[int, int]:
        """Auto-generated builder for TakeNode."""
        # Import the MODULE (not class) to access builder functions like Start(), Add*(), End()
        from executorch.backends.mlx.serialization._generated.mlx_delegate import TakeNode as FBTakeNodeModule
        from executorch.backends.mlx.serialization._generated.mlx_delegate import OpNode as FBOpNodeModule
        from executorch.backends.mlx.serialization._generated.mlx_delegate.Tid import CreateTid
        from executorch.backends.mlx.serialization._generated.mlx_delegate.Vid import CreateVid

        index_off = self._build_int_or_vid_or_tid(builder, op.index)

        FBTakeNodeModule.Start(builder)
        FBTakeNodeModule.AddX(builder, CreateTid(builder, op.x.idx))
        FBTakeNodeModule.AddOut(builder, CreateTid(builder, op.out.idx))
        FBTakeNodeModule.AddIndex(builder, index_off)
        FBTakeNodeModule.AddAxis(builder, op.axis)
        offset = FBTakeNodeModule.End(builder)
        return offset, FBOpNodeModule.OpNode.TakeNode

    def _build_RMSNormNode(
        self, builder: flatbuffers.Builder, op: RMSNormNode
    ) -> Tuple[int, int]:
        """Auto-generated builder for RMSNormNode."""
        # Import the MODULE (not class) to access builder functions like Start(), Add*(), End()
        from executorch.backends.mlx.serialization._generated.mlx_delegate import RMSNormNode as FBRMSNormNodeModule
        from executorch.backends.mlx.serialization._generated.mlx_delegate import OpNode as FBOpNodeModule
        from executorch.backends.mlx.serialization._generated.mlx_delegate.Tid import CreateTid
        from executorch.backends.mlx.serialization._generated.mlx_delegate.Vid import CreateVid

        FBRMSNormNodeModule.Start(builder)
        FBRMSNormNodeModule.AddX(builder, CreateTid(builder, op.x.idx))
        if op.weight is not None:
            FBRMSNormNodeModule.AddWeight(builder, CreateTid(builder, op.weight.idx))
        FBRMSNormNodeModule.AddOut(builder, CreateTid(builder, op.out.idx))
        FBRMSNormNodeModule.AddEps(builder, op.eps)
        offset = FBRMSNormNodeModule.End(builder)
        return offset, FBOpNodeModule.OpNode.RMSNormNode

    def _build_LayerNormNode(
        self, builder: flatbuffers.Builder, op: LayerNormNode
    ) -> Tuple[int, int]:
        """Auto-generated builder for LayerNormNode."""
        # Import the MODULE (not class) to access builder functions like Start(), Add*(), End()
        from executorch.backends.mlx.serialization._generated.mlx_delegate import LayerNormNode as FBLayerNormNodeModule
        from executorch.backends.mlx.serialization._generated.mlx_delegate import OpNode as FBOpNodeModule
        from executorch.backends.mlx.serialization._generated.mlx_delegate.Tid import CreateTid
        from executorch.backends.mlx.serialization._generated.mlx_delegate.Vid import CreateVid

        FBLayerNormNodeModule.Start(builder)
        FBLayerNormNodeModule.AddX(builder, CreateTid(builder, op.x.idx))
        FBLayerNormNodeModule.AddOut(builder, CreateTid(builder, op.out.idx))
        if op.weight is not None:
            FBLayerNormNodeModule.AddWeight(builder, CreateTid(builder, op.weight.idx))
        if op.bias is not None:
            FBLayerNormNodeModule.AddBias(builder, CreateTid(builder, op.bias.idx))
        FBLayerNormNodeModule.AddEps(builder, op.eps)
        offset = FBLayerNormNodeModule.End(builder)
        return offset, FBOpNodeModule.OpNode.LayerNormNode

    def _build_RopeNode(
        self, builder: flatbuffers.Builder, op: RopeNode
    ) -> Tuple[int, int]:
        """Auto-generated builder for RopeNode."""
        # Import the MODULE (not class) to access builder functions like Start(), Add*(), End()
        from executorch.backends.mlx.serialization._generated.mlx_delegate import RopeNode as FBRopeNodeModule
        from executorch.backends.mlx.serialization._generated.mlx_delegate import OpNode as FBOpNodeModule
        from executorch.backends.mlx.serialization._generated.mlx_delegate.Tid import CreateTid
        from executorch.backends.mlx.serialization._generated.mlx_delegate.Vid import CreateVid

        offset_off = self._build_vid_or_tid(builder, op.offset)

        FBRopeNodeModule.Start(builder)
        FBRopeNodeModule.AddX(builder, CreateTid(builder, op.x.idx))
        FBRopeNodeModule.AddOut(builder, CreateTid(builder, op.out.idx))
        FBRopeNodeModule.AddDims(builder, op.dims)
        FBRopeNodeModule.AddOffset(builder, offset_off)
        if op.freqs is not None:
            FBRopeNodeModule.AddFreqs(builder, CreateTid(builder, op.freqs.idx))
        FBRopeNodeModule.AddTraditional(builder, op.traditional)
        FBRopeNodeModule.AddBase(builder, op.base)
        FBRopeNodeModule.AddScale(builder, op.scale)
        offset = FBRopeNodeModule.End(builder)
        return offset, FBOpNodeModule.OpNode.RopeNode

    def _build_SdpaNode(
        self, builder: flatbuffers.Builder, op: SdpaNode
    ) -> Tuple[int, int]:
        """Auto-generated builder for SdpaNode."""
        # Import the MODULE (not class) to access builder functions like Start(), Add*(), End()
        from executorch.backends.mlx.serialization._generated.mlx_delegate import SdpaNode as FBSdpaNodeModule
        from executorch.backends.mlx.serialization._generated.mlx_delegate import OpNode as FBOpNodeModule
        from executorch.backends.mlx.serialization._generated.mlx_delegate.Tid import CreateTid
        from executorch.backends.mlx.serialization._generated.mlx_delegate.Vid import CreateVid

        FBSdpaNodeModule.Start(builder)
        FBSdpaNodeModule.AddQ(builder, CreateTid(builder, op.q.idx))
        FBSdpaNodeModule.AddK(builder, CreateTid(builder, op.k.idx))
        FBSdpaNodeModule.AddV(builder, CreateTid(builder, op.v.idx))
        FBSdpaNodeModule.AddOut(builder, CreateTid(builder, op.out.idx))
        FBSdpaNodeModule.AddScale(builder, op.scale)
        if op.mask is not None:
            FBSdpaNodeModule.AddMask(builder, CreateTid(builder, op.mask.idx))
        FBSdpaNodeModule.AddCausal(builder, op.causal)
        offset = FBSdpaNodeModule.End(builder)
        return offset, FBOpNodeModule.OpNode.SdpaNode

    def _build_AddNode(
        self, builder: flatbuffers.Builder, op: AddNode
    ) -> Tuple[int, int]:
        """Auto-generated builder for AddNode."""
        # Import the MODULE (not class) to access builder functions like Start(), Add*(), End()
        from executorch.backends.mlx.serialization._generated.mlx_delegate import AddNode as FBAddNodeModule
        from executorch.backends.mlx.serialization._generated.mlx_delegate import OpNode as FBOpNodeModule
        from executorch.backends.mlx.serialization._generated.mlx_delegate.Tid import CreateTid
        from executorch.backends.mlx.serialization._generated.mlx_delegate.Vid import CreateVid

        FBAddNodeModule.Start(builder)
        FBAddNodeModule.AddA(builder, CreateTid(builder, op.a.idx))
        FBAddNodeModule.AddB(builder, CreateTid(builder, op.b.idx))
        FBAddNodeModule.AddOut(builder, CreateTid(builder, op.out.idx))
        offset = FBAddNodeModule.End(builder)
        return offset, FBOpNodeModule.OpNode.AddNode

    def _build_AddIntNode(
        self, builder: flatbuffers.Builder, op: AddIntNode
    ) -> Tuple[int, int]:
        """Auto-generated builder for AddIntNode."""
        # Import the MODULE (not class) to access builder functions like Start(), Add*(), End()
        from executorch.backends.mlx.serialization._generated.mlx_delegate import AddIntNode as FBAddIntNodeModule
        from executorch.backends.mlx.serialization._generated.mlx_delegate import OpNode as FBOpNodeModule
        from executorch.backends.mlx.serialization._generated.mlx_delegate.Tid import CreateTid
        from executorch.backends.mlx.serialization._generated.mlx_delegate.Vid import CreateVid

        a_off = self._build_int_or_vid(builder, op.a)
        b_off = self._build_int_or_vid(builder, op.b)

        FBAddIntNodeModule.Start(builder)
        FBAddIntNodeModule.AddA(builder, a_off)
        FBAddIntNodeModule.AddB(builder, b_off)
        FBAddIntNodeModule.AddOut(builder, CreateVid(builder, op.out.idx))
        offset = FBAddIntNodeModule.End(builder)
        return offset, FBOpNodeModule.OpNode.AddIntNode

    def _build_SubtractIntNode(
        self, builder: flatbuffers.Builder, op: SubtractIntNode
    ) -> Tuple[int, int]:
        """Auto-generated builder for SubtractIntNode."""
        # Import the MODULE (not class) to access builder functions like Start(), Add*(), End()
        from executorch.backends.mlx.serialization._generated.mlx_delegate import SubtractIntNode as FBSubtractIntNodeModule
        from executorch.backends.mlx.serialization._generated.mlx_delegate import OpNode as FBOpNodeModule
        from executorch.backends.mlx.serialization._generated.mlx_delegate.Tid import CreateTid
        from executorch.backends.mlx.serialization._generated.mlx_delegate.Vid import CreateVid

        a_off = self._build_int_or_vid(builder, op.a)
        b_off = self._build_int_or_vid(builder, op.b)

        FBSubtractIntNodeModule.Start(builder)
        FBSubtractIntNodeModule.AddA(builder, a_off)
        FBSubtractIntNodeModule.AddB(builder, b_off)
        FBSubtractIntNodeModule.AddOut(builder, CreateVid(builder, op.out.idx))
        offset = FBSubtractIntNodeModule.End(builder)
        return offset, FBOpNodeModule.OpNode.SubtractIntNode

    def _build_MultiplyIntNode(
        self, builder: flatbuffers.Builder, op: MultiplyIntNode
    ) -> Tuple[int, int]:
        """Auto-generated builder for MultiplyIntNode."""
        # Import the MODULE (not class) to access builder functions like Start(), Add*(), End()
        from executorch.backends.mlx.serialization._generated.mlx_delegate import MultiplyIntNode as FBMultiplyIntNodeModule
        from executorch.backends.mlx.serialization._generated.mlx_delegate import OpNode as FBOpNodeModule
        from executorch.backends.mlx.serialization._generated.mlx_delegate.Tid import CreateTid
        from executorch.backends.mlx.serialization._generated.mlx_delegate.Vid import CreateVid

        a_off = self._build_int_or_vid(builder, op.a)
        b_off = self._build_int_or_vid(builder, op.b)

        FBMultiplyIntNodeModule.Start(builder)
        FBMultiplyIntNodeModule.AddA(builder, a_off)
        FBMultiplyIntNodeModule.AddB(builder, b_off)
        FBMultiplyIntNodeModule.AddOut(builder, CreateVid(builder, op.out.idx))
        offset = FBMultiplyIntNodeModule.End(builder)
        return offset, FBOpNodeModule.OpNode.MultiplyIntNode

    def _build_FloorDivideIntNode(
        self, builder: flatbuffers.Builder, op: FloorDivideIntNode
    ) -> Tuple[int, int]:
        """Auto-generated builder for FloorDivideIntNode."""
        # Import the MODULE (not class) to access builder functions like Start(), Add*(), End()
        from executorch.backends.mlx.serialization._generated.mlx_delegate import FloorDivideIntNode as FBFloorDivideIntNodeModule
        from executorch.backends.mlx.serialization._generated.mlx_delegate import OpNode as FBOpNodeModule
        from executorch.backends.mlx.serialization._generated.mlx_delegate.Tid import CreateTid
        from executorch.backends.mlx.serialization._generated.mlx_delegate.Vid import CreateVid

        a_off = self._build_int_or_vid(builder, op.a)
        b_off = self._build_int_or_vid(builder, op.b)

        FBFloorDivideIntNodeModule.Start(builder)
        FBFloorDivideIntNodeModule.AddA(builder, a_off)
        FBFloorDivideIntNodeModule.AddB(builder, b_off)
        FBFloorDivideIntNodeModule.AddOut(builder, CreateVid(builder, op.out.idx))
        offset = FBFloorDivideIntNodeModule.End(builder)
        return offset, FBOpNodeModule.OpNode.FloorDivideIntNode

    def _build_ModIntNode(
        self, builder: flatbuffers.Builder, op: ModIntNode
    ) -> Tuple[int, int]:
        """Auto-generated builder for ModIntNode."""
        # Import the MODULE (not class) to access builder functions like Start(), Add*(), End()
        from executorch.backends.mlx.serialization._generated.mlx_delegate import ModIntNode as FBModIntNodeModule
        from executorch.backends.mlx.serialization._generated.mlx_delegate import OpNode as FBOpNodeModule
        from executorch.backends.mlx.serialization._generated.mlx_delegate.Tid import CreateTid
        from executorch.backends.mlx.serialization._generated.mlx_delegate.Vid import CreateVid

        a_off = self._build_int_or_vid(builder, op.a)
        b_off = self._build_int_or_vid(builder, op.b)

        FBModIntNodeModule.Start(builder)
        FBModIntNodeModule.AddA(builder, a_off)
        FBModIntNodeModule.AddB(builder, b_off)
        FBModIntNodeModule.AddOut(builder, CreateVid(builder, op.out.idx))
        offset = FBModIntNodeModule.End(builder)
        return offset, FBOpNodeModule.OpNode.ModIntNode

    def _build_SymSizeNode(
        self, builder: flatbuffers.Builder, op: SymSizeNode
    ) -> Tuple[int, int]:
        """Auto-generated builder for SymSizeNode."""
        # Import the MODULE (not class) to access builder functions like Start(), Add*(), End()
        from executorch.backends.mlx.serialization._generated.mlx_delegate import SymSizeNode as FBSymSizeNodeModule
        from executorch.backends.mlx.serialization._generated.mlx_delegate import OpNode as FBOpNodeModule
        from executorch.backends.mlx.serialization._generated.mlx_delegate.Tid import CreateTid
        from executorch.backends.mlx.serialization._generated.mlx_delegate.Vid import CreateVid

        FBSymSizeNodeModule.Start(builder)
        FBSymSizeNodeModule.AddA(builder, CreateTid(builder, op.a.idx))
        FBSymSizeNodeModule.AddDim(builder, op.dim)
        FBSymSizeNodeModule.AddOut(builder, CreateVid(builder, op.out.idx))
        offset = FBSymSizeNodeModule.End(builder)
        return offset, FBOpNodeModule.OpNode.SymSizeNode

    def _build_MultiplyNode(
        self, builder: flatbuffers.Builder, op: MultiplyNode
    ) -> Tuple[int, int]:
        """Auto-generated builder for MultiplyNode."""
        # Import the MODULE (not class) to access builder functions like Start(), Add*(), End()
        from executorch.backends.mlx.serialization._generated.mlx_delegate import MultiplyNode as FBMultiplyNodeModule
        from executorch.backends.mlx.serialization._generated.mlx_delegate import OpNode as FBOpNodeModule
        from executorch.backends.mlx.serialization._generated.mlx_delegate.Tid import CreateTid
        from executorch.backends.mlx.serialization._generated.mlx_delegate.Vid import CreateVid

        FBMultiplyNodeModule.Start(builder)
        FBMultiplyNodeModule.AddA(builder, CreateTid(builder, op.a.idx))
        FBMultiplyNodeModule.AddB(builder, CreateTid(builder, op.b.idx))
        FBMultiplyNodeModule.AddOut(builder, CreateTid(builder, op.out.idx))
        offset = FBMultiplyNodeModule.End(builder)
        return offset, FBOpNodeModule.OpNode.MultiplyNode

    def _build_DivideNode(
        self, builder: flatbuffers.Builder, op: DivideNode
    ) -> Tuple[int, int]:
        """Auto-generated builder for DivideNode."""
        # Import the MODULE (not class) to access builder functions like Start(), Add*(), End()
        from executorch.backends.mlx.serialization._generated.mlx_delegate import DivideNode as FBDivideNodeModule
        from executorch.backends.mlx.serialization._generated.mlx_delegate import OpNode as FBOpNodeModule
        from executorch.backends.mlx.serialization._generated.mlx_delegate.Tid import CreateTid
        from executorch.backends.mlx.serialization._generated.mlx_delegate.Vid import CreateVid

        FBDivideNodeModule.Start(builder)
        FBDivideNodeModule.AddA(builder, CreateTid(builder, op.a.idx))
        FBDivideNodeModule.AddB(builder, CreateTid(builder, op.b.idx))
        FBDivideNodeModule.AddOut(builder, CreateTid(builder, op.out.idx))
        offset = FBDivideNodeModule.End(builder)
        return offset, FBOpNodeModule.OpNode.DivideNode

    def _build_SubtractNode(
        self, builder: flatbuffers.Builder, op: SubtractNode
    ) -> Tuple[int, int]:
        """Auto-generated builder for SubtractNode."""
        # Import the MODULE (not class) to access builder functions like Start(), Add*(), End()
        from executorch.backends.mlx.serialization._generated.mlx_delegate import SubtractNode as FBSubtractNodeModule
        from executorch.backends.mlx.serialization._generated.mlx_delegate import OpNode as FBOpNodeModule
        from executorch.backends.mlx.serialization._generated.mlx_delegate.Tid import CreateTid
        from executorch.backends.mlx.serialization._generated.mlx_delegate.Vid import CreateVid

        FBSubtractNodeModule.Start(builder)
        FBSubtractNodeModule.AddA(builder, CreateTid(builder, op.a.idx))
        FBSubtractNodeModule.AddB(builder, CreateTid(builder, op.b.idx))
        FBSubtractNodeModule.AddOut(builder, CreateTid(builder, op.out.idx))
        offset = FBSubtractNodeModule.End(builder)
        return offset, FBOpNodeModule.OpNode.SubtractNode

    def _build_Conv1DNode(
        self, builder: flatbuffers.Builder, op: Conv1DNode
    ) -> Tuple[int, int]:
        """Auto-generated builder for Conv1DNode."""
        # Import the MODULE (not class) to access builder functions like Start(), Add*(), End()
        from executorch.backends.mlx.serialization._generated.mlx_delegate import Conv1DNode as FBConv1DNodeModule
        from executorch.backends.mlx.serialization._generated.mlx_delegate import OpNode as FBOpNodeModule
        from executorch.backends.mlx.serialization._generated.mlx_delegate.Tid import CreateTid
        from executorch.backends.mlx.serialization._generated.mlx_delegate.Vid import CreateVid

        FBConv1DNodeModule.Start(builder)
        FBConv1DNodeModule.AddX(builder, CreateTid(builder, op.x.idx))
        FBConv1DNodeModule.AddW(builder, CreateTid(builder, op.w.idx))
        FBConv1DNodeModule.AddOut(builder, CreateTid(builder, op.out.idx))
        FBConv1DNodeModule.AddStride(builder, op.stride)
        FBConv1DNodeModule.AddPadding(builder, op.padding)
        FBConv1DNodeModule.AddDilation(builder, op.dilation)
        FBConv1DNodeModule.AddGroups(builder, op.groups)
        offset = FBConv1DNodeModule.End(builder)
        return offset, FBOpNodeModule.OpNode.Conv1DNode

    def _build_Conv2DNode(
        self, builder: flatbuffers.Builder, op: Conv2DNode
    ) -> Tuple[int, int]:
        """Auto-generated builder for Conv2DNode."""
        # Import the MODULE (not class) to access builder functions like Start(), Add*(), End()
        from executorch.backends.mlx.serialization._generated.mlx_delegate import Conv2DNode as FBConv2DNodeModule
        from executorch.backends.mlx.serialization._generated.mlx_delegate import OpNode as FBOpNodeModule
        from executorch.backends.mlx.serialization._generated.mlx_delegate.Tid import CreateTid
        from executorch.backends.mlx.serialization._generated.mlx_delegate.Vid import CreateVid

        FBConv2DNodeModule.Start(builder)
        FBConv2DNodeModule.AddX(builder, CreateTid(builder, op.x.idx))
        FBConv2DNodeModule.AddW(builder, CreateTid(builder, op.w.idx))
        FBConv2DNodeModule.AddOut(builder, CreateTid(builder, op.out.idx))
        FBConv2DNodeModule.AddStrideH(builder, op.stride_h)
        FBConv2DNodeModule.AddStrideW(builder, op.stride_w)
        FBConv2DNodeModule.AddPaddingH(builder, op.padding_h)
        FBConv2DNodeModule.AddPaddingW(builder, op.padding_w)
        FBConv2DNodeModule.AddDilationH(builder, op.dilation_h)
        FBConv2DNodeModule.AddDilationW(builder, op.dilation_w)
        FBConv2DNodeModule.AddGroups(builder, op.groups)
        offset = FBConv2DNodeModule.End(builder)
        return offset, FBOpNodeModule.OpNode.Conv2DNode

    def _build_Conv3DNode(
        self, builder: flatbuffers.Builder, op: Conv3DNode
    ) -> Tuple[int, int]:
        """Auto-generated builder for Conv3DNode."""
        # Import the MODULE (not class) to access builder functions like Start(), Add*(), End()
        from executorch.backends.mlx.serialization._generated.mlx_delegate import Conv3DNode as FBConv3DNodeModule
        from executorch.backends.mlx.serialization._generated.mlx_delegate import OpNode as FBOpNodeModule
        from executorch.backends.mlx.serialization._generated.mlx_delegate.Tid import CreateTid
        from executorch.backends.mlx.serialization._generated.mlx_delegate.Vid import CreateVid

        FBConv3DNodeModule.Start(builder)
        FBConv3DNodeModule.AddX(builder, CreateTid(builder, op.x.idx))
        FBConv3DNodeModule.AddW(builder, CreateTid(builder, op.w.idx))
        FBConv3DNodeModule.AddOut(builder, CreateTid(builder, op.out.idx))
        FBConv3DNodeModule.AddStrideD(builder, op.stride_d)
        FBConv3DNodeModule.AddStrideH(builder, op.stride_h)
        FBConv3DNodeModule.AddStrideW(builder, op.stride_w)
        FBConv3DNodeModule.AddPaddingD(builder, op.padding_d)
        FBConv3DNodeModule.AddPaddingH(builder, op.padding_h)
        FBConv3DNodeModule.AddPaddingW(builder, op.padding_w)
        FBConv3DNodeModule.AddDilationD(builder, op.dilation_d)
        FBConv3DNodeModule.AddDilationH(builder, op.dilation_h)
        FBConv3DNodeModule.AddDilationW(builder, op.dilation_w)
        FBConv3DNodeModule.AddGroups(builder, op.groups)
        offset = FBConv3DNodeModule.End(builder)
        return offset, FBOpNodeModule.OpNode.Conv3DNode

    def _build_ConvTranspose1DNode(
        self, builder: flatbuffers.Builder, op: ConvTranspose1DNode
    ) -> Tuple[int, int]:
        """Auto-generated builder for ConvTranspose1DNode."""
        # Import the MODULE (not class) to access builder functions like Start(), Add*(), End()
        from executorch.backends.mlx.serialization._generated.mlx_delegate import ConvTranspose1DNode as FBConvTranspose1DNodeModule
        from executorch.backends.mlx.serialization._generated.mlx_delegate import OpNode as FBOpNodeModule
        from executorch.backends.mlx.serialization._generated.mlx_delegate.Tid import CreateTid
        from executorch.backends.mlx.serialization._generated.mlx_delegate.Vid import CreateVid

        FBConvTranspose1DNodeModule.Start(builder)
        FBConvTranspose1DNodeModule.AddX(builder, CreateTid(builder, op.x.idx))
        FBConvTranspose1DNodeModule.AddW(builder, CreateTid(builder, op.w.idx))
        FBConvTranspose1DNodeModule.AddOut(builder, CreateTid(builder, op.out.idx))
        FBConvTranspose1DNodeModule.AddStride(builder, op.stride)
        FBConvTranspose1DNodeModule.AddPadding(builder, op.padding)
        FBConvTranspose1DNodeModule.AddDilation(builder, op.dilation)
        FBConvTranspose1DNodeModule.AddOutputPadding(builder, op.output_padding)
        FBConvTranspose1DNodeModule.AddGroups(builder, op.groups)
        offset = FBConvTranspose1DNodeModule.End(builder)
        return offset, FBOpNodeModule.OpNode.ConvTranspose1DNode

    def _build_ConvTranspose2DNode(
        self, builder: flatbuffers.Builder, op: ConvTranspose2DNode
    ) -> Tuple[int, int]:
        """Auto-generated builder for ConvTranspose2DNode."""
        # Import the MODULE (not class) to access builder functions like Start(), Add*(), End()
        from executorch.backends.mlx.serialization._generated.mlx_delegate import ConvTranspose2DNode as FBConvTranspose2DNodeModule
        from executorch.backends.mlx.serialization._generated.mlx_delegate import OpNode as FBOpNodeModule
        from executorch.backends.mlx.serialization._generated.mlx_delegate.Tid import CreateTid
        from executorch.backends.mlx.serialization._generated.mlx_delegate.Vid import CreateVid

        FBConvTranspose2DNodeModule.Start(builder)
        FBConvTranspose2DNodeModule.AddX(builder, CreateTid(builder, op.x.idx))
        FBConvTranspose2DNodeModule.AddW(builder, CreateTid(builder, op.w.idx))
        FBConvTranspose2DNodeModule.AddOut(builder, CreateTid(builder, op.out.idx))
        FBConvTranspose2DNodeModule.AddStrideH(builder, op.stride_h)
        FBConvTranspose2DNodeModule.AddStrideW(builder, op.stride_w)
        FBConvTranspose2DNodeModule.AddPaddingH(builder, op.padding_h)
        FBConvTranspose2DNodeModule.AddPaddingW(builder, op.padding_w)
        FBConvTranspose2DNodeModule.AddDilationH(builder, op.dilation_h)
        FBConvTranspose2DNodeModule.AddDilationW(builder, op.dilation_w)
        FBConvTranspose2DNodeModule.AddOutputPaddingH(builder, op.output_padding_h)
        FBConvTranspose2DNodeModule.AddOutputPaddingW(builder, op.output_padding_w)
        FBConvTranspose2DNodeModule.AddGroups(builder, op.groups)
        offset = FBConvTranspose2DNodeModule.End(builder)
        return offset, FBOpNodeModule.OpNode.ConvTranspose2DNode

    def _build_ConvTranspose3DNode(
        self, builder: flatbuffers.Builder, op: ConvTranspose3DNode
    ) -> Tuple[int, int]:
        """Auto-generated builder for ConvTranspose3DNode."""
        # Import the MODULE (not class) to access builder functions like Start(), Add*(), End()
        from executorch.backends.mlx.serialization._generated.mlx_delegate import ConvTranspose3DNode as FBConvTranspose3DNodeModule
        from executorch.backends.mlx.serialization._generated.mlx_delegate import OpNode as FBOpNodeModule
        from executorch.backends.mlx.serialization._generated.mlx_delegate.Tid import CreateTid
        from executorch.backends.mlx.serialization._generated.mlx_delegate.Vid import CreateVid

        FBConvTranspose3DNodeModule.Start(builder)
        FBConvTranspose3DNodeModule.AddX(builder, CreateTid(builder, op.x.idx))
        FBConvTranspose3DNodeModule.AddW(builder, CreateTid(builder, op.w.idx))
        FBConvTranspose3DNodeModule.AddOut(builder, CreateTid(builder, op.out.idx))
        FBConvTranspose3DNodeModule.AddStrideD(builder, op.stride_d)
        FBConvTranspose3DNodeModule.AddStrideH(builder, op.stride_h)
        FBConvTranspose3DNodeModule.AddStrideW(builder, op.stride_w)
        FBConvTranspose3DNodeModule.AddPaddingD(builder, op.padding_d)
        FBConvTranspose3DNodeModule.AddPaddingH(builder, op.padding_h)
        FBConvTranspose3DNodeModule.AddPaddingW(builder, op.padding_w)
        FBConvTranspose3DNodeModule.AddDilationD(builder, op.dilation_d)
        FBConvTranspose3DNodeModule.AddDilationH(builder, op.dilation_h)
        FBConvTranspose3DNodeModule.AddDilationW(builder, op.dilation_w)
        FBConvTranspose3DNodeModule.AddOutputPaddingD(builder, op.output_padding_d)
        FBConvTranspose3DNodeModule.AddOutputPaddingH(builder, op.output_padding_h)
        FBConvTranspose3DNodeModule.AddOutputPaddingW(builder, op.output_padding_w)
        FBConvTranspose3DNodeModule.AddGroups(builder, op.groups)
        offset = FBConvTranspose3DNodeModule.End(builder)
        return offset, FBOpNodeModule.OpNode.ConvTranspose3DNode

    def _build_GeluNode(
        self, builder: flatbuffers.Builder, op: GeluNode
    ) -> Tuple[int, int]:
        """Auto-generated builder for GeluNode."""
        # Import the MODULE (not class) to access builder functions like Start(), Add*(), End()
        from executorch.backends.mlx.serialization._generated.mlx_delegate import GeluNode as FBGeluNodeModule
        from executorch.backends.mlx.serialization._generated.mlx_delegate import OpNode as FBOpNodeModule
        from executorch.backends.mlx.serialization._generated.mlx_delegate.Tid import CreateTid
        from executorch.backends.mlx.serialization._generated.mlx_delegate.Vid import CreateVid

        approximate_off = builder.CreateString(op.approximate)

        FBGeluNodeModule.Start(builder)
        FBGeluNodeModule.AddX(builder, CreateTid(builder, op.x.idx))
        FBGeluNodeModule.AddOut(builder, CreateTid(builder, op.out.idx))
        FBGeluNodeModule.AddApproximate(builder, approximate_off)
        offset = FBGeluNodeModule.End(builder)
        return offset, FBOpNodeModule.OpNode.GeluNode

    def _build_ARangeNode(
        self, builder: flatbuffers.Builder, op: ARangeNode
    ) -> Tuple[int, int]:
        """Auto-generated builder for ARangeNode."""
        # Import the MODULE (not class) to access builder functions like Start(), Add*(), End()
        from executorch.backends.mlx.serialization._generated.mlx_delegate import ARangeNode as FBARangeNodeModule
        from executorch.backends.mlx.serialization._generated.mlx_delegate import OpNode as FBOpNodeModule
        from executorch.backends.mlx.serialization._generated.mlx_delegate.Tid import CreateTid
        from executorch.backends.mlx.serialization._generated.mlx_delegate.Vid import CreateVid

        start_off = self._build_int_or_vid(builder, op.start)
        stop_off = self._build_int_or_vid(builder, op.stop)
        step_off = self._build_int_or_vid(builder, op.step)

        FBARangeNodeModule.Start(builder)
        FBARangeNodeModule.AddOut(builder, CreateTid(builder, op.out.idx))
        FBARangeNodeModule.AddStart(builder, start_off)
        FBARangeNodeModule.AddStop(builder, stop_off)
        FBARangeNodeModule.AddStep(builder, step_off)
        if op.scalar_type is not None:
            FBARangeNodeModule.AddScalarType(builder, op.scalar_type)
        offset = FBARangeNodeModule.End(builder)
        return offset, FBOpNodeModule.OpNode.ARangeNode

    def _build_SiluNode(
        self, builder: flatbuffers.Builder, op: SiluNode
    ) -> Tuple[int, int]:
        """Auto-generated builder for SiluNode."""
        # Import the MODULE (not class) to access builder functions like Start(), Add*(), End()
        from executorch.backends.mlx.serialization._generated.mlx_delegate import SiluNode as FBSiluNodeModule
        from executorch.backends.mlx.serialization._generated.mlx_delegate import OpNode as FBOpNodeModule
        from executorch.backends.mlx.serialization._generated.mlx_delegate.Tid import CreateTid
        from executorch.backends.mlx.serialization._generated.mlx_delegate.Vid import CreateVid

        FBSiluNodeModule.Start(builder)
        FBSiluNodeModule.AddX(builder, CreateTid(builder, op.x.idx))
        FBSiluNodeModule.AddOut(builder, CreateTid(builder, op.out.idx))
        offset = FBSiluNodeModule.End(builder)
        return offset, FBOpNodeModule.OpNode.SiluNode

    def _build_SigmoidNode(
        self, builder: flatbuffers.Builder, op: SigmoidNode
    ) -> Tuple[int, int]:
        """Auto-generated builder for SigmoidNode."""
        # Import the MODULE (not class) to access builder functions like Start(), Add*(), End()
        from executorch.backends.mlx.serialization._generated.mlx_delegate import SigmoidNode as FBSigmoidNodeModule
        from executorch.backends.mlx.serialization._generated.mlx_delegate import OpNode as FBOpNodeModule
        from executorch.backends.mlx.serialization._generated.mlx_delegate.Tid import CreateTid
        from executorch.backends.mlx.serialization._generated.mlx_delegate.Vid import CreateVid

        FBSigmoidNodeModule.Start(builder)
        FBSigmoidNodeModule.AddX(builder, CreateTid(builder, op.x.idx))
        FBSigmoidNodeModule.AddOut(builder, CreateTid(builder, op.out.idx))
        offset = FBSigmoidNodeModule.End(builder)
        return offset, FBOpNodeModule.OpNode.SigmoidNode

    def _build_TanhNode(
        self, builder: flatbuffers.Builder, op: TanhNode
    ) -> Tuple[int, int]:
        """Auto-generated builder for TanhNode."""
        # Import the MODULE (not class) to access builder functions like Start(), Add*(), End()
        from executorch.backends.mlx.serialization._generated.mlx_delegate import TanhNode as FBTanhNodeModule
        from executorch.backends.mlx.serialization._generated.mlx_delegate import OpNode as FBOpNodeModule
        from executorch.backends.mlx.serialization._generated.mlx_delegate.Tid import CreateTid
        from executorch.backends.mlx.serialization._generated.mlx_delegate.Vid import CreateVid

        FBTanhNodeModule.Start(builder)
        FBTanhNodeModule.AddX(builder, CreateTid(builder, op.x.idx))
        FBTanhNodeModule.AddOut(builder, CreateTid(builder, op.out.idx))
        offset = FBTanhNodeModule.End(builder)
        return offset, FBOpNodeModule.OpNode.TanhNode

    def _build_SqueezeNode(
        self, builder: flatbuffers.Builder, op: SqueezeNode
    ) -> Tuple[int, int]:
        """Auto-generated builder for SqueezeNode."""
        # Import the MODULE (not class) to access builder functions like Start(), Add*(), End()
        from executorch.backends.mlx.serialization._generated.mlx_delegate import SqueezeNode as FBSqueezeNodeModule
        from executorch.backends.mlx.serialization._generated.mlx_delegate import OpNode as FBOpNodeModule
        from executorch.backends.mlx.serialization._generated.mlx_delegate.Tid import CreateTid
        from executorch.backends.mlx.serialization._generated.mlx_delegate.Vid import CreateVid

        dims_vec = _build_int_vector(builder, op.dims) if op.dims is not None else None

        FBSqueezeNodeModule.Start(builder)
        FBSqueezeNodeModule.AddX(builder, CreateTid(builder, op.x.idx))
        FBSqueezeNodeModule.AddOut(builder, CreateTid(builder, op.out.idx))
        if dims_vec is not None:
            FBSqueezeNodeModule.AddDims(builder, dims_vec)
        offset = FBSqueezeNodeModule.End(builder)
        return offset, FBOpNodeModule.OpNode.SqueezeNode

    def _build_SplitNode(
        self, builder: flatbuffers.Builder, op: SplitNode
    ) -> Tuple[int, int]:
        """Auto-generated builder for SplitNode."""
        # Import the MODULE (not class) to access builder functions like Start(), Add*(), End()
        from executorch.backends.mlx.serialization._generated.mlx_delegate import SplitNode as FBSplitNodeModule
        from executorch.backends.mlx.serialization._generated.mlx_delegate import OpNode as FBOpNodeModule
        from executorch.backends.mlx.serialization._generated.mlx_delegate.Tid import CreateTid
        from executorch.backends.mlx.serialization._generated.mlx_delegate.Vid import CreateVid

        outs_vec = self._build_tid_vector(builder, op.outs)
        sizes_vec = self._build_int_or_vid_vector(builder, op.sizes)

        FBSplitNodeModule.Start(builder)
        FBSplitNodeModule.AddX(builder, CreateTid(builder, op.x.idx))
        FBSplitNodeModule.AddOuts(builder, outs_vec)
        FBSplitNodeModule.AddSizes(builder, sizes_vec)
        FBSplitNodeModule.AddAxis(builder, op.axis)
        offset = FBSplitNodeModule.End(builder)
        return offset, FBOpNodeModule.OpNode.SplitNode

    def _build_RsqrtNode(
        self, builder: flatbuffers.Builder, op: RsqrtNode
    ) -> Tuple[int, int]:
        """Auto-generated builder for RsqrtNode."""
        # Import the MODULE (not class) to access builder functions like Start(), Add*(), End()
        from executorch.backends.mlx.serialization._generated.mlx_delegate import RsqrtNode as FBRsqrtNodeModule
        from executorch.backends.mlx.serialization._generated.mlx_delegate import OpNode as FBOpNodeModule
        from executorch.backends.mlx.serialization._generated.mlx_delegate.Tid import CreateTid
        from executorch.backends.mlx.serialization._generated.mlx_delegate.Vid import CreateVid

        FBRsqrtNodeModule.Start(builder)
        FBRsqrtNodeModule.AddX(builder, CreateTid(builder, op.x.idx))
        FBRsqrtNodeModule.AddOut(builder, CreateTid(builder, op.out.idx))
        offset = FBRsqrtNodeModule.End(builder)
        return offset, FBOpNodeModule.OpNode.RsqrtNode

    def _build_MaximumNode(
        self, builder: flatbuffers.Builder, op: MaximumNode
    ) -> Tuple[int, int]:
        """Auto-generated builder for MaximumNode."""
        # Import the MODULE (not class) to access builder functions like Start(), Add*(), End()
        from executorch.backends.mlx.serialization._generated.mlx_delegate import MaximumNode as FBMaximumNodeModule
        from executorch.backends.mlx.serialization._generated.mlx_delegate import OpNode as FBOpNodeModule
        from executorch.backends.mlx.serialization._generated.mlx_delegate.Tid import CreateTid
        from executorch.backends.mlx.serialization._generated.mlx_delegate.Vid import CreateVid

        FBMaximumNodeModule.Start(builder)
        FBMaximumNodeModule.AddA(builder, CreateTid(builder, op.a.idx))
        FBMaximumNodeModule.AddB(builder, CreateTid(builder, op.b.idx))
        FBMaximumNodeModule.AddOut(builder, CreateTid(builder, op.out.idx))
        offset = FBMaximumNodeModule.End(builder)
        return offset, FBOpNodeModule.OpNode.MaximumNode

    def _build_MinimumNode(
        self, builder: flatbuffers.Builder, op: MinimumNode
    ) -> Tuple[int, int]:
        """Auto-generated builder for MinimumNode."""
        # Import the MODULE (not class) to access builder functions like Start(), Add*(), End()
        from executorch.backends.mlx.serialization._generated.mlx_delegate import MinimumNode as FBMinimumNodeModule
        from executorch.backends.mlx.serialization._generated.mlx_delegate import OpNode as FBOpNodeModule
        from executorch.backends.mlx.serialization._generated.mlx_delegate.Tid import CreateTid
        from executorch.backends.mlx.serialization._generated.mlx_delegate.Vid import CreateVid

        FBMinimumNodeModule.Start(builder)
        FBMinimumNodeModule.AddA(builder, CreateTid(builder, op.a.idx))
        FBMinimumNodeModule.AddB(builder, CreateTid(builder, op.b.idx))
        FBMinimumNodeModule.AddOut(builder, CreateTid(builder, op.out.idx))
        offset = FBMinimumNodeModule.End(builder)
        return offset, FBOpNodeModule.OpNode.MinimumNode

    def _build_LogNode(
        self, builder: flatbuffers.Builder, op: LogNode
    ) -> Tuple[int, int]:
        """Auto-generated builder for LogNode."""
        # Import the MODULE (not class) to access builder functions like Start(), Add*(), End()
        from executorch.backends.mlx.serialization._generated.mlx_delegate import LogNode as FBLogNodeModule
        from executorch.backends.mlx.serialization._generated.mlx_delegate import OpNode as FBOpNodeModule
        from executorch.backends.mlx.serialization._generated.mlx_delegate.Tid import CreateTid
        from executorch.backends.mlx.serialization._generated.mlx_delegate.Vid import CreateVid

        FBLogNodeModule.Start(builder)
        FBLogNodeModule.AddX(builder, CreateTid(builder, op.x.idx))
        FBLogNodeModule.AddOut(builder, CreateTid(builder, op.out.idx))
        offset = FBLogNodeModule.End(builder)
        return offset, FBOpNodeModule.OpNode.LogNode

    def _build_SoftmaxNode(
        self, builder: flatbuffers.Builder, op: SoftmaxNode
    ) -> Tuple[int, int]:
        """Auto-generated builder for SoftmaxNode."""
        # Import the MODULE (not class) to access builder functions like Start(), Add*(), End()
        from executorch.backends.mlx.serialization._generated.mlx_delegate import SoftmaxNode as FBSoftmaxNodeModule
        from executorch.backends.mlx.serialization._generated.mlx_delegate import OpNode as FBOpNodeModule
        from executorch.backends.mlx.serialization._generated.mlx_delegate.Tid import CreateTid
        from executorch.backends.mlx.serialization._generated.mlx_delegate.Vid import CreateVid

        FBSoftmaxNodeModule.Start(builder)
        FBSoftmaxNodeModule.AddX(builder, CreateTid(builder, op.x.idx))
        FBSoftmaxNodeModule.AddOut(builder, CreateTid(builder, op.out.idx))
        FBSoftmaxNodeModule.AddAxis(builder, op.axis)
        FBSoftmaxNodeModule.AddPrecise(builder, op.precise)
        offset = FBSoftmaxNodeModule.End(builder)
        return offset, FBOpNodeModule.OpNode.SoftmaxNode

    def _build_BroadcastToNode(
        self, builder: flatbuffers.Builder, op: BroadcastToNode
    ) -> Tuple[int, int]:
        """Auto-generated builder for BroadcastToNode."""
        # Import the MODULE (not class) to access builder functions like Start(), Add*(), End()
        from executorch.backends.mlx.serialization._generated.mlx_delegate import BroadcastToNode as FBBroadcastToNodeModule
        from executorch.backends.mlx.serialization._generated.mlx_delegate import OpNode as FBOpNodeModule
        from executorch.backends.mlx.serialization._generated.mlx_delegate.Tid import CreateTid
        from executorch.backends.mlx.serialization._generated.mlx_delegate.Vid import CreateVid

        shape_vec = self._build_int_or_vid_vector(builder, op.shape)

        FBBroadcastToNodeModule.Start(builder)
        FBBroadcastToNodeModule.AddX(builder, CreateTid(builder, op.x.idx))
        FBBroadcastToNodeModule.AddOut(builder, CreateTid(builder, op.out.idx))
        FBBroadcastToNodeModule.AddShape(builder, shape_vec)
        offset = FBBroadcastToNodeModule.End(builder)
        return offset, FBOpNodeModule.OpNode.BroadcastToNode

    def _build_PadNode(
        self, builder: flatbuffers.Builder, op: PadNode
    ) -> Tuple[int, int]:
        """Auto-generated builder for PadNode."""
        # Import the MODULE (not class) to access builder functions like Start(), Add*(), End()
        from executorch.backends.mlx.serialization._generated.mlx_delegate import PadNode as FBPadNodeModule
        from executorch.backends.mlx.serialization._generated.mlx_delegate import OpNode as FBOpNodeModule
        from executorch.backends.mlx.serialization._generated.mlx_delegate.Tid import CreateTid
        from executorch.backends.mlx.serialization._generated.mlx_delegate.Vid import CreateVid

        pad_width_vec = self._build_int_or_vid_vector(builder, op.pad_width)
        mode_off = builder.CreateString(op.mode)

        FBPadNodeModule.Start(builder)
        FBPadNodeModule.AddX(builder, CreateTid(builder, op.x.idx))
        FBPadNodeModule.AddOut(builder, CreateTid(builder, op.out.idx))
        FBPadNodeModule.AddPadWidth(builder, pad_width_vec)
        FBPadNodeModule.AddMode(builder, mode_off)
        FBPadNodeModule.AddConstantValue(builder, op.constant_value)
        offset = FBPadNodeModule.End(builder)
        return offset, FBOpNodeModule.OpNode.PadNode

    def _build_WhereNode(
        self, builder: flatbuffers.Builder, op: WhereNode
    ) -> Tuple[int, int]:
        """Auto-generated builder for WhereNode."""
        # Import the MODULE (not class) to access builder functions like Start(), Add*(), End()
        from executorch.backends.mlx.serialization._generated.mlx_delegate import WhereNode as FBWhereNodeModule
        from executorch.backends.mlx.serialization._generated.mlx_delegate import OpNode as FBOpNodeModule
        from executorch.backends.mlx.serialization._generated.mlx_delegate.Tid import CreateTid
        from executorch.backends.mlx.serialization._generated.mlx_delegate.Vid import CreateVid

        FBWhereNodeModule.Start(builder)
        FBWhereNodeModule.AddCondition(builder, CreateTid(builder, op.condition.idx))
        FBWhereNodeModule.AddX(builder, CreateTid(builder, op.x.idx))
        FBWhereNodeModule.AddY(builder, CreateTid(builder, op.y.idx))
        FBWhereNodeModule.AddOut(builder, CreateTid(builder, op.out.idx))
        offset = FBWhereNodeModule.End(builder)
        return offset, FBOpNodeModule.OpNode.WhereNode

    def _build_ReshapeNode(
        self, builder: flatbuffers.Builder, op: ReshapeNode
    ) -> Tuple[int, int]:
        """Auto-generated builder for ReshapeNode."""
        # Import the MODULE (not class) to access builder functions like Start(), Add*(), End()
        from executorch.backends.mlx.serialization._generated.mlx_delegate import ReshapeNode as FBReshapeNodeModule
        from executorch.backends.mlx.serialization._generated.mlx_delegate import OpNode as FBOpNodeModule
        from executorch.backends.mlx.serialization._generated.mlx_delegate.Tid import CreateTid
        from executorch.backends.mlx.serialization._generated.mlx_delegate.Vid import CreateVid

        shape_vec = self._build_int_or_vid_vector(builder, op.shape)

        FBReshapeNodeModule.Start(builder)
        FBReshapeNodeModule.AddX(builder, CreateTid(builder, op.x.idx))
        FBReshapeNodeModule.AddOut(builder, CreateTid(builder, op.out.idx))
        FBReshapeNodeModule.AddShape(builder, shape_vec)
        offset = FBReshapeNodeModule.End(builder)
        return offset, FBOpNodeModule.OpNode.ReshapeNode

    def _build_TransposeNode(
        self, builder: flatbuffers.Builder, op: TransposeNode
    ) -> Tuple[int, int]:
        """Auto-generated builder for TransposeNode."""
        # Import the MODULE (not class) to access builder functions like Start(), Add*(), End()
        from executorch.backends.mlx.serialization._generated.mlx_delegate import TransposeNode as FBTransposeNodeModule
        from executorch.backends.mlx.serialization._generated.mlx_delegate import OpNode as FBOpNodeModule
        from executorch.backends.mlx.serialization._generated.mlx_delegate.Tid import CreateTid
        from executorch.backends.mlx.serialization._generated.mlx_delegate.Vid import CreateVid

        perm_vec = _build_int_vector(builder, op.perm)

        FBTransposeNodeModule.Start(builder)
        FBTransposeNodeModule.AddX(builder, CreateTid(builder, op.x.idx))
        FBTransposeNodeModule.AddOut(builder, CreateTid(builder, op.out.idx))
        FBTransposeNodeModule.AddPerm(builder, perm_vec)
        offset = FBTransposeNodeModule.End(builder)
        return offset, FBOpNodeModule.OpNode.TransposeNode

    def _build_AsStridedNode(
        self, builder: flatbuffers.Builder, op: AsStridedNode
    ) -> Tuple[int, int]:
        """Auto-generated builder for AsStridedNode."""
        # Import the MODULE (not class) to access builder functions like Start(), Add*(), End()
        from executorch.backends.mlx.serialization._generated.mlx_delegate import AsStridedNode as FBAsStridedNodeModule
        from executorch.backends.mlx.serialization._generated.mlx_delegate import OpNode as FBOpNodeModule
        from executorch.backends.mlx.serialization._generated.mlx_delegate.Tid import CreateTid
        from executorch.backends.mlx.serialization._generated.mlx_delegate.Vid import CreateVid

        shape_vec = self._build_int_or_vid_vector(builder, op.shape)
        strides_vec = self._build_int_or_vid_vector(builder, op.strides)

        FBAsStridedNodeModule.Start(builder)
        FBAsStridedNodeModule.AddX(builder, CreateTid(builder, op.x.idx))
        FBAsStridedNodeModule.AddOut(builder, CreateTid(builder, op.out.idx))
        FBAsStridedNodeModule.AddShape(builder, shape_vec)
        FBAsStridedNodeModule.AddStrides(builder, strides_vec)
        FBAsStridedNodeModule.AddOffset(builder, op.offset)
        offset = FBAsStridedNodeModule.End(builder)
        return offset, FBOpNodeModule.OpNode.AsStridedNode

    def _build_ContiguousNode(
        self, builder: flatbuffers.Builder, op: ContiguousNode
    ) -> Tuple[int, int]:
        """Auto-generated builder for ContiguousNode."""
        # Import the MODULE (not class) to access builder functions like Start(), Add*(), End()
        from executorch.backends.mlx.serialization._generated.mlx_delegate import ContiguousNode as FBContiguousNodeModule
        from executorch.backends.mlx.serialization._generated.mlx_delegate import OpNode as FBOpNodeModule
        from executorch.backends.mlx.serialization._generated.mlx_delegate.Tid import CreateTid
        from executorch.backends.mlx.serialization._generated.mlx_delegate.Vid import CreateVid

        FBContiguousNodeModule.Start(builder)
        FBContiguousNodeModule.AddX(builder, CreateTid(builder, op.x.idx))
        FBContiguousNodeModule.AddOut(builder, CreateTid(builder, op.out.idx))
        offset = FBContiguousNodeModule.End(builder)
        return offset, FBOpNodeModule.OpNode.ContiguousNode

    def _build_GatherNode(
        self, builder: flatbuffers.Builder, op: GatherNode
    ) -> Tuple[int, int]:
        """Auto-generated builder for GatherNode."""
        # Import the MODULE (not class) to access builder functions like Start(), Add*(), End()
        from executorch.backends.mlx.serialization._generated.mlx_delegate import GatherNode as FBGatherNodeModule
        from executorch.backends.mlx.serialization._generated.mlx_delegate import OpNode as FBOpNodeModule
        from executorch.backends.mlx.serialization._generated.mlx_delegate.Tid import CreateTid
        from executorch.backends.mlx.serialization._generated.mlx_delegate.Vid import CreateVid

        indices_vec = self._build_tid_vector(builder, op.indices)
        axes_vec = _build_int_vector(builder, op.axes)
        slice_sizes_vec = _build_int_vector(builder, op.slice_sizes)

        FBGatherNodeModule.Start(builder)
        FBGatherNodeModule.AddX(builder, CreateTid(builder, op.x.idx))
        FBGatherNodeModule.AddIndices(builder, indices_vec)
        FBGatherNodeModule.AddOut(builder, CreateTid(builder, op.out.idx))
        FBGatherNodeModule.AddAxes(builder, axes_vec)
        FBGatherNodeModule.AddSliceSizes(builder, slice_sizes_vec)
        offset = FBGatherNodeModule.End(builder)
        return offset, FBOpNodeModule.OpNode.GatherNode

    def _build_SliceNode(
        self, builder: flatbuffers.Builder, op: SliceNode
    ) -> Tuple[int, int]:
        """Auto-generated builder for SliceNode."""
        # Import the MODULE (not class) to access builder functions like Start(), Add*(), End()
        from executorch.backends.mlx.serialization._generated.mlx_delegate import SliceNode as FBSliceNodeModule
        from executorch.backends.mlx.serialization._generated.mlx_delegate import OpNode as FBOpNodeModule
        from executorch.backends.mlx.serialization._generated.mlx_delegate.Tid import CreateTid
        from executorch.backends.mlx.serialization._generated.mlx_delegate.Vid import CreateVid

        axis_off = self._build_int_or_vid(builder, op.axis)
        start_off = self._build_int_or_vid(builder, op.start)
        stop_off = self._build_int_or_vid(builder, op.stop)

        FBSliceNodeModule.Start(builder)
        FBSliceNodeModule.AddX(builder, CreateTid(builder, op.x.idx))
        FBSliceNodeModule.AddOut(builder, CreateTid(builder, op.out.idx))
        FBSliceNodeModule.AddAxis(builder, axis_off)
        FBSliceNodeModule.AddStart(builder, start_off)
        FBSliceNodeModule.AddStop(builder, stop_off)
        FBSliceNodeModule.AddStep(builder, op.step)
        offset = FBSliceNodeModule.End(builder)
        return offset, FBOpNodeModule.OpNode.SliceNode

    def _build_AsTypeNode(
        self, builder: flatbuffers.Builder, op: AsTypeNode
    ) -> Tuple[int, int]:
        """Auto-generated builder for AsTypeNode."""
        # Import the MODULE (not class) to access builder functions like Start(), Add*(), End()
        from executorch.backends.mlx.serialization._generated.mlx_delegate import AsTypeNode as FBAsTypeNodeModule
        from executorch.backends.mlx.serialization._generated.mlx_delegate import OpNode as FBOpNodeModule
        from executorch.backends.mlx.serialization._generated.mlx_delegate.Tid import CreateTid
        from executorch.backends.mlx.serialization._generated.mlx_delegate.Vid import CreateVid

        FBAsTypeNodeModule.Start(builder)
        FBAsTypeNodeModule.AddX(builder, CreateTid(builder, op.x.idx))
        FBAsTypeNodeModule.AddOut(builder, CreateTid(builder, op.out.idx))
        FBAsTypeNodeModule.AddScalarType(builder, op.scalar_type)
        offset = FBAsTypeNodeModule.End(builder)
        return offset, FBOpNodeModule.OpNode.AsTypeNode

    def _build_QuantizedMatmulNode(
        self, builder: flatbuffers.Builder, op: QuantizedMatmulNode
    ) -> Tuple[int, int]:
        """Auto-generated builder for QuantizedMatmulNode."""
        # Import the MODULE (not class) to access builder functions like Start(), Add*(), End()
        from executorch.backends.mlx.serialization._generated.mlx_delegate import QuantizedMatmulNode as FBQuantizedMatmulNodeModule
        from executorch.backends.mlx.serialization._generated.mlx_delegate import OpNode as FBOpNodeModule
        from executorch.backends.mlx.serialization._generated.mlx_delegate.Tid import CreateTid
        from executorch.backends.mlx.serialization._generated.mlx_delegate.Vid import CreateVid

        mode_off = builder.CreateString(op.mode)

        FBQuantizedMatmulNodeModule.Start(builder)
        FBQuantizedMatmulNodeModule.AddX(builder, CreateTid(builder, op.x.idx))
        FBQuantizedMatmulNodeModule.AddW(builder, CreateTid(builder, op.w.idx))
        FBQuantizedMatmulNodeModule.AddScales(builder, CreateTid(builder, op.scales.idx))
        FBQuantizedMatmulNodeModule.AddOut(builder, CreateTid(builder, op.out.idx))
        if op.biases is not None:
            FBQuantizedMatmulNodeModule.AddBiases(builder, CreateTid(builder, op.biases.idx))
        FBQuantizedMatmulNodeModule.AddGroupSize(builder, op.group_size)
        FBQuantizedMatmulNodeModule.AddBits(builder, op.bits)
        FBQuantizedMatmulNodeModule.AddMode(builder, mode_off)
        FBQuantizedMatmulNodeModule.AddTranspose(builder, op.transpose)
        offset = FBQuantizedMatmulNodeModule.End(builder)
        return offset, FBOpNodeModule.OpNode.QuantizedMatmulNode

    def _build_ScatterAddNode(
        self, builder: flatbuffers.Builder, op: ScatterAddNode
    ) -> Tuple[int, int]:
        """Auto-generated builder for ScatterAddNode."""
        # Import the MODULE (not class) to access builder functions like Start(), Add*(), End()
        from executorch.backends.mlx.serialization._generated.mlx_delegate import ScatterAddNode as FBScatterAddNodeModule
        from executorch.backends.mlx.serialization._generated.mlx_delegate import OpNode as FBOpNodeModule
        from executorch.backends.mlx.serialization._generated.mlx_delegate.Tid import CreateTid
        from executorch.backends.mlx.serialization._generated.mlx_delegate.Vid import CreateVid

        FBScatterAddNodeModule.Start(builder)
        FBScatterAddNodeModule.AddX(builder, CreateTid(builder, op.x.idx))
        FBScatterAddNodeModule.AddIndices(builder, CreateTid(builder, op.indices.idx))
        FBScatterAddNodeModule.AddUpdates(builder, CreateTid(builder, op.updates.idx))
        FBScatterAddNodeModule.AddOut(builder, CreateTid(builder, op.out.idx))
        FBScatterAddNodeModule.AddAxis(builder, op.axis)
        offset = FBScatterAddNodeModule.End(builder)
        return offset, FBOpNodeModule.OpNode.ScatterAddNode

    def _build_ConcatenateNode(
        self, builder: flatbuffers.Builder, op: ConcatenateNode
    ) -> Tuple[int, int]:
        """Auto-generated builder for ConcatenateNode."""
        # Import the MODULE (not class) to access builder functions like Start(), Add*(), End()
        from executorch.backends.mlx.serialization._generated.mlx_delegate import ConcatenateNode as FBConcatenateNodeModule
        from executorch.backends.mlx.serialization._generated.mlx_delegate import OpNode as FBOpNodeModule
        from executorch.backends.mlx.serialization._generated.mlx_delegate.Tid import CreateTid
        from executorch.backends.mlx.serialization._generated.mlx_delegate.Vid import CreateVid

        tensors_vec = self._build_tid_vector(builder, op.tensors)

        FBConcatenateNodeModule.Start(builder)
        FBConcatenateNodeModule.AddTensors(builder, tensors_vec)
        FBConcatenateNodeModule.AddOut(builder, CreateTid(builder, op.out.idx))
        FBConcatenateNodeModule.AddAxis(builder, op.axis)
        offset = FBConcatenateNodeModule.End(builder)
        return offset, FBOpNodeModule.OpNode.ConcatenateNode

    def _build_FullNode(
        self, builder: flatbuffers.Builder, op: FullNode
    ) -> Tuple[int, int]:
        """Auto-generated builder for FullNode."""
        # Import the MODULE (not class) to access builder functions like Start(), Add*(), End()
        from executorch.backends.mlx.serialization._generated.mlx_delegate import FullNode as FBFullNodeModule
        from executorch.backends.mlx.serialization._generated.mlx_delegate import OpNode as FBOpNodeModule
        from executorch.backends.mlx.serialization._generated.mlx_delegate.Tid import CreateTid
        from executorch.backends.mlx.serialization._generated.mlx_delegate.Vid import CreateVid

        shape_vec = self._build_int_or_vid_vector(builder, op.shape)
        v_off = self._build_float_or_vid(builder, op.v)

        FBFullNodeModule.Start(builder)
        FBFullNodeModule.AddOut(builder, CreateTid(builder, op.out.idx))
        FBFullNodeModule.AddShape(builder, shape_vec)
        FBFullNodeModule.AddV(builder, v_off)
        FBFullNodeModule.AddScalarType(builder, op.scalar_type)
        offset = FBFullNodeModule.End(builder)
        return offset, FBOpNodeModule.OpNode.FullNode

    def _build_FullLikeNode(
        self, builder: flatbuffers.Builder, op: FullLikeNode
    ) -> Tuple[int, int]:
        """Auto-generated builder for FullLikeNode."""
        # Import the MODULE (not class) to access builder functions like Start(), Add*(), End()
        from executorch.backends.mlx.serialization._generated.mlx_delegate import FullLikeNode as FBFullLikeNodeModule
        from executorch.backends.mlx.serialization._generated.mlx_delegate import OpNode as FBOpNodeModule
        from executorch.backends.mlx.serialization._generated.mlx_delegate.Tid import CreateTid
        from executorch.backends.mlx.serialization._generated.mlx_delegate.Vid import CreateVid

        v_off = self._build_float_or_vid(builder, op.v)

        FBFullLikeNodeModule.Start(builder)
        FBFullLikeNodeModule.AddX(builder, CreateTid(builder, op.x.idx))
        FBFullLikeNodeModule.AddOut(builder, CreateTid(builder, op.out.idx))
        FBFullLikeNodeModule.AddV(builder, v_off)
        if op.scalar_type is not None:
            FBFullLikeNodeModule.AddScalarType(builder, op.scalar_type)
        offset = FBFullLikeNodeModule.End(builder)
        return offset, FBOpNodeModule.OpNode.FullLikeNode

    def _build_ArgmaxNode(
        self, builder: flatbuffers.Builder, op: ArgmaxNode
    ) -> Tuple[int, int]:
        """Auto-generated builder for ArgmaxNode."""
        # Import the MODULE (not class) to access builder functions like Start(), Add*(), End()
        from executorch.backends.mlx.serialization._generated.mlx_delegate import ArgmaxNode as FBArgmaxNodeModule
        from executorch.backends.mlx.serialization._generated.mlx_delegate import OpNode as FBOpNodeModule
        from executorch.backends.mlx.serialization._generated.mlx_delegate.Tid import CreateTid
        from executorch.backends.mlx.serialization._generated.mlx_delegate.Vid import CreateVid

        FBArgmaxNodeModule.Start(builder)
        FBArgmaxNodeModule.AddX(builder, CreateTid(builder, op.x.idx))
        FBArgmaxNodeModule.AddOut(builder, CreateTid(builder, op.out.idx))
        FBArgmaxNodeModule.AddAxis(builder, op.axis)
        FBArgmaxNodeModule.AddKeepdims(builder, op.keepdims)
        offset = FBArgmaxNodeModule.End(builder)
        return offset, FBOpNodeModule.OpNode.ArgmaxNode

    def _build_SliceUpdateNode(
        self, builder: flatbuffers.Builder, op: SliceUpdateNode
    ) -> Tuple[int, int]:
        """Auto-generated builder for SliceUpdateNode."""
        # Import the MODULE (not class) to access builder functions like Start(), Add*(), End()
        from executorch.backends.mlx.serialization._generated.mlx_delegate import SliceUpdateNode as FBSliceUpdateNodeModule
        from executorch.backends.mlx.serialization._generated.mlx_delegate import OpNode as FBOpNodeModule
        from executorch.backends.mlx.serialization._generated.mlx_delegate.Tid import CreateTid
        from executorch.backends.mlx.serialization._generated.mlx_delegate.Vid import CreateVid

        axis_off = self._build_int_or_vid(builder, op.axis)
        start_off = self._build_int_or_vid(builder, op.start)
        stop_off = self._build_int_or_vid(builder, op.stop)

        FBSliceUpdateNodeModule.Start(builder)
        FBSliceUpdateNodeModule.AddDst(builder, CreateTid(builder, op.dst.idx))
        FBSliceUpdateNodeModule.AddUpdate(builder, CreateTid(builder, op.update.idx))
        FBSliceUpdateNodeModule.AddOut(builder, CreateTid(builder, op.out.idx))
        FBSliceUpdateNodeModule.AddAxis(builder, axis_off)
        FBSliceUpdateNodeModule.AddStart(builder, start_off)
        FBSliceUpdateNodeModule.AddStop(builder, stop_off)
        FBSliceUpdateNodeModule.AddStep(builder, op.step)
        offset = FBSliceUpdateNodeModule.End(builder)
        return offset, FBOpNodeModule.OpNode.SliceUpdateNode

    def _build_IndexCopyNode(
        self, builder: flatbuffers.Builder, op: IndexCopyNode
    ) -> Tuple[int, int]:
        """Auto-generated builder for IndexCopyNode."""
        # Import the MODULE (not class) to access builder functions like Start(), Add*(), End()
        from executorch.backends.mlx.serialization._generated.mlx_delegate import IndexCopyNode as FBIndexCopyNodeModule
        from executorch.backends.mlx.serialization._generated.mlx_delegate import OpNode as FBOpNodeModule
        from executorch.backends.mlx.serialization._generated.mlx_delegate.Tid import CreateTid
        from executorch.backends.mlx.serialization._generated.mlx_delegate.Vid import CreateVid

        FBIndexCopyNodeModule.Start(builder)
        FBIndexCopyNodeModule.AddDst(builder, CreateTid(builder, op.dst.idx))
        FBIndexCopyNodeModule.AddUpdate(builder, CreateTid(builder, op.update.idx))
        FBIndexCopyNodeModule.AddIndices(builder, CreateTid(builder, op.indices.idx))
        FBIndexCopyNodeModule.AddOut(builder, CreateTid(builder, op.out.idx))
        FBIndexCopyNodeModule.AddAxis(builder, op.axis)
        offset = FBIndexCopyNodeModule.End(builder)
        return offset, FBOpNodeModule.OpNode.IndexCopyNode

    def _build_DequantizeNode(
        self, builder: flatbuffers.Builder, op: DequantizeNode
    ) -> Tuple[int, int]:
        """Auto-generated builder for DequantizeNode."""
        # Import the MODULE (not class) to access builder functions like Start(), Add*(), End()
        from executorch.backends.mlx.serialization._generated.mlx_delegate import DequantizeNode as FBDequantizeNodeModule
        from executorch.backends.mlx.serialization._generated.mlx_delegate import OpNode as FBOpNodeModule
        from executorch.backends.mlx.serialization._generated.mlx_delegate.Tid import CreateTid
        from executorch.backends.mlx.serialization._generated.mlx_delegate.Vid import CreateVid

        mode_off = builder.CreateString(op.mode)

        FBDequantizeNodeModule.Start(builder)
        FBDequantizeNodeModule.AddW(builder, CreateTid(builder, op.w.idx))
        FBDequantizeNodeModule.AddScales(builder, CreateTid(builder, op.scales.idx))
        FBDequantizeNodeModule.AddOut(builder, CreateTid(builder, op.out.idx))
        if op.biases is not None:
            FBDequantizeNodeModule.AddBiases(builder, CreateTid(builder, op.biases.idx))
        FBDequantizeNodeModule.AddGroupSize(builder, op.group_size)
        FBDequantizeNodeModule.AddBits(builder, op.bits)
        FBDequantizeNodeModule.AddMode(builder, mode_off)
        if op.global_scale is not None:
            FBDequantizeNodeModule.AddGlobalScale(builder, CreateTid(builder, op.global_scale.idx))
        if op.dtype is not None:
            FBDequantizeNodeModule.AddDtype(builder, op.dtype)
        offset = FBDequantizeNodeModule.End(builder)
        return offset, FBOpNodeModule.OpNode.DequantizeNode

    def _build_LessNode(
        self, builder: flatbuffers.Builder, op: LessNode
    ) -> Tuple[int, int]:
        """Auto-generated builder for LessNode."""
        # Import the MODULE (not class) to access builder functions like Start(), Add*(), End()
        from executorch.backends.mlx.serialization._generated.mlx_delegate import LessNode as FBLessNodeModule
        from executorch.backends.mlx.serialization._generated.mlx_delegate import OpNode as FBOpNodeModule
        from executorch.backends.mlx.serialization._generated.mlx_delegate.Tid import CreateTid
        from executorch.backends.mlx.serialization._generated.mlx_delegate.Vid import CreateVid

        FBLessNodeModule.Start(builder)
        FBLessNodeModule.AddA(builder, CreateTid(builder, op.a.idx))
        FBLessNodeModule.AddB(builder, CreateTid(builder, op.b.idx))
        FBLessNodeModule.AddOut(builder, CreateTid(builder, op.out.idx))
        offset = FBLessNodeModule.End(builder)
        return offset, FBOpNodeModule.OpNode.LessNode

    def _build_LessEqualNode(
        self, builder: flatbuffers.Builder, op: LessEqualNode
    ) -> Tuple[int, int]:
        """Auto-generated builder for LessEqualNode."""
        # Import the MODULE (not class) to access builder functions like Start(), Add*(), End()
        from executorch.backends.mlx.serialization._generated.mlx_delegate import LessEqualNode as FBLessEqualNodeModule
        from executorch.backends.mlx.serialization._generated.mlx_delegate import OpNode as FBOpNodeModule
        from executorch.backends.mlx.serialization._generated.mlx_delegate.Tid import CreateTid
        from executorch.backends.mlx.serialization._generated.mlx_delegate.Vid import CreateVid

        FBLessEqualNodeModule.Start(builder)
        FBLessEqualNodeModule.AddA(builder, CreateTid(builder, op.a.idx))
        FBLessEqualNodeModule.AddB(builder, CreateTid(builder, op.b.idx))
        FBLessEqualNodeModule.AddOut(builder, CreateTid(builder, op.out.idx))
        offset = FBLessEqualNodeModule.End(builder)
        return offset, FBOpNodeModule.OpNode.LessEqualNode

    def _build_GreaterNode(
        self, builder: flatbuffers.Builder, op: GreaterNode
    ) -> Tuple[int, int]:
        """Auto-generated builder for GreaterNode."""
        # Import the MODULE (not class) to access builder functions like Start(), Add*(), End()
        from executorch.backends.mlx.serialization._generated.mlx_delegate import GreaterNode as FBGreaterNodeModule
        from executorch.backends.mlx.serialization._generated.mlx_delegate import OpNode as FBOpNodeModule
        from executorch.backends.mlx.serialization._generated.mlx_delegate.Tid import CreateTid
        from executorch.backends.mlx.serialization._generated.mlx_delegate.Vid import CreateVid

        FBGreaterNodeModule.Start(builder)
        FBGreaterNodeModule.AddA(builder, CreateTid(builder, op.a.idx))
        FBGreaterNodeModule.AddB(builder, CreateTid(builder, op.b.idx))
        FBGreaterNodeModule.AddOut(builder, CreateTid(builder, op.out.idx))
        offset = FBGreaterNodeModule.End(builder)
        return offset, FBOpNodeModule.OpNode.GreaterNode

    def _build_GreaterEqualNode(
        self, builder: flatbuffers.Builder, op: GreaterEqualNode
    ) -> Tuple[int, int]:
        """Auto-generated builder for GreaterEqualNode."""
        # Import the MODULE (not class) to access builder functions like Start(), Add*(), End()
        from executorch.backends.mlx.serialization._generated.mlx_delegate import GreaterEqualNode as FBGreaterEqualNodeModule
        from executorch.backends.mlx.serialization._generated.mlx_delegate import OpNode as FBOpNodeModule
        from executorch.backends.mlx.serialization._generated.mlx_delegate.Tid import CreateTid
        from executorch.backends.mlx.serialization._generated.mlx_delegate.Vid import CreateVid

        FBGreaterEqualNodeModule.Start(builder)
        FBGreaterEqualNodeModule.AddA(builder, CreateTid(builder, op.a.idx))
        FBGreaterEqualNodeModule.AddB(builder, CreateTid(builder, op.b.idx))
        FBGreaterEqualNodeModule.AddOut(builder, CreateTid(builder, op.out.idx))
        offset = FBGreaterEqualNodeModule.End(builder)
        return offset, FBOpNodeModule.OpNode.GreaterEqualNode

    def _build_EqualNode(
        self, builder: flatbuffers.Builder, op: EqualNode
    ) -> Tuple[int, int]:
        """Auto-generated builder for EqualNode."""
        # Import the MODULE (not class) to access builder functions like Start(), Add*(), End()
        from executorch.backends.mlx.serialization._generated.mlx_delegate import EqualNode as FBEqualNodeModule
        from executorch.backends.mlx.serialization._generated.mlx_delegate import OpNode as FBOpNodeModule
        from executorch.backends.mlx.serialization._generated.mlx_delegate.Tid import CreateTid
        from executorch.backends.mlx.serialization._generated.mlx_delegate.Vid import CreateVid

        FBEqualNodeModule.Start(builder)
        FBEqualNodeModule.AddA(builder, CreateTid(builder, op.a.idx))
        FBEqualNodeModule.AddB(builder, CreateTid(builder, op.b.idx))
        FBEqualNodeModule.AddOut(builder, CreateTid(builder, op.out.idx))
        offset = FBEqualNodeModule.End(builder)
        return offset, FBOpNodeModule.OpNode.EqualNode

    def _build_NotEqualNode(
        self, builder: flatbuffers.Builder, op: NotEqualNode
    ) -> Tuple[int, int]:
        """Auto-generated builder for NotEqualNode."""
        # Import the MODULE (not class) to access builder functions like Start(), Add*(), End()
        from executorch.backends.mlx.serialization._generated.mlx_delegate import NotEqualNode as FBNotEqualNodeModule
        from executorch.backends.mlx.serialization._generated.mlx_delegate import OpNode as FBOpNodeModule
        from executorch.backends.mlx.serialization._generated.mlx_delegate.Tid import CreateTid
        from executorch.backends.mlx.serialization._generated.mlx_delegate.Vid import CreateVid

        FBNotEqualNodeModule.Start(builder)
        FBNotEqualNodeModule.AddA(builder, CreateTid(builder, op.a.idx))
        FBNotEqualNodeModule.AddB(builder, CreateTid(builder, op.b.idx))
        FBNotEqualNodeModule.AddOut(builder, CreateTid(builder, op.out.idx))
        offset = FBNotEqualNodeModule.End(builder)
        return offset, FBOpNodeModule.OpNode.NotEqualNode

    def _build_LogicalNotNode(
        self, builder: flatbuffers.Builder, op: LogicalNotNode
    ) -> Tuple[int, int]:
        """Auto-generated builder for LogicalNotNode."""
        # Import the MODULE (not class) to access builder functions like Start(), Add*(), End()
        from executorch.backends.mlx.serialization._generated.mlx_delegate import LogicalNotNode as FBLogicalNotNodeModule
        from executorch.backends.mlx.serialization._generated.mlx_delegate import OpNode as FBOpNodeModule
        from executorch.backends.mlx.serialization._generated.mlx_delegate.Tid import CreateTid
        from executorch.backends.mlx.serialization._generated.mlx_delegate.Vid import CreateVid

        FBLogicalNotNodeModule.Start(builder)
        FBLogicalNotNodeModule.AddX(builder, CreateTid(builder, op.x.idx))
        FBLogicalNotNodeModule.AddOut(builder, CreateTid(builder, op.out.idx))
        offset = FBLogicalNotNodeModule.End(builder)
        return offset, FBOpNodeModule.OpNode.LogicalNotNode

    def _build_LogicalAndNode(
        self, builder: flatbuffers.Builder, op: LogicalAndNode
    ) -> Tuple[int, int]:
        """Auto-generated builder for LogicalAndNode."""
        # Import the MODULE (not class) to access builder functions like Start(), Add*(), End()
        from executorch.backends.mlx.serialization._generated.mlx_delegate import LogicalAndNode as FBLogicalAndNodeModule
        from executorch.backends.mlx.serialization._generated.mlx_delegate import OpNode as FBOpNodeModule
        from executorch.backends.mlx.serialization._generated.mlx_delegate.Tid import CreateTid
        from executorch.backends.mlx.serialization._generated.mlx_delegate.Vid import CreateVid

        FBLogicalAndNodeModule.Start(builder)
        FBLogicalAndNodeModule.AddA(builder, CreateTid(builder, op.a.idx))
        FBLogicalAndNodeModule.AddB(builder, CreateTid(builder, op.b.idx))
        FBLogicalAndNodeModule.AddOut(builder, CreateTid(builder, op.out.idx))
        offset = FBLogicalAndNodeModule.End(builder)
        return offset, FBOpNodeModule.OpNode.LogicalAndNode

    def _build_LogicalOrNode(
        self, builder: flatbuffers.Builder, op: LogicalOrNode
    ) -> Tuple[int, int]:
        """Auto-generated builder for LogicalOrNode."""
        # Import the MODULE (not class) to access builder functions like Start(), Add*(), End()
        from executorch.backends.mlx.serialization._generated.mlx_delegate import LogicalOrNode as FBLogicalOrNodeModule
        from executorch.backends.mlx.serialization._generated.mlx_delegate import OpNode as FBOpNodeModule
        from executorch.backends.mlx.serialization._generated.mlx_delegate.Tid import CreateTid
        from executorch.backends.mlx.serialization._generated.mlx_delegate.Vid import CreateVid

        FBLogicalOrNodeModule.Start(builder)
        FBLogicalOrNodeModule.AddA(builder, CreateTid(builder, op.a.idx))
        FBLogicalOrNodeModule.AddB(builder, CreateTid(builder, op.b.idx))
        FBLogicalOrNodeModule.AddOut(builder, CreateTid(builder, op.out.idx))
        offset = FBLogicalOrNodeModule.End(builder)
        return offset, FBOpNodeModule.OpNode.LogicalOrNode

    def _build_TriNode(
        self, builder: flatbuffers.Builder, op: TriNode
    ) -> Tuple[int, int]:
        """Auto-generated builder for TriNode."""
        # Import the MODULE (not class) to access builder functions like Start(), Add*(), End()
        from executorch.backends.mlx.serialization._generated.mlx_delegate import TriNode as FBTriNodeModule
        from executorch.backends.mlx.serialization._generated.mlx_delegate import OpNode as FBOpNodeModule
        from executorch.backends.mlx.serialization._generated.mlx_delegate.Tid import CreateTid
        from executorch.backends.mlx.serialization._generated.mlx_delegate.Vid import CreateVid

        n_off = self._build_int_or_vid(builder, op.n)
        m_off = self._build_int_or_vid(builder, op.m)

        FBTriNodeModule.Start(builder)
        FBTriNodeModule.AddOut(builder, CreateTid(builder, op.out.idx))
        FBTriNodeModule.AddN(builder, n_off)
        FBTriNodeModule.AddM(builder, m_off)
        FBTriNodeModule.AddK(builder, op.k)
        FBTriNodeModule.AddScalarType(builder, op.scalar_type)
        offset = FBTriNodeModule.End(builder)
        return offset, FBOpNodeModule.OpNode.TriNode

    def _build_TrilNode(
        self, builder: flatbuffers.Builder, op: TrilNode
    ) -> Tuple[int, int]:
        """Auto-generated builder for TrilNode."""
        # Import the MODULE (not class) to access builder functions like Start(), Add*(), End()
        from executorch.backends.mlx.serialization._generated.mlx_delegate import TrilNode as FBTrilNodeModule
        from executorch.backends.mlx.serialization._generated.mlx_delegate import OpNode as FBOpNodeModule
        from executorch.backends.mlx.serialization._generated.mlx_delegate.Tid import CreateTid
        from executorch.backends.mlx.serialization._generated.mlx_delegate.Vid import CreateVid

        FBTrilNodeModule.Start(builder)
        FBTrilNodeModule.AddX(builder, CreateTid(builder, op.x.idx))
        FBTrilNodeModule.AddOut(builder, CreateTid(builder, op.out.idx))
        FBTrilNodeModule.AddK(builder, op.k)
        offset = FBTrilNodeModule.End(builder)
        return offset, FBOpNodeModule.OpNode.TrilNode

    def _build_TriuNode(
        self, builder: flatbuffers.Builder, op: TriuNode
    ) -> Tuple[int, int]:
        """Auto-generated builder for TriuNode."""
        # Import the MODULE (not class) to access builder functions like Start(), Add*(), End()
        from executorch.backends.mlx.serialization._generated.mlx_delegate import TriuNode as FBTriuNodeModule
        from executorch.backends.mlx.serialization._generated.mlx_delegate import OpNode as FBOpNodeModule
        from executorch.backends.mlx.serialization._generated.mlx_delegate.Tid import CreateTid
        from executorch.backends.mlx.serialization._generated.mlx_delegate.Vid import CreateVid

        FBTriuNodeModule.Start(builder)
        FBTriuNodeModule.AddX(builder, CreateTid(builder, op.x.idx))
        FBTriuNodeModule.AddOut(builder, CreateTid(builder, op.out.idx))
        FBTriuNodeModule.AddK(builder, op.k)
        offset = FBTriuNodeModule.End(builder)
        return offset, FBOpNodeModule.OpNode.TriuNode

    def _build_ClipNode(
        self, builder: flatbuffers.Builder, op: ClipNode
    ) -> Tuple[int, int]:
        """Auto-generated builder for ClipNode."""
        # Import the MODULE (not class) to access builder functions like Start(), Add*(), End()
        from executorch.backends.mlx.serialization._generated.mlx_delegate import ClipNode as FBClipNodeModule
        from executorch.backends.mlx.serialization._generated.mlx_delegate import OpNode as FBOpNodeModule
        from executorch.backends.mlx.serialization._generated.mlx_delegate.Tid import CreateTid
        from executorch.backends.mlx.serialization._generated.mlx_delegate.Vid import CreateVid

        FBClipNodeModule.Start(builder)
        FBClipNodeModule.AddX(builder, CreateTid(builder, op.x.idx))
        FBClipNodeModule.AddOut(builder, CreateTid(builder, op.out.idx))
        if op.a_min is not None:
            FBClipNodeModule.AddAMin(builder, CreateTid(builder, op.a_min.idx))
        if op.a_max is not None:
            FBClipNodeModule.AddAMax(builder, CreateTid(builder, op.a_max.idx))
        offset = FBClipNodeModule.End(builder)
        return offset, FBOpNodeModule.OpNode.ClipNode

    def _build_CumsumNode(
        self, builder: flatbuffers.Builder, op: CumsumNode
    ) -> Tuple[int, int]:
        """Auto-generated builder for CumsumNode."""
        # Import the MODULE (not class) to access builder functions like Start(), Add*(), End()
        from executorch.backends.mlx.serialization._generated.mlx_delegate import CumsumNode as FBCumsumNodeModule
        from executorch.backends.mlx.serialization._generated.mlx_delegate import OpNode as FBOpNodeModule
        from executorch.backends.mlx.serialization._generated.mlx_delegate.Tid import CreateTid
        from executorch.backends.mlx.serialization._generated.mlx_delegate.Vid import CreateVid

        FBCumsumNodeModule.Start(builder)
        FBCumsumNodeModule.AddX(builder, CreateTid(builder, op.x.idx))
        FBCumsumNodeModule.AddOut(builder, CreateTid(builder, op.out.idx))
        FBCumsumNodeModule.AddAxis(builder, op.axis)
        FBCumsumNodeModule.AddReverse(builder, op.reverse)
        FBCumsumNodeModule.AddInclusive(builder, op.inclusive)
        offset = FBCumsumNodeModule.End(builder)
        return offset, FBOpNodeModule.OpNode.CumsumNode

    def _build_StackNode(
        self, builder: flatbuffers.Builder, op: StackNode
    ) -> Tuple[int, int]:
        """Auto-generated builder for StackNode."""
        # Import the MODULE (not class) to access builder functions like Start(), Add*(), End()
        from executorch.backends.mlx.serialization._generated.mlx_delegate import StackNode as FBStackNodeModule
        from executorch.backends.mlx.serialization._generated.mlx_delegate import OpNode as FBOpNodeModule
        from executorch.backends.mlx.serialization._generated.mlx_delegate.Tid import CreateTid
        from executorch.backends.mlx.serialization._generated.mlx_delegate.Vid import CreateVid

        tensors_vec = self._build_tid_vector(builder, op.tensors)

        FBStackNodeModule.Start(builder)
        FBStackNodeModule.AddTensors(builder, tensors_vec)
        FBStackNodeModule.AddOut(builder, CreateTid(builder, op.out.idx))
        FBStackNodeModule.AddAxis(builder, op.axis)
        offset = FBStackNodeModule.End(builder)
        return offset, FBOpNodeModule.OpNode.StackNode

    def _build_SignNode(
        self, builder: flatbuffers.Builder, op: SignNode
    ) -> Tuple[int, int]:
        """Auto-generated builder for SignNode."""
        # Import the MODULE (not class) to access builder functions like Start(), Add*(), End()
        from executorch.backends.mlx.serialization._generated.mlx_delegate import SignNode as FBSignNodeModule
        from executorch.backends.mlx.serialization._generated.mlx_delegate import OpNode as FBOpNodeModule
        from executorch.backends.mlx.serialization._generated.mlx_delegate.Tid import CreateTid
        from executorch.backends.mlx.serialization._generated.mlx_delegate.Vid import CreateVid

        FBSignNodeModule.Start(builder)
        FBSignNodeModule.AddX(builder, CreateTid(builder, op.x.idx))
        FBSignNodeModule.AddOut(builder, CreateTid(builder, op.out.idx))
        offset = FBSignNodeModule.End(builder)
        return offset, FBOpNodeModule.OpNode.SignNode

    def _build_AnyNode(
        self, builder: flatbuffers.Builder, op: AnyNode
    ) -> Tuple[int, int]:
        """Auto-generated builder for AnyNode."""
        # Import the MODULE (not class) to access builder functions like Start(), Add*(), End()
        from executorch.backends.mlx.serialization._generated.mlx_delegate import AnyNode as FBAnyNodeModule
        from executorch.backends.mlx.serialization._generated.mlx_delegate import OpNode as FBOpNodeModule
        from executorch.backends.mlx.serialization._generated.mlx_delegate.Tid import CreateTid
        from executorch.backends.mlx.serialization._generated.mlx_delegate.Vid import CreateVid

        axes_vec = _build_int_vector(builder, op.axes) if op.axes is not None else None

        FBAnyNodeModule.Start(builder)
        FBAnyNodeModule.AddX(builder, CreateTid(builder, op.x.idx))
        FBAnyNodeModule.AddOut(builder, CreateTid(builder, op.out.idx))
        if axes_vec is not None:
            FBAnyNodeModule.AddAxes(builder, axes_vec)
        FBAnyNodeModule.AddKeepdims(builder, op.keepdims)
        offset = FBAnyNodeModule.End(builder)
        return offset, FBOpNodeModule.OpNode.AnyNode

    def _build_AllNode(
        self, builder: flatbuffers.Builder, op: AllNode
    ) -> Tuple[int, int]:
        """Auto-generated builder for AllNode."""
        # Import the MODULE (not class) to access builder functions like Start(), Add*(), End()
        from executorch.backends.mlx.serialization._generated.mlx_delegate import AllNode as FBAllNodeModule
        from executorch.backends.mlx.serialization._generated.mlx_delegate import OpNode as FBOpNodeModule
        from executorch.backends.mlx.serialization._generated.mlx_delegate.Tid import CreateTid
        from executorch.backends.mlx.serialization._generated.mlx_delegate.Vid import CreateVid

        axes_vec = _build_int_vector(builder, op.axes) if op.axes is not None else None

        FBAllNodeModule.Start(builder)
        FBAllNodeModule.AddX(builder, CreateTid(builder, op.x.idx))
        FBAllNodeModule.AddOut(builder, CreateTid(builder, op.out.idx))
        if axes_vec is not None:
            FBAllNodeModule.AddAxes(builder, axes_vec)
        FBAllNodeModule.AddKeepdims(builder, op.keepdims)
        offset = FBAllNodeModule.End(builder)
        return offset, FBOpNodeModule.OpNode.AllNode

    def _build_RepeatNode(
        self, builder: flatbuffers.Builder, op: RepeatNode
    ) -> Tuple[int, int]:
        """Auto-generated builder for RepeatNode."""
        # Import the MODULE (not class) to access builder functions like Start(), Add*(), End()
        from executorch.backends.mlx.serialization._generated.mlx_delegate import RepeatNode as FBRepeatNodeModule
        from executorch.backends.mlx.serialization._generated.mlx_delegate import OpNode as FBOpNodeModule
        from executorch.backends.mlx.serialization._generated.mlx_delegate.Tid import CreateTid
        from executorch.backends.mlx.serialization._generated.mlx_delegate.Vid import CreateVid

        repeats_off = self._build_int_or_vid(builder, op.repeats)

        FBRepeatNodeModule.Start(builder)
        FBRepeatNodeModule.AddX(builder, CreateTid(builder, op.x.idx))
        FBRepeatNodeModule.AddOut(builder, CreateTid(builder, op.out.idx))
        FBRepeatNodeModule.AddRepeats(builder, repeats_off)
        FBRepeatNodeModule.AddAxis(builder, op.axis)
        offset = FBRepeatNodeModule.End(builder)
        return offset, FBOpNodeModule.OpNode.RepeatNode

    def _build_SortNode(
        self, builder: flatbuffers.Builder, op: SortNode
    ) -> Tuple[int, int]:
        """Auto-generated builder for SortNode."""
        # Import the MODULE (not class) to access builder functions like Start(), Add*(), End()
        from executorch.backends.mlx.serialization._generated.mlx_delegate import SortNode as FBSortNodeModule
        from executorch.backends.mlx.serialization._generated.mlx_delegate import OpNode as FBOpNodeModule
        from executorch.backends.mlx.serialization._generated.mlx_delegate.Tid import CreateTid
        from executorch.backends.mlx.serialization._generated.mlx_delegate.Vid import CreateVid

        FBSortNodeModule.Start(builder)
        FBSortNodeModule.AddX(builder, CreateTid(builder, op.x.idx))
        FBSortNodeModule.AddOut(builder, CreateTid(builder, op.out.idx))
        FBSortNodeModule.AddAxis(builder, op.axis)
        offset = FBSortNodeModule.End(builder)
        return offset, FBOpNodeModule.OpNode.SortNode

    def _build_ArgsortNode(
        self, builder: flatbuffers.Builder, op: ArgsortNode
    ) -> Tuple[int, int]:
        """Auto-generated builder for ArgsortNode."""
        # Import the MODULE (not class) to access builder functions like Start(), Add*(), End()
        from executorch.backends.mlx.serialization._generated.mlx_delegate import ArgsortNode as FBArgsortNodeModule
        from executorch.backends.mlx.serialization._generated.mlx_delegate import OpNode as FBOpNodeModule
        from executorch.backends.mlx.serialization._generated.mlx_delegate.Tid import CreateTid
        from executorch.backends.mlx.serialization._generated.mlx_delegate.Vid import CreateVid

        FBArgsortNodeModule.Start(builder)
        FBArgsortNodeModule.AddX(builder, CreateTid(builder, op.x.idx))
        FBArgsortNodeModule.AddOut(builder, CreateTid(builder, op.out.idx))
        FBArgsortNodeModule.AddAxis(builder, op.axis)
        offset = FBArgsortNodeModule.End(builder)
        return offset, FBOpNodeModule.OpNode.ArgsortNode

    def _build_PartitionNode(
        self, builder: flatbuffers.Builder, op: PartitionNode
    ) -> Tuple[int, int]:
        """Auto-generated builder for PartitionNode."""
        # Import the MODULE (not class) to access builder functions like Start(), Add*(), End()
        from executorch.backends.mlx.serialization._generated.mlx_delegate import PartitionNode as FBPartitionNodeModule
        from executorch.backends.mlx.serialization._generated.mlx_delegate import OpNode as FBOpNodeModule
        from executorch.backends.mlx.serialization._generated.mlx_delegate.Tid import CreateTid
        from executorch.backends.mlx.serialization._generated.mlx_delegate.Vid import CreateVid

        kth_off = self._build_int_or_vid(builder, op.kth)

        FBPartitionNodeModule.Start(builder)
        FBPartitionNodeModule.AddX(builder, CreateTid(builder, op.x.idx))
        FBPartitionNodeModule.AddOut(builder, CreateTid(builder, op.out.idx))
        FBPartitionNodeModule.AddKth(builder, kth_off)
        FBPartitionNodeModule.AddAxis(builder, op.axis)
        offset = FBPartitionNodeModule.End(builder)
        return offset, FBOpNodeModule.OpNode.PartitionNode

    def _build_ArgPartitionNode(
        self, builder: flatbuffers.Builder, op: ArgPartitionNode
    ) -> Tuple[int, int]:
        """Auto-generated builder for ArgPartitionNode."""
        # Import the MODULE (not class) to access builder functions like Start(), Add*(), End()
        from executorch.backends.mlx.serialization._generated.mlx_delegate import ArgPartitionNode as FBArgPartitionNodeModule
        from executorch.backends.mlx.serialization._generated.mlx_delegate import OpNode as FBOpNodeModule
        from executorch.backends.mlx.serialization._generated.mlx_delegate.Tid import CreateTid
        from executorch.backends.mlx.serialization._generated.mlx_delegate.Vid import CreateVid

        kth_off = self._build_int_or_vid(builder, op.kth)

        FBArgPartitionNodeModule.Start(builder)
        FBArgPartitionNodeModule.AddX(builder, CreateTid(builder, op.x.idx))
        FBArgPartitionNodeModule.AddOut(builder, CreateTid(builder, op.out.idx))
        FBArgPartitionNodeModule.AddKth(builder, kth_off)
        FBArgPartitionNodeModule.AddAxis(builder, op.axis)
        offset = FBArgPartitionNodeModule.End(builder)
        return offset, FBOpNodeModule.OpNode.ArgPartitionNode

    def _build_FloorNode(
        self, builder: flatbuffers.Builder, op: FloorNode
    ) -> Tuple[int, int]:
        """Auto-generated builder for FloorNode."""
        # Import the MODULE (not class) to access builder functions like Start(), Add*(), End()
        from executorch.backends.mlx.serialization._generated.mlx_delegate import FloorNode as FBFloorNodeModule
        from executorch.backends.mlx.serialization._generated.mlx_delegate import OpNode as FBOpNodeModule
        from executorch.backends.mlx.serialization._generated.mlx_delegate.Tid import CreateTid
        from executorch.backends.mlx.serialization._generated.mlx_delegate.Vid import CreateVid

        FBFloorNodeModule.Start(builder)
        FBFloorNodeModule.AddX(builder, CreateTid(builder, op.x.idx))
        FBFloorNodeModule.AddOut(builder, CreateTid(builder, op.out.idx))
        offset = FBFloorNodeModule.End(builder)
        return offset, FBOpNodeModule.OpNode.FloorNode

    def _build_CeilNode(
        self, builder: flatbuffers.Builder, op: CeilNode
    ) -> Tuple[int, int]:
        """Auto-generated builder for CeilNode."""
        # Import the MODULE (not class) to access builder functions like Start(), Add*(), End()
        from executorch.backends.mlx.serialization._generated.mlx_delegate import CeilNode as FBCeilNodeModule
        from executorch.backends.mlx.serialization._generated.mlx_delegate import OpNode as FBOpNodeModule
        from executorch.backends.mlx.serialization._generated.mlx_delegate.Tid import CreateTid
        from executorch.backends.mlx.serialization._generated.mlx_delegate.Vid import CreateVid

        FBCeilNodeModule.Start(builder)
        FBCeilNodeModule.AddX(builder, CreateTid(builder, op.x.idx))
        FBCeilNodeModule.AddOut(builder, CreateTid(builder, op.out.idx))
        offset = FBCeilNodeModule.End(builder)
        return offset, FBOpNodeModule.OpNode.CeilNode

    def _build_SquareNode(
        self, builder: flatbuffers.Builder, op: SquareNode
    ) -> Tuple[int, int]:
        """Auto-generated builder for SquareNode."""
        # Import the MODULE (not class) to access builder functions like Start(), Add*(), End()
        from executorch.backends.mlx.serialization._generated.mlx_delegate import SquareNode as FBSquareNodeModule
        from executorch.backends.mlx.serialization._generated.mlx_delegate import OpNode as FBOpNodeModule
        from executorch.backends.mlx.serialization._generated.mlx_delegate.Tid import CreateTid
        from executorch.backends.mlx.serialization._generated.mlx_delegate.Vid import CreateVid

        FBSquareNodeModule.Start(builder)
        FBSquareNodeModule.AddX(builder, CreateTid(builder, op.x.idx))
        FBSquareNodeModule.AddOut(builder, CreateTid(builder, op.out.idx))
        offset = FBSquareNodeModule.End(builder)
        return offset, FBOpNodeModule.OpNode.SquareNode

    def _build_ExpNode(
        self, builder: flatbuffers.Builder, op: ExpNode
    ) -> Tuple[int, int]:
        """Auto-generated builder for ExpNode."""
        # Import the MODULE (not class) to access builder functions like Start(), Add*(), End()
        from executorch.backends.mlx.serialization._generated.mlx_delegate import ExpNode as FBExpNodeModule
        from executorch.backends.mlx.serialization._generated.mlx_delegate import OpNode as FBOpNodeModule
        from executorch.backends.mlx.serialization._generated.mlx_delegate.Tid import CreateTid
        from executorch.backends.mlx.serialization._generated.mlx_delegate.Vid import CreateVid

        FBExpNodeModule.Start(builder)
        FBExpNodeModule.AddX(builder, CreateTid(builder, op.x.idx))
        FBExpNodeModule.AddOut(builder, CreateTid(builder, op.out.idx))
        offset = FBExpNodeModule.End(builder)
        return offset, FBOpNodeModule.OpNode.ExpNode

    def _build_SinNode(
        self, builder: flatbuffers.Builder, op: SinNode
    ) -> Tuple[int, int]:
        """Auto-generated builder for SinNode."""
        # Import the MODULE (not class) to access builder functions like Start(), Add*(), End()
        from executorch.backends.mlx.serialization._generated.mlx_delegate import SinNode as FBSinNodeModule
        from executorch.backends.mlx.serialization._generated.mlx_delegate import OpNode as FBOpNodeModule
        from executorch.backends.mlx.serialization._generated.mlx_delegate.Tid import CreateTid
        from executorch.backends.mlx.serialization._generated.mlx_delegate.Vid import CreateVid

        FBSinNodeModule.Start(builder)
        FBSinNodeModule.AddX(builder, CreateTid(builder, op.x.idx))
        FBSinNodeModule.AddOut(builder, CreateTid(builder, op.out.idx))
        offset = FBSinNodeModule.End(builder)
        return offset, FBOpNodeModule.OpNode.SinNode

    def _build_CosNode(
        self, builder: flatbuffers.Builder, op: CosNode
    ) -> Tuple[int, int]:
        """Auto-generated builder for CosNode."""
        # Import the MODULE (not class) to access builder functions like Start(), Add*(), End()
        from executorch.backends.mlx.serialization._generated.mlx_delegate import CosNode as FBCosNodeModule
        from executorch.backends.mlx.serialization._generated.mlx_delegate import OpNode as FBOpNodeModule
        from executorch.backends.mlx.serialization._generated.mlx_delegate.Tid import CreateTid
        from executorch.backends.mlx.serialization._generated.mlx_delegate.Vid import CreateVid

        FBCosNodeModule.Start(builder)
        FBCosNodeModule.AddX(builder, CreateTid(builder, op.x.idx))
        FBCosNodeModule.AddOut(builder, CreateTid(builder, op.out.idx))
        offset = FBCosNodeModule.End(builder)
        return offset, FBOpNodeModule.OpNode.CosNode

    def _build_TanNode(
        self, builder: flatbuffers.Builder, op: TanNode
    ) -> Tuple[int, int]:
        """Auto-generated builder for TanNode."""
        # Import the MODULE (not class) to access builder functions like Start(), Add*(), End()
        from executorch.backends.mlx.serialization._generated.mlx_delegate import TanNode as FBTanNodeModule
        from executorch.backends.mlx.serialization._generated.mlx_delegate import OpNode as FBOpNodeModule
        from executorch.backends.mlx.serialization._generated.mlx_delegate.Tid import CreateTid
        from executorch.backends.mlx.serialization._generated.mlx_delegate.Vid import CreateVid

        FBTanNodeModule.Start(builder)
        FBTanNodeModule.AddX(builder, CreateTid(builder, op.x.idx))
        FBTanNodeModule.AddOut(builder, CreateTid(builder, op.out.idx))
        offset = FBTanNodeModule.End(builder)
        return offset, FBOpNodeModule.OpNode.TanNode

    def _build_ArcsinNode(
        self, builder: flatbuffers.Builder, op: ArcsinNode
    ) -> Tuple[int, int]:
        """Auto-generated builder for ArcsinNode."""
        # Import the MODULE (not class) to access builder functions like Start(), Add*(), End()
        from executorch.backends.mlx.serialization._generated.mlx_delegate import ArcsinNode as FBArcsinNodeModule
        from executorch.backends.mlx.serialization._generated.mlx_delegate import OpNode as FBOpNodeModule
        from executorch.backends.mlx.serialization._generated.mlx_delegate.Tid import CreateTid
        from executorch.backends.mlx.serialization._generated.mlx_delegate.Vid import CreateVid

        FBArcsinNodeModule.Start(builder)
        FBArcsinNodeModule.AddX(builder, CreateTid(builder, op.x.idx))
        FBArcsinNodeModule.AddOut(builder, CreateTid(builder, op.out.idx))
        offset = FBArcsinNodeModule.End(builder)
        return offset, FBOpNodeModule.OpNode.ArcsinNode

    def _build_ArccosNode(
        self, builder: flatbuffers.Builder, op: ArccosNode
    ) -> Tuple[int, int]:
        """Auto-generated builder for ArccosNode."""
        # Import the MODULE (not class) to access builder functions like Start(), Add*(), End()
        from executorch.backends.mlx.serialization._generated.mlx_delegate import ArccosNode as FBArccosNodeModule
        from executorch.backends.mlx.serialization._generated.mlx_delegate import OpNode as FBOpNodeModule
        from executorch.backends.mlx.serialization._generated.mlx_delegate.Tid import CreateTid
        from executorch.backends.mlx.serialization._generated.mlx_delegate.Vid import CreateVid

        FBArccosNodeModule.Start(builder)
        FBArccosNodeModule.AddX(builder, CreateTid(builder, op.x.idx))
        FBArccosNodeModule.AddOut(builder, CreateTid(builder, op.out.idx))
        offset = FBArccosNodeModule.End(builder)
        return offset, FBOpNodeModule.OpNode.ArccosNode

    def _build_ArctanNode(
        self, builder: flatbuffers.Builder, op: ArctanNode
    ) -> Tuple[int, int]:
        """Auto-generated builder for ArctanNode."""
        # Import the MODULE (not class) to access builder functions like Start(), Add*(), End()
        from executorch.backends.mlx.serialization._generated.mlx_delegate import ArctanNode as FBArctanNodeModule
        from executorch.backends.mlx.serialization._generated.mlx_delegate import OpNode as FBOpNodeModule
        from executorch.backends.mlx.serialization._generated.mlx_delegate.Tid import CreateTid
        from executorch.backends.mlx.serialization._generated.mlx_delegate.Vid import CreateVid

        FBArctanNodeModule.Start(builder)
        FBArctanNodeModule.AddX(builder, CreateTid(builder, op.x.idx))
        FBArctanNodeModule.AddOut(builder, CreateTid(builder, op.out.idx))
        offset = FBArctanNodeModule.End(builder)
        return offset, FBOpNodeModule.OpNode.ArctanNode

    def _build_SinhNode(
        self, builder: flatbuffers.Builder, op: SinhNode
    ) -> Tuple[int, int]:
        """Auto-generated builder for SinhNode."""
        # Import the MODULE (not class) to access builder functions like Start(), Add*(), End()
        from executorch.backends.mlx.serialization._generated.mlx_delegate import SinhNode as FBSinhNodeModule
        from executorch.backends.mlx.serialization._generated.mlx_delegate import OpNode as FBOpNodeModule
        from executorch.backends.mlx.serialization._generated.mlx_delegate.Tid import CreateTid
        from executorch.backends.mlx.serialization._generated.mlx_delegate.Vid import CreateVid

        FBSinhNodeModule.Start(builder)
        FBSinhNodeModule.AddX(builder, CreateTid(builder, op.x.idx))
        FBSinhNodeModule.AddOut(builder, CreateTid(builder, op.out.idx))
        offset = FBSinhNodeModule.End(builder)
        return offset, FBOpNodeModule.OpNode.SinhNode

    def _build_CoshNode(
        self, builder: flatbuffers.Builder, op: CoshNode
    ) -> Tuple[int, int]:
        """Auto-generated builder for CoshNode."""
        # Import the MODULE (not class) to access builder functions like Start(), Add*(), End()
        from executorch.backends.mlx.serialization._generated.mlx_delegate import CoshNode as FBCoshNodeModule
        from executorch.backends.mlx.serialization._generated.mlx_delegate import OpNode as FBOpNodeModule
        from executorch.backends.mlx.serialization._generated.mlx_delegate.Tid import CreateTid
        from executorch.backends.mlx.serialization._generated.mlx_delegate.Vid import CreateVid

        FBCoshNodeModule.Start(builder)
        FBCoshNodeModule.AddX(builder, CreateTid(builder, op.x.idx))
        FBCoshNodeModule.AddOut(builder, CreateTid(builder, op.out.idx))
        offset = FBCoshNodeModule.End(builder)
        return offset, FBOpNodeModule.OpNode.CoshNode

    def _build_ArcsinhNode(
        self, builder: flatbuffers.Builder, op: ArcsinhNode
    ) -> Tuple[int, int]:
        """Auto-generated builder for ArcsinhNode."""
        # Import the MODULE (not class) to access builder functions like Start(), Add*(), End()
        from executorch.backends.mlx.serialization._generated.mlx_delegate import ArcsinhNode as FBArcsinhNodeModule
        from executorch.backends.mlx.serialization._generated.mlx_delegate import OpNode as FBOpNodeModule
        from executorch.backends.mlx.serialization._generated.mlx_delegate.Tid import CreateTid
        from executorch.backends.mlx.serialization._generated.mlx_delegate.Vid import CreateVid

        FBArcsinhNodeModule.Start(builder)
        FBArcsinhNodeModule.AddX(builder, CreateTid(builder, op.x.idx))
        FBArcsinhNodeModule.AddOut(builder, CreateTid(builder, op.out.idx))
        offset = FBArcsinhNodeModule.End(builder)
        return offset, FBOpNodeModule.OpNode.ArcsinhNode

    def _build_ArccoshNode(
        self, builder: flatbuffers.Builder, op: ArccoshNode
    ) -> Tuple[int, int]:
        """Auto-generated builder for ArccoshNode."""
        # Import the MODULE (not class) to access builder functions like Start(), Add*(), End()
        from executorch.backends.mlx.serialization._generated.mlx_delegate import ArccoshNode as FBArccoshNodeModule
        from executorch.backends.mlx.serialization._generated.mlx_delegate import OpNode as FBOpNodeModule
        from executorch.backends.mlx.serialization._generated.mlx_delegate.Tid import CreateTid
        from executorch.backends.mlx.serialization._generated.mlx_delegate.Vid import CreateVid

        FBArccoshNodeModule.Start(builder)
        FBArccoshNodeModule.AddX(builder, CreateTid(builder, op.x.idx))
        FBArccoshNodeModule.AddOut(builder, CreateTid(builder, op.out.idx))
        offset = FBArccoshNodeModule.End(builder)
        return offset, FBOpNodeModule.OpNode.ArccoshNode

    def _build_ArctanhNode(
        self, builder: flatbuffers.Builder, op: ArctanhNode
    ) -> Tuple[int, int]:
        """Auto-generated builder for ArctanhNode."""
        # Import the MODULE (not class) to access builder functions like Start(), Add*(), End()
        from executorch.backends.mlx.serialization._generated.mlx_delegate import ArctanhNode as FBArctanhNodeModule
        from executorch.backends.mlx.serialization._generated.mlx_delegate import OpNode as FBOpNodeModule
        from executorch.backends.mlx.serialization._generated.mlx_delegate.Tid import CreateTid
        from executorch.backends.mlx.serialization._generated.mlx_delegate.Vid import CreateVid

        FBArctanhNodeModule.Start(builder)
        FBArctanhNodeModule.AddX(builder, CreateTid(builder, op.x.idx))
        FBArctanhNodeModule.AddOut(builder, CreateTid(builder, op.out.idx))
        offset = FBArctanhNodeModule.End(builder)
        return offset, FBOpNodeModule.OpNode.ArctanhNode

    def _build_Log2Node(
        self, builder: flatbuffers.Builder, op: Log2Node
    ) -> Tuple[int, int]:
        """Auto-generated builder for Log2Node."""
        # Import the MODULE (not class) to access builder functions like Start(), Add*(), End()
        from executorch.backends.mlx.serialization._generated.mlx_delegate import Log2Node as FBLog2NodeModule
        from executorch.backends.mlx.serialization._generated.mlx_delegate import OpNode as FBOpNodeModule
        from executorch.backends.mlx.serialization._generated.mlx_delegate.Tid import CreateTid
        from executorch.backends.mlx.serialization._generated.mlx_delegate.Vid import CreateVid

        FBLog2NodeModule.Start(builder)
        FBLog2NodeModule.AddX(builder, CreateTid(builder, op.x.idx))
        FBLog2NodeModule.AddOut(builder, CreateTid(builder, op.out.idx))
        offset = FBLog2NodeModule.End(builder)
        return offset, FBOpNodeModule.OpNode.Log2Node

    def _build_Log10Node(
        self, builder: flatbuffers.Builder, op: Log10Node
    ) -> Tuple[int, int]:
        """Auto-generated builder for Log10Node."""
        # Import the MODULE (not class) to access builder functions like Start(), Add*(), End()
        from executorch.backends.mlx.serialization._generated.mlx_delegate import Log10Node as FBLog10NodeModule
        from executorch.backends.mlx.serialization._generated.mlx_delegate import OpNode as FBOpNodeModule
        from executorch.backends.mlx.serialization._generated.mlx_delegate.Tid import CreateTid
        from executorch.backends.mlx.serialization._generated.mlx_delegate.Vid import CreateVid

        FBLog10NodeModule.Start(builder)
        FBLog10NodeModule.AddX(builder, CreateTid(builder, op.x.idx))
        FBLog10NodeModule.AddOut(builder, CreateTid(builder, op.out.idx))
        offset = FBLog10NodeModule.End(builder)
        return offset, FBOpNodeModule.OpNode.Log10Node

    def _build_Log1pNode(
        self, builder: flatbuffers.Builder, op: Log1pNode
    ) -> Tuple[int, int]:
        """Auto-generated builder for Log1pNode."""
        # Import the MODULE (not class) to access builder functions like Start(), Add*(), End()
        from executorch.backends.mlx.serialization._generated.mlx_delegate import Log1pNode as FBLog1pNodeModule
        from executorch.backends.mlx.serialization._generated.mlx_delegate import OpNode as FBOpNodeModule
        from executorch.backends.mlx.serialization._generated.mlx_delegate.Tid import CreateTid
        from executorch.backends.mlx.serialization._generated.mlx_delegate.Vid import CreateVid

        FBLog1pNodeModule.Start(builder)
        FBLog1pNodeModule.AddX(builder, CreateTid(builder, op.x.idx))
        FBLog1pNodeModule.AddOut(builder, CreateTid(builder, op.out.idx))
        offset = FBLog1pNodeModule.End(builder)
        return offset, FBOpNodeModule.OpNode.Log1pNode

    def _build_ErfNode(
        self, builder: flatbuffers.Builder, op: ErfNode
    ) -> Tuple[int, int]:
        """Auto-generated builder for ErfNode."""
        # Import the MODULE (not class) to access builder functions like Start(), Add*(), End()
        from executorch.backends.mlx.serialization._generated.mlx_delegate import ErfNode as FBErfNodeModule
        from executorch.backends.mlx.serialization._generated.mlx_delegate import OpNode as FBOpNodeModule
        from executorch.backends.mlx.serialization._generated.mlx_delegate.Tid import CreateTid
        from executorch.backends.mlx.serialization._generated.mlx_delegate.Vid import CreateVid

        FBErfNodeModule.Start(builder)
        FBErfNodeModule.AddX(builder, CreateTid(builder, op.x.idx))
        FBErfNodeModule.AddOut(builder, CreateTid(builder, op.out.idx))
        offset = FBErfNodeModule.End(builder)
        return offset, FBOpNodeModule.OpNode.ErfNode

    def _build_Expm1Node(
        self, builder: flatbuffers.Builder, op: Expm1Node
    ) -> Tuple[int, int]:
        """Auto-generated builder for Expm1Node."""
        # Import the MODULE (not class) to access builder functions like Start(), Add*(), End()
        from executorch.backends.mlx.serialization._generated.mlx_delegate import Expm1Node as FBExpm1NodeModule
        from executorch.backends.mlx.serialization._generated.mlx_delegate import OpNode as FBOpNodeModule
        from executorch.backends.mlx.serialization._generated.mlx_delegate.Tid import CreateTid
        from executorch.backends.mlx.serialization._generated.mlx_delegate.Vid import CreateVid

        FBExpm1NodeModule.Start(builder)
        FBExpm1NodeModule.AddX(builder, CreateTid(builder, op.x.idx))
        FBExpm1NodeModule.AddOut(builder, CreateTid(builder, op.out.idx))
        offset = FBExpm1NodeModule.End(builder)
        return offset, FBOpNodeModule.OpNode.Expm1Node

    def _build_RoundNode(
        self, builder: flatbuffers.Builder, op: RoundNode
    ) -> Tuple[int, int]:
        """Auto-generated builder for RoundNode."""
        # Import the MODULE (not class) to access builder functions like Start(), Add*(), End()
        from executorch.backends.mlx.serialization._generated.mlx_delegate import RoundNode as FBRoundNodeModule
        from executorch.backends.mlx.serialization._generated.mlx_delegate import OpNode as FBOpNodeModule
        from executorch.backends.mlx.serialization._generated.mlx_delegate.Tid import CreateTid
        from executorch.backends.mlx.serialization._generated.mlx_delegate.Vid import CreateVid

        FBRoundNodeModule.Start(builder)
        FBRoundNodeModule.AddX(builder, CreateTid(builder, op.x.idx))
        FBRoundNodeModule.AddOut(builder, CreateTid(builder, op.out.idx))
        FBRoundNodeModule.AddDecimals(builder, op.decimals)
        offset = FBRoundNodeModule.End(builder)
        return offset, FBOpNodeModule.OpNode.RoundNode

    def _build_ReciprocalNode(
        self, builder: flatbuffers.Builder, op: ReciprocalNode
    ) -> Tuple[int, int]:
        """Auto-generated builder for ReciprocalNode."""
        # Import the MODULE (not class) to access builder functions like Start(), Add*(), End()
        from executorch.backends.mlx.serialization._generated.mlx_delegate import ReciprocalNode as FBReciprocalNodeModule
        from executorch.backends.mlx.serialization._generated.mlx_delegate import OpNode as FBOpNodeModule
        from executorch.backends.mlx.serialization._generated.mlx_delegate.Tid import CreateTid
        from executorch.backends.mlx.serialization._generated.mlx_delegate.Vid import CreateVid

        FBReciprocalNodeModule.Start(builder)
        FBReciprocalNodeModule.AddX(builder, CreateTid(builder, op.x.idx))
        FBReciprocalNodeModule.AddOut(builder, CreateTid(builder, op.out.idx))
        offset = FBReciprocalNodeModule.End(builder)
        return offset, FBOpNodeModule.OpNode.ReciprocalNode

    def _build_SqrtNode(
        self, builder: flatbuffers.Builder, op: SqrtNode
    ) -> Tuple[int, int]:
        """Auto-generated builder for SqrtNode."""
        # Import the MODULE (not class) to access builder functions like Start(), Add*(), End()
        from executorch.backends.mlx.serialization._generated.mlx_delegate import SqrtNode as FBSqrtNodeModule
        from executorch.backends.mlx.serialization._generated.mlx_delegate import OpNode as FBOpNodeModule
        from executorch.backends.mlx.serialization._generated.mlx_delegate.Tid import CreateTid
        from executorch.backends.mlx.serialization._generated.mlx_delegate.Vid import CreateVid

        FBSqrtNodeModule.Start(builder)
        FBSqrtNodeModule.AddX(builder, CreateTid(builder, op.x.idx))
        FBSqrtNodeModule.AddOut(builder, CreateTid(builder, op.out.idx))
        offset = FBSqrtNodeModule.End(builder)
        return offset, FBOpNodeModule.OpNode.SqrtNode

    def _build_AbsNode(
        self, builder: flatbuffers.Builder, op: AbsNode
    ) -> Tuple[int, int]:
        """Auto-generated builder for AbsNode."""
        # Import the MODULE (not class) to access builder functions like Start(), Add*(), End()
        from executorch.backends.mlx.serialization._generated.mlx_delegate import AbsNode as FBAbsNodeModule
        from executorch.backends.mlx.serialization._generated.mlx_delegate import OpNode as FBOpNodeModule
        from executorch.backends.mlx.serialization._generated.mlx_delegate.Tid import CreateTid
        from executorch.backends.mlx.serialization._generated.mlx_delegate.Vid import CreateVid

        FBAbsNodeModule.Start(builder)
        FBAbsNodeModule.AddX(builder, CreateTid(builder, op.x.idx))
        FBAbsNodeModule.AddOut(builder, CreateTid(builder, op.out.idx))
        offset = FBAbsNodeModule.End(builder)
        return offset, FBOpNodeModule.OpNode.AbsNode

    def _build_NegNode(
        self, builder: flatbuffers.Builder, op: NegNode
    ) -> Tuple[int, int]:
        """Auto-generated builder for NegNode."""
        # Import the MODULE (not class) to access builder functions like Start(), Add*(), End()
        from executorch.backends.mlx.serialization._generated.mlx_delegate import NegNode as FBNegNodeModule
        from executorch.backends.mlx.serialization._generated.mlx_delegate import OpNode as FBOpNodeModule
        from executorch.backends.mlx.serialization._generated.mlx_delegate.Tid import CreateTid
        from executorch.backends.mlx.serialization._generated.mlx_delegate.Vid import CreateVid

        FBNegNodeModule.Start(builder)
        FBNegNodeModule.AddX(builder, CreateTid(builder, op.x.idx))
        FBNegNodeModule.AddOut(builder, CreateTid(builder, op.out.idx))
        offset = FBNegNodeModule.End(builder)
        return offset, FBOpNodeModule.OpNode.NegNode

    def _build_Atan2Node(
        self, builder: flatbuffers.Builder, op: Atan2Node
    ) -> Tuple[int, int]:
        """Auto-generated builder for Atan2Node."""
        # Import the MODULE (not class) to access builder functions like Start(), Add*(), End()
        from executorch.backends.mlx.serialization._generated.mlx_delegate import Atan2Node as FBAtan2NodeModule
        from executorch.backends.mlx.serialization._generated.mlx_delegate import OpNode as FBOpNodeModule
        from executorch.backends.mlx.serialization._generated.mlx_delegate.Tid import CreateTid
        from executorch.backends.mlx.serialization._generated.mlx_delegate.Vid import CreateVid

        FBAtan2NodeModule.Start(builder)
        FBAtan2NodeModule.AddA(builder, CreateTid(builder, op.a.idx))
        FBAtan2NodeModule.AddB(builder, CreateTid(builder, op.b.idx))
        FBAtan2NodeModule.AddOut(builder, CreateTid(builder, op.out.idx))
        offset = FBAtan2NodeModule.End(builder)
        return offset, FBOpNodeModule.OpNode.Atan2Node

    def _build_LogAddExpNode(
        self, builder: flatbuffers.Builder, op: LogAddExpNode
    ) -> Tuple[int, int]:
        """Auto-generated builder for LogAddExpNode."""
        # Import the MODULE (not class) to access builder functions like Start(), Add*(), End()
        from executorch.backends.mlx.serialization._generated.mlx_delegate import LogAddExpNode as FBLogAddExpNodeModule
        from executorch.backends.mlx.serialization._generated.mlx_delegate import OpNode as FBOpNodeModule
        from executorch.backends.mlx.serialization._generated.mlx_delegate.Tid import CreateTid
        from executorch.backends.mlx.serialization._generated.mlx_delegate.Vid import CreateVid

        FBLogAddExpNodeModule.Start(builder)
        FBLogAddExpNodeModule.AddA(builder, CreateTid(builder, op.a.idx))
        FBLogAddExpNodeModule.AddB(builder, CreateTid(builder, op.b.idx))
        FBLogAddExpNodeModule.AddOut(builder, CreateTid(builder, op.out.idx))
        offset = FBLogAddExpNodeModule.End(builder)
        return offset, FBOpNodeModule.OpNode.LogAddExpNode

    def _build_FloorDivideNode(
        self, builder: flatbuffers.Builder, op: FloorDivideNode
    ) -> Tuple[int, int]:
        """Auto-generated builder for FloorDivideNode."""
        # Import the MODULE (not class) to access builder functions like Start(), Add*(), End()
        from executorch.backends.mlx.serialization._generated.mlx_delegate import FloorDivideNode as FBFloorDivideNodeModule
        from executorch.backends.mlx.serialization._generated.mlx_delegate import OpNode as FBOpNodeModule
        from executorch.backends.mlx.serialization._generated.mlx_delegate.Tid import CreateTid
        from executorch.backends.mlx.serialization._generated.mlx_delegate.Vid import CreateVid

        FBFloorDivideNodeModule.Start(builder)
        FBFloorDivideNodeModule.AddA(builder, CreateTid(builder, op.a.idx))
        FBFloorDivideNodeModule.AddB(builder, CreateTid(builder, op.b.idx))
        FBFloorDivideNodeModule.AddOut(builder, CreateTid(builder, op.out.idx))
        offset = FBFloorDivideNodeModule.End(builder)
        return offset, FBOpNodeModule.OpNode.FloorDivideNode

    def _build_RemainderNode(
        self, builder: flatbuffers.Builder, op: RemainderNode
    ) -> Tuple[int, int]:
        """Auto-generated builder for RemainderNode."""
        # Import the MODULE (not class) to access builder functions like Start(), Add*(), End()
        from executorch.backends.mlx.serialization._generated.mlx_delegate import RemainderNode as FBRemainderNodeModule
        from executorch.backends.mlx.serialization._generated.mlx_delegate import OpNode as FBOpNodeModule
        from executorch.backends.mlx.serialization._generated.mlx_delegate.Tid import CreateTid
        from executorch.backends.mlx.serialization._generated.mlx_delegate.Vid import CreateVid

        FBRemainderNodeModule.Start(builder)
        FBRemainderNodeModule.AddA(builder, CreateTid(builder, op.a.idx))
        FBRemainderNodeModule.AddB(builder, CreateTid(builder, op.b.idx))
        FBRemainderNodeModule.AddOut(builder, CreateTid(builder, op.out.idx))
        offset = FBRemainderNodeModule.End(builder)
        return offset, FBOpNodeModule.OpNode.RemainderNode

    def _build_PowerNode(
        self, builder: flatbuffers.Builder, op: PowerNode
    ) -> Tuple[int, int]:
        """Auto-generated builder for PowerNode."""
        # Import the MODULE (not class) to access builder functions like Start(), Add*(), End()
        from executorch.backends.mlx.serialization._generated.mlx_delegate import PowerNode as FBPowerNodeModule
        from executorch.backends.mlx.serialization._generated.mlx_delegate import OpNode as FBOpNodeModule
        from executorch.backends.mlx.serialization._generated.mlx_delegate.Tid import CreateTid
        from executorch.backends.mlx.serialization._generated.mlx_delegate.Vid import CreateVid

        FBPowerNodeModule.Start(builder)
        FBPowerNodeModule.AddA(builder, CreateTid(builder, op.a.idx))
        FBPowerNodeModule.AddB(builder, CreateTid(builder, op.b.idx))
        FBPowerNodeModule.AddOut(builder, CreateTid(builder, op.out.idx))
        offset = FBPowerNodeModule.End(builder)
        return offset, FBOpNodeModule.OpNode.PowerNode

    def _build_LogSumExpNode(
        self, builder: flatbuffers.Builder, op: LogSumExpNode
    ) -> Tuple[int, int]:
        """Auto-generated builder for LogSumExpNode."""
        # Import the MODULE (not class) to access builder functions like Start(), Add*(), End()
        from executorch.backends.mlx.serialization._generated.mlx_delegate import LogSumExpNode as FBLogSumExpNodeModule
        from executorch.backends.mlx.serialization._generated.mlx_delegate import OpNode as FBOpNodeModule
        from executorch.backends.mlx.serialization._generated.mlx_delegate.Tid import CreateTid
        from executorch.backends.mlx.serialization._generated.mlx_delegate.Vid import CreateVid

        axes_vec = _build_int_vector(builder, op.axes) if op.axes is not None else None

        FBLogSumExpNodeModule.Start(builder)
        FBLogSumExpNodeModule.AddX(builder, CreateTid(builder, op.x.idx))
        FBLogSumExpNodeModule.AddOut(builder, CreateTid(builder, op.out.idx))
        if axes_vec is not None:
            FBLogSumExpNodeModule.AddAxes(builder, axes_vec)
        FBLogSumExpNodeModule.AddKeepdims(builder, op.keepdims)
        offset = FBLogSumExpNodeModule.End(builder)
        return offset, FBOpNodeModule.OpNode.LogSumExpNode

    def _build_SumNode(
        self, builder: flatbuffers.Builder, op: SumNode
    ) -> Tuple[int, int]:
        """Auto-generated builder for SumNode."""
        # Import the MODULE (not class) to access builder functions like Start(), Add*(), End()
        from executorch.backends.mlx.serialization._generated.mlx_delegate import SumNode as FBSumNodeModule
        from executorch.backends.mlx.serialization._generated.mlx_delegate import OpNode as FBOpNodeModule
        from executorch.backends.mlx.serialization._generated.mlx_delegate.Tid import CreateTid
        from executorch.backends.mlx.serialization._generated.mlx_delegate.Vid import CreateVid

        axes_vec = _build_int_vector(builder, op.axes) if op.axes is not None else None

        FBSumNodeModule.Start(builder)
        FBSumNodeModule.AddX(builder, CreateTid(builder, op.x.idx))
        FBSumNodeModule.AddOut(builder, CreateTid(builder, op.out.idx))
        if axes_vec is not None:
            FBSumNodeModule.AddAxes(builder, axes_vec)
        FBSumNodeModule.AddKeepdims(builder, op.keepdims)
        offset = FBSumNodeModule.End(builder)
        return offset, FBOpNodeModule.OpNode.SumNode

    def _build_MeanNode(
        self, builder: flatbuffers.Builder, op: MeanNode
    ) -> Tuple[int, int]:
        """Auto-generated builder for MeanNode."""
        # Import the MODULE (not class) to access builder functions like Start(), Add*(), End()
        from executorch.backends.mlx.serialization._generated.mlx_delegate import MeanNode as FBMeanNodeModule
        from executorch.backends.mlx.serialization._generated.mlx_delegate import OpNode as FBOpNodeModule
        from executorch.backends.mlx.serialization._generated.mlx_delegate.Tid import CreateTid
        from executorch.backends.mlx.serialization._generated.mlx_delegate.Vid import CreateVid

        axes_vec = _build_int_vector(builder, op.axes) if op.axes is not None else None

        FBMeanNodeModule.Start(builder)
        FBMeanNodeModule.AddX(builder, CreateTid(builder, op.x.idx))
        FBMeanNodeModule.AddOut(builder, CreateTid(builder, op.out.idx))
        if axes_vec is not None:
            FBMeanNodeModule.AddAxes(builder, axes_vec)
        FBMeanNodeModule.AddKeepdims(builder, op.keepdims)
        offset = FBMeanNodeModule.End(builder)
        return offset, FBOpNodeModule.OpNode.MeanNode

    def _build_VarNode(
        self, builder: flatbuffers.Builder, op: VarNode
    ) -> Tuple[int, int]:
        """Auto-generated builder for VarNode."""
        # Import the MODULE (not class) to access builder functions like Start(), Add*(), End()
        from executorch.backends.mlx.serialization._generated.mlx_delegate import VarNode as FBVarNodeModule
        from executorch.backends.mlx.serialization._generated.mlx_delegate import OpNode as FBOpNodeModule
        from executorch.backends.mlx.serialization._generated.mlx_delegate.Tid import CreateTid
        from executorch.backends.mlx.serialization._generated.mlx_delegate.Vid import CreateVid

        axes_vec = _build_int_vector(builder, op.axes) if op.axes is not None else None

        FBVarNodeModule.Start(builder)
        FBVarNodeModule.AddX(builder, CreateTid(builder, op.x.idx))
        FBVarNodeModule.AddOut(builder, CreateTid(builder, op.out.idx))
        if axes_vec is not None:
            FBVarNodeModule.AddAxes(builder, axes_vec)
        FBVarNodeModule.AddKeepdims(builder, op.keepdims)
        FBVarNodeModule.AddDdof(builder, op.ddof)
        offset = FBVarNodeModule.End(builder)
        return offset, FBOpNodeModule.OpNode.VarNode

    def _build_StdNode(
        self, builder: flatbuffers.Builder, op: StdNode
    ) -> Tuple[int, int]:
        """Auto-generated builder for StdNode."""
        # Import the MODULE (not class) to access builder functions like Start(), Add*(), End()
        from executorch.backends.mlx.serialization._generated.mlx_delegate import StdNode as FBStdNodeModule
        from executorch.backends.mlx.serialization._generated.mlx_delegate import OpNode as FBOpNodeModule
        from executorch.backends.mlx.serialization._generated.mlx_delegate.Tid import CreateTid
        from executorch.backends.mlx.serialization._generated.mlx_delegate.Vid import CreateVid

        axes_vec = _build_int_vector(builder, op.axes) if op.axes is not None else None

        FBStdNodeModule.Start(builder)
        FBStdNodeModule.AddX(builder, CreateTid(builder, op.x.idx))
        FBStdNodeModule.AddOut(builder, CreateTid(builder, op.out.idx))
        if axes_vec is not None:
            FBStdNodeModule.AddAxes(builder, axes_vec)
        FBStdNodeModule.AddKeepdims(builder, op.keepdims)
        FBStdNodeModule.AddDdof(builder, op.ddof)
        offset = FBStdNodeModule.End(builder)
        return offset, FBOpNodeModule.OpNode.StdNode

    def _build_ProdNode(
        self, builder: flatbuffers.Builder, op: ProdNode
    ) -> Tuple[int, int]:
        """Auto-generated builder for ProdNode."""
        # Import the MODULE (not class) to access builder functions like Start(), Add*(), End()
        from executorch.backends.mlx.serialization._generated.mlx_delegate import ProdNode as FBProdNodeModule
        from executorch.backends.mlx.serialization._generated.mlx_delegate import OpNode as FBOpNodeModule
        from executorch.backends.mlx.serialization._generated.mlx_delegate.Tid import CreateTid
        from executorch.backends.mlx.serialization._generated.mlx_delegate.Vid import CreateVid

        axes_vec = _build_int_vector(builder, op.axes) if op.axes is not None else None

        FBProdNodeModule.Start(builder)
        FBProdNodeModule.AddX(builder, CreateTid(builder, op.x.idx))
        FBProdNodeModule.AddOut(builder, CreateTid(builder, op.out.idx))
        if axes_vec is not None:
            FBProdNodeModule.AddAxes(builder, axes_vec)
        FBProdNodeModule.AddKeepdims(builder, op.keepdims)
        offset = FBProdNodeModule.End(builder)
        return offset, FBOpNodeModule.OpNode.ProdNode

    def _build_MaxNode(
        self, builder: flatbuffers.Builder, op: MaxNode
    ) -> Tuple[int, int]:
        """Auto-generated builder for MaxNode."""
        # Import the MODULE (not class) to access builder functions like Start(), Add*(), End()
        from executorch.backends.mlx.serialization._generated.mlx_delegate import MaxNode as FBMaxNodeModule
        from executorch.backends.mlx.serialization._generated.mlx_delegate import OpNode as FBOpNodeModule
        from executorch.backends.mlx.serialization._generated.mlx_delegate.Tid import CreateTid
        from executorch.backends.mlx.serialization._generated.mlx_delegate.Vid import CreateVid

        axes_vec = _build_int_vector(builder, op.axes) if op.axes is not None else None

        FBMaxNodeModule.Start(builder)
        FBMaxNodeModule.AddX(builder, CreateTid(builder, op.x.idx))
        FBMaxNodeModule.AddOut(builder, CreateTid(builder, op.out.idx))
        if axes_vec is not None:
            FBMaxNodeModule.AddAxes(builder, axes_vec)
        FBMaxNodeModule.AddKeepdims(builder, op.keepdims)
        offset = FBMaxNodeModule.End(builder)
        return offset, FBOpNodeModule.OpNode.MaxNode

    def _build_MinNode(
        self, builder: flatbuffers.Builder, op: MinNode
    ) -> Tuple[int, int]:
        """Auto-generated builder for MinNode."""
        # Import the MODULE (not class) to access builder functions like Start(), Add*(), End()
        from executorch.backends.mlx.serialization._generated.mlx_delegate import MinNode as FBMinNodeModule
        from executorch.backends.mlx.serialization._generated.mlx_delegate import OpNode as FBOpNodeModule
        from executorch.backends.mlx.serialization._generated.mlx_delegate.Tid import CreateTid
        from executorch.backends.mlx.serialization._generated.mlx_delegate.Vid import CreateVid

        axes_vec = _build_int_vector(builder, op.axes) if op.axes is not None else None

        FBMinNodeModule.Start(builder)
        FBMinNodeModule.AddX(builder, CreateTid(builder, op.x.idx))
        FBMinNodeModule.AddOut(builder, CreateTid(builder, op.out.idx))
        if axes_vec is not None:
            FBMinNodeModule.AddAxes(builder, axes_vec)
        FBMinNodeModule.AddKeepdims(builder, op.keepdims)
        offset = FBMinNodeModule.End(builder)
        return offset, FBOpNodeModule.OpNode.MinNode

    def _build_ArgminNode(
        self, builder: flatbuffers.Builder, op: ArgminNode
    ) -> Tuple[int, int]:
        """Auto-generated builder for ArgminNode."""
        # Import the MODULE (not class) to access builder functions like Start(), Add*(), End()
        from executorch.backends.mlx.serialization._generated.mlx_delegate import ArgminNode as FBArgminNodeModule
        from executorch.backends.mlx.serialization._generated.mlx_delegate import OpNode as FBOpNodeModule
        from executorch.backends.mlx.serialization._generated.mlx_delegate.Tid import CreateTid
        from executorch.backends.mlx.serialization._generated.mlx_delegate.Vid import CreateVid

        FBArgminNodeModule.Start(builder)
        FBArgminNodeModule.AddX(builder, CreateTid(builder, op.x.idx))
        FBArgminNodeModule.AddOut(builder, CreateTid(builder, op.out.idx))
        FBArgminNodeModule.AddAxis(builder, op.axis)
        FBArgminNodeModule.AddKeepdims(builder, op.keepdims)
        offset = FBArgminNodeModule.End(builder)
        return offset, FBOpNodeModule.OpNode.ArgminNode

    def _build_MedianNode(
        self, builder: flatbuffers.Builder, op: MedianNode
    ) -> Tuple[int, int]:
        """Auto-generated builder for MedianNode."""
        # Import the MODULE (not class) to access builder functions like Start(), Add*(), End()
        from executorch.backends.mlx.serialization._generated.mlx_delegate import MedianNode as FBMedianNodeModule
        from executorch.backends.mlx.serialization._generated.mlx_delegate import OpNode as FBOpNodeModule
        from executorch.backends.mlx.serialization._generated.mlx_delegate.Tid import CreateTid
        from executorch.backends.mlx.serialization._generated.mlx_delegate.Vid import CreateVid

        axes_vec = _build_int_vector(builder, op.axes) if op.axes is not None else None

        FBMedianNodeModule.Start(builder)
        FBMedianNodeModule.AddX(builder, CreateTid(builder, op.x.idx))
        FBMedianNodeModule.AddOut(builder, CreateTid(builder, op.out.idx))
        if axes_vec is not None:
            FBMedianNodeModule.AddAxes(builder, axes_vec)
        FBMedianNodeModule.AddKeepdims(builder, op.keepdims)
        offset = FBMedianNodeModule.End(builder)
        return offset, FBOpNodeModule.OpNode.MedianNode

    def _build_GatherMmNode(
        self, builder: flatbuffers.Builder, op: GatherMmNode
    ) -> Tuple[int, int]:
        """Auto-generated builder for GatherMmNode."""
        # Import the MODULE (not class) to access builder functions like Start(), Add*(), End()
        from executorch.backends.mlx.serialization._generated.mlx_delegate import GatherMmNode as FBGatherMmNodeModule
        from executorch.backends.mlx.serialization._generated.mlx_delegate import OpNode as FBOpNodeModule
        from executorch.backends.mlx.serialization._generated.mlx_delegate.Tid import CreateTid
        from executorch.backends.mlx.serialization._generated.mlx_delegate.Vid import CreateVid

        FBGatherMmNodeModule.Start(builder)
        FBGatherMmNodeModule.AddA(builder, CreateTid(builder, op.a.idx))
        FBGatherMmNodeModule.AddB(builder, CreateTid(builder, op.b.idx))
        FBGatherMmNodeModule.AddOut(builder, CreateTid(builder, op.out.idx))
        if op.lhs_indices is not None:
            FBGatherMmNodeModule.AddLhsIndices(builder, CreateTid(builder, op.lhs_indices.idx))
        if op.rhs_indices is not None:
            FBGatherMmNodeModule.AddRhsIndices(builder, CreateTid(builder, op.rhs_indices.idx))
        FBGatherMmNodeModule.AddSortedIndices(builder, op.sorted_indices)
        offset = FBGatherMmNodeModule.End(builder)
        return offset, FBOpNodeModule.OpNode.GatherMmNode

    def _build_GatherQmmNode(
        self, builder: flatbuffers.Builder, op: GatherQmmNode
    ) -> Tuple[int, int]:
        """Auto-generated builder for GatherQmmNode."""
        # Import the MODULE (not class) to access builder functions like Start(), Add*(), End()
        from executorch.backends.mlx.serialization._generated.mlx_delegate import GatherQmmNode as FBGatherQmmNodeModule
        from executorch.backends.mlx.serialization._generated.mlx_delegate import OpNode as FBOpNodeModule
        from executorch.backends.mlx.serialization._generated.mlx_delegate.Tid import CreateTid
        from executorch.backends.mlx.serialization._generated.mlx_delegate.Vid import CreateVid

        mode_off = builder.CreateString(op.mode)

        FBGatherQmmNodeModule.Start(builder)
        FBGatherQmmNodeModule.AddX(builder, CreateTid(builder, op.x.idx))
        FBGatherQmmNodeModule.AddW(builder, CreateTid(builder, op.w.idx))
        FBGatherQmmNodeModule.AddScales(builder, CreateTid(builder, op.scales.idx))
        FBGatherQmmNodeModule.AddOut(builder, CreateTid(builder, op.out.idx))
        FBGatherQmmNodeModule.AddMode(builder, mode_off)
        if op.biases is not None:
            FBGatherQmmNodeModule.AddBiases(builder, CreateTid(builder, op.biases.idx))
        if op.lhs_indices is not None:
            FBGatherQmmNodeModule.AddLhsIndices(builder, CreateTid(builder, op.lhs_indices.idx))
        if op.rhs_indices is not None:
            FBGatherQmmNodeModule.AddRhsIndices(builder, CreateTid(builder, op.rhs_indices.idx))
        FBGatherQmmNodeModule.AddTranspose(builder, op.transpose)
        FBGatherQmmNodeModule.AddGroupSize(builder, op.group_size)
        FBGatherQmmNodeModule.AddBits(builder, op.bits)
        FBGatherQmmNodeModule.AddSortedIndices(builder, op.sorted_indices)
        offset = FBGatherQmmNodeModule.End(builder)
        return offset, FBOpNodeModule.OpNode.GatherQmmNode

    def _build_ScanNode(
        self, builder: flatbuffers.Builder, op: ScanNode
    ) -> Tuple[int, int]:
        """Auto-generated builder for ScanNode."""
        # Import the MODULE (not class) to access builder functions like Start(), Add*(), End()
        from executorch.backends.mlx.serialization._generated.mlx_delegate import ScanNode as FBScanNodeModule
        from executorch.backends.mlx.serialization._generated.mlx_delegate import OpNode as FBOpNodeModule
        from executorch.backends.mlx.serialization._generated.mlx_delegate.Tid import CreateTid
        from executorch.backends.mlx.serialization._generated.mlx_delegate.Vid import CreateVid

        originals_vec = self._build_tid_vector(builder, op.originals)
        sliced_vec = self._build_tid_vector(builder, op.sliced)
        outputs_vec = self._build_tid_vector(builder, op.outputs)
        carry_vec = self._build_tid_vector(builder, op.carry)

        FBScanNodeModule.Start(builder)
        FBScanNodeModule.AddOriginals(builder, originals_vec)
        FBScanNodeModule.AddSliced(builder, sliced_vec)
        FBScanNodeModule.AddOutputs(builder, outputs_vec)
        FBScanNodeModule.AddCarry(builder, carry_vec)
        FBScanNodeModule.AddBodyChainIdx(builder, op.body_chain_idx)
        FBScanNodeModule.AddScanAxis(builder, op.scan_axis)
        offset = FBScanNodeModule.End(builder)
        return offset, FBOpNodeModule.OpNode.ScanNode

    def _build_MetalKernelNode(
        self, builder: flatbuffers.Builder, op: MetalKernelNode
    ) -> Tuple[int, int]:
        """Auto-generated builder for MetalKernelNode."""
        # Import the MODULE (not class) to access builder functions like Start(), Add*(), End()
        from executorch.backends.mlx.serialization._generated.mlx_delegate import MetalKernelNode as FBMetalKernelNodeModule
        from executorch.backends.mlx.serialization._generated.mlx_delegate import OpNode as FBOpNodeModule
        from executorch.backends.mlx.serialization._generated.mlx_delegate.Tid import CreateTid
        from executorch.backends.mlx.serialization._generated.mlx_delegate.Vid import CreateVid

        name_off = builder.CreateString(op.name)
        source_off = builder.CreateString(op.source)
        inputs_vec = self._build_tid_vector(builder, op.inputs)
        outputs_vec = self._build_tid_vector(builder, op.outputs)
        grid_vec = self._build_int_or_vid_vector(builder, op.grid)
        threadgroup_vec = self._build_int_or_vid_vector(builder, op.threadgroup)
        header_off = builder.CreateString(op.header) if op.header is not None else None
        input_names_vec = self._build_string_vector(builder, op.input_names) if op.input_names is not None else None
        output_names_vec = self._build_string_vector(builder, op.output_names) if op.output_names is not None else None
        output_shapes_flat_vec = self._build_int_or_vid_vector(builder, op.output_shapes_flat) if op.output_shapes_flat is not None else None
        output_shape_lengths_vec = _build_int_vector(builder, op.output_shape_lengths) if op.output_shape_lengths is not None else None
        output_dtypes_vec = _build_int8_vector(builder, op.output_dtypes) if op.output_dtypes is not None else None
        template_arg_names_vec = self._build_string_vector(builder, op.template_arg_names) if op.template_arg_names is not None else None
        template_arg_kinds_vec = _build_int8_vector(builder, op.template_arg_kinds) if op.template_arg_kinds is not None else None
        template_arg_values_vec = _build_int_vector(builder, op.template_arg_values) if op.template_arg_values is not None else None

        FBMetalKernelNodeModule.Start(builder)
        FBMetalKernelNodeModule.AddName(builder, name_off)
        FBMetalKernelNodeModule.AddSource(builder, source_off)
        FBMetalKernelNodeModule.AddInputs(builder, inputs_vec)
        FBMetalKernelNodeModule.AddOutputs(builder, outputs_vec)
        FBMetalKernelNodeModule.AddGrid(builder, grid_vec)
        FBMetalKernelNodeModule.AddThreadgroup(builder, threadgroup_vec)
        if header_off is not None:
            FBMetalKernelNodeModule.AddHeader(builder, header_off)
        if input_names_vec is not None:
            FBMetalKernelNodeModule.AddInputNames(builder, input_names_vec)
        if output_names_vec is not None:
            FBMetalKernelNodeModule.AddOutputNames(builder, output_names_vec)
        FBMetalKernelNodeModule.AddEnsureRowContiguous(builder, op.ensure_row_contiguous)
        FBMetalKernelNodeModule.AddAtomicOutputs(builder, op.atomic_outputs)
        if output_shapes_flat_vec is not None:
            FBMetalKernelNodeModule.AddOutputShapesFlat(builder, output_shapes_flat_vec)
        if output_shape_lengths_vec is not None:
            FBMetalKernelNodeModule.AddOutputShapeLengths(builder, output_shape_lengths_vec)
        if output_dtypes_vec is not None:
            FBMetalKernelNodeModule.AddOutputDtypes(builder, output_dtypes_vec)
        if template_arg_names_vec is not None:
            FBMetalKernelNodeModule.AddTemplateArgNames(builder, template_arg_names_vec)
        if template_arg_kinds_vec is not None:
            FBMetalKernelNodeModule.AddTemplateArgKinds(builder, template_arg_kinds_vec)
        if template_arg_values_vec is not None:
            FBMetalKernelNodeModule.AddTemplateArgValues(builder, template_arg_values_vec)
        if op.init_value is not None:
            FBMetalKernelNodeModule.AddInitValue(builder, op.init_value)
        offset = FBMetalKernelNodeModule.End(builder)
        return offset, FBOpNodeModule.OpNode.MetalKernelNode

    def _build_BitwiseXorNode(
        self, builder: flatbuffers.Builder, op: BitwiseXorNode
    ) -> Tuple[int, int]:
        """Auto-generated builder for BitwiseXorNode."""
        # Import the MODULE (not class) to access builder functions like Start(), Add*(), End()
        from executorch.backends.mlx.serialization._generated.mlx_delegate import BitwiseXorNode as FBBitwiseXorNodeModule
        from executorch.backends.mlx.serialization._generated.mlx_delegate import OpNode as FBOpNodeModule
        from executorch.backends.mlx.serialization._generated.mlx_delegate.Tid import CreateTid
        from executorch.backends.mlx.serialization._generated.mlx_delegate.Vid import CreateVid

        FBBitwiseXorNodeModule.Start(builder)
        if op.a is not None:
            FBBitwiseXorNodeModule.AddA(builder, CreateTid(builder, op.a.idx))
        if op.b is not None:
            FBBitwiseXorNodeModule.AddB(builder, CreateTid(builder, op.b.idx))
        if op.out is not None:
            FBBitwiseXorNodeModule.AddOut(builder, CreateTid(builder, op.out.idx))
        offset = FBBitwiseXorNodeModule.End(builder)
        return offset, FBOpNodeModule.OpNode.BitwiseXorNode
