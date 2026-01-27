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
#
# This file contains auto-generated serializer methods for all op types.

from __future__ import annotations

from typing import List, Tuple, Dict

import flatbuffers

# FlatBuffer union indices: 0 = NONE, then 1-indexed from union order
MLX_OP_TYPE_NAMES = {
    0: "NONE",
    1: "NoopNode",
    2: "LinearNode",
    3: "ItemIntNode",
    4: "ExpandDimsNode",
    5: "TileNode",
    6: "TakeAlongAxisNode",
    7: "RMSNormNode",
    8: "LayerNormNode",
    9: "RopeNode",
    10: "SdpaNode",
    11: "AddNode",
    12: "AddScalarNode",
    13: "SymSizeNode",
    14: "MulNode",
    15: "Conv1DNode",
    16: "GeluNode",
    17: "ARangeNode",
    18: "SiluNode",
    19: "ReshapeNode",
    20: "TransposeNode",
    21: "ContiguousNode",
    22: "IdCopyNode",
    23: "GatherNode",
    24: "SliceNode",
    25: "CastNode",
    26: "QuantizedLinearNode",
    27: "ConcatNode",
    28: "FullNode",
    29: "ZerosNode",
    30: "OnesNode",
    31: "ArgmaxNode",
    32: "SliceUpdateNode",
    33: "QuantizedGatherNode",
}

from executorch.backends.apple.mlx.serialization.mlx_graph_schema import (
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
    IntOrVid,
    FloatOrVid,
    Tid,
    Vid,
)


def _build_int_vector(builder: flatbuffers.Builder, vec: List[int]) -> int:
    """Build a vector of int32."""
    builder.StartVector(4, len(vec), 4)
    for v in reversed(vec):
        builder.PrependInt32(v)
    return builder.EndVector()


class GeneratedOpBuilders:
    """Mixin class with auto-generated op builder methods."""

    def _build_int_or_vid(self, builder: flatbuffers.Builder, iov: IntOrVid) -> int:
        """Build an IntOrVid table."""
        from executorch.backends.apple.mlx.serialization._generated.mlx_delegate import IntOrVid as FBIntOrVidModule
        from executorch.backends.apple.mlx.serialization._generated.mlx_delegate.Vid import CreateVid

        FBIntOrVidModule.Start(builder)
        FBIntOrVidModule.AddLiteral(builder, iov.literal)
        FBIntOrVidModule.AddIsVid(builder, iov.is_vid)
        if iov.vid is not None:
            # Vid is an inline struct - must be added last for proper FlatBuffer layout
            FBIntOrVidModule.AddVid(builder, CreateVid(builder, iov.vid.idx))
        return FBIntOrVidModule.End(builder)

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

    def _build_NoopNode(
        self, builder: flatbuffers.Builder, op: NoopNode
    ) -> Tuple[int, int]:
        """Auto-generated builder for NoopNode."""
        # Import the MODULE (not class) to access builder functions like Start(), Add*(), End()
        from executorch.backends.apple.mlx.serialization._generated.mlx_delegate import NoopNode as FBNoopNodeModule
        from executorch.backends.apple.mlx.serialization._generated.mlx_delegate import OpNode as FBOpNodeModule
        from executorch.backends.apple.mlx.serialization._generated.mlx_delegate.Tid import CreateTid
        from executorch.backends.apple.mlx.serialization._generated.mlx_delegate.Vid import CreateVid

        FBNoopNodeModule.Start(builder)
        offset = FBNoopNodeModule.End(builder)
        return offset, FBOpNodeModule.OpNode.NoopNode

    def _build_LinearNode(
        self, builder: flatbuffers.Builder, op: LinearNode
    ) -> Tuple[int, int]:
        """Auto-generated builder for LinearNode."""
        # Import the MODULE (not class) to access builder functions like Start(), Add*(), End()
        from executorch.backends.apple.mlx.serialization._generated.mlx_delegate import LinearNode as FBLinearNodeModule
        from executorch.backends.apple.mlx.serialization._generated.mlx_delegate import OpNode as FBOpNodeModule
        from executorch.backends.apple.mlx.serialization._generated.mlx_delegate.Tid import CreateTid
        from executorch.backends.apple.mlx.serialization._generated.mlx_delegate.Vid import CreateVid

        FBLinearNodeModule.Start(builder)
        FBLinearNodeModule.AddX(builder, CreateTid(builder, op.x.idx))
        FBLinearNodeModule.AddWeight(builder, CreateTid(builder, op.weight.idx))
        FBLinearNodeModule.AddOut(builder, CreateTid(builder, op.out.idx))
        if op.bias is not None:
            FBLinearNodeModule.AddBias(builder, CreateTid(builder, op.bias.idx))
        offset = FBLinearNodeModule.End(builder)
        return offset, FBOpNodeModule.OpNode.LinearNode

    def _build_ItemIntNode(
        self, builder: flatbuffers.Builder, op: ItemIntNode
    ) -> Tuple[int, int]:
        """Auto-generated builder for ItemIntNode."""
        # Import the MODULE (not class) to access builder functions like Start(), Add*(), End()
        from executorch.backends.apple.mlx.serialization._generated.mlx_delegate import ItemIntNode as FBItemIntNodeModule
        from executorch.backends.apple.mlx.serialization._generated.mlx_delegate import OpNode as FBOpNodeModule
        from executorch.backends.apple.mlx.serialization._generated.mlx_delegate.Tid import CreateTid
        from executorch.backends.apple.mlx.serialization._generated.mlx_delegate.Vid import CreateVid

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
        from executorch.backends.apple.mlx.serialization._generated.mlx_delegate import ExpandDimsNode as FBExpandDimsNodeModule
        from executorch.backends.apple.mlx.serialization._generated.mlx_delegate import OpNode as FBOpNodeModule
        from executorch.backends.apple.mlx.serialization._generated.mlx_delegate.Tid import CreateTid
        from executorch.backends.apple.mlx.serialization._generated.mlx_delegate.Vid import CreateVid

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
        from executorch.backends.apple.mlx.serialization._generated.mlx_delegate import TileNode as FBTileNodeModule
        from executorch.backends.apple.mlx.serialization._generated.mlx_delegate import OpNode as FBOpNodeModule
        from executorch.backends.apple.mlx.serialization._generated.mlx_delegate.Tid import CreateTid
        from executorch.backends.apple.mlx.serialization._generated.mlx_delegate.Vid import CreateVid

        reps_vec = _build_int_vector(builder, op.reps)

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
        from executorch.backends.apple.mlx.serialization._generated.mlx_delegate import TakeAlongAxisNode as FBTakeAlongAxisNodeModule
        from executorch.backends.apple.mlx.serialization._generated.mlx_delegate import OpNode as FBOpNodeModule
        from executorch.backends.apple.mlx.serialization._generated.mlx_delegate.Tid import CreateTid
        from executorch.backends.apple.mlx.serialization._generated.mlx_delegate.Vid import CreateVid

        FBTakeAlongAxisNodeModule.Start(builder)
        FBTakeAlongAxisNodeModule.AddX(builder, CreateTid(builder, op.x.idx))
        FBTakeAlongAxisNodeModule.AddIndices(builder, CreateTid(builder, op.indices.idx))
        FBTakeAlongAxisNodeModule.AddOut(builder, CreateTid(builder, op.out.idx))
        FBTakeAlongAxisNodeModule.AddAxis(builder, op.axis)
        offset = FBTakeAlongAxisNodeModule.End(builder)
        return offset, FBOpNodeModule.OpNode.TakeAlongAxisNode

    def _build_RMSNormNode(
        self, builder: flatbuffers.Builder, op: RMSNormNode
    ) -> Tuple[int, int]:
        """Auto-generated builder for RMSNormNode."""
        # Import the MODULE (not class) to access builder functions like Start(), Add*(), End()
        from executorch.backends.apple.mlx.serialization._generated.mlx_delegate import RMSNormNode as FBRMSNormNodeModule
        from executorch.backends.apple.mlx.serialization._generated.mlx_delegate import OpNode as FBOpNodeModule
        from executorch.backends.apple.mlx.serialization._generated.mlx_delegate.Tid import CreateTid
        from executorch.backends.apple.mlx.serialization._generated.mlx_delegate.Vid import CreateVid

        FBRMSNormNodeModule.Start(builder)
        FBRMSNormNodeModule.AddX(builder, CreateTid(builder, op.x.idx))
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
        from executorch.backends.apple.mlx.serialization._generated.mlx_delegate import LayerNormNode as FBLayerNormNodeModule
        from executorch.backends.apple.mlx.serialization._generated.mlx_delegate import OpNode as FBOpNodeModule
        from executorch.backends.apple.mlx.serialization._generated.mlx_delegate.Tid import CreateTid
        from executorch.backends.apple.mlx.serialization._generated.mlx_delegate.Vid import CreateVid

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
        from executorch.backends.apple.mlx.serialization._generated.mlx_delegate import RopeNode as FBRopeNodeModule
        from executorch.backends.apple.mlx.serialization._generated.mlx_delegate import OpNode as FBOpNodeModule
        from executorch.backends.apple.mlx.serialization._generated.mlx_delegate.Tid import CreateTid
        from executorch.backends.apple.mlx.serialization._generated.mlx_delegate.Vid import CreateVid

        FBRopeNodeModule.Start(builder)
        FBRopeNodeModule.AddQIn(builder, CreateTid(builder, op.q_in.idx))
        FBRopeNodeModule.AddKIn(builder, CreateTid(builder, op.k_in.idx))
        FBRopeNodeModule.AddQOut(builder, CreateTid(builder, op.q_out.idx))
        FBRopeNodeModule.AddKOut(builder, CreateTid(builder, op.k_out.idx))
        FBRopeNodeModule.AddHeadDim(builder, op.head_dim)
        FBRopeNodeModule.AddPos(builder, CreateVid(builder, op.pos.idx))
        if op.freqs is not None:
            FBRopeNodeModule.AddFreqs(builder, CreateTid(builder, op.freqs.idx))
        FBRopeNodeModule.AddTraditional(builder, op.traditional)
        if op.base is not None:
            FBRopeNodeModule.AddBase(builder, op.base)
            FBRopeNodeModule.AddBaseIsSet(builder, True)
        FBRopeNodeModule.AddScale(builder, op.scale)
        offset = FBRopeNodeModule.End(builder)
        return offset, FBOpNodeModule.OpNode.RopeNode

    def _build_SdpaNode(
        self, builder: flatbuffers.Builder, op: SdpaNode
    ) -> Tuple[int, int]:
        """Auto-generated builder for SdpaNode."""
        # Import the MODULE (not class) to access builder functions like Start(), Add*(), End()
        from executorch.backends.apple.mlx.serialization._generated.mlx_delegate import SdpaNode as FBSdpaNodeModule
        from executorch.backends.apple.mlx.serialization._generated.mlx_delegate import OpNode as FBOpNodeModule
        from executorch.backends.apple.mlx.serialization._generated.mlx_delegate.Tid import CreateTid
        from executorch.backends.apple.mlx.serialization._generated.mlx_delegate.Vid import CreateVid

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
        from executorch.backends.apple.mlx.serialization._generated.mlx_delegate import AddNode as FBAddNodeModule
        from executorch.backends.apple.mlx.serialization._generated.mlx_delegate import OpNode as FBOpNodeModule
        from executorch.backends.apple.mlx.serialization._generated.mlx_delegate.Tid import CreateTid
        from executorch.backends.apple.mlx.serialization._generated.mlx_delegate.Vid import CreateVid

        FBAddNodeModule.Start(builder)
        FBAddNodeModule.AddA(builder, CreateTid(builder, op.a.idx))
        FBAddNodeModule.AddB(builder, CreateTid(builder, op.b.idx))
        FBAddNodeModule.AddOut(builder, CreateTid(builder, op.out.idx))
        offset = FBAddNodeModule.End(builder)
        return offset, FBOpNodeModule.OpNode.AddNode

    def _build_AddScalarNode(
        self, builder: flatbuffers.Builder, op: AddScalarNode
    ) -> Tuple[int, int]:
        """Auto-generated builder for AddScalarNode."""
        # Import the MODULE (not class) to access builder functions like Start(), Add*(), End()
        from executorch.backends.apple.mlx.serialization._generated.mlx_delegate import AddScalarNode as FBAddScalarNodeModule
        from executorch.backends.apple.mlx.serialization._generated.mlx_delegate import OpNode as FBOpNodeModule
        from executorch.backends.apple.mlx.serialization._generated.mlx_delegate.Tid import CreateTid
        from executorch.backends.apple.mlx.serialization._generated.mlx_delegate.Vid import CreateVid

        a_off = self._build_int_or_vid(builder, op.a)
        b_off = self._build_int_or_vid(builder, op.b)

        FBAddScalarNodeModule.Start(builder)
        FBAddScalarNodeModule.AddA(builder, a_off)
        FBAddScalarNodeModule.AddB(builder, b_off)
        FBAddScalarNodeModule.AddOut(builder, CreateVid(builder, op.out.idx))
        offset = FBAddScalarNodeModule.End(builder)
        return offset, FBOpNodeModule.OpNode.AddScalarNode

    def _build_SymSizeNode(
        self, builder: flatbuffers.Builder, op: SymSizeNode
    ) -> Tuple[int, int]:
        """Auto-generated builder for SymSizeNode."""
        # Import the MODULE (not class) to access builder functions like Start(), Add*(), End()
        from executorch.backends.apple.mlx.serialization._generated.mlx_delegate import SymSizeNode as FBSymSizeNodeModule
        from executorch.backends.apple.mlx.serialization._generated.mlx_delegate import OpNode as FBOpNodeModule
        from executorch.backends.apple.mlx.serialization._generated.mlx_delegate.Tid import CreateTid
        from executorch.backends.apple.mlx.serialization._generated.mlx_delegate.Vid import CreateVid

        FBSymSizeNodeModule.Start(builder)
        FBSymSizeNodeModule.AddA(builder, CreateTid(builder, op.a.idx))
        FBSymSizeNodeModule.AddDim(builder, op.dim)
        FBSymSizeNodeModule.AddOut(builder, CreateVid(builder, op.out.idx))
        offset = FBSymSizeNodeModule.End(builder)
        return offset, FBOpNodeModule.OpNode.SymSizeNode

    def _build_MulNode(
        self, builder: flatbuffers.Builder, op: MulNode
    ) -> Tuple[int, int]:
        """Auto-generated builder for MulNode."""
        # Import the MODULE (not class) to access builder functions like Start(), Add*(), End()
        from executorch.backends.apple.mlx.serialization._generated.mlx_delegate import MulNode as FBMulNodeModule
        from executorch.backends.apple.mlx.serialization._generated.mlx_delegate import OpNode as FBOpNodeModule
        from executorch.backends.apple.mlx.serialization._generated.mlx_delegate.Tid import CreateTid
        from executorch.backends.apple.mlx.serialization._generated.mlx_delegate.Vid import CreateVid

        FBMulNodeModule.Start(builder)
        FBMulNodeModule.AddA(builder, CreateTid(builder, op.a.idx))
        FBMulNodeModule.AddB(builder, CreateTid(builder, op.b.idx))
        FBMulNodeModule.AddOut(builder, CreateTid(builder, op.out.idx))
        offset = FBMulNodeModule.End(builder)
        return offset, FBOpNodeModule.OpNode.MulNode

    def _build_Conv1DNode(
        self, builder: flatbuffers.Builder, op: Conv1DNode
    ) -> Tuple[int, int]:
        """Auto-generated builder for Conv1DNode."""
        # Import the MODULE (not class) to access builder functions like Start(), Add*(), End()
        from executorch.backends.apple.mlx.serialization._generated.mlx_delegate import Conv1DNode as FBConv1DNodeModule
        from executorch.backends.apple.mlx.serialization._generated.mlx_delegate import OpNode as FBOpNodeModule
        from executorch.backends.apple.mlx.serialization._generated.mlx_delegate.Tid import CreateTid
        from executorch.backends.apple.mlx.serialization._generated.mlx_delegate.Vid import CreateVid

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

    def _build_GeluNode(
        self, builder: flatbuffers.Builder, op: GeluNode
    ) -> Tuple[int, int]:
        """Auto-generated builder for GeluNode."""
        # Import the MODULE (not class) to access builder functions like Start(), Add*(), End()
        from executorch.backends.apple.mlx.serialization._generated.mlx_delegate import GeluNode as FBGeluNodeModule
        from executorch.backends.apple.mlx.serialization._generated.mlx_delegate import OpNode as FBOpNodeModule
        from executorch.backends.apple.mlx.serialization._generated.mlx_delegate.Tid import CreateTid
        from executorch.backends.apple.mlx.serialization._generated.mlx_delegate.Vid import CreateVid

        FBGeluNodeModule.Start(builder)
        FBGeluNodeModule.AddX(builder, CreateTid(builder, op.x.idx))
        FBGeluNodeModule.AddOut(builder, CreateTid(builder, op.out.idx))
        offset = FBGeluNodeModule.End(builder)
        return offset, FBOpNodeModule.OpNode.GeluNode

    def _build_ARangeNode(
        self, builder: flatbuffers.Builder, op: ARangeNode
    ) -> Tuple[int, int]:
        """Auto-generated builder for ARangeNode."""
        # Import the MODULE (not class) to access builder functions like Start(), Add*(), End()
        from executorch.backends.apple.mlx.serialization._generated.mlx_delegate import ARangeNode as FBARangeNodeModule
        from executorch.backends.apple.mlx.serialization._generated.mlx_delegate import OpNode as FBOpNodeModule
        from executorch.backends.apple.mlx.serialization._generated.mlx_delegate.Tid import CreateTid
        from executorch.backends.apple.mlx.serialization._generated.mlx_delegate.Vid import CreateVid

        FBARangeNodeModule.Start(builder)
        FBARangeNodeModule.AddOut(builder, CreateTid(builder, op.out.idx))
        FBARangeNodeModule.AddStart(builder, op.start)
        FBARangeNodeModule.AddStop(builder, op.stop)
        FBARangeNodeModule.AddStep(builder, op.step)
        if op.dtype is not None:
            FBARangeNodeModule.AddDtype(builder, op.dtype)
            FBARangeNodeModule.AddDtypeIsSet(builder, True)
        offset = FBARangeNodeModule.End(builder)
        return offset, FBOpNodeModule.OpNode.ARangeNode

    def _build_SiluNode(
        self, builder: flatbuffers.Builder, op: SiluNode
    ) -> Tuple[int, int]:
        """Auto-generated builder for SiluNode."""
        # Import the MODULE (not class) to access builder functions like Start(), Add*(), End()
        from executorch.backends.apple.mlx.serialization._generated.mlx_delegate import SiluNode as FBSiluNodeModule
        from executorch.backends.apple.mlx.serialization._generated.mlx_delegate import OpNode as FBOpNodeModule
        from executorch.backends.apple.mlx.serialization._generated.mlx_delegate.Tid import CreateTid
        from executorch.backends.apple.mlx.serialization._generated.mlx_delegate.Vid import CreateVid

        FBSiluNodeModule.Start(builder)
        FBSiluNodeModule.AddX(builder, CreateTid(builder, op.x.idx))
        FBSiluNodeModule.AddOut(builder, CreateTid(builder, op.out.idx))
        offset = FBSiluNodeModule.End(builder)
        return offset, FBOpNodeModule.OpNode.SiluNode

    def _build_ReshapeNode(
        self, builder: flatbuffers.Builder, op: ReshapeNode
    ) -> Tuple[int, int]:
        """Auto-generated builder for ReshapeNode."""
        # Import the MODULE (not class) to access builder functions like Start(), Add*(), End()
        from executorch.backends.apple.mlx.serialization._generated.mlx_delegate import ReshapeNode as FBReshapeNodeModule
        from executorch.backends.apple.mlx.serialization._generated.mlx_delegate import OpNode as FBOpNodeModule
        from executorch.backends.apple.mlx.serialization._generated.mlx_delegate.Tid import CreateTid
        from executorch.backends.apple.mlx.serialization._generated.mlx_delegate.Vid import CreateVid

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
        from executorch.backends.apple.mlx.serialization._generated.mlx_delegate import TransposeNode as FBTransposeNodeModule
        from executorch.backends.apple.mlx.serialization._generated.mlx_delegate import OpNode as FBOpNodeModule
        from executorch.backends.apple.mlx.serialization._generated.mlx_delegate.Tid import CreateTid
        from executorch.backends.apple.mlx.serialization._generated.mlx_delegate.Vid import CreateVid

        perm_vec = _build_int_vector(builder, op.perm)

        FBTransposeNodeModule.Start(builder)
        FBTransposeNodeModule.AddX(builder, CreateTid(builder, op.x.idx))
        FBTransposeNodeModule.AddOut(builder, CreateTid(builder, op.out.idx))
        FBTransposeNodeModule.AddPerm(builder, perm_vec)
        offset = FBTransposeNodeModule.End(builder)
        return offset, FBOpNodeModule.OpNode.TransposeNode

    def _build_ContiguousNode(
        self, builder: flatbuffers.Builder, op: ContiguousNode
    ) -> Tuple[int, int]:
        """Auto-generated builder for ContiguousNode."""
        # Import the MODULE (not class) to access builder functions like Start(), Add*(), End()
        from executorch.backends.apple.mlx.serialization._generated.mlx_delegate import ContiguousNode as FBContiguousNodeModule
        from executorch.backends.apple.mlx.serialization._generated.mlx_delegate import OpNode as FBOpNodeModule
        from executorch.backends.apple.mlx.serialization._generated.mlx_delegate.Tid import CreateTid
        from executorch.backends.apple.mlx.serialization._generated.mlx_delegate.Vid import CreateVid

        FBContiguousNodeModule.Start(builder)
        FBContiguousNodeModule.AddX(builder, CreateTid(builder, op.x.idx))
        FBContiguousNodeModule.AddOut(builder, CreateTid(builder, op.out.idx))
        offset = FBContiguousNodeModule.End(builder)
        return offset, FBOpNodeModule.OpNode.ContiguousNode

    def _build_IdCopyNode(
        self, builder: flatbuffers.Builder, op: IdCopyNode
    ) -> Tuple[int, int]:
        """Auto-generated builder for IdCopyNode."""
        # Import the MODULE (not class) to access builder functions like Start(), Add*(), End()
        from executorch.backends.apple.mlx.serialization._generated.mlx_delegate import IdCopyNode as FBIdCopyNodeModule
        from executorch.backends.apple.mlx.serialization._generated.mlx_delegate import OpNode as FBOpNodeModule
        from executorch.backends.apple.mlx.serialization._generated.mlx_delegate.Tid import CreateTid
        from executorch.backends.apple.mlx.serialization._generated.mlx_delegate.Vid import CreateVid

        FBIdCopyNodeModule.Start(builder)
        FBIdCopyNodeModule.AddX(builder, CreateTid(builder, op.x.idx))
        FBIdCopyNodeModule.AddOut(builder, CreateTid(builder, op.out.idx))
        offset = FBIdCopyNodeModule.End(builder)
        return offset, FBOpNodeModule.OpNode.IdCopyNode

    def _build_GatherNode(
        self, builder: flatbuffers.Builder, op: GatherNode
    ) -> Tuple[int, int]:
        """Auto-generated builder for GatherNode."""
        # Import the MODULE (not class) to access builder functions like Start(), Add*(), End()
        from executorch.backends.apple.mlx.serialization._generated.mlx_delegate import GatherNode as FBGatherNodeModule
        from executorch.backends.apple.mlx.serialization._generated.mlx_delegate import OpNode as FBOpNodeModule
        from executorch.backends.apple.mlx.serialization._generated.mlx_delegate.Tid import CreateTid
        from executorch.backends.apple.mlx.serialization._generated.mlx_delegate.Vid import CreateVid

        FBGatherNodeModule.Start(builder)
        FBGatherNodeModule.AddTable_(builder, CreateTid(builder, op.table_.idx))
        FBGatherNodeModule.AddIds(builder, CreateTid(builder, op.ids.idx))
        FBGatherNodeModule.AddOut(builder, CreateTid(builder, op.out.idx))
        offset = FBGatherNodeModule.End(builder)
        return offset, FBOpNodeModule.OpNode.GatherNode

    def _build_SliceNode(
        self, builder: flatbuffers.Builder, op: SliceNode
    ) -> Tuple[int, int]:
        """Auto-generated builder for SliceNode."""
        # Import the MODULE (not class) to access builder functions like Start(), Add*(), End()
        from executorch.backends.apple.mlx.serialization._generated.mlx_delegate import SliceNode as FBSliceNodeModule
        from executorch.backends.apple.mlx.serialization._generated.mlx_delegate import OpNode as FBOpNodeModule
        from executorch.backends.apple.mlx.serialization._generated.mlx_delegate.Tid import CreateTid
        from executorch.backends.apple.mlx.serialization._generated.mlx_delegate.Vid import CreateVid

        axis_off = self._build_int_or_vid(builder, op.axis)
        start_off = self._build_int_or_vid(builder, op.start)
        end_off = self._build_int_or_vid(builder, op.end)

        FBSliceNodeModule.Start(builder)
        FBSliceNodeModule.AddX(builder, CreateTid(builder, op.x.idx))
        FBSliceNodeModule.AddOut(builder, CreateTid(builder, op.out.idx))
        FBSliceNodeModule.AddAxis(builder, axis_off)
        FBSliceNodeModule.AddStart(builder, start_off)
        FBSliceNodeModule.AddEnd(builder, end_off)
        offset = FBSliceNodeModule.End(builder)
        return offset, FBOpNodeModule.OpNode.SliceNode

    def _build_CastNode(
        self, builder: flatbuffers.Builder, op: CastNode
    ) -> Tuple[int, int]:
        """Auto-generated builder for CastNode."""
        # Import the MODULE (not class) to access builder functions like Start(), Add*(), End()
        from executorch.backends.apple.mlx.serialization._generated.mlx_delegate import CastNode as FBCastNodeModule
        from executorch.backends.apple.mlx.serialization._generated.mlx_delegate import OpNode as FBOpNodeModule
        from executorch.backends.apple.mlx.serialization._generated.mlx_delegate.Tid import CreateTid
        from executorch.backends.apple.mlx.serialization._generated.mlx_delegate.Vid import CreateVid

        FBCastNodeModule.Start(builder)
        FBCastNodeModule.AddX(builder, CreateTid(builder, op.x.idx))
        FBCastNodeModule.AddOut(builder, CreateTid(builder, op.out.idx))
        FBCastNodeModule.AddDtype(builder, op.dtype)
        offset = FBCastNodeModule.End(builder)
        return offset, FBOpNodeModule.OpNode.CastNode

    def _build_QuantizedLinearNode(
        self, builder: flatbuffers.Builder, op: QuantizedLinearNode
    ) -> Tuple[int, int]:
        """Auto-generated builder for QuantizedLinearNode."""
        # Import the MODULE (not class) to access builder functions like Start(), Add*(), End()
        from executorch.backends.apple.mlx.serialization._generated.mlx_delegate import QuantizedLinearNode as FBQuantizedLinearNodeModule
        from executorch.backends.apple.mlx.serialization._generated.mlx_delegate import OpNode as FBOpNodeModule
        from executorch.backends.apple.mlx.serialization._generated.mlx_delegate.Tid import CreateTid
        from executorch.backends.apple.mlx.serialization._generated.mlx_delegate.Vid import CreateVid

        mode_off = builder.CreateString(op.mode)

        FBQuantizedLinearNodeModule.Start(builder)
        FBQuantizedLinearNodeModule.AddX(builder, CreateTid(builder, op.x.idx))
        FBQuantizedLinearNodeModule.AddW(builder, CreateTid(builder, op.w.idx))
        FBQuantizedLinearNodeModule.AddScales(builder, CreateTid(builder, op.scales.idx))
        FBQuantizedLinearNodeModule.AddOut(builder, CreateTid(builder, op.out.idx))
        if op.biases is not None:
            FBQuantizedLinearNodeModule.AddBiases(builder, CreateTid(builder, op.biases.idx))
        if op.bias is not None:
            FBQuantizedLinearNodeModule.AddBias(builder, CreateTid(builder, op.bias.idx))
        FBQuantizedLinearNodeModule.AddGroupSize(builder, op.group_size)
        FBQuantizedLinearNodeModule.AddBits(builder, op.bits)
        FBQuantizedLinearNodeModule.AddMode(builder, mode_off)
        FBQuantizedLinearNodeModule.AddOutDtype(builder, op.out_dtype)
        offset = FBQuantizedLinearNodeModule.End(builder)
        return offset, FBOpNodeModule.OpNode.QuantizedLinearNode

    def _build_ConcatNode(
        self, builder: flatbuffers.Builder, op: ConcatNode
    ) -> Tuple[int, int]:
        """Auto-generated builder for ConcatNode."""
        # Import the MODULE (not class) to access builder functions like Start(), Add*(), End()
        from executorch.backends.apple.mlx.serialization._generated.mlx_delegate import ConcatNode as FBConcatNodeModule
        from executorch.backends.apple.mlx.serialization._generated.mlx_delegate import OpNode as FBOpNodeModule
        from executorch.backends.apple.mlx.serialization._generated.mlx_delegate.Tid import CreateTid
        from executorch.backends.apple.mlx.serialization._generated.mlx_delegate.Vid import CreateVid

        FBConcatNodeModule.Start(builder)
        FBConcatNodeModule.AddA(builder, CreateTid(builder, op.a.idx))
        FBConcatNodeModule.AddB(builder, CreateTid(builder, op.b.idx))
        FBConcatNodeModule.AddOut(builder, CreateTid(builder, op.out.idx))
        FBConcatNodeModule.AddAxis(builder, op.axis)
        offset = FBConcatNodeModule.End(builder)
        return offset, FBOpNodeModule.OpNode.ConcatNode

    def _build_FullNode(
        self, builder: flatbuffers.Builder, op: FullNode
    ) -> Tuple[int, int]:
        """Auto-generated builder for FullNode."""
        # Import the MODULE (not class) to access builder functions like Start(), Add*(), End()
        from executorch.backends.apple.mlx.serialization._generated.mlx_delegate import FullNode as FBFullNodeModule
        from executorch.backends.apple.mlx.serialization._generated.mlx_delegate import OpNode as FBOpNodeModule
        from executorch.backends.apple.mlx.serialization._generated.mlx_delegate.Tid import CreateTid
        from executorch.backends.apple.mlx.serialization._generated.mlx_delegate.Vid import CreateVid

        shape_vec = _build_int_vector(builder, op.shape)

        FBFullNodeModule.Start(builder)
        FBFullNodeModule.AddOut(builder, CreateTid(builder, op.out.idx))
        FBFullNodeModule.AddShape(builder, shape_vec)
        FBFullNodeModule.AddV(builder, op.v)
        FBFullNodeModule.AddDtype(builder, op.dtype)
        offset = FBFullNodeModule.End(builder)
        return offset, FBOpNodeModule.OpNode.FullNode

    def _build_ZerosNode(
        self, builder: flatbuffers.Builder, op: ZerosNode
    ) -> Tuple[int, int]:
        """Auto-generated builder for ZerosNode."""
        # Import the MODULE (not class) to access builder functions like Start(), Add*(), End()
        from executorch.backends.apple.mlx.serialization._generated.mlx_delegate import ZerosNode as FBZerosNodeModule
        from executorch.backends.apple.mlx.serialization._generated.mlx_delegate import OpNode as FBOpNodeModule
        from executorch.backends.apple.mlx.serialization._generated.mlx_delegate.Tid import CreateTid
        from executorch.backends.apple.mlx.serialization._generated.mlx_delegate.Vid import CreateVid

        shape_vec = _build_int_vector(builder, op.shape)

        FBZerosNodeModule.Start(builder)
        FBZerosNodeModule.AddOut(builder, CreateTid(builder, op.out.idx))
        FBZerosNodeModule.AddShape(builder, shape_vec)
        FBZerosNodeModule.AddDtype(builder, op.dtype)
        offset = FBZerosNodeModule.End(builder)
        return offset, FBOpNodeModule.OpNode.ZerosNode

    def _build_OnesNode(
        self, builder: flatbuffers.Builder, op: OnesNode
    ) -> Tuple[int, int]:
        """Auto-generated builder for OnesNode."""
        # Import the MODULE (not class) to access builder functions like Start(), Add*(), End()
        from executorch.backends.apple.mlx.serialization._generated.mlx_delegate import OnesNode as FBOnesNodeModule
        from executorch.backends.apple.mlx.serialization._generated.mlx_delegate import OpNode as FBOpNodeModule
        from executorch.backends.apple.mlx.serialization._generated.mlx_delegate.Tid import CreateTid
        from executorch.backends.apple.mlx.serialization._generated.mlx_delegate.Vid import CreateVid

        shape_vec = _build_int_vector(builder, op.shape)

        FBOnesNodeModule.Start(builder)
        FBOnesNodeModule.AddOut(builder, CreateTid(builder, op.out.idx))
        FBOnesNodeModule.AddShape(builder, shape_vec)
        FBOnesNodeModule.AddDtype(builder, op.dtype)
        offset = FBOnesNodeModule.End(builder)
        return offset, FBOpNodeModule.OpNode.OnesNode

    def _build_ArgmaxNode(
        self, builder: flatbuffers.Builder, op: ArgmaxNode
    ) -> Tuple[int, int]:
        """Auto-generated builder for ArgmaxNode."""
        # Import the MODULE (not class) to access builder functions like Start(), Add*(), End()
        from executorch.backends.apple.mlx.serialization._generated.mlx_delegate import ArgmaxNode as FBArgmaxNodeModule
        from executorch.backends.apple.mlx.serialization._generated.mlx_delegate import OpNode as FBOpNodeModule
        from executorch.backends.apple.mlx.serialization._generated.mlx_delegate.Tid import CreateTid
        from executorch.backends.apple.mlx.serialization._generated.mlx_delegate.Vid import CreateVid

        FBArgmaxNodeModule.Start(builder)
        FBArgmaxNodeModule.AddX(builder, CreateTid(builder, op.x.idx))
        FBArgmaxNodeModule.AddOut(builder, CreateTid(builder, op.out.idx))
        FBArgmaxNodeModule.AddAxis(builder, op.axis)
        offset = FBArgmaxNodeModule.End(builder)
        return offset, FBOpNodeModule.OpNode.ArgmaxNode

    def _build_SliceUpdateNode(
        self, builder: flatbuffers.Builder, op: SliceUpdateNode
    ) -> Tuple[int, int]:
        """Auto-generated builder for SliceUpdateNode."""
        # Import the MODULE (not class) to access builder functions like Start(), Add*(), End()
        from executorch.backends.apple.mlx.serialization._generated.mlx_delegate import SliceUpdateNode as FBSliceUpdateNodeModule
        from executorch.backends.apple.mlx.serialization._generated.mlx_delegate import OpNode as FBOpNodeModule
        from executorch.backends.apple.mlx.serialization._generated.mlx_delegate.Tid import CreateTid
        from executorch.backends.apple.mlx.serialization._generated.mlx_delegate.Vid import CreateVid

        axis_off = self._build_int_or_vid(builder, op.axis)
        start_off = self._build_int_or_vid(builder, op.start)
        stop_off = self._build_int_or_vid(builder, op.stop)

        FBSliceUpdateNodeModule.Start(builder)
        FBSliceUpdateNodeModule.AddDst(builder, CreateTid(builder, op.dst.idx))
        FBSliceUpdateNodeModule.AddUpdate(builder, CreateTid(builder, op.update.idx))
        FBSliceUpdateNodeModule.AddAxis(builder, axis_off)
        FBSliceUpdateNodeModule.AddStart(builder, start_off)
        FBSliceUpdateNodeModule.AddStop(builder, stop_off)
        offset = FBSliceUpdateNodeModule.End(builder)
        return offset, FBOpNodeModule.OpNode.SliceUpdateNode

    def _build_QuantizedGatherNode(
        self, builder: flatbuffers.Builder, op: QuantizedGatherNode
    ) -> Tuple[int, int]:
        """Auto-generated builder for QuantizedGatherNode."""
        # Import the MODULE (not class) to access builder functions like Start(), Add*(), End()
        from executorch.backends.apple.mlx.serialization._generated.mlx_delegate import QuantizedGatherNode as FBQuantizedGatherNodeModule
        from executorch.backends.apple.mlx.serialization._generated.mlx_delegate import OpNode as FBOpNodeModule
        from executorch.backends.apple.mlx.serialization._generated.mlx_delegate.Tid import CreateTid
        from executorch.backends.apple.mlx.serialization._generated.mlx_delegate.Vid import CreateVid

        mode_off = builder.CreateString(op.mode)

        FBQuantizedGatherNodeModule.Start(builder)
        FBQuantizedGatherNodeModule.AddTableQ(builder, CreateTid(builder, op.table_q.idx))
        FBQuantizedGatherNodeModule.AddScales(builder, CreateTid(builder, op.scales.idx))
        FBQuantizedGatherNodeModule.AddIds(builder, CreateTid(builder, op.ids.idx))
        FBQuantizedGatherNodeModule.AddOut(builder, CreateTid(builder, op.out.idx))
        if op.biases is not None:
            FBQuantizedGatherNodeModule.AddBiases(builder, CreateTid(builder, op.biases.idx))
        FBQuantizedGatherNodeModule.AddGroupSize(builder, op.group_size)
        FBQuantizedGatherNodeModule.AddBits(builder, op.bits)
        FBQuantizedGatherNodeModule.AddMode(builder, mode_off)
        FBQuantizedGatherNodeModule.AddOutDtype(builder, op.out_dtype)
        offset = FBQuantizedGatherNodeModule.End(builder)
        return offset, FBOpNodeModule.OpNode.QuantizedGatherNode
