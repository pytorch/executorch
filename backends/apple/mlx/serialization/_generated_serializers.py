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
        from executorch.backends.apple.mlx.serialization._generated import (
            IntOrVid as FBIntOrVid,
        )
        from executorch.backends.apple.mlx.serialization._generated.mlx_delegate.Vid import CreateVid

        FBIntOrVid.Start(builder)
        FBIntOrVid.AddLiteral(builder, iov.literal)
        FBIntOrVid.AddIsVid(builder, iov.is_vid)
        if iov.vid is not None:
            # Vid is an inline struct - must be added last for proper FlatBuffer layout
            FBIntOrVid.AddVid(builder, CreateVid(builder, iov.vid.idx))
        return FBIntOrVid.End(builder)

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
        from executorch.backends.apple.mlx.serialization._generated import (
            NoopNode as FBNoopNode,
            OpNode as FBOpNode,
        )
        from executorch.backends.apple.mlx.serialization._generated.mlx_delegate.Tid import CreateTid
        from executorch.backends.apple.mlx.serialization._generated.mlx_delegate.Vid import CreateVid

        FBNoopNode.Start(builder)
        offset = FBNoopNode.End(builder)
        return offset, FBOpNode.OpNode.NoopNode

    def _build_LinearNode(
        self, builder: flatbuffers.Builder, op: LinearNode
    ) -> Tuple[int, int]:
        """Auto-generated builder for LinearNode."""
        from executorch.backends.apple.mlx.serialization._generated import (
            LinearNode as FBLinearNode,
            OpNode as FBOpNode,
        )
        from executorch.backends.apple.mlx.serialization._generated.mlx_delegate.Tid import CreateTid
        from executorch.backends.apple.mlx.serialization._generated.mlx_delegate.Vid import CreateVid

        FBLinearNode.Start(builder)
        FBLinearNode.AddX(builder, CreateTid(builder, op.x.idx))
        FBLinearNode.AddWeight(builder, CreateTid(builder, op.weight.idx))
        FBLinearNode.AddOut(builder, CreateTid(builder, op.out.idx))
        if op.bias is not None:
            FBLinearNode.AddBias(builder, CreateTid(builder, op.bias.idx))
        offset = FBLinearNode.End(builder)
        return offset, FBOpNode.OpNode.LinearNode

    def _build_ItemIntNode(
        self, builder: flatbuffers.Builder, op: ItemIntNode
    ) -> Tuple[int, int]:
        """Auto-generated builder for ItemIntNode."""
        from executorch.backends.apple.mlx.serialization._generated import (
            ItemIntNode as FBItemIntNode,
            OpNode as FBOpNode,
        )
        from executorch.backends.apple.mlx.serialization._generated.mlx_delegate.Tid import CreateTid
        from executorch.backends.apple.mlx.serialization._generated.mlx_delegate.Vid import CreateVid

        FBItemIntNode.Start(builder)
        FBItemIntNode.AddX(builder, CreateTid(builder, op.x.idx))
        FBItemIntNode.AddOut(builder, CreateVid(builder, op.out.idx))
        offset = FBItemIntNode.End(builder)
        return offset, FBOpNode.OpNode.ItemIntNode

    def _build_ExpandDimsNode(
        self, builder: flatbuffers.Builder, op: ExpandDimsNode
    ) -> Tuple[int, int]:
        """Auto-generated builder for ExpandDimsNode."""
        from executorch.backends.apple.mlx.serialization._generated import (
            ExpandDimsNode as FBExpandDimsNode,
            OpNode as FBOpNode,
        )
        from executorch.backends.apple.mlx.serialization._generated.mlx_delegate.Tid import CreateTid
        from executorch.backends.apple.mlx.serialization._generated.mlx_delegate.Vid import CreateVid

        FBExpandDimsNode.Start(builder)
        FBExpandDimsNode.AddX(builder, CreateTid(builder, op.x.idx))
        FBExpandDimsNode.AddOut(builder, CreateTid(builder, op.out.idx))
        FBExpandDimsNode.AddAxis(builder, op.axis)
        offset = FBExpandDimsNode.End(builder)
        return offset, FBOpNode.OpNode.ExpandDimsNode

    def _build_TileNode(
        self, builder: flatbuffers.Builder, op: TileNode
    ) -> Tuple[int, int]:
        """Auto-generated builder for TileNode."""
        from executorch.backends.apple.mlx.serialization._generated import (
            TileNode as FBTileNode,
            OpNode as FBOpNode,
        )
        from executorch.backends.apple.mlx.serialization._generated.mlx_delegate.Tid import CreateTid
        from executorch.backends.apple.mlx.serialization._generated.mlx_delegate.Vid import CreateVid

        reps_vec = _build_int_vector(builder, op.reps)

        FBTileNode.Start(builder)
        FBTileNode.AddX(builder, CreateTid(builder, op.x.idx))
        FBTileNode.AddOut(builder, CreateTid(builder, op.out.idx))
        FBTileNode.AddReps(builder, reps_vec)
        offset = FBTileNode.End(builder)
        return offset, FBOpNode.OpNode.TileNode

    def _build_TakeAlongAxisNode(
        self, builder: flatbuffers.Builder, op: TakeAlongAxisNode
    ) -> Tuple[int, int]:
        """Auto-generated builder for TakeAlongAxisNode."""
        from executorch.backends.apple.mlx.serialization._generated import (
            TakeAlongAxisNode as FBTakeAlongAxisNode,
            OpNode as FBOpNode,
        )
        from executorch.backends.apple.mlx.serialization._generated.mlx_delegate.Tid import CreateTid
        from executorch.backends.apple.mlx.serialization._generated.mlx_delegate.Vid import CreateVid

        FBTakeAlongAxisNode.Start(builder)
        FBTakeAlongAxisNode.AddX(builder, CreateTid(builder, op.x.idx))
        FBTakeAlongAxisNode.AddIndices(builder, CreateTid(builder, op.indices.idx))
        FBTakeAlongAxisNode.AddOut(builder, CreateTid(builder, op.out.idx))
        FBTakeAlongAxisNode.AddAxis(builder, op.axis)
        offset = FBTakeAlongAxisNode.End(builder)
        return offset, FBOpNode.OpNode.TakeAlongAxisNode

    def _build_RMSNormNode(
        self, builder: flatbuffers.Builder, op: RMSNormNode
    ) -> Tuple[int, int]:
        """Auto-generated builder for RMSNormNode."""
        from executorch.backends.apple.mlx.serialization._generated import (
            RMSNormNode as FBRMSNormNode,
            OpNode as FBOpNode,
        )
        from executorch.backends.apple.mlx.serialization._generated.mlx_delegate.Tid import CreateTid
        from executorch.backends.apple.mlx.serialization._generated.mlx_delegate.Vid import CreateVid

        FBRMSNormNode.Start(builder)
        FBRMSNormNode.AddX(builder, CreateTid(builder, op.x.idx))
        FBRMSNormNode.AddWeight(builder, CreateTid(builder, op.weight.idx))
        FBRMSNormNode.AddOut(builder, CreateTid(builder, op.out.idx))
        FBRMSNormNode.AddEps(builder, op.eps)
        offset = FBRMSNormNode.End(builder)
        return offset, FBOpNode.OpNode.RMSNormNode

    def _build_LayerNormNode(
        self, builder: flatbuffers.Builder, op: LayerNormNode
    ) -> Tuple[int, int]:
        """Auto-generated builder for LayerNormNode."""
        from executorch.backends.apple.mlx.serialization._generated import (
            LayerNormNode as FBLayerNormNode,
            OpNode as FBOpNode,
        )
        from executorch.backends.apple.mlx.serialization._generated.mlx_delegate.Tid import CreateTid
        from executorch.backends.apple.mlx.serialization._generated.mlx_delegate.Vid import CreateVid

        FBLayerNormNode.Start(builder)
        FBLayerNormNode.AddX(builder, CreateTid(builder, op.x.idx))
        FBLayerNormNode.AddOut(builder, CreateTid(builder, op.out.idx))
        if op.weight is not None:
            FBLayerNormNode.AddWeight(builder, CreateTid(builder, op.weight.idx))
        if op.bias is not None:
            FBLayerNormNode.AddBias(builder, CreateTid(builder, op.bias.idx))
        FBLayerNormNode.AddEps(builder, op.eps)
        offset = FBLayerNormNode.End(builder)
        return offset, FBOpNode.OpNode.LayerNormNode

    def _build_RopeNode(
        self, builder: flatbuffers.Builder, op: RopeNode
    ) -> Tuple[int, int]:
        """Auto-generated builder for RopeNode."""
        from executorch.backends.apple.mlx.serialization._generated import (
            RopeNode as FBRopeNode,
            OpNode as FBOpNode,
        )
        from executorch.backends.apple.mlx.serialization._generated.mlx_delegate.Tid import CreateTid
        from executorch.backends.apple.mlx.serialization._generated.mlx_delegate.Vid import CreateVid

        FBRopeNode.Start(builder)
        FBRopeNode.AddQIn(builder, CreateTid(builder, op.q_in.idx))
        FBRopeNode.AddKIn(builder, CreateTid(builder, op.k_in.idx))
        FBRopeNode.AddQOut(builder, CreateTid(builder, op.q_out.idx))
        FBRopeNode.AddKOut(builder, CreateTid(builder, op.k_out.idx))
        FBRopeNode.AddHeadDim(builder, op.head_dim)
        FBRopeNode.AddPos(builder, CreateVid(builder, op.pos.idx))
        if op.freqs is not None:
            FBRopeNode.AddFreqs(builder, CreateTid(builder, op.freqs.idx))
        FBRopeNode.AddTraditional(builder, op.traditional)
        if op.base is not None:
            FBRopeNode.AddBase(builder, op.base)
            FBRopeNode.AddBaseIsSet(builder, True)
        FBRopeNode.AddScale(builder, op.scale)
        offset = FBRopeNode.End(builder)
        return offset, FBOpNode.OpNode.RopeNode

    def _build_SdpaNode(
        self, builder: flatbuffers.Builder, op: SdpaNode
    ) -> Tuple[int, int]:
        """Auto-generated builder for SdpaNode."""
        from executorch.backends.apple.mlx.serialization._generated import (
            SdpaNode as FBSdpaNode,
            OpNode as FBOpNode,
        )
        from executorch.backends.apple.mlx.serialization._generated.mlx_delegate.Tid import CreateTid
        from executorch.backends.apple.mlx.serialization._generated.mlx_delegate.Vid import CreateVid

        FBSdpaNode.Start(builder)
        FBSdpaNode.AddQ(builder, CreateTid(builder, op.q.idx))
        FBSdpaNode.AddK(builder, CreateTid(builder, op.k.idx))
        FBSdpaNode.AddV(builder, CreateTid(builder, op.v.idx))
        FBSdpaNode.AddOut(builder, CreateTid(builder, op.out.idx))
        FBSdpaNode.AddScale(builder, op.scale)
        if op.mask is not None:
            FBSdpaNode.AddMask(builder, CreateTid(builder, op.mask.idx))
        FBSdpaNode.AddCausal(builder, op.causal)
        offset = FBSdpaNode.End(builder)
        return offset, FBOpNode.OpNode.SdpaNode

    def _build_AddNode(
        self, builder: flatbuffers.Builder, op: AddNode
    ) -> Tuple[int, int]:
        """Auto-generated builder for AddNode."""
        from executorch.backends.apple.mlx.serialization._generated import (
            AddNode as FBAddNode,
            OpNode as FBOpNode,
        )
        from executorch.backends.apple.mlx.serialization._generated.mlx_delegate.Tid import CreateTid
        from executorch.backends.apple.mlx.serialization._generated.mlx_delegate.Vid import CreateVid

        FBAddNode.Start(builder)
        FBAddNode.AddA(builder, CreateTid(builder, op.a.idx))
        FBAddNode.AddB(builder, CreateTid(builder, op.b.idx))
        FBAddNode.AddOut(builder, CreateTid(builder, op.out.idx))
        offset = FBAddNode.End(builder)
        return offset, FBOpNode.OpNode.AddNode

    def _build_AddScalarNode(
        self, builder: flatbuffers.Builder, op: AddScalarNode
    ) -> Tuple[int, int]:
        """Auto-generated builder for AddScalarNode."""
        from executorch.backends.apple.mlx.serialization._generated import (
            AddScalarNode as FBAddScalarNode,
            OpNode as FBOpNode,
        )
        from executorch.backends.apple.mlx.serialization._generated.mlx_delegate.Tid import CreateTid
        from executorch.backends.apple.mlx.serialization._generated.mlx_delegate.Vid import CreateVid

        a_off = self._build_int_or_vid(builder, op.a)
        b_off = self._build_int_or_vid(builder, op.b)

        FBAddScalarNode.Start(builder)
        FBAddScalarNode.AddA(builder, a_off)
        FBAddScalarNode.AddB(builder, b_off)
        FBAddScalarNode.AddOut(builder, CreateVid(builder, op.out.idx))
        offset = FBAddScalarNode.End(builder)
        return offset, FBOpNode.OpNode.AddScalarNode

    def _build_SymSizeNode(
        self, builder: flatbuffers.Builder, op: SymSizeNode
    ) -> Tuple[int, int]:
        """Auto-generated builder for SymSizeNode."""
        from executorch.backends.apple.mlx.serialization._generated import (
            SymSizeNode as FBSymSizeNode,
            OpNode as FBOpNode,
        )
        from executorch.backends.apple.mlx.serialization._generated.mlx_delegate.Tid import CreateTid
        from executorch.backends.apple.mlx.serialization._generated.mlx_delegate.Vid import CreateVid

        FBSymSizeNode.Start(builder)
        FBSymSizeNode.AddA(builder, CreateTid(builder, op.a.idx))
        FBSymSizeNode.AddDim(builder, op.dim)
        FBSymSizeNode.AddOut(builder, CreateVid(builder, op.out.idx))
        offset = FBSymSizeNode.End(builder)
        return offset, FBOpNode.OpNode.SymSizeNode

    def _build_MulNode(
        self, builder: flatbuffers.Builder, op: MulNode
    ) -> Tuple[int, int]:
        """Auto-generated builder for MulNode."""
        from executorch.backends.apple.mlx.serialization._generated import (
            MulNode as FBMulNode,
            OpNode as FBOpNode,
        )
        from executorch.backends.apple.mlx.serialization._generated.mlx_delegate.Tid import CreateTid
        from executorch.backends.apple.mlx.serialization._generated.mlx_delegate.Vid import CreateVid

        FBMulNode.Start(builder)
        FBMulNode.AddA(builder, CreateTid(builder, op.a.idx))
        FBMulNode.AddB(builder, CreateTid(builder, op.b.idx))
        FBMulNode.AddOut(builder, CreateTid(builder, op.out.idx))
        offset = FBMulNode.End(builder)
        return offset, FBOpNode.OpNode.MulNode

    def _build_Conv1DNode(
        self, builder: flatbuffers.Builder, op: Conv1DNode
    ) -> Tuple[int, int]:
        """Auto-generated builder for Conv1DNode."""
        from executorch.backends.apple.mlx.serialization._generated import (
            Conv1DNode as FBConv1DNode,
            OpNode as FBOpNode,
        )
        from executorch.backends.apple.mlx.serialization._generated.mlx_delegate.Tid import CreateTid
        from executorch.backends.apple.mlx.serialization._generated.mlx_delegate.Vid import CreateVid

        FBConv1DNode.Start(builder)
        FBConv1DNode.AddX(builder, CreateTid(builder, op.x.idx))
        FBConv1DNode.AddW(builder, CreateTid(builder, op.w.idx))
        FBConv1DNode.AddOut(builder, CreateTid(builder, op.out.idx))
        FBConv1DNode.AddStride(builder, op.stride)
        FBConv1DNode.AddPadding(builder, op.padding)
        FBConv1DNode.AddDilation(builder, op.dilation)
        FBConv1DNode.AddGroups(builder, op.groups)
        offset = FBConv1DNode.End(builder)
        return offset, FBOpNode.OpNode.Conv1DNode

    def _build_GeluNode(
        self, builder: flatbuffers.Builder, op: GeluNode
    ) -> Tuple[int, int]:
        """Auto-generated builder for GeluNode."""
        from executorch.backends.apple.mlx.serialization._generated import (
            GeluNode as FBGeluNode,
            OpNode as FBOpNode,
        )
        from executorch.backends.apple.mlx.serialization._generated.mlx_delegate.Tid import CreateTid
        from executorch.backends.apple.mlx.serialization._generated.mlx_delegate.Vid import CreateVid

        FBGeluNode.Start(builder)
        FBGeluNode.AddX(builder, CreateTid(builder, op.x.idx))
        FBGeluNode.AddOut(builder, CreateTid(builder, op.out.idx))
        offset = FBGeluNode.End(builder)
        return offset, FBOpNode.OpNode.GeluNode

    def _build_ARangeNode(
        self, builder: flatbuffers.Builder, op: ARangeNode
    ) -> Tuple[int, int]:
        """Auto-generated builder for ARangeNode."""
        from executorch.backends.apple.mlx.serialization._generated import (
            ARangeNode as FBARangeNode,
            OpNode as FBOpNode,
        )
        from executorch.backends.apple.mlx.serialization._generated.mlx_delegate.Tid import CreateTid
        from executorch.backends.apple.mlx.serialization._generated.mlx_delegate.Vid import CreateVid

        FBARangeNode.Start(builder)
        FBARangeNode.AddOut(builder, CreateTid(builder, op.out.idx))
        FBARangeNode.AddStart(builder, op.start)
        FBARangeNode.AddStop(builder, op.stop)
        FBARangeNode.AddStep(builder, op.step)
        if op.dtype is not None:
            FBARangeNode.AddDtype(builder, op.dtype)
            FBARangeNode.AddDtypeIsSet(builder, True)
        offset = FBARangeNode.End(builder)
        return offset, FBOpNode.OpNode.ARangeNode

    def _build_SiluNode(
        self, builder: flatbuffers.Builder, op: SiluNode
    ) -> Tuple[int, int]:
        """Auto-generated builder for SiluNode."""
        from executorch.backends.apple.mlx.serialization._generated import (
            SiluNode as FBSiluNode,
            OpNode as FBOpNode,
        )
        from executorch.backends.apple.mlx.serialization._generated.mlx_delegate.Tid import CreateTid
        from executorch.backends.apple.mlx.serialization._generated.mlx_delegate.Vid import CreateVid

        FBSiluNode.Start(builder)
        FBSiluNode.AddX(builder, CreateTid(builder, op.x.idx))
        FBSiluNode.AddOut(builder, CreateTid(builder, op.out.idx))
        offset = FBSiluNode.End(builder)
        return offset, FBOpNode.OpNode.SiluNode

    def _build_ReshapeNode(
        self, builder: flatbuffers.Builder, op: ReshapeNode
    ) -> Tuple[int, int]:
        """Auto-generated builder for ReshapeNode."""
        from executorch.backends.apple.mlx.serialization._generated import (
            ReshapeNode as FBReshapeNode,
            OpNode as FBOpNode,
        )
        from executorch.backends.apple.mlx.serialization._generated.mlx_delegate.Tid import CreateTid
        from executorch.backends.apple.mlx.serialization._generated.mlx_delegate.Vid import CreateVid

        shape_vec = self._build_int_or_vid_vector(builder, op.shape)

        FBReshapeNode.Start(builder)
        FBReshapeNode.AddX(builder, CreateTid(builder, op.x.idx))
        FBReshapeNode.AddOut(builder, CreateTid(builder, op.out.idx))
        FBReshapeNode.AddShape(builder, shape_vec)
        offset = FBReshapeNode.End(builder)
        return offset, FBOpNode.OpNode.ReshapeNode

    def _build_TransposeNode(
        self, builder: flatbuffers.Builder, op: TransposeNode
    ) -> Tuple[int, int]:
        """Auto-generated builder for TransposeNode."""
        from executorch.backends.apple.mlx.serialization._generated import (
            TransposeNode as FBTransposeNode,
            OpNode as FBOpNode,
        )
        from executorch.backends.apple.mlx.serialization._generated.mlx_delegate.Tid import CreateTid
        from executorch.backends.apple.mlx.serialization._generated.mlx_delegate.Vid import CreateVid

        perm_vec = _build_int_vector(builder, op.perm)

        FBTransposeNode.Start(builder)
        FBTransposeNode.AddX(builder, CreateTid(builder, op.x.idx))
        FBTransposeNode.AddOut(builder, CreateTid(builder, op.out.idx))
        FBTransposeNode.AddPerm(builder, perm_vec)
        offset = FBTransposeNode.End(builder)
        return offset, FBOpNode.OpNode.TransposeNode

    def _build_ContiguousNode(
        self, builder: flatbuffers.Builder, op: ContiguousNode
    ) -> Tuple[int, int]:
        """Auto-generated builder for ContiguousNode."""
        from executorch.backends.apple.mlx.serialization._generated import (
            ContiguousNode as FBContiguousNode,
            OpNode as FBOpNode,
        )
        from executorch.backends.apple.mlx.serialization._generated.mlx_delegate.Tid import CreateTid
        from executorch.backends.apple.mlx.serialization._generated.mlx_delegate.Vid import CreateVid

        FBContiguousNode.Start(builder)
        FBContiguousNode.AddX(builder, CreateTid(builder, op.x.idx))
        FBContiguousNode.AddOut(builder, CreateTid(builder, op.out.idx))
        offset = FBContiguousNode.End(builder)
        return offset, FBOpNode.OpNode.ContiguousNode

    def _build_IdCopyNode(
        self, builder: flatbuffers.Builder, op: IdCopyNode
    ) -> Tuple[int, int]:
        """Auto-generated builder for IdCopyNode."""
        from executorch.backends.apple.mlx.serialization._generated import (
            IdCopyNode as FBIdCopyNode,
            OpNode as FBOpNode,
        )
        from executorch.backends.apple.mlx.serialization._generated.mlx_delegate.Tid import CreateTid
        from executorch.backends.apple.mlx.serialization._generated.mlx_delegate.Vid import CreateVid

        FBIdCopyNode.Start(builder)
        FBIdCopyNode.AddX(builder, CreateTid(builder, op.x.idx))
        FBIdCopyNode.AddOut(builder, CreateTid(builder, op.out.idx))
        offset = FBIdCopyNode.End(builder)
        return offset, FBOpNode.OpNode.IdCopyNode

    def _build_GatherNode(
        self, builder: flatbuffers.Builder, op: GatherNode
    ) -> Tuple[int, int]:
        """Auto-generated builder for GatherNode."""
        from executorch.backends.apple.mlx.serialization._generated import (
            GatherNode as FBGatherNode,
            OpNode as FBOpNode,
        )
        from executorch.backends.apple.mlx.serialization._generated.mlx_delegate.Tid import CreateTid
        from executorch.backends.apple.mlx.serialization._generated.mlx_delegate.Vid import CreateVid

        FBGatherNode.Start(builder)
        FBGatherNode.AddTable_(builder, CreateTid(builder, op.table_.idx))
        FBGatherNode.AddIds(builder, CreateTid(builder, op.ids.idx))
        FBGatherNode.AddOut(builder, CreateTid(builder, op.out.idx))
        offset = FBGatherNode.End(builder)
        return offset, FBOpNode.OpNode.GatherNode

    def _build_SliceNode(
        self, builder: flatbuffers.Builder, op: SliceNode
    ) -> Tuple[int, int]:
        """Auto-generated builder for SliceNode."""
        from executorch.backends.apple.mlx.serialization._generated import (
            SliceNode as FBSliceNode,
            OpNode as FBOpNode,
        )
        from executorch.backends.apple.mlx.serialization._generated.mlx_delegate.Tid import CreateTid
        from executorch.backends.apple.mlx.serialization._generated.mlx_delegate.Vid import CreateVid

        axis_off = self._build_int_or_vid(builder, op.axis)
        start_off = self._build_int_or_vid(builder, op.start)
        end_off = self._build_int_or_vid(builder, op.end)

        FBSliceNode.Start(builder)
        FBSliceNode.AddX(builder, CreateTid(builder, op.x.idx))
        FBSliceNode.AddOut(builder, CreateTid(builder, op.out.idx))
        FBSliceNode.AddAxis(builder, axis_off)
        FBSliceNode.AddStart(builder, start_off)
        FBSliceNode.AddEnd(builder, end_off)
        offset = FBSliceNode.End(builder)
        return offset, FBOpNode.OpNode.SliceNode

    def _build_CastNode(
        self, builder: flatbuffers.Builder, op: CastNode
    ) -> Tuple[int, int]:
        """Auto-generated builder for CastNode."""
        from executorch.backends.apple.mlx.serialization._generated import (
            CastNode as FBCastNode,
            OpNode as FBOpNode,
        )
        from executorch.backends.apple.mlx.serialization._generated.mlx_delegate.Tid import CreateTid
        from executorch.backends.apple.mlx.serialization._generated.mlx_delegate.Vid import CreateVid

        FBCastNode.Start(builder)
        FBCastNode.AddX(builder, CreateTid(builder, op.x.idx))
        FBCastNode.AddOut(builder, CreateTid(builder, op.out.idx))
        FBCastNode.AddDtype(builder, op.dtype)
        offset = FBCastNode.End(builder)
        return offset, FBOpNode.OpNode.CastNode

    def _build_QuantizedLinearNode(
        self, builder: flatbuffers.Builder, op: QuantizedLinearNode
    ) -> Tuple[int, int]:
        """Auto-generated builder for QuantizedLinearNode."""
        from executorch.backends.apple.mlx.serialization._generated import (
            QuantizedLinearNode as FBQuantizedLinearNode,
            OpNode as FBOpNode,
        )
        from executorch.backends.apple.mlx.serialization._generated.mlx_delegate.Tid import CreateTid
        from executorch.backends.apple.mlx.serialization._generated.mlx_delegate.Vid import CreateVid

        mode_off = builder.CreateString(op.mode)

        FBQuantizedLinearNode.Start(builder)
        FBQuantizedLinearNode.AddX(builder, CreateTid(builder, op.x.idx))
        FBQuantizedLinearNode.AddW(builder, CreateTid(builder, op.w.idx))
        FBQuantizedLinearNode.AddScales(builder, CreateTid(builder, op.scales.idx))
        FBQuantizedLinearNode.AddOut(builder, CreateTid(builder, op.out.idx))
        if op.biases is not None:
            FBQuantizedLinearNode.AddBiases(builder, CreateTid(builder, op.biases.idx))
        if op.bias is not None:
            FBQuantizedLinearNode.AddBias(builder, CreateTid(builder, op.bias.idx))
        FBQuantizedLinearNode.AddGroupSize(builder, op.group_size)
        FBQuantizedLinearNode.AddBits(builder, op.bits)
        FBQuantizedLinearNode.AddMode(builder, mode_off)
        FBQuantizedLinearNode.AddOutDtype(builder, op.out_dtype)
        offset = FBQuantizedLinearNode.End(builder)
        return offset, FBOpNode.OpNode.QuantizedLinearNode

    def _build_ConcatNode(
        self, builder: flatbuffers.Builder, op: ConcatNode
    ) -> Tuple[int, int]:
        """Auto-generated builder for ConcatNode."""
        from executorch.backends.apple.mlx.serialization._generated import (
            ConcatNode as FBConcatNode,
            OpNode as FBOpNode,
        )
        from executorch.backends.apple.mlx.serialization._generated.mlx_delegate.Tid import CreateTid
        from executorch.backends.apple.mlx.serialization._generated.mlx_delegate.Vid import CreateVid

        FBConcatNode.Start(builder)
        FBConcatNode.AddA(builder, CreateTid(builder, op.a.idx))
        FBConcatNode.AddB(builder, CreateTid(builder, op.b.idx))
        FBConcatNode.AddOut(builder, CreateTid(builder, op.out.idx))
        FBConcatNode.AddAxis(builder, op.axis)
        offset = FBConcatNode.End(builder)
        return offset, FBOpNode.OpNode.ConcatNode

    def _build_FullNode(
        self, builder: flatbuffers.Builder, op: FullNode
    ) -> Tuple[int, int]:
        """Auto-generated builder for FullNode."""
        from executorch.backends.apple.mlx.serialization._generated import (
            FullNode as FBFullNode,
            OpNode as FBOpNode,
        )
        from executorch.backends.apple.mlx.serialization._generated.mlx_delegate.Tid import CreateTid
        from executorch.backends.apple.mlx.serialization._generated.mlx_delegate.Vid import CreateVid

        shape_vec = _build_int_vector(builder, op.shape)

        FBFullNode.Start(builder)
        FBFullNode.AddOut(builder, CreateTid(builder, op.out.idx))
        FBFullNode.AddShape(builder, shape_vec)
        FBFullNode.AddV(builder, op.v)
        FBFullNode.AddDtype(builder, op.dtype)
        offset = FBFullNode.End(builder)
        return offset, FBOpNode.OpNode.FullNode

    def _build_ZerosNode(
        self, builder: flatbuffers.Builder, op: ZerosNode
    ) -> Tuple[int, int]:
        """Auto-generated builder for ZerosNode."""
        from executorch.backends.apple.mlx.serialization._generated import (
            ZerosNode as FBZerosNode,
            OpNode as FBOpNode,
        )
        from executorch.backends.apple.mlx.serialization._generated.mlx_delegate.Tid import CreateTid
        from executorch.backends.apple.mlx.serialization._generated.mlx_delegate.Vid import CreateVid

        shape_vec = _build_int_vector(builder, op.shape)

        FBZerosNode.Start(builder)
        FBZerosNode.AddOut(builder, CreateTid(builder, op.out.idx))
        FBZerosNode.AddShape(builder, shape_vec)
        FBZerosNode.AddDtype(builder, op.dtype)
        offset = FBZerosNode.End(builder)
        return offset, FBOpNode.OpNode.ZerosNode

    def _build_OnesNode(
        self, builder: flatbuffers.Builder, op: OnesNode
    ) -> Tuple[int, int]:
        """Auto-generated builder for OnesNode."""
        from executorch.backends.apple.mlx.serialization._generated import (
            OnesNode as FBOnesNode,
            OpNode as FBOpNode,
        )
        from executorch.backends.apple.mlx.serialization._generated.mlx_delegate.Tid import CreateTid
        from executorch.backends.apple.mlx.serialization._generated.mlx_delegate.Vid import CreateVid

        shape_vec = _build_int_vector(builder, op.shape)

        FBOnesNode.Start(builder)
        FBOnesNode.AddOut(builder, CreateTid(builder, op.out.idx))
        FBOnesNode.AddShape(builder, shape_vec)
        FBOnesNode.AddDtype(builder, op.dtype)
        offset = FBOnesNode.End(builder)
        return offset, FBOpNode.OpNode.OnesNode

    def _build_ArgmaxNode(
        self, builder: flatbuffers.Builder, op: ArgmaxNode
    ) -> Tuple[int, int]:
        """Auto-generated builder for ArgmaxNode."""
        from executorch.backends.apple.mlx.serialization._generated import (
            ArgmaxNode as FBArgmaxNode,
            OpNode as FBOpNode,
        )
        from executorch.backends.apple.mlx.serialization._generated.mlx_delegate.Tid import CreateTid
        from executorch.backends.apple.mlx.serialization._generated.mlx_delegate.Vid import CreateVid

        FBArgmaxNode.Start(builder)
        FBArgmaxNode.AddX(builder, CreateTid(builder, op.x.idx))
        FBArgmaxNode.AddOut(builder, CreateTid(builder, op.out.idx))
        FBArgmaxNode.AddAxis(builder, op.axis)
        offset = FBArgmaxNode.End(builder)
        return offset, FBOpNode.OpNode.ArgmaxNode

    def _build_SliceUpdateNode(
        self, builder: flatbuffers.Builder, op: SliceUpdateNode
    ) -> Tuple[int, int]:
        """Auto-generated builder for SliceUpdateNode."""
        from executorch.backends.apple.mlx.serialization._generated import (
            SliceUpdateNode as FBSliceUpdateNode,
            OpNode as FBOpNode,
        )
        from executorch.backends.apple.mlx.serialization._generated.mlx_delegate.Tid import CreateTid
        from executorch.backends.apple.mlx.serialization._generated.mlx_delegate.Vid import CreateVid

        axis_off = self._build_int_or_vid(builder, op.axis)
        start_off = self._build_int_or_vid(builder, op.start)
        stop_off = self._build_int_or_vid(builder, op.stop)

        FBSliceUpdateNode.Start(builder)
        FBSliceUpdateNode.AddDst(builder, CreateTid(builder, op.dst.idx))
        FBSliceUpdateNode.AddUpdate(builder, CreateTid(builder, op.update.idx))
        FBSliceUpdateNode.AddAxis(builder, axis_off)
        FBSliceUpdateNode.AddStart(builder, start_off)
        FBSliceUpdateNode.AddStop(builder, stop_off)
        offset = FBSliceUpdateNode.End(builder)
        return offset, FBOpNode.OpNode.SliceUpdateNode

    def _build_QuantizedGatherNode(
        self, builder: flatbuffers.Builder, op: QuantizedGatherNode
    ) -> Tuple[int, int]:
        """Auto-generated builder for QuantizedGatherNode."""
        from executorch.backends.apple.mlx.serialization._generated import (
            QuantizedGatherNode as FBQuantizedGatherNode,
            OpNode as FBOpNode,
        )
        from executorch.backends.apple.mlx.serialization._generated.mlx_delegate.Tid import CreateTid
        from executorch.backends.apple.mlx.serialization._generated.mlx_delegate.Vid import CreateVid

        mode_off = builder.CreateString(op.mode)

        FBQuantizedGatherNode.Start(builder)
        FBQuantizedGatherNode.AddTableQ(builder, CreateTid(builder, op.table_q.idx))
        FBQuantizedGatherNode.AddScales(builder, CreateTid(builder, op.scales.idx))
        FBQuantizedGatherNode.AddIds(builder, CreateTid(builder, op.ids.idx))
        FBQuantizedGatherNode.AddOut(builder, CreateTid(builder, op.out.idx))
        if op.biases is not None:
            FBQuantizedGatherNode.AddBiases(builder, CreateTid(builder, op.biases.idx))
        FBQuantizedGatherNode.AddGroupSize(builder, op.group_size)
        FBQuantizedGatherNode.AddBits(builder, op.bits)
        FBQuantizedGatherNode.AddMode(builder, mode_off)
        FBQuantizedGatherNode.AddOutDtype(builder, op.out_dtype)
        offset = FBQuantizedGatherNode.End(builder)
        return offset, FBOpNode.OpNode.QuantizedGatherNode
