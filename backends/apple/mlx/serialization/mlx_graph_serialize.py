#
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
#

"""
Serialization utilities for MLX delegate.

Converts MLXGraph dataclasses to FlatBuffer binary format.

Constants are NOT embedded in the delegate payload - they are provided by
ExecuTorch via named_data_map at runtime.

Layout:
    [Header: 24 bytes]
        - Padding: 4 bytes (zeros)
        - Magic: 4 bytes ("MLX0")
        - Reserved: 16 bytes (zeros, for future use)
    [FlatBuffer payload]
"""

from __future__ import annotations

import struct
from typing import Any, List, Tuple

import flatbuffers

# Import auto-generated serializers
from executorch.backends.apple.mlx.serialization._generated_serializers import (
    GeneratedOpBuilders,
)
from executorch.backends.apple.mlx.serialization.mlx_graph_schema import (  # noqa: F401
    AddIntNode,
    AddNode,
    ARangeNode,
    ArgmaxNode,
    AsTypeNode,
    ConcatenateNode,
    ContiguousNode,
    Conv1DNode,
    ExpandDimsNode,
    FloatOrVid,
    FloorDivideIntNode,
    FullNode,
    GatherNode,
    GeluNode,
    IdCopyNode,
    Instruction,
    IntOrVid,
    ItemIntNode,
    LayerNormNode,
    LinearNode,
    MLXGraph,
    MultiplyIntNode,
    MultiplyNode,
    NamedSlot,
    NoopNode,
    OpNodeUnion,
    QuantizedGatherNode,
    QuantizedLinearNode,
    ReshapeNode,
    RMSNormNode,
    RopeNode,
    SdpaNode,
    SiluNode,
    SliceNode,
    SliceUpdateNode,
    SlotType,
    SlotVariant,
    SubtractIntNode,
    SymSizeNode,
    TakeAlongAxisNode,
    TensorMeta,
    Tid,
    TileNode,
    TransposeNode,
    Vid,
)
from executorch.exir._serialize._program import Cord

# =============================================================================
# Constants
# =============================================================================

HEADER_LENGTH = 24
MAGIC = b"MLX0"
ALIGNMENT = 16


# =============================================================================
# FlatBuffer Builder Helpers
# =============================================================================


def _padding_required(offset: int, alignment: int) -> int:
    """Returns padding needed to align offset to alignment boundary."""
    remainder = offset % alignment
    return (alignment - remainder) % alignment


def _build_tid(builder: flatbuffers.Builder, tid: Tid) -> int:
    """Build a Tid struct (inline, returns 0 - structs are written inline)."""
    # Structs in FlatBuffers are written inline, not as offsets
    # We'll handle this in the parent table
    return tid.idx


def _build_vid(builder: flatbuffers.Builder, vid: Vid) -> int:
    """Build a Vid struct (inline, returns 0 - structs are written inline)."""
    return vid.idx


def _build_int_or_vid(builder: flatbuffers.Builder, iov: IntOrVid) -> int:
    """Build an IntOrVid table."""
    # Import the MODULE (not class) to access builder functions
    from executorch.backends.apple.mlx.serialization._generated.mlx_delegate import (
        IntOrVid as FBIntOrVidModule,
    )
    from executorch.backends.apple.mlx.serialization._generated.mlx_delegate.Vid import (
        CreateVid,
    )

    FBIntOrVidModule.Start(builder)
    FBIntOrVidModule.AddLiteral(builder, iov.literal)
    FBIntOrVidModule.AddIsVid(builder, iov.is_vid)
    if iov.vid is not None:
        # Vid is an inline struct - must be added last for proper FlatBuffer layout
        FBIntOrVidModule.AddVid(builder, CreateVid(builder, iov.vid.idx))
    return FBIntOrVidModule.End(builder)


def _build_string(builder: flatbuffers.Builder, s: str) -> int:
    """Build a string and return its offset."""
    return builder.CreateString(s)


def _build_int_vector(builder: flatbuffers.Builder, vec: List[int]) -> int:
    """Build a vector of int32 and return its offset."""
    # FlatBuffers vectors must be created before the table that contains them
    builder.StartVector(4, len(vec), 4)  # elem_size=4, num_elems, alignment
    for v in reversed(vec):
        builder.PrependInt32(v)
    return builder.EndVector()


# =============================================================================
# Serialization Cord Builder
# =============================================================================


class MLXGraphSerializer(GeneratedOpBuilders):
    """
    Serializes MLXGraph to bytes with separate constant data segment.

    Inherits auto-generated op builders from GeneratedOpBuilders mixin.
    """

    def __init__(self, graph: MLXGraph, constant_data: bytes = b""):
        self.graph = graph
        self.constant_data = constant_data

    def serialize(self) -> bytes:
        """
        Serialize the graph to bytes.

        Returns:
            Complete serialized payload with header, flatbuffer, and data segment.
        """
        # Build FlatBuffer
        fb_bytes = self._build_flatbuffer()

        # Calculate offsets
        data_segment_offset = HEADER_LENGTH + len(fb_bytes)
        padding_len = _padding_required(data_segment_offset, ALIGNMENT)
        data_segment_offset += padding_len
        data_segment_size = len(self.constant_data)

        # Build header
        header = (
            b"\x00\x00\x00\x00"  # 4 bytes padding
            + MAGIC  # 4 bytes magic
            + struct.pack("<Q", data_segment_offset)  # 8 bytes offset
            + struct.pack("<Q", data_segment_size)  # 8 bytes size
        )
        assert len(header) == HEADER_LENGTH

        # Combine all parts
        result = Cord()
        result.append(header)
        result.append(fb_bytes)
        if padding_len > 0:
            result.append(b"\x00" * padding_len)
        result.append(self.constant_data)

        return bytes(result)

    def _build_flatbuffer(self) -> bytes:
        """Build the FlatBuffer portion of the payload."""
        builder = flatbuffers.Builder(4096)

        # Build all components bottom-up (FlatBuffers requirement)

        # 1. Build instruction chains
        chain_offsets = []
        for chain in self.graph.instruction_chains:
            instr_offsets = []
            for instr in chain.instructions:
                instr_offsets.append(self._build_instruction(builder, instr))
            instr_vec = self._build_offset_vector(builder, instr_offsets)

            from executorch.backends.apple.mlx.serialization._generated.mlx_delegate import (
                InstructionChain as FBInstructionChainModule,
            )

            FBInstructionChainModule.Start(builder)
            FBInstructionChainModule.AddInstructions(builder, instr_vec)
            chain_offsets.append(FBInstructionChainModule.End(builder))

        chains_vec = self._build_offset_vector(builder, chain_offsets)

        # 2. Build I/O maps
        input_map_vec = self._build_slot_variant_vector(builder, self.graph.input_map)
        output_map_vec = self._build_slot_variant_vector(builder, self.graph.output_map)
        mutable_buffer_map_vec = self._build_slot_variant_vector(
            builder, self.graph.mutable_buffer_map
        )

        # 3. Build named slots
        named_slots_offsets = []
        for ns in self.graph.named_slots:
            named_slots_offsets.append(self._build_named_slot(builder, ns))
        named_slots_vec = self._build_offset_vector(builder, named_slots_offsets)

        # 4. Build tensor metadata
        tensor_meta_offsets = []
        for tm in self.graph.tensor_meta:
            if tm is not None:
                tensor_meta_offsets.append(self._build_tensor_meta(builder, tm))
            else:
                tensor_meta_offsets.append(0)  # null
        tensor_meta_vec = self._build_offset_vector(builder, tensor_meta_offsets)

        # 5. Build version string (must be created before the table that uses it)
        version_off = builder.CreateString(self.graph.version)

        # 6. Build the root MLXGraph table
        from executorch.backends.apple.mlx.serialization._generated.mlx_delegate import (
            MLXGraph as FBMLXGraphModule,
        )

        FBMLXGraphModule.Start(builder)
        FBMLXGraphModule.AddVersion(builder, version_off)
        FBMLXGraphModule.AddNumConstantTensors(builder, self.graph.num_constant_tensors)
        FBMLXGraphModule.AddNumInputTensors(builder, self.graph.num_input_tensors)
        FBMLXGraphModule.AddNumOutputTensors(builder, self.graph.num_output_tensors)
        FBMLXGraphModule.AddNumMutableBufferTensors(
            builder, self.graph.num_mutable_buffer_tensors
        )
        FBMLXGraphModule.AddNumTempTensors(builder, self.graph.num_temp_tensors)
        FBMLXGraphModule.AddNumValues(builder, self.graph.num_values)
        FBMLXGraphModule.AddInstructionChains(builder, chains_vec)
        FBMLXGraphModule.AddMainChainIdx(builder, self.graph.main_chain_idx)
        FBMLXGraphModule.AddInitChainIdx(builder, self.graph.init_chain_idx)
        FBMLXGraphModule.AddInputMap(builder, input_map_vec)
        FBMLXGraphModule.AddOutputMap(builder, output_map_vec)
        FBMLXGraphModule.AddMutableBufferMap(builder, mutable_buffer_map_vec)
        FBMLXGraphModule.AddNamedSlots(builder, named_slots_vec)
        FBMLXGraphModule.AddTensorMeta(builder, tensor_meta_vec)
        root = FBMLXGraphModule.End(builder)

        builder.Finish(root)
        return bytes(builder.Output())

    def _build_instruction(
        self, builder: flatbuffers.Builder, instr: Instruction
    ) -> int:
        """Build an Instruction table containing an op."""
        op_offset, op_type = self._build_op_node(builder, instr.op)

        from executorch.backends.apple.mlx.serialization._generated.mlx_delegate import (
            Instruction as FBInstructionModule,
        )

        FBInstructionModule.Start(builder)
        FBInstructionModule.AddOpType(builder, op_type)
        FBInstructionModule.AddOp(builder, op_offset)
        return FBInstructionModule.End(builder)

    def _build_op_node(
        self, builder: flatbuffers.Builder, op: OpNodeUnion
    ) -> Tuple[int, int]:
        """
        Build an op node and return (offset, union_type).

        This is the main dispatch for all op types.
        """
        # Map Python class to FlatBuffer union type and builder
        # This would ideally be auto-generated

        op_type = type(op).__name__
        builder_method = getattr(self, f"_build_{op_type}", None)

        if builder_method is None:
            raise NotImplementedError(f"No builder for op type: {op_type}")

        return builder_method(builder, op)

    # =========================================================================
    # Op Node Builders - From GeneratedOpBuilders mixin
    # =========================================================================
    # Individual op builders are inherited from GeneratedOpBuilders.
    # Only override here if custom behavior is needed.

    def _build_offset_vector(
        self, builder: flatbuffers.Builder, offsets: List[int]
    ) -> int:
        """Build a vector of table offsets."""
        builder.StartVector(4, len(offsets), 4)
        for off in reversed(offsets):
            builder.PrependUOffsetTRelative(off)
        return builder.EndVector()

    def _build_slot_variant_vector(
        self, builder: flatbuffers.Builder, slots: List[SlotVariant]
    ) -> int:
        """Build a vector of SlotVariant tables."""
        offsets = []
        for slot in slots:
            offsets.append(self._build_slot_variant(builder, slot))
        return self._build_offset_vector(builder, offsets)

    def _build_slot_variant(
        self, builder: flatbuffers.Builder, slot: SlotVariant
    ) -> int:
        """Build a SlotVariant table."""
        from executorch.backends.apple.mlx.serialization._generated.mlx_delegate import (
            SlotVariant as FBSlotVariantModule,
        )

        FBSlotVariantModule.Start(builder)
        FBSlotVariantModule.AddIdx(builder, slot.idx)
        FBSlotVariantModule.AddSlotType(builder, slot.slot_type)
        return FBSlotVariantModule.End(builder)

    def _build_named_slot(self, builder: flatbuffers.Builder, ns: NamedSlot) -> int:
        """Build a NamedSlot table."""
        name_off = builder.CreateString(ns.name)
        slot_off = self._build_slot_variant(builder, ns.slot)

        from executorch.backends.apple.mlx.serialization._generated.mlx_delegate import (
            NamedSlot as FBNamedSlotModule,
        )

        FBNamedSlotModule.Start(builder)
        FBNamedSlotModule.AddName(builder, name_off)
        FBNamedSlotModule.AddSlot(builder, slot_off)
        return FBNamedSlotModule.End(builder)

    def _build_tensor_meta(self, builder: flatbuffers.Builder, tm: TensorMeta) -> int:
        """Build a TensorMeta table."""
        # Shape is now a vector of IntOrVid tables
        shape_offsets = []
        for dim in tm.shape:
            shape_offsets.append(_build_int_or_vid(builder, dim))
        # Build vector of table offsets
        builder.StartVector(4, len(shape_offsets), 4)
        for off in reversed(shape_offsets):
            builder.PrependUOffsetTRelative(off)
        shape_vec = builder.EndVector()

        # Build dim_order vector (uint8)
        dim_order_vec = 0
        if tm.dim_order:
            builder.StartVector(1, len(tm.dim_order), 1)  # elem_size=1 for uint8
            for d in reversed(tm.dim_order):
                builder.PrependUint8(d)
            dim_order_vec = builder.EndVector()

        from executorch.backends.apple.mlx.serialization._generated.mlx_delegate import (
            TensorMeta as FBTensorMetaModule,
        )

        FBTensorMetaModule.Start(builder)
        FBTensorMetaModule.AddShape(builder, shape_vec)
        if tm.scalar_type is not None:
            FBTensorMetaModule.AddScalarType(builder, tm.scalar_type)
        if dim_order_vec:
            FBTensorMetaModule.AddDimOrder(builder, dim_order_vec)
        return FBTensorMetaModule.End(builder)


# =============================================================================
# Convenience function
# =============================================================================


def serialize_mlx_graph(graph: MLXGraph, constant_data: bytes = b"") -> bytes:
    """
    Serialize an MLXGraph to bytes.

    Args:
        graph: The MLXGraph to serialize.
        constant_data: Raw bytes for constant tensors.

    Returns:
        Serialized bytes with header, flatbuffer, and data segment.
    """
    serializer = MLXGraphSerializer(graph, constant_data)
    return serializer.serialize()


# =============================================================================
# Deserialization (for debugging / JSON dump)
# =============================================================================


def parse_header(data: bytes) -> Tuple[int, int, int, int]:
    """
    Parse the MLX delegate header.

    Returns:
        (flatbuffer_offset, flatbuffer_size, data_segment_offset, data_segment_size)
    """
    if len(data) < HEADER_LENGTH:
        raise ValueError(f"Data too short: {len(data)} < {HEADER_LENGTH}")

    magic = data[4:8]
    if magic != MAGIC:
        raise ValueError(f"Invalid magic: {magic!r} (expected {MAGIC!r})")

    data_segment_offset = struct.unpack("<Q", data[8:16])[0]
    data_segment_size = struct.unpack("<Q", data[16:24])[0]

    flatbuffer_offset = HEADER_LENGTH
    flatbuffer_size = data_segment_offset - HEADER_LENGTH

    return flatbuffer_offset, flatbuffer_size, data_segment_offset, data_segment_size


def deserialize_to_json(data: bytes) -> dict:
    """
    Deserialize MLX delegate payload to a JSON-compatible dict.

    Useful for debugging - extracts the FlatBuffer and dumps it as JSON.
    """
    fb_off, fb_size, ds_off, ds_size = parse_header(data)

    # Extract FlatBuffer portion
    fb_data = data[fb_off : fb_off + fb_size]

    # Parse using generated FlatBuffer code
    from executorch.backends.apple.mlx.serialization._generated.mlx_delegate.MLXGraph import (
        MLXGraph as FBMLXGraphClass,
    )

    graph = FBMLXGraphClass.GetRootAs(fb_data, 0)

    # Convert to dict (recursive)
    result = _fb_to_dict(graph)
    result["_constant_segment_size"] = ds_size

    return result


def _fb_to_dict(obj: Any) -> Any:
    """Recursively convert FlatBuffer object to dict."""
    if obj is None:
        return None
    if isinstance(obj, (int, float, str, bool, bytes)):
        return obj
    if isinstance(obj, (list, tuple)):
        return [_fb_to_dict(item) for item in obj]

    # FlatBuffer object - extract fields
    result = {}
    for attr in dir(obj):
        if attr.startswith("_") or attr[0].islower():
            continue
        try:
            value = getattr(obj, attr)()
            result[attr] = _fb_to_dict(value)
        except (TypeError, AttributeError):
            pass

    return result
