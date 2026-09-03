# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# Copyright 2025 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import importlib.resources as _resources
import json
import math
import os
import tempfile
from dataclasses import dataclass
from typing import BinaryIO, ClassVar, Dict, List, Literal, Optional, Union

import executorch.extension.flat_tensor.serialize as serialize_package

import torch

from executorch.exir._serialize._cord import Cord
from executorch.exir._serialize._dataclass import _DataclassEncoder, _json_to_dataclass
from executorch.exir._serialize._flatbuffer import _flatc_compile, _flatc_decompile
from executorch.exir._serialize._named_data_store import (
    _tensor_to_bytes,
    NamedDataStoreOutput,
)
from executorch.exir._serialize._program import _insert_flatbuffer_header
from executorch.exir._serialize.data_serializer import (
    DataEntry,
    DataPayload,
    DataSerializer,
)
from executorch.exir._serialize.padding import aligned_size, pad_to, padding_required
from executorch.exir.tensor import get_scalar_type
from executorch.exir.tensor_layout import TensorLayout
from executorch.extension.flat_tensor.serialize.flat_tensor_schema import (
    DataSegment,
    FlatTensor,
    NamedData,
)

# Byte order of numbers written to flat tensor headers. Always little-endian
# regardless of the host system, since all commonly-used modern CPUs are little
# endian.
_HEADER_BYTEORDER: Literal["little"] = "little"

# Alignment of the flatbuffer (after the header).
_FLATBUFFER_ALIGNMENT: int = 16

# Current version. Keep in sync with c++ version number in serialize, and with
# FlatTensorDataMap::kMaxSupportedSchemaVersion in
# //executorch/extension/flat_tensor/flat_tensor_data_map.h, which is the
# highest version the runtime agrees to load.
_FLAT_TENSOR_VERSION: int = 0


def _serialize_to_flatbuffer(flat_tensor: FlatTensor) -> Cord:
    """Serializes a FlatTensor to a flatbuffer and returns the serialized data."""
    flat_tensor_json = json.dumps(flat_tensor, cls=_DataclassEncoder)
    with tempfile.TemporaryDirectory() as d:
        schema_path = os.path.join(d, "flat_tensor.fbs")
        with open(schema_path, "wb") as schema_file:
            schema_file.write(
                _resources.read_binary(serialize_package, "flat_tensor.fbs")
            )
        scalar_type_path = os.path.join(d, "scalar_type.fbs")
        with open(scalar_type_path, "wb") as scalar_type_file:
            scalar_type_file.write(
                _resources.read_binary(serialize_package, "scalar_type.fbs")
            )
        json_path = os.path.join(d, "flat_tensor.json")
        with open(json_path, "wb") as json_file:
            json_file.write(flat_tensor_json.encode("ascii"))

        _flatc_compile(d, schema_path, json_path)
        output_path = os.path.join(d, "flat_tensor.ptd")
        with open(output_path, "rb") as output_file:
            return Cord(output_file.read())


def _deserialize_to_flat_tensor(flatbuffer: bytes) -> FlatTensor:
    """Deserializes a flatbuffer to a FlatTensor and returns the dataclass."""
    with tempfile.TemporaryDirectory() as d:
        schema_path = os.path.join(d, "flat_tensor.fbs")
        with open(schema_path, "wb") as schema_file:
            schema_file.write(
                _resources.read_binary(serialize_package, "flat_tensor.fbs")
            )

        scalar_type_path = os.path.join(d, "scalar_type.fbs")
        with open(scalar_type_path, "wb") as scalar_type_file:
            scalar_type_file.write(
                _resources.read_binary(serialize_package, "scalar_type.fbs")
            )

        bin_path = os.path.join(d, "flat_tensor.bin")
        with open(bin_path, "wb") as bin_file:
            bin_file.write(flatbuffer)

        _flatc_decompile(d, schema_path, bin_path, ["--raw-binary"])

        json_path = os.path.join(d, "flat_tensor.json")
        with open(json_path, "rb") as output_file:
            return _json_to_dataclass(json.load(output_file), cls=FlatTensor)


@dataclass
class FlatTensorConfig:
    segment_alignment: int = 128


@dataclass
class FlatTensorHeader:
    # Class constants.
    # The magic bytes that should be at the beginning of the header.
    # This should be in sync with the magic in
    # executorch/extension/flat_tensor/serialize/flat_tensor_header.h
    EXPECTED_MAGIC: ClassVar[bytes] = b"FH01"
    EXPECTED_LENGTH: ClassVar[int] = (
        # Header magic
        4
        # Header length
        + 4
        # Flatbuffer offset
        + 8
        # Flatbuffer data size
        + 8
        # Segment base offset
        + 8
        # Data size
        + 8
    )

    # Instance attributes. @dataclass will turn these into ctor args.

    # Offset to the start of the flatbuffer data, in bytes.
    flatbuffer_offset: int
    # The size of the serialized data in bytes.
    flatbuffer_size: int
    # Offset to the start of the first segment, or zero if there
    # are no segments.
    segment_base_offset: int
    # Size of all the segment data, in bytes.
    segment_data_size: int

    # The magic bytes read from or to be written to the binary header.
    magic: bytes = EXPECTED_MAGIC
    # The header length, in bytes, read from or to be written to the binary
    # header.
    length: int = EXPECTED_LENGTH

    @staticmethod
    def from_bytes(data: bytes) -> "FlatTensorHeader":
        """Tries to read an flat_tensor header from the provided data.

        Does not validate that the header is well-formed. Callers should
        use is_valid().

        Args:
            data: The data to read from.
        Returns:
            The contents of the flat_tensor header.
        Raises:
            ValueError: If not enough data is provided.
        """
        if len(data) < FlatTensorHeader.EXPECTED_LENGTH:
            raise ValueError(
                f"Not enough data for flat_tensor header: {len(data)} "
                + f"< {FlatTensorHeader.EXPECTED_LENGTH}"
            )

        return FlatTensorHeader(
            magic=data[0:4],
            length=int.from_bytes(data[4:8], byteorder=_HEADER_BYTEORDER),
            flatbuffer_offset=int.from_bytes(data[8:16], byteorder=_HEADER_BYTEORDER),
            flatbuffer_size=int.from_bytes(data[16:24], byteorder=_HEADER_BYTEORDER),
            segment_base_offset=int.from_bytes(
                data[24:32], byteorder=_HEADER_BYTEORDER
            ),
            segment_data_size=int.from_bytes(data[32:40], byteorder=_HEADER_BYTEORDER),
        )

    def is_valid(self) -> bool:
        """Returns true if the flat_tensor header appears to be well-formed."""
        return (
            self.magic == FlatTensorHeader.EXPECTED_MAGIC
            and self.length >= FlatTensorHeader.EXPECTED_LENGTH
        )

    def to_bytes(self) -> bytes:
        """Returns the binary representation of the flat_tensor header.

        Note that this will ignore self.magic and self.length and will always
        write the proper magic/length.
        """
        data: bytes = (
            # Extended header magic. This lets consumers detect whether the
            # header was inserted or not. Always use the proper magic value
            # (i.e., ignore self.magic) since there's no reason to create an
            # invalid header.
            self.EXPECTED_MAGIC
            # uint32_t: Size of this header. This makes it easier to add new
            # fields to this header in the future. Always use the proper size
            # (i.e., ignore self.length) since there's no reason to create an
            # invalid header.
            + self.EXPECTED_LENGTH.to_bytes(4, byteorder=_HEADER_BYTEORDER)
            # uint64_t: Offset to the start of the flatbuffer data, in bytes.
            + self.flatbuffer_offset.to_bytes(8, byteorder=_HEADER_BYTEORDER)
            # uint64_t: Size of the serialized data in bytes.
            + self.flatbuffer_size.to_bytes(8, byteorder=_HEADER_BYTEORDER)
            # uint64_t: Offset to the start of the first segment, or zero if
            # there are no segments.
            + self.segment_base_offset.to_bytes(8, byteorder=_HEADER_BYTEORDER)
            # uint64_t: Size of all the segment data, in bytes.
            + self.segment_data_size.to_bytes(8, byteorder=_HEADER_BYTEORDER)
        )
        return data


@dataclass
class AlignedData:
    """
    Holds data that should be aligned, for serialization.

    Attributes:
        data: The data to serialize, as a cord.
        alignment: The alignment required for the data.
    """

    data: Cord
    alignment: int

    def __init__(self, data: Cord, alignment: Optional[int] = None) -> None:
        self.data = data
        self.alignment = alignment or 1


def _get_extended_header(flat_tensor_data: bytes) -> Optional[FlatTensorHeader]:
    """Returns the extended header of the flat_tensor data, if present and valid."""
    try:
        eh = FlatTensorHeader.from_bytes(flat_tensor_data[8:])
        if eh.is_valid():
            return eh
    except ValueError:
        pass
    return None


def _extract_named_data(
    data_payload: DataPayload,
    segments: List[AlignedData],
) -> List[NamedData]:
    """Places named data into segments and record the alignment for each.

    Args:
        key_to_data: A map from keys to opaque data entries.
        buffers: A sequence of buffers holding opaque blob data.
        segments: A list of segments to append data to. Modified in-place.

    Returns:
        A list of NamedData describing the offsets to the opaque blob data.
    """

    # Map from buffer_idx to segment_idx.
    segment_index_map: Dict[int, int] = {}

    named_data: List[NamedData] = []
    for key, data_entry in data_payload.named_data.items():
        buffer_idx = data_entry.buffer_index
        segment_index = segment_index_map.get(buffer_idx, None)
        if segment_index is None:
            segment_index = len(segments)
            segment_index_map[buffer_idx] = segment_index
            segments.append(
                AlignedData(
                    Cord(data_payload.buffers[buffer_idx]), data_entry.alignment
                )
            )
        named_data.append(
            NamedData(
                key=key,
                segment_index=segment_index,
                # pyre-ignore Incompatible parameter type [6]
                tensor_layout=data_entry.tensor_layout,
            )
        )
    return named_data


class FlatTensorSerializer(DataSerializer):
    """A concrete implementation of the DataSerializer interface that
    serializes and deserializes data to/from the FlatTensor format.
    """

    def __init__(self, config: Optional[FlatTensorConfig] = None) -> None:
        """FlatTensorConfig holds information required for serialization,
        eg. alignment.
        """
        if config is None:
            self.config: FlatTensorConfig = FlatTensorConfig()
        else:
            self.config: FlatTensorConfig = config

    def serialize(
        self,
        data: DataPayload,
    ) -> Cord:
        """Serializes a list of tensors and named data into a blob."""

        segments: List[AlignedData] = []

        # Add a config to place tensors in a single segment.
        named_data = _extract_named_data(data, segments)

        data_segments: List[DataSegment] = []
        aggregated_segment_data = Cord()
        for segment in segments:
            prev_end = (
                (data_segments[-1].offset + data_segments[-1].size)
                if data_segments
                else 0
            )
            alignment = math.lcm(self.config.segment_alignment, segment.alignment)
            data_segments.append(
                DataSegment(
                    offset=aligned_size(prev_end, alignment),
                    size=len(segment.data),
                )
            )
            # Pad aggregated_segment_data to segment alignment.
            segment_pad_length = padding_required(
                len(aggregated_segment_data), alignment
            )
            if segment_pad_length > 0:
                aggregated_segment_data.append(b"\x00" * segment_pad_length)
            aggregated_segment_data.append(segment.data)

        # Create FlatTensor, which describes of the contents of the file and
        # points to all the data segments. It will be serialized to flatbuffer.
        flat_tensor = FlatTensor(
            version=_FLAT_TENSOR_VERSION,
            segments=data_segments,
            named_data=named_data,
        )

        flatbuffer_payload = _serialize_to_flatbuffer(flat_tensor)
        padded_header_length: int = aligned_size(
            input_size=FlatTensorHeader.EXPECTED_LENGTH,
            alignment=_FLATBUFFER_ALIGNMENT,
        )

        segment_base_offset = aligned_size(
            len(flatbuffer_payload) + padded_header_length,
            self.config.segment_alignment,
        )

        # Create FlatTensorHeader, which stores the offsets and sizes of the
        # FlatTensor flatbuffer and the segment data.
        header_data: bytes = FlatTensorHeader(
            flatbuffer_offset=padded_header_length,
            flatbuffer_size=len(flatbuffer_payload),
            segment_base_offset=segment_base_offset,
            segment_data_size=len(aggregated_segment_data),
        ).to_bytes()

        # Pad header and payload to segment alignment.
        header_data = pad_to(header_data, padded_header_length)
        injected_flatbuffer_data: bytes = _insert_flatbuffer_header(
            flatbuffer_data=flatbuffer_payload.__bytes__(),
            magic_regex=r"FT[0-9a-zA-Z][0-9a-zA-Z]",
            header_data=header_data,
        )
        injected_flatbuffer_data = pad_to(injected_flatbuffer_data, segment_base_offset)

        eh = _get_extended_header(injected_flatbuffer_data)
        assert eh is not None
        assert eh.flatbuffer_size == len(flatbuffer_payload)
        assert eh.segment_base_offset == segment_base_offset
        assert eh.flatbuffer_offset == padded_header_length
        assert eh.segment_data_size == len(aggregated_segment_data)

        del header_data
        del flatbuffer_payload

        # Place everything into one segment.
        payload = Cord()
        payload.append(injected_flatbuffer_data)
        payload.append(aggregated_segment_data)

        return payload

    def deserialize(self, blob: Cord) -> DataPayload:
        """
        Deserializes a flat_tensor blob into a list of tensor metadata and tensors.

        Note: deserialization does not preserve alignment information.
        """

        data = bytes(blob)

        # Read header. Verify that it's valid.
        header = FlatTensorHeader.from_bytes(data[8:])
        if not header.is_valid():
            raise RuntimeError(
                "Flat tensor header is invalid. File is likely incorrect format or corrupt."
            )

        # Deserialize the flat tensor data, which contains the data offsets and tensor metadata.
        flat_tensor_bytes = data[0 : header.flatbuffer_offset + header.flatbuffer_size]
        flat_tensor = _deserialize_to_flat_tensor(flat_tensor_bytes)

        # Verify that this is a supported version. Older files still load; only
        # a file newer than this reader understands is refused, matching the
        # runtime readers and the append-only schema policy in schema/README.md.
        if flat_tensor.version > _FLAT_TENSOR_VERSION:
            raise NotImplementedError(
                f"Flat tensor file reports version {flat_tensor.version}, which is newer than this reader supports (max {_FLAT_TENSOR_VERSION})."
            )

        # Extract the buffers.
        buffers = [
            data[
                header.segment_base_offset
                + segment.offset : header.segment_base_offset
                + segment.offset
                + segment.size
            ]
            for segment in flat_tensor.segments
        ]

        payload = DataPayload(
            buffers=buffers,
            named_data={},
        )

        # Read the named data entries.
        for named_data in flat_tensor.named_data:
            entry = DataEntry(
                buffer_index=named_data.segment_index,
                alignment=1,
                tensor_layout=named_data.tensor_layout,
            )
            payload.named_data[named_data.key] = entry

        return payload

    def deserialize_to_named_data_store_output(
        self, blob: bytes, name: str
    ) -> NamedDataStoreOutput:
        bytes = Cord(blob)
        data_payload = self.deserialize(bytes)
        return NamedDataStoreOutput(
            buffers=data_payload.buffers,
            pte_data={},
            external_data={name: data_payload.named_data},
        )


# Matches the tensor_alignment passed by the C++ save_ptd callers in
# extension/training/examples/{CIFAR,XOR}/train.cpp.
_DEFAULT_TENSOR_ALIGNMENT = 16


def _tensor_map_to_payload(
    tensor_map: Dict[str, torch.Tensor],
    tensor_alignment: int,
) -> DataPayload:
    """Builds a DataPayload from a map of tensor names to tensors."""
    buffers: List[bytes] = []
    named_data: Dict[str, DataEntry] = {}

    # Tied weights (eg. an embedding matrix reused as the output projection)
    # show up under more than one key in a state dict. Point the keys at a
    # single buffer instead of writing identical bytes twice. Every tensor is
    # kept alive by tensor_map for the duration of the loop, so id() is stable.
    buffer_index_by_tensor: Dict[int, int] = {}

    for key, tensor in tensor_map.items():
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(
                f"Expected a torch.Tensor for key '{key}', got {type(tensor).__name__}."
            )
        buffer_index = buffer_index_by_tensor.get(id(tensor))
        if buffer_index is None:
            buffer_index = len(buffers)
            buffer_index_by_tensor[id(tensor)] = buffer_index
            buffers.append(_tensor_to_bytes(tensor))
        named_data[key] = DataEntry(
            buffer_index=buffer_index,
            alignment=tensor_alignment,
            tensor_layout=TensorLayout.from_tensor(tensor),
        )

    return DataPayload(buffers=buffers, named_data=named_data)


def _tensor_from_layout(buffer: bytes, tensor_layout: TensorLayout) -> torch.Tensor:
    """Rebuilds a tensor from its serialized bytes and layout metadata."""
    dtype = get_scalar_type(tensor_layout.scalar_type)
    sizes = list(tensor_layout.sizes)

    # An empty tensor has no bytes to read; torch.frombuffer rejects a
    # zero-length buffer, so construct it directly.
    if any(size == 0 for size in sizes):
        return torch.empty(sizes, dtype=dtype)

    # bytearray gives torch a writable copy, so the tensor owns its memory and
    # torch does not warn about a read-only buffer.
    flat = torch.frombuffer(bytearray(buffer), dtype=dtype)

    # dim_order lists dimensions outermost-first as laid out in memory, so the
    # bytes on disk are the sizes permuted into that order. Reshape to the
    # physical shape, then invert the permutation to recover the logical one.
    dim_order = list(tensor_layout.dim_order)
    physical_sizes = [sizes[dim] for dim in dim_order]
    tensor = flat.reshape(physical_sizes)

    inverse_dim_order = [0] * len(dim_order)
    for position, dim in enumerate(dim_order):
        inverse_dim_order[dim] = position
    return tensor.permute(inverse_dim_order)


def save_ptd(
    path: Union[str, os.PathLike, BinaryIO],
    tensor_map: Dict[str, torch.Tensor],
    tensor_alignment: int = _DEFAULT_TENSOR_ALIGNMENT,
) -> None:
    """Creates a .ptd from the given tensor map.

    Mirrors the C++ save_ptd in extension/flat_tensor/serialize/serialize.h,
    which has both a path and a stream overload.

    Args:
        path: The file path to save the .ptd to, or a binary file-like object
            to write the .ptd data to.
        tensor_map: The map of tensor names to tensors to save.
        tensor_alignment: The bytes tensor data should be aligned to.

    Raises:
        TypeError: If a value in tensor_map is not a torch.Tensor.
        ValueError: If a tensor is neither contiguous nor channels-last.
    """
    payload = _tensor_map_to_payload(tensor_map, tensor_alignment)
    blob = FlatTensorSerializer(FlatTensorConfig()).serialize(payload)

    if hasattr(path, "write"):
        blob.write_to_file(path)
    else:
        with open(path, "wb") as outfile:
            blob.write_to_file(outfile)


def load_ptd(path: Union[str, os.PathLike, BinaryIO]) -> Dict[str, torch.Tensor]:
    """Loads a .ptd into a map of tensor names to tensors.

    Reverses save_ptd. Named data entries that carry no tensor layout are not
    tensors and are skipped.

    Args:
        path: The file path to load the .ptd from, or a binary file-like
            object to read the .ptd data from.

    Returns:
        A map of tensor names to tensors.
    """
    if hasattr(path, "read"):
        data = path.read()
    else:
        with open(path, "rb") as infile:
            data = infile.read()

    payload = FlatTensorSerializer().deserialize(Cord(data))

    tensor_map: Dict[str, torch.Tensor] = {}
    for key, entry in payload.named_data.items():
        if entry.tensor_layout is None:
            continue
        tensor_map[key] = _tensor_from_layout(
            bytes(payload.buffers[entry.buffer_index]), entry.tensor_layout
        )
    return tensor_map
