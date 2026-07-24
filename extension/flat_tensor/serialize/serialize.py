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
from typing import ClassVar, Dict, List, Literal, Optional, Set, Union

import executorch.extension.flat_tensor.serialize as serialize_package
import torch

from executorch.exir._serialize._cord import Cord
from executorch.exir._serialize._dataclass import _DataclassEncoder, _json_to_dataclass
from executorch.exir._serialize._flatbuffer import _flatc_compile, _flatc_decompile
from executorch.exir._serialize._named_data_store import NamedDataStoreOutput
from executorch.exir._serialize._program import _insert_flatbuffer_header
from executorch.exir._serialize.data_serializer import (
    DataEntry,
    DataPayload,
    DataSerializer,
)
from executorch.exir._serialize.padding import aligned_size, pad_to, padding_required
from executorch.exir.scalar_type import ScalarType
from executorch.exir.tensor import dim_order_from_stride, stride_from_dim_order
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

# Current version. Keep in sync with c++ version number in serialize.
_FLAT_TENSOR_VERSION: int = 0

# Keep in sync with scalar_type.fbs, which excludes complex types.
_PTD_TO_TORCH_DTYPE: Dict[ScalarType, torch.dtype] = {
    ScalarType.BYTE: torch.uint8,
    ScalarType.CHAR: torch.int8,
    ScalarType.SHORT: torch.int16,
    ScalarType.INT: torch.int32,
    ScalarType.LONG: torch.int64,
    ScalarType.HALF: torch.float16,
    ScalarType.FLOAT: torch.float32,
    ScalarType.DOUBLE: torch.float64,
    ScalarType.BOOL: torch.bool,
    ScalarType.QINT8: torch.qint8,
    ScalarType.QUINT8: torch.quint8,
    ScalarType.QINT32: torch.qint32,
    ScalarType.BFLOAT16: torch.bfloat16,
    ScalarType.QUINT4x2: torch.quint4x2,
    ScalarType.QUINT2x4: torch.quint2x4,
    ScalarType.BITS16: torch.bits16,
    ScalarType.FLOAT8E5M2: torch.float8_e5m2,
    ScalarType.FLOAT8E4M3FN: torch.float8_e4m3fn,
    ScalarType.FLOAT8E5M2FNUZ: torch.float8_e5m2fnuz,
    ScalarType.FLOAT8E4M3FNUZ: torch.float8_e4m3fnuz,
    ScalarType.UINT16: torch.uint16,
    ScalarType.UINT32: torch.uint32,
    ScalarType.UINT64: torch.uint64,
}
_TORCH_DTYPE_TO_PTD: Dict[torch.dtype, ScalarType] = {
    dtype: scalar_type for scalar_type, dtype in _PTD_TO_TORCH_DTYPE.items()
}
# PTD tensor layouts do not encode quantization parameters.
_QUANTIZED_DTYPES: Set[torch.dtype] = {
    torch.qint8,
    torch.quint8,
    torch.qint32,
    torch.quint4x2,
    torch.quint2x4,
}


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

        # Verify that this is a supported version.
        if flat_tensor.version != _FLAT_TENSOR_VERSION:
            raise NotImplementedError(
                f"Flat tensor files reports unsupported version {flat_tensor.version}. Expected {_FLAT_TENSOR_VERSION}."
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


def save_ptd(
    path: Union[str, os.PathLike[str]], tensor_map: Dict[str, torch.Tensor]
) -> None:
    """Saves a dictionary of tensors to a PTD file.

    Args:
        path: Path to the output PTD file.
        tensor_map: Mapping from tensor names to supported strided CPU tensors.
    """
    buffers: List[bytes] = []
    named_data: Dict[str, DataEntry] = {}
    for name, tensor in tensor_map.items():
        if tensor.device.type != "cpu":
            raise ValueError(
                f"Tensor '{name}' must be on CPU, received {tensor.device}."
            )
        if tensor.dtype not in _TORCH_DTYPE_TO_PTD or tensor.is_quantized:
            raise ValueError(f"Unsupported tensor dtype {tensor.dtype} for PTD files.")
        if tensor.layout != torch.strided:
            raise ValueError(f"Tensor '{name}' must use a strided layout.")

        tensor_to_save = tensor.detach().resolve_neg()
        if 0 in tensor_to_save.stride():
            tensor_to_save = tensor_to_save.clone(memory_format=torch.contiguous_format)
        elif not (
            tensor_to_save.is_contiguous()
            or tensor_to_save.is_contiguous(memory_format=torch.channels_last)
        ):
            tensor_to_save = tensor_to_save.contiguous()
        if tensor_to_save.nbytes != tensor_to_save.untyped_storage().nbytes():
            tensor_to_save = tensor_to_save.clone(memory_format=torch.preserve_format)
        tensor_layout = TensorLayout(
            scalar_type=_TORCH_DTYPE_TO_PTD[tensor_to_save.dtype],
            sizes=list(tensor_to_save.shape),
            dim_order=list(dim_order_from_stride(tensor_to_save.stride())),
        )

        buffer_index = len(buffers)
        buffers.append(bytes(tensor_to_save.untyped_storage()))
        named_data[name] = DataEntry(
            buffer_index=buffer_index,
            alignment=1,
            tensor_layout=tensor_layout,
        )

    data_payload = DataPayload(buffers=buffers, named_data=named_data)
    serialized_data = FlatTensorSerializer().serialize(data_payload)
    with open(path, "wb") as file:
        file.write(bytes(serialized_data))


def load_ptd(path: Union[str, os.PathLike[str]]) -> Dict[str, torch.Tensor]:
    """Loads a dictionary of tensors from a PTD file.

    Args:
        path: Path to the PTD file.

    Returns:
        A mapping from tensor names to mutable CPU tensors.
    """
    with open(path, "rb") as file:
        data_payload = FlatTensorSerializer().deserialize(Cord(file.read()))

    tensor_map: Dict[str, torch.Tensor] = {}
    for name, data_entry in data_payload.named_data.items():
        if not 0 <= data_entry.buffer_index < len(data_payload.buffers):
            raise ValueError(
                f"Tensor '{name}' references invalid buffer index "
                f"{data_entry.buffer_index}."
            )

        tensor_layout = data_entry.tensor_layout
        if tensor_layout is None:
            raise ValueError(f"Named data '{name}' does not contain a tensor layout.")

        try:
            dtype = _PTD_TO_TORCH_DTYPE[tensor_layout.scalar_type]
        except KeyError as error:
            raise ValueError(
                f"Unsupported scalar type {tensor_layout.scalar_type} for tensor "
                f"'{name}'."
            ) from error
        if dtype in _QUANTIZED_DTYPES:
            raise ValueError(f"Unsupported tensor dtype {dtype} for PTD files.")

        strides = stride_from_dim_order(tensor_layout.sizes, tensor_layout.dim_order)
        numel = math.prod(tensor_layout.sizes)
        buffer = data_payload.buffers[data_entry.buffer_index]
        expected_nbytes = (
            numel * torch.empty((), dtype=dtype, device="cpu").element_size()
        )
        if len(buffer) < expected_nbytes:
            raise ValueError(
                f"Tensor '{name}' requires {expected_nbytes} bytes, but its buffer "
                f"contains {len(buffer)} bytes."
            )
        if numel == 0:
            tensor = torch.empty_strided(
                tensor_layout.sizes, strides, dtype=dtype, device="cpu"
            )
        else:
            tensor = torch.frombuffer(
                bytearray(buffer),
                dtype=dtype,
                count=numel,
            ).as_strided(tensor_layout.sizes, strides)
        tensor_map[name] = tensor

    return tensor_map
