import json
import os
import tempfile
from dataclasses import dataclass
from typing import ClassVar, Dict, List, Optional

import pkg_resources
from executorch.exir._serialize._cord import Cord
from executorch.exir._serialize._dataclass import _DataclassEncoder

from executorch.exir._serialize._flatbuffer import _flatc_compile
from executorch.exir._serialize.data_serializer import DataSerializer, SerializationInfo

from executorch.exir._serialize.utils import (
    _aligned_size,
    _HEADER_BYTEORDER,
    _pad_to,
    _padding_required,
)

from executorch.extension.flat_tensor.serialize.flat_tensor_schema import (
    DataSegment,
    FlatTensor,
    TensorMetadata,
)


def _convert_to_flatbuffer(flat_tensor: FlatTensor) -> Cord:
    flat_tensor_json = json.dumps(flat_tensor, cls=_DataclassEncoder)
    with tempfile.TemporaryDirectory() as d:
        schema_path = os.path.join(d, "flat_tensor.fbs")
        with open(schema_path, "wb") as schema_file:
            schema_file.write(
                pkg_resources.resource_string(__name__, "flat_tensor.fbs")
            )
        scalar_type_path = os.path.join(d, "scalar_type.fbs")
        with open(scalar_type_path, "wb") as scalar_type_file:
            scalar_type_file.write(
                pkg_resources.resource_string(__name__, "scalar_type.fbs")
            )
        json_path = os.path.join(d, "flat_tensor.json")
        with open(json_path, "wb") as json_file:
            json_file.write(flat_tensor_json.encode("ascii"))

        _flatc_compile(d, schema_path, json_path)
        output_path = os.path.join(d, "flat_tensor.ptd")
        with open(output_path, "rb") as output_file:
            return Cord(output_file.read())


@dataclass
class FlatTensorConfig:
    tensor_alignment: int = 16
    segment_alignment: int = 16


@dataclass
class FlatTensorHeader:
    # Class constants.
    # The magic bytes that should be at the beginning of the header.
    EXPECTED_MAGIC: ClassVar[bytes] = b"FT01"
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
    data_size: int

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
            data_size=int.from_bytes(data[32:40], byteorder=_HEADER_BYTEORDER),
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
            + self.data_size.to_bytes(8, byteorder=_HEADER_BYTEORDER)
        )
        return data


class FlatTensorSerializer(DataSerializer):
    def __init__(self, config: Optional[FlatTensorConfig] = None) -> None:
        if config is None:
            self.config = FlatTensorConfig()
        else:
            self.config = config

    def serialize_tensors(
        self,
        serialization_info: SerializationInfo,
    ) -> Cord:
        flat_tensor_metadata: List[TensorMetadata] = []
        flat_tensor_data: Cord = Cord()

        # {idx, offset}
        saved_offsets: Dict[int, int] = {}

        for fqn, idx in serialization_info.fqn_to_buffer_index.items():
            tensor_layout = serialization_info.fqn_to_tensor_layout.get(fqn, None)
            assert tensor_layout is not None
            # Check index into the tensor buffers is valid.
            assert idx < len(serialization_info.tensor_buffers)

            # Check if the tensor has already been saved.
            offset = saved_offsets.get(idx, -1)
            if offset == -1:
                if len(flat_tensor_data) > 0:
                    # Add padding to round off the previous tensor offset.
                    pad_length = _padding_required(
                        len(flat_tensor_data), self.config.tensor_alignment
                    )
                    flat_tensor_data.append(b"\x00" * pad_length)
                # Add to saved offsets.
                offset = len(flat_tensor_data)
                saved_offsets[idx] = offset
                # Append to flat_tensor_data at the offset.
                flat_tensor_data.append(serialization_info.tensor_buffers[idx])

            flat_tensor_metadata.append(
                TensorMetadata(
                    fully_qualified_name=fqn,
                    scalar_type=tensor_layout.scalar_type,
                    dim_sizes=tensor_layout.dim_sizes,
                    dim_order=tensor_layout.dim_order,
                    segment_index=0,
                    offset=offset,
                )
            )
        # Pad to segment alignment.
        segment_pad_length = _padding_required(
            len(flat_tensor_data), self.config.segment_alignment
        )
        if segment_pad_length > 0:
            flat_tensor_data.append(b"\x00" * segment_pad_length)

        # Organize the tensors and segments.
        flat_tensor = FlatTensor(
            version=0,
            tensor_alignment=self.config.tensor_alignment,
            tensors=flat_tensor_metadata,
            segments=[DataSegment(offset=0, size=len(flat_tensor_data))],
        )

        flatbuffer_payload = _convert_to_flatbuffer(flat_tensor)
        padded_flatbuffer_length: int = _aligned_size(
            input_size=len(flatbuffer_payload),
            alignment=self.config.tensor_alignment,
        )

        padded_header_length: int = _aligned_size(
            input_size=FlatTensorHeader.EXPECTED_LENGTH,
            alignment=self.config.tensor_alignment,
        )

        segment_base_offset = _aligned_size(
            padded_flatbuffer_length + padded_header_length,
            self.config.segment_alignment,
        )

        header_data: bytes = FlatTensorHeader(
            flatbuffer_offset=padded_header_length,
            flatbuffer_size=len(flatbuffer_payload),
            segment_base_offset=segment_base_offset,
            data_size=len(flat_tensor_data),
        ).to_bytes()

        # Pad header and payload to segment alignment.
        header_data = _pad_to(header_data, padded_header_length)
        flatbuffer_payload.append(
            b"\x00" * (padded_flatbuffer_length - len(flatbuffer_payload))
        )

        # Place everything into one segment.
        payload = Cord()
        payload.append(header_data)
        payload.append(flatbuffer_payload)
        payload.append(flat_tensor_data)

        return payload

    def deserialize_tensors(self, blob: Cord) -> SerializationInfo:
        """
        Deserializes a blob into a list of tensor metadata and tensors.
        """
        raise NotImplementedError("deserialize_data")
