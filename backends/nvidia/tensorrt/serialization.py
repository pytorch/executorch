# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Serialization format for TensorRT backend.

Defines the binary blob format for storing TensorRT engines in .pte files.
The format is designed to be:
- Simple and self-contained
- 16-byte aligned for efficient memory access
- Forward compatible with reserved header space

Blob Layout:
    [Header: 32 bytes]
    [I/O Metadata JSON: variable]
    [Padding to 16-byte alignment]
    [Engine Data: variable]
"""

import json
import struct
from dataclasses import asdict, dataclass, field
from typing import List, Optional

# Magic bytes identifying a TensorRT blob.
# "TR01" = TensorRT version 1 format with I/O metadata.
# Must match kTensorRTMagic in TensorRTBlobHeader.h for C++ runtime compatibility.
TENSORRT_MAGIC = b"TR01"

# Header is 32 bytes, 16-byte aligned
# Layout:
#   magic (4 bytes) - "TR01"
#   metadata_offset (4 bytes) - offset to metadata JSON from start
#   metadata_size (4 bytes) - size of metadata JSON in bytes
#   engine_offset (4 bytes) - offset to engine data from start
#   engine_size (8 bytes) - size of engine data in bytes
#   reserved (8 bytes) - for future use
HEADER_SIZE = 32
HEADER_FORMAT = "<4sIIIQ8s"  # little-endian


@dataclass
class TensorRTIOBinding:
    """I/O binding metadata for a TensorRT engine tensor.

    Attributes:
        name: Name of the tensor binding.
        dtype: Data type as string (e.g., "float32", "float16", "int32").
        shape: Shape of the tensor as list of dimensions.
        is_input: True if this is an input binding, False for output.
    """

    name: str
    dtype: str
    shape: List[int]
    is_input: bool
    is_shape_tensor: bool = False

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "TensorRTIOBinding":
        """Create from dictionary."""
        return cls(
            name=data["name"],
            dtype=data["dtype"],
            shape=data["shape"],
            is_input=data["is_input"],
            is_shape_tensor=data.get("is_shape_tensor", False),
        )


@dataclass
class TensorRTBlobMetadata:
    """Metadata stored in TensorRT blob.

    Attributes:
        io_bindings: List of input/output tensor bindings.
    """

    io_bindings: List[TensorRTIOBinding] = field(default_factory=list)

    def to_json(self) -> bytes:
        """Serialize metadata to JSON bytes."""
        data = {
            "io_bindings": [b.to_dict() for b in self.io_bindings],
        }
        return json.dumps(data, separators=(",", ":")).encode("utf-8")

    @classmethod
    def from_json(cls, data: bytes) -> "TensorRTBlobMetadata":
        """Deserialize metadata from JSON bytes."""
        parsed = json.loads(data.decode("utf-8"))
        io_bindings = [
            TensorRTIOBinding.from_dict(b) for b in parsed.get("io_bindings", [])
        ]
        return cls(io_bindings=io_bindings)


@dataclass
class TensorRTBlobHeader:
    """Header for TensorRT serialized blob.

    Attributes:
        magic: Magic bytes identifying blob type (b"TR01").
        metadata_offset: Offset in bytes from start to metadata.
        metadata_size: Size of metadata JSON in bytes.
        engine_offset: Offset in bytes from start to engine data.
        engine_size: Size of engine data in bytes.
    """

    magic: bytes
    metadata_offset: int
    metadata_size: int
    engine_offset: int
    engine_size: int

    def is_valid(self) -> bool:
        """Check if this is a valid TensorRT blob header."""
        return self.magic == TENSORRT_MAGIC


def _align_to_16(offset: int) -> int:
    """Align offset to 16-byte boundary."""
    return (offset + 15) & ~15


def serialize_blob(
    engine_bytes: bytes,
    metadata: Optional[TensorRTBlobMetadata] = None,
) -> bytes:
    """Serialize TensorRT engine to blob format with metadata.

    Args:
        engine_bytes: Serialized TensorRT engine bytes.
        metadata: Optional metadata including I/O bindings.

    Returns:
        Complete blob with header, metadata, and engine data.
    """
    if metadata is None:
        metadata = TensorRTBlobMetadata()

    # Serialize metadata to JSON
    metadata_json = metadata.to_json()
    metadata_size = len(metadata_json)

    # Calculate offsets with alignment
    metadata_offset = HEADER_SIZE
    engine_offset = _align_to_16(metadata_offset + metadata_size)
    engine_size = len(engine_bytes)

    # Build header
    reserved = b"\x00" * 8
    header = struct.pack(
        HEADER_FORMAT,
        TENSORRT_MAGIC,
        metadata_offset,
        metadata_size,
        engine_offset,
        engine_size,
        reserved,
    )

    # Build padding between metadata and engine
    padding_size = engine_offset - (metadata_offset + metadata_size)
    padding = b"\x00" * padding_size

    return header + metadata_json + padding + engine_bytes


def deserialize_blob_header(data: bytes) -> Optional[TensorRTBlobHeader]:
    """Deserialize blob header from binary data.

    Args:
        data: Binary data containing at least the header.

    Returns:
        TensorRTBlobHeader if valid, None otherwise.
    """
    if len(data) < HEADER_SIZE:
        return None

    magic, metadata_offset, metadata_size, engine_offset, engine_size, _ = (
        struct.unpack(HEADER_FORMAT, data[:HEADER_SIZE])
    )

    return TensorRTBlobHeader(
        magic=magic,
        metadata_offset=metadata_offset,
        metadata_size=metadata_size,
        engine_offset=engine_offset,
        engine_size=engine_size,
    )


def get_metadata_from_blob(data: bytes) -> Optional[TensorRTBlobMetadata]:
    """Extract metadata from blob.

    Args:
        data: Complete blob data.

    Returns:
        TensorRTBlobMetadata if valid blob, None otherwise.
    """
    header = deserialize_blob_header(data)
    if header is None or not header.is_valid():
        return None

    end_offset = header.metadata_offset + header.metadata_size
    if len(data) < end_offset:
        return None

    metadata_json = data[header.metadata_offset : end_offset]
    return TensorRTBlobMetadata.from_json(metadata_json)


def get_engine_from_blob(data: bytes) -> Optional[bytes]:
    """Extract TensorRT engine bytes from blob.

    Args:
        data: Complete blob data.

    Returns:
        Engine bytes if valid blob, None otherwise.
    """
    header = deserialize_blob_header(data)
    if header is None or not header.is_valid():
        return None

    end_offset = header.engine_offset + header.engine_size
    if len(data) < end_offset:
        return None

    return data[header.engine_offset : end_offset]
