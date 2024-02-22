# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import json
import os
import tempfile

from dataclasses import dataclass, fields, is_dataclass
from typing import ClassVar, Literal

import pkg_resources
from executorch.backends.xnnpack.serialization.xnnpack_graph_schema import XNNGraph
from executorch.exir._serialize._dataclass import _DataclassEncoder

from executorch.exir._serialize._flatbuffer import _flatc_compile

# Byte order of numbers written to program headers. Always little-endian
# regardless of the host system, since all commonly-used modern CPUs are little
# endian.
_HEADER_BYTEORDER: Literal["little"] = "little"

# Constant Tensor alignment for serializaing XNNPACK payloads
CONSTANT_TENSOR_ALIGNMENT = 16


def sanity_check_xnngraph_dataclass(table, name: str = ""):
    """
    Make sure no SymInt sneaked in during the preparation of XNNGraph.
    """
    assert is_dataclass(table), f"Expecting a dataclass but got {type(table)}"

    def get_cls_name(obj, field_name=None):
        return (
            f"<{obj.__class__.__name__}>{field_name}"
            if field_name
            else obj.__class__.__name__
        )

    def check_for_sym(obj, name):
        """
        Basic check against the class name of the given obj and
        if it starts from "Sym" or not to catch SymInt the main culprit.
        """
        class_name = get_cls_name(obj)
        assert (
            "Sym" not in class_name
        ), f"Non serializable type {class_name} found at type {name}"

    _name = name if len(name) else get_cls_name(table)

    for field in fields(table):
        o = getattr(table, field.name)

        # Skip str and bytes
        if isinstance(o, str) or isinstance(o, bytes):
            continue

        _name_field = f"{_name}.{get_cls_name(o, field.name)}"

        # Recurse
        if is_dataclass(o):
            sanity_check_xnngraph_dataclass(o, _name_field)

        # Only handles List type, add more if needed
        elif isinstance(o, list):
            for i, v in enumerate(o):
                _name_field_i = _name_field + f"[{i}]"
                # Recurse
                if is_dataclass(v):
                    sanity_check_xnngraph_dataclass(v, f"{_name_field_i}")
                else:
                    check_for_sym(v, _name_field_i)
        else:
            check_for_sym(o, _name_field)


@dataclass
class XNNHeader:
    # Class Constants
    MAGIC_OFFSET: ClassVar[slice] = slice(4, 8)
    HEADER_SIZE_OFFSET: ClassVar[slice] = slice(8, 10)
    FLATBUFFER_OFFSET_OFFSET: ClassVar[slice] = slice(10, 14)
    FLATBUFFER_SIZE_OFFSET: ClassVar[slice] = slice(14, 18)
    CONSTANT_DATA_OFFSET_OFFSET: ClassVar[slice] = slice(18, 22)
    CONSTANT_DATA_SIZE_OFFSET: ClassVar[slice] = slice(22, 30)

    # magic bytes that should be at the beginning of the header
    EXPECTED_MAGIC: ClassVar[bytes] = b"XH00"
    # The length of the header in bytes.
    EXPECTED_LENGTH: ClassVar[int] = (
        # Zeros magic
        # We offset the magic by 4 bytes so that it is in the same location
        # as the flatbuffer payload's magic. This way we can dynamically
        # choose between the XNNPACK Header and Flatbuffer Header
        4
        # Header magic
        + 4
        # Header Length
        + 2
        # Flatbuffer offset
        + 4
        # Flatbuffer size
        + 4
        # Constant Data offset
        + 4
        # Constant Data size
        + 8
    )

    # Instance attributes. @dataclass will turn these into ctor args.

    # offset to the flatbuffer data
    flatbuffer_offset: int

    # flatbuffer size
    flatbuffer_size: int

    # offset to the constant data
    constant_data_offset: int

    # constant data size
    constant_data_size: int

    @staticmethod
    def from_bytes(data: bytes) -> "XNNHeader":
        """
        Converts the given bytes into an XNNHeader object.

        We check that the magic and length is valid, but do not check that the offset and
        size values are valid. We ensure here that the XNNHeader metadata is valid (magic and length)
        but not the offsets and sizes themselves. Callers should use is_valid() to validate the
        header contents

        Args:
            data: Data to read from
        Returns:
            XNNHeader object that contains the parsed data
        Raises:
            ValueError: if not enough data is provided, or if parsed length/magic are invalid
        """
        if len(data) > XNNHeader.EXPECTED_LENGTH:
            raise ValueError(
                f"Invalid XNNHeader: expected no more than {XNNHeader.EXPECTED_LENGTH} bytes, got {len(data)}"
            )

        magic: bytes = data[XNNHeader.MAGIC_OFFSET]
        length_bytes: bytes = data[XNNHeader.HEADER_SIZE_OFFSET]
        flatbuffer_offset_bytes: bytes = data[XNNHeader.FLATBUFFER_OFFSET_OFFSET]
        flatbuffer_size_bytes: bytes = data[XNNHeader.FLATBUFFER_SIZE_OFFSET]
        constant_data_offset_bytes: bytes = data[XNNHeader.CONSTANT_DATA_OFFSET_OFFSET]
        constant_data_size_bytes: bytes = data[XNNHeader.CONSTANT_DATA_SIZE_OFFSET]

        length = int.from_bytes(length_bytes, byteorder=_HEADER_BYTEORDER)

        if magic != XNNHeader.EXPECTED_MAGIC:
            raise ValueError(
                f"Invalid XNNHeader: invalid magic bytes {magic}, expected {XNNHeader.EXPECTED_MAGIC}"
            )
        if length != len(data):
            raise ValueError(
                f"Invalid XNNHeader: Invalid parsed length: data given was {len(data)} bytes, parsed length was {length} bytes"
            )

        return XNNHeader(
            flatbuffer_offset=int.from_bytes(
                flatbuffer_offset_bytes, byteorder=_HEADER_BYTEORDER
            ),
            flatbuffer_size=int.from_bytes(
                flatbuffer_size_bytes, byteorder=_HEADER_BYTEORDER
            ),
            constant_data_offset=int.from_bytes(
                constant_data_offset_bytes, byteorder=_HEADER_BYTEORDER
            ),
            constant_data_size=int.from_bytes(
                constant_data_size_bytes, byteorder=_HEADER_BYTEORDER
            ),
        )

    def is_valid(self) -> bool:
        """
        Sanity checks the the XNNHeader.

        We check that the flatbuffer size is non_zero and that the constant data offset
        is after the flatbuffer payload. We check that the constant data size is non-negative.

        Returns:
            True if the XNNHeader is valid, False otherwise
        """
        # flatbuffer payload must have a non-zero size
        valid_flatbuffer_size = self.flatbuffer_size > 0
        # constant data offset is after flatbuffer payload
        valid_const_data_offset = (
            self.constant_data_offset >= self.flatbuffer_offset + self.flatbuffer_size
        )
        valid_const_data_size = self.constant_data_size >= 0

        return (
            valid_flatbuffer_size and valid_const_data_offset and valid_const_data_size
        )

    def to_bytes(self) -> bytes:
        """
        Converts XNNHeader to bytes for serialization.

        Returns:
            Returns the binary representation of the XNNPACK Header.
        """

        # We expect the given offsets and sizes to be valid
        if not self.is_valid():
            raise ValueError("Invalid XNNHeader: header failed is_valid() check")

        data: bytes = (
            # Padding for magic bytes. This is so that header magic is in the same position
            # as the flatbuffer magic, and allows consumer to detect whether the header is
            # being used or not
            b"\x00\x00\x00\x00"
            # XNNPACK Header's magic. This allows consumer to detect whether or not the header
            # is being used or the flatbuffer header is being used
            + self.EXPECTED_MAGIC
            # uint16_t: Size of this header. This makes it easier to add new fields to the header
            # in the future.
            + self.EXPECTED_LENGTH.to_bytes(2, byteorder=_HEADER_BYTEORDER)
            # uint32_t: Offset to the start of the flatbuffer data
            + self.flatbuffer_offset.to_bytes(4, byteorder=_HEADER_BYTEORDER)
            # uint32_t: Size of the flatbuffer data payload
            + self.flatbuffer_size.to_bytes(4, byteorder=_HEADER_BYTEORDER)
            # uint32_t: Offset to the start of the constant data
            + self.constant_data_offset.to_bytes(4, byteorder=_HEADER_BYTEORDER)
            # uint64_t: Size of the constant data
            + self.constant_data_size.to_bytes(8, byteorder=_HEADER_BYTEORDER)
        )

        assert len(data) == XNNHeader.EXPECTED_LENGTH

        return data


def _padding_required(offset: int, alignment: int) -> int:
    """Returns the padding required to align `offset` to `alignment`."""
    remainder: int = offset % alignment
    if remainder != 0:
        return alignment - remainder
    return 0


def _aligned_size(input_size: int, alignment: int) -> int:
    """Returns input_size padded up to the next whole multiple of alignment."""
    aligned_size = input_size + _padding_required(input_size, alignment)
    assert aligned_size % alignment == 0
    return aligned_size


def _pad_to(data: bytes, length: int) -> bytes:
    """Returns the input followed by enough zero bytes to become the requested length.

    Args:
        data: The data to pad.
        length: The length of the returned data.
    Returns:
        The padded data.
    Raises:
        ValueError: If the requested length is less than the input length.
    """
    if length < len(data):
        raise ValueError(f"Data length {len(data)} > padded length {length}")
    if length > len(data):
        data = data + b"\x00" * (length - len(data))
    assert len(data) == length
    return data


def pretty_print_xnngraph(xnnpack_graph_json: str):
    """
    Pretty print the XNNGraph
    """
    from pprint import pprint

    d = json.loads(xnnpack_graph_json)
    pprint(d)


def convert_to_flatbuffer(xnnpack_graph: XNNGraph) -> bytes:
    sanity_check_xnngraph_dataclass(xnnpack_graph)
    xnnpack_graph_json = json.dumps(xnnpack_graph, cls=_DataclassEncoder)
    with tempfile.TemporaryDirectory() as d:
        schema_path = os.path.join(d, "schema.fbs")
        with open(schema_path, "wb") as schema_file:
            schema_file.write(pkg_resources.resource_string(__name__, "schema.fbs"))
        json_path = os.path.join(d, "schema.json")
        with open(json_path, "wb") as json_file:
            json_file.write(xnnpack_graph_json.encode("ascii"))

        _flatc_compile(d, schema_path, json_path)
        output_path = os.path.join(d, "schema.bin")
        with open(output_path, "rb") as output_file:
            return output_file.read()


def serialize_xnnpack_binary(
    xnnpack_graph: XNNGraph, constant_data_bytes: bytearray
) -> bytes:
    """Returns the runtime binary representation of the given XNNGraph.

    Args:
        xnnpack_graph: XNNGraph object to serialize.

    Returns:
        The serialized form of the XNNGraph, ready for execution by XNNPACK Backend
    """

    # Convert the XNNGraph to a flatbuffer
    flatbuffer_payload = convert_to_flatbuffer(xnnpack_graph)

    # size of flatbuffer data, padded to be `constant_tensor_alignment`  byte aligned
    padded_flatbuffer_length: int = _aligned_size(
        input_size=len(flatbuffer_payload),
        alignment=CONSTANT_TENSOR_ALIGNMENT,
    )
    # size of header to insert, padded to be `constant_tensor_alignment` byte aligned
    padded_header_length: int = _aligned_size(
        input_size=XNNHeader.EXPECTED_LENGTH, alignment=CONSTANT_TENSOR_ALIGNMENT
    )

    # Create the XNNPACK Header
    header: bytes = XNNHeader(
        flatbuffer_offset=padded_header_length,
        flatbuffer_size=len(flatbuffer_payload),
        constant_data_offset=padded_header_length + padded_flatbuffer_length,
        constant_data_size=len(constant_data_bytes),
    ).to_bytes()

    return b"".join(
        [
            _pad_to(header, padded_header_length),
            _pad_to(flatbuffer_payload, padded_flatbuffer_length),
            constant_data_bytes,
        ]
    )
