# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest

from executorch.backends.xnnpack.serialization.xnnpack_graph_serialize import XNNHeader

EXAMPLE_FLATBUFFER_OFFSET: int = 0x11223344
EXAMPLE_FLATBUFFER_SIZE: int = 0x55667788
EXAMPLE_CONSTANT_DATA_OFFSET: int = EXAMPLE_FLATBUFFER_OFFSET + EXAMPLE_FLATBUFFER_SIZE
EXAMPLE_CONSTANT_DATA_SIZE: int = 0x99AABBCC99AABBCC

# If header layout or magic changes, this test must change too.
# The layout of the header is a contract, not an implementation detail
EXAMPLE_HEADER_DATA: bytes = (
    # zeros
    b"\x00\x00\x00\x00"
    # magic
    + b"XH00"
    # All Values below are littl Endian
    # header length
    + b"\x1E\x00"
    # Flatbuffer Offset
    + b"\x44\x33\x22\x11"
    # Flatbuffer Size
    + b"\x88\x77\x66\x55"
    # Constant Data Offset
    + b"\xCC\xAA\x88\x66"
    # Constant Data Size
    + b"\xCC\xBB\xAA\x99\xCC\xBB\xAA\x99"
)


class TestXNNHeader(unittest.TestCase):
    def test_to_bytes(self) -> None:
        header = XNNHeader(
            EXAMPLE_FLATBUFFER_OFFSET,
            EXAMPLE_FLATBUFFER_SIZE,
            EXAMPLE_CONSTANT_DATA_OFFSET,
            EXAMPLE_CONSTANT_DATA_SIZE,
        )
        self.assertEqual(header.to_bytes(), EXAMPLE_HEADER_DATA)
        self.assertTrue(header.is_valid())

    def test_from_bytes(self) -> None:
        header = XNNHeader.from_bytes(EXAMPLE_HEADER_DATA)
        self.assertEqual(header.flatbuffer_offset, EXAMPLE_FLATBUFFER_OFFSET)
        self.assertEqual(header.flatbuffer_size, EXAMPLE_FLATBUFFER_SIZE)
        self.assertEqual(header.constant_data_offset, EXAMPLE_CONSTANT_DATA_OFFSET)
        self.assertEqual(header.constant_data_size, EXAMPLE_CONSTANT_DATA_SIZE)

    def test_invalid_metadata(self) -> None:
        WRONG_MAGIC_DATA = EXAMPLE_HEADER_DATA[0:4] + b"YT01" + EXAMPLE_HEADER_DATA[8:]
        with self.assertRaisesRegex(
            ValueError,
            "Invalid XNNHeader: invalid magic bytes b'YT01', expected b'XH00'",
        ):
            XNNHeader.from_bytes(WRONG_MAGIC_DATA)

        WRONG_LENGTH_DATA = (
            EXAMPLE_HEADER_DATA[0:8] + b"\x1D\x00" + EXAMPLE_HEADER_DATA[10:]
        )
        with self.assertRaisesRegex(
            ValueError,
            "Invalid XNNHeader: Invalid parsed length: data given was 30 bytes, parsed length was 29 bytes",
        ):
            XNNHeader.from_bytes(WRONG_LENGTH_DATA)

        with self.assertRaisesRegex(
            ValueError,
            "Invalid XNNHeader: expected no more than 30 bytes, got 31",
        ):
            XNNHeader.from_bytes(EXAMPLE_HEADER_DATA + b"\x00")

    def test_invalid_flatbuffer_size(self) -> None:
        header = XNNHeader(
            EXAMPLE_FLATBUFFER_OFFSET,
            0,
            EXAMPLE_CONSTANT_DATA_OFFSET,
            EXAMPLE_CONSTANT_DATA_SIZE,
        )

        with self.assertRaises(ValueError):
            header.to_bytes()

    def test_invalid_constant_data_offset(self) -> None:
        header = XNNHeader(
            EXAMPLE_FLATBUFFER_OFFSET,
            EXAMPLE_FLATBUFFER_SIZE,
            EXAMPLE_FLATBUFFER_OFFSET + EXAMPLE_FLATBUFFER_SIZE - 1,
            EXAMPLE_CONSTANT_DATA_SIZE,
        )

        with self.assertRaises(ValueError):
            header.to_bytes()

    def test_to_bytes_same_as_from_bytes(self) -> None:
        header = XNNHeader.from_bytes(EXAMPLE_HEADER_DATA)

        to_bytes = header.to_bytes()
        self.assertEqual(EXAMPLE_HEADER_DATA, to_bytes)
