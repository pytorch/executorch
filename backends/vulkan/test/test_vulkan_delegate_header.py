# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest

from executorch.backends.vulkan.serialization.vulkan_graph_serialize import (
    VulkanDelegateHeader,
)

EXAMPLE_FLATBUFFER_OFFSET: int = 0x11223344
EXAMPLE_FLATBUFFER_SIZE: int = 0x55667788
EXAMPLE_CONSTANTS_OFFSET: int = EXAMPLE_FLATBUFFER_OFFSET + EXAMPLE_FLATBUFFER_SIZE
EXAMPLE_CONSTANTS_SIZE: int = 0x99AABB
EXAMPLE_SHADERS_OFFSET: int = EXAMPLE_CONSTANTS_OFFSET + EXAMPLE_CONSTANTS_SIZE
EXAMPLE_SHADERS_SIZE: int = 0x11223344

# If header layout or magic changes, this test must change too.
# The layout of the header is a contract, not an implementation detail
EXAMPLE_HEADER_DATA: bytes = (
    # zeros
    b"\x00\x00\x00\x00"
    # magic
    + b"VKDG"
    # All Values below are littl Endian
    # header length
    + b"\x2A\x00"
    # Flatbuffer Offset
    + b"\x44\x33\x22\x11"
    # Flatbuffer Size
    + b"\x88\x77\x66\x55"
    # Constant Data Offset
    + b"\xCC\xAA\x88\x66"
    # Constant Data Size
    + b"\xBB\xAA\x99\x00\x00\x00\x00\x00"
    # Shader Offset
    + b"\x87\x55\x22\x67"
    # Shader Size
    + b"\x44\x33\x22\x11\x00\x00\x00\x00"
)


class TestVulkanDelegateHeader(unittest.TestCase):
    def test_to_bytes(self) -> None:
        header = VulkanDelegateHeader(
            EXAMPLE_FLATBUFFER_OFFSET,
            EXAMPLE_FLATBUFFER_SIZE,
            EXAMPLE_CONSTANTS_OFFSET,
            EXAMPLE_CONSTANTS_SIZE,
            EXAMPLE_SHADERS_OFFSET,
            EXAMPLE_SHADERS_SIZE,
        )
        self.assertEqual(header.to_bytes(), EXAMPLE_HEADER_DATA)
        self.assertTrue(header.is_valid())

    def test_from_bytes(self) -> None:
        header = VulkanDelegateHeader.from_bytes(EXAMPLE_HEADER_DATA)
        self.assertEqual(header.flatbuffer_offset, EXAMPLE_FLATBUFFER_OFFSET)
        self.assertEqual(header.flatbuffer_size, EXAMPLE_FLATBUFFER_SIZE)
        self.assertEqual(header.constants_offset, EXAMPLE_CONSTANTS_OFFSET)
        self.assertEqual(header.constants_size, EXAMPLE_CONSTANTS_SIZE)
        self.assertEqual(header.shaders_offset, EXAMPLE_SHADERS_OFFSET)
        self.assertEqual(header.shaders_size, EXAMPLE_SHADERS_SIZE)

    def test_invalid_metadata(self) -> None:
        WRONG_MAGIC_DATA = EXAMPLE_HEADER_DATA[0:4] + b"YT01" + EXAMPLE_HEADER_DATA[8:]
        with self.assertRaisesRegex(
            ValueError,
            "Expected magic bytes to be b'VKDG', but got b'YT01'",
        ):
            VulkanDelegateHeader.from_bytes(WRONG_MAGIC_DATA)

        WRONG_LENGTH_DATA = (
            EXAMPLE_HEADER_DATA[0:8] + b"\x1D\x00" + EXAMPLE_HEADER_DATA[10:]
        )
        with self.assertRaisesRegex(
            ValueError, "Expected header to be 42 bytes, but got 29 bytes."
        ):
            VulkanDelegateHeader.from_bytes(WRONG_LENGTH_DATA)

        with self.assertRaisesRegex(
            ValueError, "Expected header to be 42 bytes, but got 43 bytes."
        ):
            VulkanDelegateHeader.from_bytes(EXAMPLE_HEADER_DATA + b"\x00")

    def test_invalid_flatbuffer_size(self) -> None:
        header = VulkanDelegateHeader(
            EXAMPLE_FLATBUFFER_OFFSET,
            0,
            EXAMPLE_CONSTANTS_OFFSET,
            EXAMPLE_CONSTANTS_SIZE,
            EXAMPLE_SHADERS_OFFSET,
            EXAMPLE_SHADERS_SIZE,
        )

        with self.assertRaises(ValueError):
            header.to_bytes()

    def test_invalid_constants_offset(self) -> None:
        header = VulkanDelegateHeader(
            EXAMPLE_FLATBUFFER_OFFSET,
            EXAMPLE_FLATBUFFER_SIZE,
            EXAMPLE_FLATBUFFER_OFFSET + EXAMPLE_FLATBUFFER_SIZE - 1,
            EXAMPLE_CONSTANTS_SIZE,
            EXAMPLE_SHADERS_OFFSET,
            EXAMPLE_SHADERS_SIZE,
        )

        with self.assertRaises(ValueError):
            header.to_bytes()

    def test_to_bytes_same_as_from_bytes(self) -> None:
        header = VulkanDelegateHeader.from_bytes(EXAMPLE_HEADER_DATA)

        to_bytes = header.to_bytes()
        self.assertEquals(EXAMPLE_HEADER_DATA, to_bytes)
