# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


import os
import unittest

from executorch.exir._serialize._cord import Cord


TENSOR_ALIGNMENT = 16


def _padding_required(offset: int, alignment: int) -> int:
    """Returns the padding required to align `offset` to `alignment`."""
    remainder: int = offset % alignment
    if remainder != 0:
        return alignment - remainder
    return 0


class TestCord(unittest.TestCase):
    def setUp(self) -> None:
        super().setUp()

    def test_cord_init(self) -> None:
        cord = Cord()
        self.assertEqual(0, cord.get_byte_size())
        self.assertEqual(0, len(cord))

    def test_cord_append_data(self) -> None:
        cord = Cord()
        cord.append(b"Hello")  # bytes
        self.assertEqual(5, cord.get_byte_size())
        self.assertEqual(1, len(cord))
        self.assertEqual(b"Hello", cord.to_bytes())

        cord.append(bytearray(b"World"))  # bytearray
        self.assertEqual(10, cord.get_byte_size())
        self.assertEqual(2, len(cord))
        self.assertEqual(b"HelloWorld", cord.to_bytes())

    def test_cord_append_empty_data(self) -> None:
        cord = Cord()
        with self.assertRaises(AssertionError):
            cord.append(b"")

    def test_cord_append_cord(self) -> None:
        cord = Cord()
        cord.append(b"Hello")
        cord.append(bytearray(b"World"))

        cord2 = Cord()
        cord2.append(b"Prefix")
        cord2.append_cord(cord)

        self.assertEqual(16, cord2.get_byte_size())
        self.assertEqual(3, len(cord2))
        self.assertEqual(b"PrefixHelloWorld", cord2.to_bytes())

        # Confirm that no copies were made when appending a Cord.
        self.assertEqual(id(cord2[1]), id(cord[0]))
        self.assertEqual(id(cord2[2]), id(cord[1]))

    def test_cord_insert_padding(self) -> None:
        const_buffers = Cord()
        const_buffers.append(b"c")
        const_buffers.append(b"const")
        const_buffers.append(b"constant")

        const_buffers_padded = Cord()

        for i, buffer in enumerate(const_buffers):
            const_buffers_padded.append(buffer)
            if i < len(const_buffers) - 1:
                padding_required = _padding_required(len(buffer), TENSOR_ALIGNMENT)
                const_buffers_padded.append(b"\x00" * padding_required)

        self.assertEqual(TENSOR_ALIGNMENT * 2 + 8, const_buffers_padded.get_byte_size())
        self.assertEqual(5, len(const_buffers_padded))
        self.assertEqual(
            b"c" + b"\x00" * 15 + b"const" + b"\x00" * 11 + b"constant",
            const_buffers_padded.to_bytes(),
        )

        # Confirm that no copies of original data were made when inserting padding.
        self.assertEqual(id(const_buffers[0]), id(const_buffers_padded[0]))
        self.assertEqual(id(const_buffers[1]), id(const_buffers_padded[2]))
        self.assertEqual(id(const_buffers[2]), id(const_buffers_padded[4]))

    def test_cord_write_to_file(self) -> None:
        cord = Cord()
        cord.append(b"Hello")
        cord.append(bytearray(b"World"))

        cord.write_to_file("/tmp/test_cord.bin")

        with open("/tmp/test_cord.bin", "rb") as f:
            contents = f.read()
            self.assertEqual(b"HelloWorld", contents)

        os.remove("/tmp/test_cord.bin")
