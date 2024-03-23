# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


import io
import unittest

from executorch.exir._serialize._cord import Cord


class TestCord(unittest.TestCase):
    def test_cord_init(self) -> None:
        cord_empty = Cord()
        self.assertEqual(0, len(cord_empty))

        cord = Cord(b"HelloWorld")
        self.assertEqual(10, len(cord))
        self.assertEqual(b"HelloWorld", bytes(cord))

        cord2 = Cord(cord)
        self.assertEqual(10, len(cord2))
        self.assertEqual(b"HelloWorld", bytes(cord))

        # Confirm no copies were made.
        self.assertEqual(id(cord._buffers[0]), id(cord2._buffers[0]))

    def test_cord_append(self) -> None:
        cord = Cord()
        cord.append(b"Hello")
        self.assertEqual(5, len(cord))
        self.assertEqual(b"Hello", bytes(cord))

        cord.append(b"World")
        self.assertEqual(10, len(cord))
        self.assertEqual(b"HelloWorld", bytes(cord))

    def test_cord_append_cord(self) -> None:
        cord = Cord()
        cord.append(b"Hello")
        cord.append((b"World"))

        cord2 = Cord()
        cord2.append(b"Prefix")
        cord2.append(cord)

        self.assertEqual(16, len(cord2))
        self.assertEqual(b"PrefixHelloWorld", bytes(cord2))

        # Confirm that no copies were made when appending a Cord.
        self.assertEqual(id(cord2._buffers[1]), id(cord._buffers[0]))
        self.assertEqual(id(cord2._buffers[2]), id(cord._buffers[1]))

    def test_cord_write_to_file(self) -> None:
        cord = Cord()
        cord.append(b"Hello")
        cord.append(b"World")

        outfile = io.BytesIO()
        cord.write_to_file(outfile)
        self.assertEqual(b"HelloWorld", outfile.getvalue())
