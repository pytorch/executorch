# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


import io
import os
import tempfile
import unittest
from unittest import mock

from executorch.exir._serialize._cord import Cord, FileBackedData


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

    def test_file_backed_data(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            source_path = os.path.join(directory, "source.bin")
            with open(source_path, "wb") as f:
                f.write(b"FileBacked")

            with FileBackedData.move_from(source_path) as data:
                self.assertFalse(os.path.exists(source_path))
                self.assertEqual(10, len(data))

                cord = Cord(b"Prefix")
                cord.append(data)
                outfile = io.BytesIO()
                cord.write_to_file(outfile)
                self.assertEqual(b"PrefixFileBacked", outfile.getvalue())
                self.assertEqual(b"PrefixFileBacked", bytes(cord))

            self.assertEqual([], os.listdir(directory))

    def test_file_backed_data_moves_into_a_file_without_a_second_copy(self) -> None:
        payload = bytes(range(256)) * 41  # 10 chunks and part of one
        chunk = 1000
        with tempfile.TemporaryDirectory() as directory:
            source_path = os.path.join(directory, "source.bin")
            with open(source_path, "wb") as f:
                f.write(payload)
            written, on_disk = [0], []

            class Recording(io.FileIO):
                # Bytes on disk after each write: the source left plus all written.
                def write(self, b) -> int:
                    size = super().write(b)
                    written[0] += size
                    on_disk.append(os.path.getsize(source_path) + written[0])
                    return size

            data = FileBackedData(source_path, consume_on_write=True)
            cord = Cord(b"Prefix")
            cord.append(data)
            cord.append(b"Suffix")
            out_path = os.path.join(directory, "out.bin")
            with mock.patch.object(FileBackedData, "_COPY_CHUNK_SIZE", chunk):
                with io.BufferedWriter(Recording(out_path, "wb")) as outfile:
                    cord.write_to_file(outfile)

            with open(out_path, "rb") as result:
                self.assertEqual(b"Prefix" + payload + b"Suffix", result.read())
            self.assertEqual(0, os.path.getsize(source_path))
            self.assertLessEqual(
                max(on_disk), len(b"PrefixSuffix") + len(payload) + chunk
            )
            with self.assertRaises(RuntimeError):
                data.to_bytes()

    def test_file_backed_data_is_copied_into_a_stream_it_cannot_seek(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            source_path = os.path.join(directory, "source.bin")
            with open(source_path, "wb") as f:
                f.write(b"FileBacked")
            data = FileBackedData(source_path, consume_on_write=True)
            read, write = os.pipe()
            with open(write, "wb") as outfile:
                data.write_to_file(outfile)
            with open(read, "rb") as infile:
                self.assertEqual(b"FileBacked", infile.read())
            self.assertEqual(b"FileBacked", data.to_bytes())

    def test_file_backed_data_is_copied_into_a_file_opened_to_append(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            source_path = os.path.join(directory, "source.bin")
            with open(source_path, "wb") as f:
                f.write(b"FileBacked")
            out_path = os.path.join(directory, "out.bin")
            with open(out_path, "wb") as f:
                f.write(b"Prefix")
            data = FileBackedData(source_path, consume_on_write=True)
            with mock.patch.object(FileBackedData, "_COPY_CHUNK_SIZE", 4):
                with open(out_path, "ab") as outfile:
                    data.write_to_file(outfile)
            with open(out_path, "rb") as result:
                self.assertEqual(b"PrefixFileBacked", result.read())

    def test_file_backed_data_refuses_to_move_a_file_that_shrank(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            source_path = os.path.join(directory, "source.bin")
            with open(source_path, "wb") as f:
                f.write(b"FileBacked")
            first = FileBackedData(source_path, consume_on_write=True)
            second = FileBackedData(source_path, consume_on_write=True)
            with open(os.path.join(directory, "out.bin"), "wb") as outfile:
                first.write_to_file(outfile)
                with self.assertRaises(RuntimeError):
                    second.write_to_file(outfile)
