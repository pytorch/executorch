# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


import copy
import errno
import io
import os
import sys
import tempfile
import unittest
from unittest import mock

from executorch.exir._serialize import _cord
from executorch.exir._serialize._cord import Cord, FileBackedData

# No two 256-byte chunks are the same, so a chunk written to the wrong place shows.
_DATA = bytes(i % 251 for i in range(1024))
_HEAD = b"head" * 4


class _Disk(io.BufferedWriter):
    """A file on a disk with `space` bytes free."""

    def __init__(self, path, space):
        super().__init__(io.FileIO(path, "w"))
        self.space = space

    def write(self, data):
        if len(data) > self.space:
            raise OSError(errno.ENOSPC, "No space left on device")
        self.space -= len(data)
        return super().write(data)


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


@mock.patch.object(FileBackedData, "_MOVE_CHUNK_SIZE", 256)
class TestMove(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.dir = directory.name
        self.output = os.path.join(self.dir, "output")

    def file(self, name="data", content=_DATA):
        path = os.path.join(self.dir, name)
        with open(path, "wb") as f:
            f.write(content)
        return path

    def owned(self, name="data", content=_DATA):
        return FileBackedData.move_from(self.file(name, content))

    def cord(self, *data):
        cord = Cord(_HEAD)
        for item in data:
            cord.append(item)
        cord.append(b"tail")
        return cord

    def write(self, data, space=None):
        with _Disk(self.output, sys.maxsize if space is None else space) as f:
            self.cord(data)._write_and_move(f, {data})

    def read_output(self):
        with open(self.output, "rb") as f:
            return f.read()

    def test_moves_owned_data_and_keeps_owning_the_emptied_file(self):
        data = self.owned()
        self.assertTrue(data._can_move())
        self.write(data)
        self.assertEqual(_HEAD + _DATA + b"tail", self.read_output())
        self.assertEqual(0, os.path.getsize(data.path))
        for read in (data.to_bytes, data.sha256, lambda: data.prefix(1)):
            with self.assertRaises(RuntimeError):
                read()
        data.close()
        self.assertFalse(os.path.exists(data.path))

    def test_moves_several_buffers_and_stops_at_the_first_that_fails(self):
        first = self.owned("first")
        second = self.owned("second", _DATA[::-1])
        with open(self.output, "wb") as f:
            self.cord(first, b"middle", second)._write_and_move(f, {first, second})
        self.assertEqual(
            _HEAD + _DATA + b"middle" + _DATA[::-1] + b"tail", self.read_output()
        )
        self.assertEqual([0, 0], [os.path.getsize(d.path) for d in (first, second)])

        first = self.owned("first")
        second = self.owned("second")
        with open(self.output, "wb") as f, mock.patch.object(
            second, "_move_to", side_effect=OSError(errno.EIO, "")
        ), self.assertRaises(OSError):
            self.cord(first, second)._write_and_move(f, {first, second})
        self.assertEqual(0, os.path.getsize(first.path))
        self.assertEqual(_DATA, second.to_bytes())
        self.assertEqual(bytes(16), self.read_output()[:16])

    def test_data_moves_only_into_a_cord_that_starts_with_16_bytes(self):
        data = self.owned()
        cord = Cord(b"short")
        cord.append(data)
        with open(self.output, "wb") as f, self.assertRaises(ValueError):
            cord._write_and_move(f, {data})
        self.assertEqual(_DATA, data.to_bytes())

    def test_a_read_raises_if_the_file_is_cut_while_it_is_read(self):
        data = self.owned()
        open_data = FileBackedData._open

        def open_then_cut(data):
            f = open_data(data)
            os.truncate(data.path, 10)
            return f

        reads = {
            "to_bytes": data.to_bytes,
            "sha256": data.sha256,
            "write_to_file": lambda: data.write_to_file(io.BytesIO()),
        }
        for name, read in reads.items():
            with self.subTest(name), mock.patch.object(
                FileBackedData, "_open", open_then_cut
            ):
                with open(data.path, "wb") as f:
                    f.write(_DATA)
                with self.assertRaises(RuntimeError):
                    read()

    def test_data_that_another_save_is_moving_cannot_move(self):
        data = self.owned()
        data._finalizer.detach()
        with mock.patch.object(data, "_can_move", return_value=True):
            with self.assertRaisesRegex(ValueError, "already being moved"):
                self.write(data)
        self.assertEqual(len(_DATA), os.path.getsize(data.path))

    def test_a_deep_copy_can_move_the_data(self):
        data = self.owned()
        self.write(copy.deepcopy(data))
        self.assertEqual(0, os.path.getsize(data.path))

    def test_data_that_is_not_safe_to_cut_short_cannot_move(self):
        def not_owned():
            return FileBackedData(self.file("plain"))

        def linked():
            data = self.owned()
            os.link(data.path, os.path.join(self.dir, "other"))
            return data

        def symlink():
            link = os.path.join(self.dir, "link")
            os.symlink(self.file("target"), link)
            return FileBackedData(link, cleanup=True)

        def read_only():
            data = self.owned()
            os.chmod(data.path, 0o444)
            return data

        cases = {"not owned": not_owned, "hard link": linked}
        if sys.platform != "win32":
            cases["symlink"] = symlink
            if os.geteuid() != 0:
                cases["read-only"] = read_only
        for name, make in cases.items():
            with self.subTest(name):
                data = make()
                self.assertFalse(data._can_move())
                with self.assertRaises(ValueError):
                    self.write(data)
                self.assertEqual(_DATA, data.to_bytes())

    def test_data_whose_file_changed_size_does_not_move(self):
        grown = self.owned()
        with open(grown.path, "ab") as f:
            f.write(b"more")
        with self.assertRaises(RuntimeError):
            self.write(grown)
        self.assertEqual(len(_DATA) + 4, os.path.getsize(grown.path))
        grown.close()
        self.assertFalse(os.path.exists(grown.path))

        # As if the file shrank after the size check.
        shrunk = self.owned()
        os.truncate(shrunk.path, len(_DATA) - 1)
        with mock.patch.object(shrunk, "_check_size"):
            with self.assertRaisesRegex(OSError, "shrank while it was moved"):
                self.write(shrunk)

    @unittest.skipIf(sys.platform == "win32", "no fcntl there")
    def test_a_sync_on_macos_falls_back_only_where_a_full_sync_is_unsupported(self):
        import fcntl

        for error in (errno.ENOTSUP, errno.EIO):
            with self.subTest(error=errno.errorcode[error]), mock.patch.object(
                sys, "platform", "darwin"
            ), mock.patch.object(
                fcntl, "F_FULLFSYNC", 51, create=True
            ), mock.patch.object(
                fcntl, "fcntl", side_effect=OSError(error, "")
            ), mock.patch.object(
                os, "fsync"
            ) as fsync:
                if error == errno.EIO:
                    with self.assertRaises(OSError):
                        _cord._sync(0)
                    fsync.assert_not_called()
                else:
                    _cord._sync(0)
                    fsync.assert_called_once_with(0)

    def test_each_chunk_is_in_the_output_before_it_leaves_the_file_and_head_is_last(
        self,
    ):
        data = self.owned()
        synced = []

        def sync(fd):
            self.assertTrue(os.path.samestat(os.fstat(fd), os.stat(self.output)))
            size = os.path.getsize(data.path)
            begin = max(size - 256, 0)
            output = self.read_output()
            moved = output[len(_HEAD) + begin : len(_HEAD) + len(_DATA)]
            synced.append((output[:16], size, moved == _DATA[begin:]))

        with mock.patch.object(_cord, "_sync", sync):
            self.write(data)
        zeros = bytes(16)
        self.assertEqual(
            [(zeros, size, True) for size in (1024, 768, 512, 256)]
            + [(_HEAD, 0, True)],
            synced,
        )

    def test_a_full_disk_before_data_leaves_the_file_changes_nothing(self):
        data = self.owned()
        with self.assertRaises(OSError):
            # Room for the head and the tail, but not also a chunk.
            self.write(data, space=len(_HEAD + b"tail") + 256 - 1)
        self.assertEqual(_DATA, data.to_bytes())
        data.close()
        self.assertFalse(os.path.exists(data.path))

    def test_a_full_disk_during_the_move_keeps_every_byte(self):
        data = self.owned()
        with self.assertRaises(OSError) as raised:
            self.write(data, space=len(_HEAD + b"tail") + 2 * 256)
        split = len(_DATA) - 2 * 256
        self.assertIn(
            f"Bytes [0, {split}) of the data are still in {data.path}",
            "".join(raised.exception.__notes__),
        )
        data.close()
        with open(data.path, "rb") as f:
            self.assertEqual(_DATA[:split], f.read())
        output = self.read_output()
        self.assertEqual(_DATA[split:], output[16 + split : 16 + len(_DATA)])
        self.assertEqual(bytes(16), output[:16])
