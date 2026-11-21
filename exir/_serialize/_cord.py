# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import contextlib
import errno
import hashlib
import io
import os
import shutil
import sys
import tempfile
import weakref
from typing import Collection, Dict, IO, List, Optional, Union


# Errors that mean a file system can't do a full sync.
_NO_FULL_SYNC = (errno.EINVAL, errno.ENOTSUP, errno.EOPNOTSUPP, errno.ENOTTY)


def _sync(fd: int) -> None:
    if sys.platform == "darwin":
        import fcntl

        # On macOS, fsync leaves the data in the drive's cache.
        try:
            fcntl.fcntl(fd, fcntl.F_FULLFSYNC)
            return
        except OSError as e:
            if e.errno not in _NO_FULL_SYNC:
                raise
    os.fsync(fd)


class FileBackedData:
    """A byte buffer that stays on disk until explicitly closed, or moved by a consuming
    save, after which reading its bytes raises."""

    _COPY_CHUNK_SIZE = 8 * 1024 * 1024
    # Larger, because each chunk that moves is synced to disk.
    _MOVE_CHUNK_SIZE = 64 * 1024 * 1024

    def __init__(
        self,
        path: str,
        cleanup: bool = False,
        sha256: Optional[bytes] = None,
    ) -> None:
        self._path = path
        self._size = os.path.getsize(path)
        self._sha256 = sha256
        self._finalizer = (
            weakref.finalize(self, self._remove, path) if cleanup else None
        )

    @staticmethod
    def _remove(path: str) -> None:
        try:
            os.remove(path)
        except OSError:
            pass

    @classmethod
    def move_from(cls, path: str, sha256: Optional[bytes] = None) -> "FileBackedData":
        """Take ownership of ``path`` without loading its contents."""
        directory = os.path.dirname(path) or "."
        fd, owned_path = tempfile.mkstemp(
            prefix=".executorch_", suffix=".data", dir=directory
        )
        os.close(fd)
        try:
            os.replace(path, owned_path)
        except Exception:
            os.remove(owned_path)
            raise
        return cls(owned_path, cleanup=True, sha256=sha256)

    def __len__(self) -> int:
        return self._size

    def __deepcopy__(self, memo: Dict[int, object]) -> "FileBackedData":
        # A copy would not own the file, so a program made by to_executorch() on a deep
        # copy of a lowered EdgeProgramManager could not move it. So the "copy" is this
        # object, and closing it deletes the file.
        return self

    @property
    def path(self) -> str:
        return self._path

    def _check_size(self, size: int) -> None:
        if size != self._size:
            raise RuntimeError(
                f"{self._path} has {size} bytes, expected {self._size}; it changed "
                "after it was wrapped, for example because a consuming save moved it"
            )

    def stat(self) -> os.stat_result:
        """Stat the file, raising if its size changed, as it does after a move."""
        result = os.stat(self._path)
        self._check_size(result.st_size)
        return result

    def _open(self) -> IO[bytes]:
        f = open(self._path, "rb")
        try:
            self._check_size(os.fstat(f.fileno()).st_size)
        except BaseException:
            f.close()
            raise
        return f

    def prefix(self, size: int) -> bytes:
        with self._open() as f:
            return f.read(size)

    def sha256(self) -> bytes:
        if self._sha256 is None:
            digest = hashlib.sha256()
            with self._open() as f:
                while chunk := f.read(self._COPY_CHUNK_SIZE):
                    digest.update(chunk)
            self._sha256 = digest.digest()
        return self._sha256

    def to_bytes(self) -> bytes:
        with self._open() as f:
            return f.read()

    def write_to_file(self, outfile: io.BufferedIOBase) -> None:
        with self._open() as f:
            shutil.copyfileobj(f, outfile, length=self._COPY_CHUNK_SIZE)

    def _can_move(self) -> bool:
        """Whether this object owns the file, can write it, and the file has no other
        name."""
        return (
            self._finalizer is not None
            and self._finalizer.alive
            and not os.path.islink(self._path)
            and os.access(self._path, os.W_OK)
            and os.stat(self._path).st_nlink == 1
        )

    def _move_to(self, outfile: io.BufferedIOBase, offset: int) -> None:
        """Move the data into ``outfile`` at ``offset``, from the end back to the start,
        syncing each chunk before cutting it off the file, so the data is on disk once
        plus one chunk, on file systems that leave holes in a file unallocated."""
        if not self._can_move():
            raise ValueError(f"{self._path} can't be moved")
        with open(self._path, "r+b") as source:
            self._check_size(os.fstat(source.fileno()).st_size)
            end = self._size
            try:
                # Once data leaves the file, the rest of it must outlive this object.
                self._finalizer.detach()
                while end > 0:
                    begin = max(end - self._MOVE_CHUNK_SIZE, 0)
                    source.seek(begin)
                    chunk = source.read(end - begin)
                    if len(chunk) != end - begin:
                        raise OSError(f"{self._path} shrank while it was moved")
                    outfile.seek(offset + begin)
                    outfile.write(chunk)
                    outfile.flush()
                    _sync(outfile.fileno())
                    source.truncate(begin)
                    end = begin
            except BaseException as e:
                with contextlib.suppress(OSError):
                    end = os.fstat(source.fileno()).st_size
                if end == self._size:
                    self._finalizer = weakref.finalize(self, self._remove, self._path)
                else:
                    e.add_note(
                        f"Bytes [0, {end}) of the data are still in {self._path}, and "
                        f"the rest is at offset {offset + end} of the output."
                    )
                raise
        self._finalizer = weakref.finalize(self, self._remove, self._path)

    def close(self) -> None:
        if self._finalizer is not None:
            self._finalizer()

    def __enter__(self) -> "FileBackedData":
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()


CordBuffer = Union[bytes, FileBackedData]


class Cord:
    """A `bytes`-like sequence of bytes, stored non-contiguously.

    Users can use a Cord to assemble large files and data blobs using references
    to and slices of other data, instead of copying and appending that data to a
    `bytes` or `bytearray` object.
    """

    def __init__(self, data: Optional[Union[CordBuffer, "Cord"]] = None) -> None:
        """Initialize Cord data structure."""
        self._buffers: List[CordBuffer] = []
        self._byte_size: int = 0

        if data is not None:
            self.append(data)

    def __len__(self):
        """Number of bytes in the Cord."""
        return self._byte_size

    def __bytes__(self) -> bytes:
        """Return the contents of the Cord as a single `bytes` object."""
        return b"".join(
            item if isinstance(item, bytes) else item.to_bytes()
            for item in self._buffers
        )

    def append(self, data: Union[CordBuffer, "Cord"]) -> None:
        """Append a bytes or Cord to the current Cord."""
        if isinstance(data, (bytes, FileBackedData)):
            self._buffers.append(data)
            self._byte_size += len(data)
        elif isinstance(data, Cord):
            self._buffers.extend(data._buffers)
            self._byte_size += len(data)
        else:
            raise TypeError(
                f"Can only append bytes, FileBackedData, or Cords, received {type(data)}"
            )

    def file_backed_buffers(self) -> List[FileBackedData]:
        return [item for item in self._buffers if isinstance(item, FileBackedData)]

    def write_to_file(self, outfile: io.BufferedIOBase) -> None:
        """Write the Cord to a file."""
        for item in self._buffers:
            if isinstance(item, bytes):
                outfile.write(item)
            else:
                item.write_to_file(outfile)

    def _write_and_move(
        self, outfile: io.BufferedIOBase, move: Collection[FileBackedData]
    ) -> None:
        """Write the Cord to ``outfile``, a regular file, moving the buffers in ``move``
        into it. Each of them must appear once and have ``_can_move`` True. Everything
        else is written first, so a full disk shows before any data moves."""
        offsets = [outfile.tell()]
        for item in self._buffers:
            offsets.append(offsets[-1] + len(item))
        moving = [
            isinstance(item, FileBackedData) and item in move for item in self._buffers
        ]
        for item, offset, moves in zip(self._buffers, offsets, moving):
            if not moves:
                outfile.seek(offset)
                if isinstance(item, bytes):
                    outfile.write(item)
                else:
                    item.write_to_file(outfile)
        for item, offset, moves in zip(self._buffers, offsets, moving):
            if moves:
                item._move_to(outfile, offset)
        outfile.seek(offsets[-1])
