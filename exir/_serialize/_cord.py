# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import hashlib
import io
import os
import shutil
import tempfile
import weakref
from typing import IO, List, Optional, Union


class FileBackedData:
    """A byte buffer that stays on disk until explicitly closed.

    With ``consume_on_write``, the first write into a seekable file moves the data there
    instead of copying it, so the data is never on disk twice: chunks are copied from
    the end back to the start, and ``path`` is truncated after each one, while the part
    of the output not yet written stays a hole. Afterwards ``path`` is empty and the
    data can't be read again. For a caller that deletes ``path`` after the write.
    """

    _COPY_CHUNK_SIZE = 8 * 1024 * 1024

    def __init__(
        self,
        path: str,
        cleanup: bool = False,
        sha256: Optional[bytes] = None,
        consume_on_write: bool = False,
    ) -> None:
        self._path = path
        self._size = os.path.getsize(path)
        self._sha256 = sha256
        self._consume_on_write = consume_on_write
        self._moved = False
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

    def _open(self, mode: str = "rb") -> IO[bytes]:
        if self._moved:
            raise RuntimeError(
                f"The data of {self._path} was moved into the file it was written to"
            )
        return open(self._path, mode)

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
        # In append mode every write lands at the end, not at the chunk's offset.
        if (
            self._consume_on_write
            and outfile.seekable()
            and "a" not in getattr(outfile, "mode", "")
        ):
            self._move_to_file(outfile)
            return
        with self._open() as f:
            shutil.copyfileobj(f, outfile, length=self._COPY_CHUNK_SIZE)

    def _move_to_file(self, outfile: io.BufferedIOBase) -> None:
        start = outfile.tell()
        with self._open("r+b") as source:
            # Set before any truncation, so a partly moved file is never read as whole.
            self._moved = True
            end = self._size
            while end > 0:
                begin = max(end - self._COPY_CHUNK_SIZE, 0)
                source.seek(begin)
                chunk = source.read(end - begin)
                if len(chunk) != end - begin:
                    raise RuntimeError(
                        f"{self._path} no longer holds the {self._size} bytes to move"
                    )
                outfile.seek(start + begin)
                outfile.write(chunk)
                source.truncate(begin)
                end = begin
        outfile.seek(start + self._size)

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

    def write_to_file(self, outfile: io.BufferedIOBase) -> None:
        """Write the Cord to a file."""
        for item in self._buffers:
            if isinstance(item, bytes):
                outfile.write(item)
            else:
                item.write_to_file(outfile)
