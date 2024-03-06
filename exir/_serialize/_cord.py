# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import io
from typing import List, Optional, Union


class Cord:
    """A `bytes`-like sequence of bytes, stored non-contiguously.

    Users can use a Cord to assemble large files and data blobs using references
    to and slices of other data, instead of copying and appending that data to a
    `bytes` or `bytearray` object.
    """

    def __init__(self, data: Optional[Union[bytes, "Cord"]] = None) -> None:
        """Initialize Cord data structure."""
        self._buffers: List[bytes] = []
        self._byte_size: int = 0

        if data is not None:
            self.append(data)

    def __len__(self):
        """Number of bytes in the Cord."""
        return self._byte_size

    def __bytes__(self) -> bytes:
        """Return the contents of the Cord as a single `bytes` object."""
        return b"".join(self._buffers)

    def append(self, data: Union[bytes, "Cord"]) -> None:
        """Append a bytes or Cord to the current Cord."""
        if isinstance(data, bytes):
            self._buffers.append(data)
            self._byte_size += len(data)
        elif isinstance(data, Cord):
            self._buffers.extend(data._buffers)
            self._byte_size += len(data)
        else:
            raise TypeError(f"Can only append bytes or Cords, received {type(data)}")

    def write_to_file(self, outfile: io.BufferedIOBase) -> None:
        """Write the Cord to a file."""
        for item in self._buffers:
            outfile.write(item)
