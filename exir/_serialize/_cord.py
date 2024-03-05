# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import List, Union


class Cord:
    """
    DOC_STRING
    """

    def __init__(
        self,
    ) -> None:
        """
        INIT_DOC_STRING
        """
        self.buffers: List[Union[bytes, bytearray]] = []
        self._byte_size: int = 0
        self._current_index: int = 0

    def append(self, data: Union[bytes, bytearray]) -> None:
        """
        APPEND_DOC_STRING
        """
        # assert len(data) > 0
        if len(data) == 0:
            return
        self.buffers.append(data)
        self._byte_size += len(data)

    def append_cord(self, cord: "Cord") -> None:
        """
        APPEND_DOC_STRING
        """
        self.buffers.extend(cord.buffers)
        self._byte_size += cord.get_byte_size()

    def __len__(self):
        """
        LEN_DOC_STRING
        """
        return len(self.buffers)

    def get_byte_size(self):
        """
        DATA_SIZE_DOC_STRING
        """
        return self._byte_size

    def write_to_file(self, file_path: str) -> None:
        """
        WRITE_TO_FILE_DOC_STRING
        """
        with open(file_path, "wb") as outfile:
            for item in self.buffers:
                outfile.write(item)

    def to_bytes(self) -> bytes:
        """
        TO_BYTES_DOC_STRING
        """
        return b"".join(self.buffers)

    def __iter__(self):
        return self

    def __next__(self):
        if self._current_index >= len(self.buffers):
            raise StopIteration
        else:
            result = self.buffers[self._current_index]
            self._current_index += 1
            return result

    def __getitem__(self, index):
        return self.buffers[index]
