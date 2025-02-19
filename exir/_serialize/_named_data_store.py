# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import hashlib
from dataclasses import dataclass

# from dataclasses import dataclass
from typing import Dict, List, Optional


def gcd(a: int, b: int) -> int:
    while b:
        a, b = b, a % b
    return a


def lcm(a: int, b: int) -> int:
    return (a * b) // gcd(a, b)


@dataclass
class BufferEntry:
    """A class to hold the buffer entries for serialization.

    Attributes:
        buffer: The buffer bytes.
        alignment: The alignment of the buffer.
    """

    buffer: bytes
    alignment: int


@dataclass
class NamedDataStoreOutput:
    """
    A class to hold the named data for serialization.

    Attributes:
        buffer: A list of unique buffer entries.
        pte_data: Contains data that is stored inside the PTE file. A mapping from
            {key: buffer_index}.
        external_data: Contains data that is stored external to the PTE. A mapping
            from {filename: {key: buffer_index}}.
    """

    buffers: List[BufferEntry]
    pte_data: Dict[str, int]
    external_data: Dict[str, Dict[str, int]]


class NamedDataStore:
    """
    NamedDataStore manages the data that delegates want to share. Backends add
    bytes to the store under a unique key. These bytes can be retrieved at
    runtime using the same key with the NamedDataMap.

    Note:
    - Keys are unique in the data store, regardless of whether they are stored
        in the PTE or externally.
    - Multiple keys can point to the same buffer entry.
    - The same data can be added multiple times; all keys will point to one
        buffer. If a duplicate blob is added with a different alignment, the
        lcm of the current and new alignment is taken for that blob.
    """

    # List of unique blobs.
    buffers: List[BufferEntry]
    # Named data stored inside the PTE file. Map of {key: buffer_index}.
    pte_data: Dict[str, int]
    # Named data stored outside of the PTE file.
    # Map of {filename: {key: buffer_index}}.
    external_data: Dict[str, Dict[str, int]]

    # Cache of the data hash for deduplication.
    data_cache: Dict[str, int]
    # Cache of the keys to ensure uniqueness.
    key_cache: Dict[str, int]

    def __init__(self) -> None:
        """
        Initializes a new NamedDataStore.
        """
        self.buffers = []
        self.pte_data = {}
        self.external_data = {}

        self.data_cache = {}
        self.key_cache = {}

    def _add_named_data_to_map(
        self, key: str, data: bytes, alignment: int, map: Dict[str, int]
    ) -> None:
        """
        Add data to a map and update the alignment. Ensure that the key-data
        pair is unique.
        - If the key exists, the data must be identical.
        - If multiple unique keys exist for the same data, those keys should
            point to the same buffer.

        Args:
            key (str): key associated with the data.
            data (bytes): Bytes being requested to be serialized.
            alignment (int): alignment for bytes to be serialized with.
            map (Dict[str, int]): map to add the data to.
        Raises:
            ValueError: when the key exists in the store, and corresponding data
                is different.
        """
        # Check if the key exists.
        buffer_idx = self.key_cache.get(key, -1)
        if buffer_idx != -1:
            # If the key exists, the corresponding data must be identical.
            if self.buffers[buffer_idx].buffer != data:
                raise ValueError(f"Duplicate key {key} with different data.")
            self.buffers[buffer_idx].alignment = lcm(
                self.buffers[buffer_idx].alignment, alignment
            )
        else:
            # Key doesn't exist; check if the data exists.
            hashed = hashlib.sha256(data).hexdigest()
            buffer_idx = self.data_cache.get(hashed, -1)
            if buffer_idx != -1:
                # The data exists; update the alignment.
                self.buffers[buffer_idx].alignment = lcm(
                    self.buffers[buffer_idx].alignment, alignment
                )
            else:
                # The data doesn't exist; add it to the data store.
                buffer_idx = len(self.buffers)
                self.buffers.append(BufferEntry(data, alignment))
                self.data_cache[hashed] = buffer_idx

            # Add key to the map and the key cache.
            map[key] = buffer_idx
            self.key_cache[key] = buffer_idx

    def add_named_data(
        self,
        key: str,
        data: bytes,
        alignment: Optional[int] = 1,
        external_tag: Optional[str] = None,
    ) -> None:
        """
        Adds a named blob to the NamedDataStore.
        Args:
            key (str): key associated with the data.
            data (bytes): Bytes being requested to be serialized.
            alignment (int): alignment for bytes to be serialized with.
            external (Optional[str]): the external filename that this data is saved to.
        Raises:
            ValueError: when the key exists in the store, and corresponding data
                is different.
        """

        # Set default alignment.
        if alignment is None:
            alignment = 1

        if external_tag is None:
            self._add_named_data_to_map(key, data, alignment, self.pte_data)
        else:
            if self.external_data.get(external_tag, None) is None:
                self.external_data[external_tag] = {}
            self._add_named_data_to_map(
                key, data, alignment, self.external_data[external_tag]
            )

    def get_named_data_store_output(self) -> NamedDataStoreOutput:
        # Clean up empty maps inside self.external_data
        self.external_data = {k: v for k, v in self.external_data.items() if len(v) > 0}
        return NamedDataStoreOutput(self.buffers, self.pte_data, self.external_data)
