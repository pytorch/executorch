# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import hashlib
from dataclasses import dataclass

# from dataclasses import dataclass
from typing import Dict, Optional, Sequence

from executorch.exir._serialize.data_serializer import DataEntry
from executorch.exir.tensor_layout import TensorLayout


@dataclass
class NamedDataStoreOutput:
    """
    Holds named data for serialization. Note: a DataEntry contains the index into
    `buffers`, the alignment and a tensor layout, if applicable.

    Attributes:
        buffers: A list of unique buffer entries.
        pte_data: Contains data that is stored inside the PTE file. A mapping from
            {key: DataEntry}.
        external_data: Contains data that is stored external to the PTE. A mapping
            from {filename: {key: DataEntry}}.
    """

    buffers: Sequence[bytes]
    pte_data: Dict[str, DataEntry]
    external_data: Dict[str, Dict[str, DataEntry]]


class NamedDataStore:
    """
    NamedDataStore manages the data that delegates want to share. Backends add
    bytes to the store under a unique key. These bytes can be retrieved at
    runtime using the same key with the NamedDataMap.

    Note:
    - Keys are unique in the data store, regardless of whether they are stored
        in the PTE or externally.
    - Multiple keys can point to the same buffer entry.
    - The same data can be added multiple times and all keys will point to one
        buffer. If a duplicate blob is added with a different alignment, the
        lcm of the current and new alignment is taken for that blob.
    """

    # List of unique blobs.
    buffers: Sequence[bytes]
    # Named data stored inside the PTE file. Map of {key: DataEntry}.
    pte_data: Dict[str, DataEntry]
    # Named data stored outside of the PTE file.
    # Map of {filename: {key: DataEntry}}.
    external_data: Dict[str, Dict[str, DataEntry]]

    # Cache of the data hash for deduplication.
    # Use a hash instead of the data as a key because a sha256 collision is
    # unlikely, and the data may be large.
    data_hash_to_buffer_idx: Dict[bytes, int]
    # Cache of the key to buffer idx to ensure uniqueness.
    # If a key is added multiple times, check the buffer idx to ensure that the
    # data is identical too.
    key_to_buffer_idx: Dict[str, int]

    def __init__(self) -> None:
        """
        Initializes a new NamedDataStore.
        """
        self.buffers = []
        self.pte_data = {}
        self.external_data = {}

        self.data_hash_to_buffer_idx = {}
        self.key_to_buffer_idx = {}

    def _add_named_data_to_map(
        self,
        key: str,
        data: bytes,
        alignment: int,
        local_key_to_buffer_idx: Dict[str, DataEntry],
        tensor_layout: Optional[TensorLayout] = None,
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
            local_key_to_buffer_idx (Dict[str, int]): map to add the data to.
        Raises:
            ValueError: when the key exists in the store, and corresponding data
                is different.
        """
        # Get data hash.
        hashed = hashlib.sha256(data).digest()

        # Check if the key exists.
        buffer_idx = self.key_to_buffer_idx.get(key, -1)
        # If the key exists, the corresponding data must be identical.
        if (
            buffer_idx != -1
            and self.data_hash_to_buffer_idx.get(hashed, -1) != buffer_idx
        ):
            raise ValueError(
                f"Duplicate key {key} with different data. "
                f"Existing data: {self.buffers[buffer_idx]}. "
                f"New data: {data}."
            )
        else:
            # Key doesn't exist; check if the data exists.
            buffer_idx = self.data_hash_to_buffer_idx.get(hashed, -1)
            if buffer_idx == -1:
                # The data doesn't exist; add it to the data store.
                buffer_idx = len(self.buffers)
                self.buffers.append(data)
                self.data_hash_to_buffer_idx[hashed] = buffer_idx

            # Add key to the map and the key cache.
            local_key_to_buffer_idx[key] = DataEntry(
                buffer_index=buffer_idx,
                alignment=alignment,
                tensor_layout=tensor_layout,
            )
            self.key_to_buffer_idx[key] = buffer_idx

    def add_named_data(
        self,
        key: str,
        data: bytes,
        alignment: Optional[int] = 1,
        external_tag: Optional[str] = None,
        tensor_layout: Optional[TensorLayout] = None,
    ) -> None:
        """
        Adds a named blob to the NamedDataStore.
        Args:
            key (str): key associated with the data.
            data (bytes): Bytes being requested to be serialized.
            alignment (int): alignment for bytes to be serialized with.
            external (Optional[str]): the external filename that this data is saved to.
            tensor_layout (Optional[TensorLayout]): layout of the tensor, if applicable.
        Raises:
            ValueError: when the key exists in the store, and corresponding data
                is different.
        """

        # Set default alignment.
        if alignment is None:
            alignment = 1
        if alignment <= 0:
            raise ValueError(f"Alignment must be greater than 0, received {alignment}.")

        if external_tag is None:
            self._add_named_data_to_map(
                key, data, alignment, self.pte_data, tensor_layout
            )
        else:
            self._add_named_data_to_map(
                key,
                data,
                alignment,
                self.external_data.setdefault(external_tag, {}),
                tensor_layout,
            )

    def get_named_data_store_output(self) -> NamedDataStoreOutput:
        # Clean up empty maps inside self.external_data
        self.external_data = {k: v for k, v in self.external_data.items() if len(v) > 0}
        return NamedDataStoreOutput(self.buffers, self.pte_data, self.external_data)

    def merge_named_data_store(self, other: NamedDataStoreOutput) -> None:
        """
        Merge another NamedDataStore into this one.
        Args:
            other (NamedDataStore): the other NamedDataStore to merge.
        Raises:
            ValueError: when the key exists in both stores, and corresponding
                data is different between them.
        """
        # Merge the pte_data.
        for key, data_entry in other.pte_data.items():
            self.add_named_data(
                key,
                other.buffers[data_entry.buffer_index],
                data_entry.alignment,
                external_tag=None,
                tensor_layout=data_entry.tensor_layout,
            )

        # Merge the external_data.
        for filename, key_to_data_entry in other.external_data.items():
            for key, data_entry in key_to_data_entry.items():
                self.add_named_data(
                    key,
                    other.buffers[data_entry.buffer_index],
                    data_entry.alignment,
                    external_tag=filename,
                    tensor_layout=data_entry.tensor_layout,
                )
