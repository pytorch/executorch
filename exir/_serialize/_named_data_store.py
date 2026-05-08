# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import hashlib
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union

import torch
from executorch.exir._serialize.data_serializer import DataEntry
from executorch.exir.tensor_layout import TensorLayout


def _tensor_to_bytes(tensor: torch.Tensor) -> bytes:
    """Convert tensor to bytes using the fastest method available.

    Uses numpy().tobytes() which is faster than bytes(untyped_storage())
    for C-contiguous tensors. Falls back to untyped_storage() for
    non-contiguous tensors (e.g., channels_last) to preserve memory layout.
    """
    if not tensor.is_contiguous():
        # For non-C-contiguous tensors (e.g., channels_last), use untyped_storage
        # to preserve the actual memory layout
        return bytes(tensor.untyped_storage())
    if tensor.dtype == torch.bfloat16:
        # BFloat16 is not supported by numpy, extract raw bytes via view
        return tensor.view(torch.uint16).numpy().tobytes()
    else:
        return tensor.numpy().tobytes()


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

    buffers: List[bytes]
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
    buffers: List[bytes]
    # Named data stored inside the PTE file. Map of {key: DataEntry}.
    pte_data: Dict[str, DataEntry]
    # Named data stored outside of the PTE file.
    # Map of {filename: {key: DataEntry}}.
    external_data: Dict[str, Dict[str, DataEntry]]

    # Fast fingerprint for dedup: (length, first 32 bytes) -> buffer indices.
    fingerprint_to_buffer_idx: Dict[Tuple[int, bytes], List[int]]
    # SHA-256 digest per buffer index, computed lazily on first dedup check.
    buffer_sha256: Dict[int, bytes]
    # Cache of key to buffer idx to detect duplicate key registration.
    key_to_buffer_idx: Dict[str, int]

    def __init__(self) -> None:
        """
        Initializes a new NamedDataStore.
        """
        self.buffers = []
        self.pte_data = {}
        self.external_data = {}
        self.fingerprint_to_buffer_idx = {}
        self.buffer_sha256 = {}
        self.key_to_buffer_idx = {}

    def _get_buffer_sha256(self, buffer_idx: int) -> bytes:
        sha = self.buffer_sha256.get(buffer_idx)
        if sha is None:
            sha = hashlib.sha256(self.buffers[buffer_idx]).digest()
            self.buffer_sha256[buffer_idx] = sha
        return sha

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
        # Check if the key exists.
        buffer_idx = self.key_to_buffer_idx.get(key, -1)
        if buffer_idx != -1:
            if data != self.buffers[buffer_idx]:
                raise ValueError(
                    f"Duplicate key {key} with different data. "
                    f"Existing data size: {len(self.buffers[buffer_idx])} bytes. "
                    f"New data size: {len(data)} bytes."
                )
        else:
            # Two-level dedup: cheap fingerprint rejects non-matches fast,
            # SHA-256 confirms matches without full byte comparison.
            fingerprint = (len(data), data[:32])
            candidates = self.fingerprint_to_buffer_idx.get(fingerprint)
            if candidates is not None:
                new_sha = hashlib.sha256(data).digest()
                for candidate in candidates:
                    if new_sha == self._get_buffer_sha256(candidate):
                        buffer_idx = candidate
                        break

            if buffer_idx == -1:
                buffer_idx = len(self.buffers)
                self.buffers.append(data)
                self.fingerprint_to_buffer_idx.setdefault(fingerprint, []).append(
                    buffer_idx
                )

            local_key_to_buffer_idx[key] = DataEntry(
                buffer_index=buffer_idx,
                alignment=alignment,
                tensor_layout=tensor_layout,
            )
            self.key_to_buffer_idx[key] = buffer_idx

    def add_named_data(
        self,
        key: str,
        data: Union[bytes, torch.Tensor],
        alignment: Optional[int] = 1,
        external_tag: Optional[str] = None,
        tensor_layout: Optional[TensorLayout] = None,
    ) -> None:
        """
        Adds a named blob to the NamedDataStore.
        Args:
            key (str): key associated with the data.
            data (Union[bytes, torch.Tensor]): Union of bytes, or torch.Tensor to serialize. Note: if a tensor is passed, it must have contiguous memory layout. The tensor_layout will be inferred from the tensor and should not be passed in.
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

        if isinstance(data, torch.Tensor):
            real_tensor_layout = TensorLayout.from_tensor(data)
            if tensor_layout is not None and not (real_tensor_layout == tensor_layout):
                raise ValueError(
                    f"Tensor {key} is a torch.Tensor, with tensor_layout {real_tensor_layout}. The provided tensor layout {tensor_layout} does not match."
                )
            tensor_layout = real_tensor_layout
            byte_data = _tensor_to_bytes(data)
        else:
            byte_data = data

        if external_tag is None:
            self._add_named_data_to_map(
                key, byte_data, alignment, self.pte_data, tensor_layout
            )
        else:
            self._add_named_data_to_map(
                key,
                byte_data,
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
