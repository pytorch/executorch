# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import contextlib
import ctypes
import gc
import hashlib
import os
import struct
import tempfile
import threading
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterator, List, Optional, Tuple

import torch
from executorch.exir._serialize._cord import FileBackedData
from executorch.exir._serialize._named_data_store import NamedDataStore
from executorch.exir.backend.backend_details import PreprocessResult
from executorch.exir.tensor import scalar_type_enum


CUDA_WEIGHT_CACHE_MAGIC = b"ETCUDAFQN3"

AOTI_DEVICE_TYPE_CPU = 0
AOTI_DEVICE_TYPE_CUDA = 1


@dataclass(frozen=True)
class CudaWeightEntry:
    fqn: str
    storage_key: str
    storage_nbytes: int
    dtype: int
    device_type: int
    storage_offset: int
    sizes: Tuple[int, ...]
    strides: Tuple[int, ...]


@dataclass
class CudaWeightArtifact:
    entries: List[CudaWeightEntry]
    storages: Dict[str, FileBackedData]


@dataclass
class _CudaWeightCapture:
    collector: "CudaWeightCollector"
    device_type_for_weight: Callable[[torch.Tensor], int]
    artifact: Optional[CudaWeightArtifact] = None


def trim_host_memory() -> None:
    gc.collect()
    try:
        ctypes.CDLL(None).malloc_trim(0)
    except AttributeError:
        pass


def _write_tensor_storage(tensor: torch.Tensor, path: str) -> bytes:
    """Stream one AOTI storage to ``path`` and return its SHA-256 digest."""
    chunk_size = 8 * 1024 * 1024
    digest = hashlib.sha256()

    def write_chunk(output, chunk) -> None:
        digest.update(chunk)
        output.write(chunk)

    if tensor.is_mkldnn:
        raise RuntimeError("MKLDNN constants are not supported by CUDA AOTI")
    storage = tensor.untyped_storage()
    nbytes = storage.nbytes()
    with open(path, "wb") as output:
        if nbytes and tensor.is_cuda:
            byte_tensor = torch.empty(0, dtype=torch.uint8, device=tensor.device).set_(
                storage, 0, (nbytes,), (1,)
            )
            for offset in range(0, nbytes, chunk_size):
                cpu_chunk = byte_tensor[offset : offset + chunk_size].cpu()
                write_chunk(output, memoryview(cpu_chunk.numpy()))
            del byte_tensor, cpu_chunk
        elif nbytes:
            raw_array = (ctypes.c_ubyte * nbytes).from_address(storage.data_ptr())
            raw_view = memoryview(raw_array).cast("B")
            for offset in range(0, nbytes, chunk_size):
                write_chunk(output, raw_view[offset : offset + chunk_size])
            del raw_view, raw_array
    del storage
    return digest.digest()


def _storage_key(fqn: str, device_type: int) -> str:
    if device_type == AOTI_DEVICE_TYPE_CPU:
        device = "cpu"
    elif device_type == AOTI_DEVICE_TYPE_CUDA:
        device = "cuda"
    else:
        raise RuntimeError(f"Unsupported AOTI device type: {device_type}")
    return f"cuda_fqn_weight:{device}:{fqn}"


def encode_cuda_weight_metadata(
    so_blob_key: str, entries: List[CudaWeightEntry]
) -> bytes:
    """Encode the per-method FQN-to-tensor metadata consumed by CUDA runtime."""
    output = bytearray(CUDA_WEIGHT_CACHE_MAGIC)

    def write_string(value: str) -> None:
        encoded = value.encode("utf-8")
        output.extend(struct.pack("<I", len(encoded)))
        output.extend(encoded)

    write_string(so_blob_key)
    output.extend(struct.pack("<I", len(entries)))
    for entry in entries:
        write_string(entry.fqn)
        write_string(entry.storage_key)
        output.extend(
            struct.pack(
                "<QiiqI",
                entry.storage_nbytes,
                entry.dtype,
                entry.device_type,
                entry.storage_offset,
                len(entry.sizes),
            )
        )
        output.extend(struct.pack(f"<{len(entry.sizes)}q", *entry.sizes))
        output.extend(struct.pack(f"<{len(entry.strides)}q", *entry.strides))
    return bytes(output)


class CudaWeightCollector:
    """Collect one global ``(device, FQN) -> value`` store for all methods."""

    _active = threading.local()

    def __init__(self) -> None:
        self._store = NamedDataStore()
        self._entries: Dict[str, CudaWeightEntry] = {}
        self._results: List[PreprocessResult] = []

    @contextlib.contextmanager
    def capture(
        self, device_type_for_weight: Callable[[torch.Tensor], int]
    ) -> Iterator[_CudaWeightCapture]:
        previous = getattr(self._active, "current", None)
        capture = _CudaWeightCapture(self, device_type_for_weight)
        self._active.current = capture
        try:
            yield capture
        finally:
            self._active.current = previous

    @classmethod
    def current_capture(cls) -> _CudaWeightCapture:
        capture = getattr(cls._active, "current", None)
        if capture is None:
            raise RuntimeError(
                "CUDA structured weights must be materialized inside preprocess"
            )
        return capture

    def materialize(
        self,
        weights: Any,
        directory: str,
        device_type_for_weight: Callable[[torch.Tensor], int],
    ) -> CudaWeightArtifact:
        """Turn AOTI ``Weights`` into one independently named blob per FQN."""
        trim_host_memory()
        entries: List[CudaWeightEntry] = []
        storages: Dict[str, FileBackedData] = {}

        for fqn, (tensor, properties) in weights.items():
            storage = tensor.untyped_storage()
            storage_nbytes = storage.nbytes()
            del storage
            device_type = device_type_for_weight(tensor)
            expected_storage_nbytes = int(
                getattr(properties, "storage_size", None) or 0
            )
            if storage_nbytes < expected_storage_nbytes:
                raise RuntimeError(
                    "AOTI cloned storage is smaller than its TensorProperties "
                    f"({storage_nbytes} < {expected_storage_nbytes} bytes)"
                )

            fd, storage_path = tempfile.mkstemp(
                prefix=".cuda_weight_", suffix=".storage", dir=directory
            )
            os.close(fd)
            try:
                digest = _write_tensor_storage(tensor, storage_path)
                data = FileBackedData.move_from(storage_path, sha256=digest)
            except Exception:
                try:
                    os.remove(storage_path)
                except OSError:
                    pass
                raise

            storage_key = _storage_key(fqn, device_type)
            if storage_key in storages:
                data.close()
                raise RuntimeError(f"Duplicate CUDA FQN weight key for {fqn!r}")
            storages[storage_key] = data

            sizes = tuple(
                int(size) for size in getattr(properties, "shape", tensor.shape)
            )
            strides = tuple(
                int(stride) for stride in getattr(properties, "stride", tensor.stride())
            )
            storage_offset = int(getattr(properties, "offset", tensor.storage_offset()))
            if (
                len(sizes) != len(strides)
                or storage_offset < 0
                or any(size < 0 for size in sizes)
                or any(stride < 0 for stride in strides)
            ):
                raise RuntimeError(f"AOTI view {fqn!r} has invalid tensor metadata")
            required_nbytes = 0
            if all(size != 0 for size in sizes):
                last_element = storage_offset + sum(
                    stride * (size - 1) for size, stride in zip(sizes, strides)
                )
                required_nbytes = (last_element + 1) * tensor.element_size()
            if required_nbytes > storage_nbytes:
                raise RuntimeError(
                    f"AOTI view {fqn!r} requires {required_nbytes} bytes from a "
                    f"{storage_nbytes}-byte cloned storage"
                )
            entries.append(
                CudaWeightEntry(
                    fqn=fqn,
                    storage_key=storage_key,
                    storage_nbytes=storage_nbytes,
                    dtype=int(scalar_type_enum(tensor.dtype)),
                    device_type=device_type,
                    storage_offset=storage_offset,
                    sizes=sizes,
                    strides=strides,
                )
            )

        trim_host_memory()
        return CudaWeightArtifact(entries=entries, storages=storages)

    def _merge_aoti_data(
        self,
        parent_store: Any,
        compatibility_blob_key: Optional[str],
        keep_compatibility_blob: bool,
    ) -> None:
        for key, entry in parent_store.pte_data.items():
            if key != compatibility_blob_key or keep_compatibility_blob:
                self._store.add_named_data(
                    key,
                    parent_store.buffers[entry.buffer_index],
                    alignment=entry.alignment,
                    tensor_layout=entry.tensor_layout,
                )
        for tag, entries in parent_store.external_data.items():
            for key, entry in entries.items():
                if key != compatibility_blob_key or keep_compatibility_blob:
                    self._store.add_named_data(
                        key,
                        parent_store.buffers[entry.buffer_index],
                        alignment=entry.alignment,
                        external_tag=tag,
                        tensor_layout=entry.tensor_layout,
                    )

    def _add_weight(
        self,
        entry: CudaWeightEntry,
        data: FileBackedData,
        external_tag: str,
    ) -> None:
        previous = self._entries.get(entry.storage_key)
        try:
            if previous is not None and previous != entry:
                raise ValueError(
                    f"Duplicate key {entry.storage_key} with different tensor "
                    "metadata."
                )
            self._store.add_named_data(
                entry.storage_key,
                data,
                alignment=1,
                external_tag=external_tag,
            )
        finally:
            if previous is not None:
                data.close()
        self._entries.setdefault(entry.storage_key, entry)

    def add_preprocess_result(
        self,
        result: PreprocessResult,
        artifact: CudaWeightArtifact,
        device_name: str,
    ) -> None:
        if result.data_store_output is None:
            raise RuntimeError("CUDA AOTI preprocess returned no named data")
        try:
            parent_keys = result.processed_bytes.decode("utf-8").splitlines()
        except UnicodeDecodeError as error:
            raise RuntimeError("Malformed CUDA AOTI named-data payload") from error
        if not parent_keys or not parent_keys[0]:
            raise RuntimeError("CUDA AOTI payload is missing its shared-object key")
        so_blob_key = parent_keys[0]
        compatibility_blob_key = parent_keys[1] if len(parent_keys) > 1 else None

        parent_store = result.data_store_output
        self._merge_aoti_data(
            parent_store,
            compatibility_blob_key,
            keep_compatibility_blob=not artifact.storages,
        )

        external_tag = f"aoti_{device_name}_blob"
        for entry in artifact.entries:
            self._add_weight(entry, artifact.storages[entry.storage_key], external_tag)

        result.processed_bytes = encode_cuda_weight_metadata(
            so_blob_key, artifact.entries
        )
        self._results.append(result)

    def finish(self) -> None:
        shared_output = self._store.get_named_data_store_output()
        for result in self._results:
            result.data_store_output = shared_output
