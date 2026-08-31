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
from dataclasses import dataclass, replace
from typing import Any, Callable, Dict, Iterator, List, Optional, Tuple

import torch
from executorch.exir._serialize._cord import FileBackedData
from executorch.exir._serialize._named_data_store import NamedDataStore
from executorch.exir.backend.backend_details import PreprocessResult
from executorch.exir.tensor import scalar_type_enum


CUDA_WEIGHT_CACHE_MAGIC = b"ETCUDAFQN3"
CUDA_MULTI_ARCH_MAGIC = b"ETCUDAFQN4"
CUDA_MULTI_ARCH_FALLBACK_MAGIC = b"ETCUDAFQN5"

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


@dataclass(frozen=True)
class CudaAotiVariant:
    """One AOTI shared library and its CUDA runtime-selection metadata."""

    target_sm: int
    ptx_compute: int
    so_blob_key: str
    fallback_only: bool = False


@dataclass(frozen=True)
class CudaAotiMetadata:
    """CUDA AOTI native and fallback variants sharing one weight manifest."""

    variants: List[CudaAotiVariant]
    entries: List[CudaWeightEntry]


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


def _storage_key(
    fqn: str, device_type: int, aoti_library_key: Optional[str] = None
) -> str:
    if device_type == AOTI_DEVICE_TYPE_CPU:
        device = "cpu"
    elif device_type == AOTI_DEVICE_TYPE_CUDA:
        device = "cuda"
    else:
        raise RuntimeError(f"Unsupported AOTI device type: {device_type}")
    if aoti_library_key is not None:
        fqn = f"{aoti_library_key}:{fqn}"
    return f"cuda_fqn_weight:{device}:{fqn}"


def _is_aoti_library_local_fqn(fqn: str) -> bool:
    # PyTorch assigns this prefix to TensorConstant entries that do not have a
    # model-level FQN. The numbering restarts in every independently compiled
    # AOTI library, so the library key is part of their global identity.
    return fqn.startswith("_tensor_constant")


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


def _validate_cuda_aoti_variant(
    variant: CudaAotiVariant, has_fallback: bool, regular_sms: set[int]
) -> None:
    if variant.target_sm <= 0:
        raise ValueError(f"Invalid CUDA target SM: {variant.target_sm}")
    if variant.ptx_compute < 0 or variant.ptx_compute > variant.target_sm:
        raise ValueError(
            f"Invalid PTX compute target {variant.ptx_compute} for sm{variant.target_sm}"
        )
    if not variant.so_blob_key:
        raise ValueError("CUDA AOTI variant is missing its shared-object key")
    if variant.fallback_only:
        if variant.ptx_compute == 0:
            raise ValueError("CUDA fallback variant must contain PTX")
        return
    if variant.target_sm in regular_sms:
        raise ValueError(f"Duplicate CUDA target SM: {variant.target_sm}")
    regular_sms.add(variant.target_sm)
    if has_fallback and variant.ptx_compute != 0:
        raise ValueError("Regular CUDA variants cannot advertise PTX fallback")


def _validate_cuda_aoti_variants(
    variants: List[CudaAotiVariant], has_fallback: bool
) -> None:
    if sum(variant.fallback_only for variant in variants) > 1:
        raise ValueError("CUDA AOTI metadata supports only one fallback variant")
    regular_sms: set[int] = set()
    for variant in variants:
        _validate_cuda_aoti_variant(variant, has_fallback, regular_sms)


def encode_cuda_multi_arch_metadata(
    variants: List[CudaAotiVariant], entries: List[CudaWeightEntry]
) -> bytes:
    """Encode CUDA AOTI variants followed by one shared weight manifest."""
    if not variants:
        raise ValueError("CUDA AOTI metadata requires at least one variant")

    has_fallback = any(variant.fallback_only for variant in variants)
    _validate_cuda_aoti_variants(variants, has_fallback)
    output = bytearray(
        CUDA_MULTI_ARCH_FALLBACK_MAGIC if has_fallback else CUDA_MULTI_ARCH_MAGIC
    )

    def write_string(value: str) -> None:
        encoded = value.encode("utf-8")
        output.extend(struct.pack("<I", len(encoded)))
        output.extend(encoded)

    output.extend(struct.pack("<I", len(variants)))
    for variant in variants:
        output.extend(struct.pack("<II", variant.target_sm, variant.ptx_compute))
        if has_fallback:
            output.extend(struct.pack("<I", int(variant.fallback_only)))
        write_string(variant.so_blob_key)

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


class _MetadataReader:
    def __init__(self, data: bytes) -> None:
        self._data = memoryview(data)
        self._offset = 0

    def read(self, size: int) -> memoryview:
        end = self._offset + size
        if size < 0 or end > len(self._data):
            raise ValueError("Truncated CUDA AOTI metadata")
        value = self._data[self._offset : end]
        self._offset = end
        return value

    def unpack(self, format: str) -> Tuple[Any, ...]:
        size = struct.calcsize(format)
        return struct.unpack(format, self.read(size))

    def read_string(self) -> str:
        (size,) = self.unpack("<I")
        try:
            return bytes(self.read(size)).decode("utf-8")
        except UnicodeDecodeError as error:
            raise ValueError("CUDA AOTI metadata contains invalid UTF-8") from error

    def finish(self) -> None:
        if self._offset != len(self._data):
            raise ValueError("CUDA AOTI metadata contains trailing bytes")


def _decode_weight_entries(reader: _MetadataReader) -> List[CudaWeightEntry]:
    (num_entries,) = reader.unpack("<I")
    if num_entries > 1 << 20:
        raise ValueError(f"CUDA AOTI metadata has too many weights: {num_entries}")

    entries = []
    for _ in range(num_entries):
        fqn = reader.read_string()
        storage_key = reader.read_string()
        storage_nbytes, dtype, device_type, storage_offset, ndim = reader.unpack(
            "<QiiqI"
        )
        if not fqn or not storage_key or ndim > 64:
            raise ValueError("CUDA AOTI metadata contains an invalid weight entry")
        sizes = reader.unpack(f"<{ndim}q") if ndim else ()
        strides = reader.unpack(f"<{ndim}q") if ndim else ()
        if storage_offset < 0 or any(value < 0 for value in sizes + strides):
            raise ValueError("CUDA AOTI metadata contains invalid tensor metadata")
        entries.append(
            CudaWeightEntry(
                fqn=fqn,
                storage_key=storage_key,
                storage_nbytes=storage_nbytes,
                dtype=dtype,
                device_type=device_type,
                storage_offset=storage_offset,
                sizes=tuple(sizes),
                strides=tuple(strides),
            )
        )
    return entries


def decode_cuda_aoti_metadata(data: bytes) -> CudaAotiMetadata:
    """Decode legacy single-SO or multi-SM CUDA AOTI metadata."""
    reader = _MetadataReader(data)
    magic = bytes(reader.read(len(CUDA_WEIGHT_CACHE_MAGIC)))
    variants = []
    if magic == CUDA_WEIGHT_CACHE_MAGIC:
        so_blob_key = reader.read_string()
        if not so_blob_key:
            raise ValueError("CUDA AOTI metadata is missing its shared-object key")
        variants.append(CudaAotiVariant(0, 0, so_blob_key))
    elif magic in (CUDA_MULTI_ARCH_MAGIC, CUDA_MULTI_ARCH_FALLBACK_MAGIC):
        has_fallback = magic == CUDA_MULTI_ARCH_FALLBACK_MAGIC
        (num_variants,) = reader.unpack("<I")
        if num_variants == 0 or num_variants > 256:
            raise ValueError(
                f"CUDA AOTI metadata has invalid variant count: {num_variants}"
            )
        target_sms = set()
        fallback_count = 0
        for _ in range(num_variants):
            target_sm, ptx_compute = reader.unpack("<II")
            (flags,) = reader.unpack("<I") if has_fallback else (0,)
            fallback_only = bool(flags & 1)
            so_blob_key = reader.read_string()
            if (
                target_sm == 0
                or ptx_compute > target_sm
                or flags & ~1
                or not so_blob_key
            ):
                raise ValueError("CUDA AOTI metadata contains an invalid variant")
            if fallback_only:
                fallback_count += 1
                if ptx_compute == 0:
                    raise ValueError(
                        "CUDA AOTI metadata contains an invalid fallback variant"
                    )
            else:
                if target_sm in target_sms or (has_fallback and ptx_compute != 0):
                    raise ValueError(
                        "CUDA AOTI metadata contains an invalid regular variant"
                    )
                target_sms.add(target_sm)
            variants.append(
                CudaAotiVariant(
                    target_sm=target_sm,
                    ptx_compute=ptx_compute,
                    so_blob_key=so_blob_key,
                    fallback_only=fallback_only,
                )
            )
        if fallback_count > 1:
            raise ValueError("CUDA AOTI metadata contains multiple fallback variants")
    else:
        raise ValueError("Unrecognized CUDA AOTI metadata")

    entries = _decode_weight_entries(reader)
    reader.finish()
    return CudaAotiMetadata(variants=variants, entries=entries)


class CudaWeightCollector:
    """Collect one global ``(device, FQN) -> value`` store for all methods."""

    _active = threading.local()

    def __init__(self) -> None:
        self._store = NamedDataStore()
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
        is_duplicate = entry.storage_key in self._store.key_to_buffer_idx
        try:
            self._store.add_named_data(
                entry.storage_key,
                data,
                alignment=1,
                external_tag=external_tag,
            )
        except Exception:
            data.close()
            raise
        if is_duplicate:
            data.close()

    def add_preprocess_result(
        self,
        result: PreprocessResult,
        artifact: CudaWeightArtifact,
        device_name: str,
        target_sm: Optional[int] = None,
        ptx_compute: int = 0,
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
        serialized_entries = []
        for entry in artifact.entries:
            data = artifact.storages[entry.storage_key]
            if _is_aoti_library_local_fqn(entry.fqn):
                entry = replace(
                    entry,
                    storage_key=_storage_key(
                        entry.fqn, entry.device_type, aoti_library_key=so_blob_key
                    ),
                )
            self._add_weight(entry, data, external_tag)
            serialized_entries.append(entry)

        if target_sm is None:
            result.processed_bytes = encode_cuda_weight_metadata(
                so_blob_key, serialized_entries
            )
        else:
            result.processed_bytes = encode_cuda_multi_arch_metadata(
                [CudaAotiVariant(target_sm, ptx_compute, so_blob_key)],
                serialized_entries,
            )
        self._results.append(result)

    def finish(self) -> None:
        shared_output = self._store.get_named_data_store_output()
        for result in self._results:
            result.data_store_output = shared_output
