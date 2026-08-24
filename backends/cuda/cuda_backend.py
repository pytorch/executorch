# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


import contextlib
import copy
import ctypes
import functools
import gc
import hashlib
import logging
import os
import shutil
import struct
import tempfile
import threading
import typing
from dataclasses import dataclass
from importlib import resources
from typing import Any, Dict, final, List, Optional, Tuple

import torch
from executorch.backends.aoti.aoti_backend import AotiBackend
from executorch.backends.cuda.passes.move_cond_predicate_to_cpu import (
    MoveCondPredicateToCpuPass,
)
from executorch.backends.cuda.passes.replace_int64_floordiv import (
    ReplaceInt64FloorDivWithFloatPass,
)
from executorch.backends.cuda.triton.replacement_pass import (
    ReplaceEdgeOpWithTritonOpPass,
)
from executorch.exir._serialize._cord import FileBackedData
from executorch.exir._serialize._named_data_store import NamedDataStore
from executorch.exir._warnings import experimental
from executorch.exir.backend.backend_details import BackendDetails, PreprocessResult
from executorch.exir.backend.compile_spec_schema import CompileSpec
from executorch.exir.tensor import scalar_type_enum
from torch._inductor.decomposition import conv1d_to_conv2d
from torch.nn.attention import SDPBackend

# ---------------------------------------------------------------------------
# AOTI compile-time CPU clones for mutated buffers
# ---------------------------------------------------------------------------
#
# Inductor's `_unlift_graph` clones every mutated buffer that gets lifted into
# the AOTI graph. By default it clones on whatever device the original tensor
# lives on — which after `move_to_device_pass` is CUDA. For Large models like
# Qwen3.5-MoE that means an extra ~18 GB GPU clone during compile, blowing past
# the 24 GB cap we want to honor for consumer GPUs (RTX 4090 and similar).
#
# The patch below side-steps that by:
#   1. Wrapping `torch._inductor.compile_fx.clone_preserve_strides` so every
#      clone the AOTI compile pipeline produces lands on CPU.
#   2. Wrapping `CppWrapperCpu.codegen_device` so the C++ wrapper still records
#      the model's original target device (e.g. cuda) in `constants_info_`,
#      not the now-CPU storage device. Without this the runtime would refuse
#      to load the constants because of a mixed-device mismatch.
#
# The wrappers are scoped via a thread-local guard and are only active while
# `_compile_time_cpu_clones(...)` is on the call stack — they are inert
# anywhere else in the process.

_CPU_CLONE_GUARD = threading.local()

_FQN_WEIGHTS_MAGIC = b"ETCUDAFQN2"
_FQN_WEIGHTS_CAPTURE = threading.local()

_AOTI_DEVICE_TYPE_CPU = 0
_AOTI_DEVICE_TYPE_CUDA = 1


@dataclass
class _FqnWeightEntry:
    fqn: str
    storage_key: str
    storage_group: int
    storage_nbytes: int
    dtype: int
    device_type: int
    storage_offset: int
    sizes: Tuple[int, ...]
    strides: Tuple[int, ...]
    shareable: bool


@dataclass
class _FqnWeightArtifact:
    entries: List[_FqnWeightEntry]
    storages: Dict[str, FileBackedData]


@dataclass
class _FqnWeightCapture:
    mutated_fqns: set[str]
    artifact: Optional[_FqnWeightArtifact] = None


def _is_cpu_clone_active() -> bool:
    return getattr(_CPU_CLONE_GUARD, "active", False)


def _trim_host_memory() -> None:
    gc.collect()
    try:
        ctypes.CDLL(None).malloc_trim(0)
    except AttributeError:
        pass


def _aoti_device_type_for_weight(tensor: torch.Tensor) -> int:
    # Low-memory compilation clones lifted CUDA buffers onto CPU, while the
    # patched wrapper records them as CUDA constants. Mirror that target-device
    # substitution in the manifest. Outside that scoped mode the serialized
    # tensor's actual device is the AOTI constant's device.
    if _is_cpu_clone_active() or tensor.device.type == "cuda":
        return _AOTI_DEVICE_TYPE_CUDA
    if tensor.device.type == "cpu":
        return _AOTI_DEVICE_TYPE_CPU
    raise RuntimeError(
        f"Unsupported AOTI constant device for CUDA export: {tensor.device}"
    )


def _stateful_buffer_fqns(graph_signature: Any) -> set[str]:
    # Some generated-code mutations (notably conditional cache updates) are
    # not surfaced through buffers_to_mutate. Treat every registered buffer as
    # model-instance-local, then include any explicitly reported mutations.
    fqns = set(getattr(graph_signature, "buffers", ()))
    fqns.update(getattr(graph_signature, "buffers_to_mutate", {}).values())
    return fqns


@contextlib.contextmanager
def _keep_triton_reduction_loads_loop_scoped():
    """Conservatively treating loads as reduction-masked keeps each definition in its
    own loop to bypass the undefined variable bug in pytorch/pytorch.
    (https://github.com/pytorch/pytorch/issues/193988)

    TODO(gasoonjia): remove this setting once bug fixed in upstream
    """
    from torch._inductor.codegen.triton import IndexingOptions

    orig_has_rmask = IndexingOptions.has_rmask

    def _has_rmask(_self) -> bool:
        return True

    IndexingOptions.has_rmask = _has_rmask
    try:
        yield
    finally:
        IndexingOptions.has_rmask = orig_has_rmask


def _full_zeros_preserving_strides(x: torch.Tensor, device) -> torch.Tensor:
    """Allocate a zero-filled tensor matching ``x``'s size/stride/dtype on ``device``.

    Used to re-synthesize KV-cache buffers whose storage was freed (``resize_(0)``)
    during the low-memory device move. KV content is all zeros, so this exactly
    reproduces the buffer for both the lifted graph value and serialization.
    """
    needed = 1
    for size, stride in zip(x.size(), x.stride()):
        needed += (size - 1) * stride
    buf = torch.zeros(int(needed), dtype=x.dtype, device=device)
    return torch.as_strided(buf, x.size(), x.stride())


def _is_emptied(x) -> bool:
    return (
        isinstance(x, torch.Tensor)
        and x.numel() > 0
        and x.untyped_storage().nbytes() == 0
    )


def _tensor_properties_for_low_memory(tensor, original):
    if _is_cpu_clone_active() and _is_emptied(tensor):
        return None
    return original(tensor)


def _required_storage_nbytes(x: torch.Tensor) -> int:
    """Return the backing storage required by ``x``'s logical view."""
    if x.numel() == 0:
        return 0
    last_element = x.storage_offset()
    for size, stride in zip(x.size(), x.stride()):
        last_element += (size - 1) * stride
    return int(last_element + 1) * x.element_size()


@contextlib.contextmanager
def _rehydrate_emptied_tensors(tensors):
    """Temporarily restore zero storage while preserving tensor aliases.

    Low-memory CUDA export keeps KV tensors' sizes and strides but releases
    their backing storage. Inductor autotuning needs those original tensor
    objects throughout cloning, reset-to-zero, and kernel launch. Restoring the
    shared storages in place preserves views/aliases that replacing individual
    arguments with fresh tensors would break.
    """
    emptied = [tensor for tensor in tensors if _is_emptied(tensor)]
    storages = {}
    for tensor in emptied:
        storage = tensor.untyped_storage()
        key = storage._cdata
        required = _required_storage_nbytes(tensor)
        if key not in storages or required > storages[key][1]:
            storages[key] = (storage, required)

    restored = []
    try:
        for storage, required in storages.values():
            storage.resize_(required)
            restored.append(storage)
        for tensor in emptied:
            tensor.zero_()
        yield
    finally:
        # The autotuner finishes with reset_to_zero_args(), whose CUDA zero_
        # launches asynchronously.  Releasing the storage before that work has
        # completed leaves the kernel writing through a freed pointer; the
        # resulting illegal access is then reported by some later CUDA API
        # (often preserve_rng_state's set_rng_state).  All users of storage that
        # is about to be resized away must be complete first.
        cuda_devices = {tensor.device for tensor in emptied if tensor.is_cuda}
        for device in cuda_devices:
            torch.cuda.synchronize(device)
        for storage in reversed(restored):
            storage.resize_(0)


@contextlib.contextmanager
def _compile_time_cpu_clones(target_device: torch.device):  # noqa: C901
    """Force AOTI's mutated-buffer clones onto CPU while preserving the
    serialized constants' target device."""
    from torch._inductor import (
        codecache as _codecache,
        compile_fx as _cfx,
        graph as _graph,
    )
    from torch._inductor.codegen.cpp_wrapper_cpu import CppWrapperCpu as _Cpp
    from torch._inductor.graph import GraphLowering as _GL
    from torch._inductor.runtime.triton_heuristics import CachingAutotuner as _Autotuner

    orig_clone = _cfx.clone_preserve_strides
    orig_codegen_device = _Cpp.codegen_device
    orig_get_const = _GL.get_original_value_of_constant
    orig_is_same = _graph.is_same_tensor
    orig_tensor_properties = _codecache.TensorProperties
    orig_determine_aoti_mmap_flags = _codecache.determine_aoti_mmap_flags

    def _force_external_weights_for_fqn_binding(consts_size):
        # Structured weights are serialized as independently named storages,
        # but the generated AOTI wrapper must still use the external-weights
        # ABI. That mode preserves the original constant view metadata when
        # the runtime replaces the dense blob with user-managed FQN tensors.
        if _is_cpu_clone_active():
            return True, False
        return orig_determine_aoti_mmap_flags(consts_size)

    orig_autotuner_run = _Autotuner.run

    def _is_same_skip_emptied(data, value):
        # KV buffers freed via resize_(0) all have data_ptr 0, so the stock
        # is_same_tensor would treat every same-shape KV constant as a duplicate
        # and collapse the 60 layers' caches into one — the runtime needs each
        # FQN's own buffer, so the collapsed ones load uninitialized garbage.
        # Never dedup an emptied tensor.
        if _is_emptied(data) or _is_emptied(value):
            return False
        return orig_is_same(data, value)

    def _cpu_clone_preserve_strides(x: torch.Tensor) -> torch.Tensor:
        # `clone_preserve_strides` is shared by `_unlift_graph` (clones lifted
        # buffers — can be safely kept on CPU) and by autotuning code in
        # `triton_heuristics.py` (clones for benchmark — must stay on GPU for
        # Triton). Discriminate by caller frame so we only force CPU clones for
        # the buffer-lifting path.
        import sys

        caller = sys._getframe(1).f_code.co_name
        if caller == "_unlift_graph":
            # KV-cache buffers are emptied (storage resize_(0)) by the low-memory
            # device move so they never occupy GPU memory during compile. Their
            # content is all zeros, so re-synthesize zeros (on CPU, strides
            # preserved) instead of cloning the now-empty storage.
            if _is_emptied(x):
                return _full_zeros_preserving_strides(x, "cpu")
            return orig_clone(x).cpu()
        return orig_clone(x)

    def _autotuner_run_with_rehydrated_emptied_args(self, *args, **kwargs):
        # CachingAutotuner.run first benchmarks configurations (where cloning and
        # reset_to_zero_args touch the inputs), then launches the winning kernel
        # once more on the original arguments. Keep their storage valid through
        # both phases and the final launch; wrapping benchmark_all_configs alone
        # would release it too early for that last call.
        tensors = (
            value
            for value in (*args, *kwargs.values())
            if isinstance(value, torch.Tensor)
        )
        with _rehydrate_emptied_tensors(tensors):
            return orig_autotuner_run(self, *args, **kwargs)

    def _get_const_synthesize_zeros(self, name):
        # AOTI serializes each constant via get_original_value_of_constant ->
        # _to_bytes. For KV buffers we freed with resize_(0) this would otherwise
        # fall back to the empty-storage constant and write 0 bytes, producing a
        # .ptd with an uninitialized cache. Re-synthesize the zeros so the blob
        # holds a correctly-zeroed KV cache.
        value = orig_get_const(self, name)
        if _is_emptied(value):
            return _full_zeros_preserving_strides(value, "cpu")
        return value

    def _codegen_device_target_aware(self, device):
        # Translate accidental CPU device strings back to the model target
        # device only when a constant we forced to CPU is being serialized.
        # Other code paths (extern op args etc.) are pass-through.
        if (
            _is_cpu_clone_active()
            and self.device != "cpu"
            and isinstance(device, torch.device)
            and device.type == "cpu"
        ):
            device = target_device
        return orig_codegen_device(self, device)

    _cfx.clone_preserve_strides = _cpu_clone_preserve_strides
    _Cpp.codegen_device = _codegen_device_target_aware
    _GL.get_original_value_of_constant = _get_const_synthesize_zeros
    _graph.is_same_tensor = _is_same_skip_emptied
    _codecache.TensorProperties = functools.partial(
        _tensor_properties_for_low_memory, original=orig_tensor_properties
    )
    _codecache.determine_aoti_mmap_flags = _force_external_weights_for_fqn_binding
    _Autotuner.run = _autotuner_run_with_rehydrated_emptied_args
    prev_active = getattr(_CPU_CLONE_GUARD, "active", False)
    _CPU_CLONE_GUARD.active = True
    try:
        yield
    finally:
        _CPU_CLONE_GUARD.active = prev_active
        _cfx.clone_preserve_strides = orig_clone
        _Cpp.codegen_device = orig_codegen_device
        _GL.get_original_value_of_constant = orig_get_const
        _graph.is_same_tensor = orig_is_same
        _codecache.TensorProperties = orig_tensor_properties
        _codecache.determine_aoti_mmap_flags = orig_determine_aoti_mmap_flags
        _Autotuner.run = orig_autotuner_run


def _is_kv_buffer(name, v) -> bool:
    """True only for an actual KV-cache *content* buffer that is safe to free.

    The low-memory path (``_move_to_device_resize_kv``) frees every buffer this
    matches and re-synthesizes it as ZEROS in both the lifted graph and the
    serialized ``.ptd`` (see ``_full_zeros_preserving_strides`` /
    ``_get_const_synthesize_zeros``). That is only valid for genuine KV *content*,
    which is all-zeros at export time (caches start empty).

    It must NOT match the non-zero constants that some KV-cache modules register
    alongside the cache — e.g. TurboQuant registers its codebook/rotation
    (``centroids``/``boundaries``/``rotation``/``rotation_T``) as buffers on the
    ``kv_cache`` module, so their FQNs also contain ``kv_cache``. Freeing+zeroing
    those silently corrupts the serialized model (TQ4 dequant -> 0 -> garbage).
    Gate on the buffer actually being all-zeros so only empty KV content is freed;
    this is robust to any future constant name (a non-zero buffer is never freed).
    """
    if not isinstance(v, torch.Tensor) or isinstance(v, torch.nn.Parameter):
        return False
    if "kv_cache" not in name or v.numel() == 0 or v.is_meta:
        return False
    # Only the genuinely all-zero KV content may be freed + re-zeroed; non-zero
    # constants (TurboQuant centroids/rotation/...) must be preserved as-is.
    return bool(torch.count_nonzero(v) == 0)


def _empty_strided_on_device(v, location):
    """A device tensor with v's shape/stride/dtype but zero (freed) storage."""
    t = torch.empty_strided(v.shape, v.stride(), dtype=v.dtype, device=location)
    t.untyped_storage().resize_(0)  # free bytes, keep device + shape/stride
    return t


def _move_graph_nodes_to_device(graph_module, location):
    """Point node device kwargs / aten.to.device targets / meta vals at location."""
    import torch.utils._pytree as pytree

    def _to_loc(v):
        return v.to(location) if isinstance(v, torch.Tensor) else v

    for m in graph_module.modules():
        if not isinstance(m, torch.fx.GraphModule):
            continue
        for node in m.graph.nodes:
            if "device" in node.kwargs:
                node.kwargs = {**node.kwargs, "device": location}
            if node.op == "call_function" and node.target is torch.ops.aten.to.device:
                args = list(node.args)
                args[1] = location
                node.args = tuple(args)
            node.meta["val"] = pytree.tree_map(_to_loc, node.meta.get("val"))


def _move_to_device_resize_kv(ep, location):
    """``move_to_device_pass`` variant that frees KV-cache storage on-device.

    Mirrors ``torch.export.passes.move_to_device_pass`` exactly, except KV-cache
    buffers (FQN contains ``kv_cache``) are placed on ``location`` but with their
    storage immediately freed via ``resize_(0)``. This keeps ``device ==
    location`` — so the fake-tensor device check on the ``index_copy`` cache
    update passes (``self`` and ``values`` both on cuda) — while no real KV bytes
    occupy the device during the AOTI compile. KV content is all zeros, so the
    emptied tensors are re-synthesized as zeros at the ``_unlift_graph`` clone
    (see ``_compile_time_cpu_clones``), which is reused as both the lifted initial
    value and the serialized ``.ptd`` constant. The empty/free is interleaved per
    tensor so the transient device peak is a single KV buffer, not the whole cache.
    Only ``kv_cache`` tensors are emptied (they are the lone large zero-buffers);
    every other tensor is moved normally so non-zero content is never lost.
    """
    import torch.utils._pytree as pytree

    for k, v in ep.state_dict.items():
        if isinstance(v, torch.nn.Parameter):
            ep._state_dict[k] = torch.nn.Parameter(v.to(location), v.requires_grad)
        elif _is_kv_buffer(k, v):
            ep._state_dict[k] = _empty_strided_on_device(v, location)
        else:
            ep._state_dict[k] = v.to(location)

    for k, v in ep.constants.items():
        if isinstance(v, torch.Tensor):
            ep._constants[k] = (
                _empty_strided_on_device(v, location)
                if _is_kv_buffer(k, v)
                else v.to(location)
            )

    if ep.example_inputs is not None:
        args, kwargs = ep.example_inputs
        ep._example_inputs = (
            pytree.tree_map_only(torch.Tensor, lambda t: t.to(location), args),
            pytree.tree_map_only(torch.Tensor, lambda t: t.to(location), kwargs),
        )

    _move_graph_nodes_to_device(ep.graph_module, location)
    ep.validate()
    return ep


def _on_off_compile_spec_value(spec: CompileSpec) -> bool:
    value = spec.value.decode("utf-8").upper()
    if value not in ["ON", "OFF"]:
        raise ValueError(f"Invalid {spec.key}: {value}. Expected 'ON' or 'OFF'.")
    return value == "ON"


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


def _materialize_fqn_weights(  # noqa: C901
    weights: Any,
    directory: str,
    mutated_fqns: set[str],
) -> _FqnWeightArtifact:
    """Turn AOTI ``Weights`` into content-addressed storage files + views."""
    # The graph can contain hundreds of independent storages.  Trimming around
    # every storage is both ineffective (``records`` below still owns all of
    # the tensors) and extremely expensive for large exported graphs.  Trim
    # once at artifact boundaries; streamed CPU chunks are released by normal
    # reference counting as they are replaced.
    _trim_host_memory()
    entries: List[_FqnWeightEntry] = []
    storages: Dict[str, FileBackedData] = {}
    records: List[Tuple[str, torch.Tensor, Any, Tuple[Any, ...], int, int]] = []
    record_indices_by_identity: Dict[Tuple[Any, ...], List[int]] = {}

    for index, (fqn, (tensor, properties)) in enumerate(weights.items()):
        storage = tensor.untyped_storage()
        storage_nbytes = storage.nbytes()
        storage_ptr = storage.data_ptr()
        device_type = _aoti_device_type_for_weight(tensor)
        property_storage_ptr = getattr(properties, "storage_ptr", None)
        if property_storage_ptr not in (None, 0):
            # TensorProperties describes the graph constant's real storage.
            # The value tensor can be a clone (including a CPU clone in CUDA
            # low-memory mode), so its data_ptr is not a stable alias key.
            identity = (
                "aoti",
                int(property_storage_ptr),
                str(tensor.dtype),
                device_type,
            )
        else:
            identity = (
                tensor.device.type,
                tensor.device.index if tensor.device.index is not None else -1,
                storage_ptr if storage_ptr != 0 else -(index + 1),
                storage_nbytes,
                device_type,
            )
        del storage
        records.append((fqn, tensor, properties, identity, storage_nbytes, device_type))
        record_indices_by_identity.setdefault(identity, []).append(index)

    storage_info_by_identity: Dict[Tuple[Any, ...], Tuple[str, int, int]] = {}
    for storage_group, (identity, record_indices) in enumerate(
        record_indices_by_identity.items()
    ):
        # AOTI's value can be a clone of a view. Pick the largest available
        # backing storage in the alias group so every declared view can be
        # reconstructed from the one serialized allocation.
        candidate_index = max(record_indices, key=lambda item: records[item][4])
        candidate_tensor = records[candidate_index][1]
        storage_nbytes = records[candidate_index][4]
        expected_storage_nbytes = max(
            (
                int(storage_size)
                for item in record_indices
                if (storage_size := getattr(records[item][2], "storage_size", None))
                is not None
            ),
            default=0,
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
            digest = _write_tensor_storage(candidate_tensor, storage_path)
            storage_key = digest.hex() + "_cuda_weight_storage"
            data = FileBackedData.move_from(storage_path, sha256=digest)
        except Exception:
            try:
                os.remove(storage_path)
            except OSError:
                pass
            raise

        existing = storages.get(storage_key)
        if existing is None:
            storages[storage_key] = data
        else:
            data.close()
        storage_info_by_identity[identity] = (
            storage_key,
            storage_nbytes,
            storage_group,
        )

    for fqn, tensor, properties, identity, _storage_nbytes, device_type in records:
        storage_key, serialized_nbytes, storage_group = storage_info_by_identity[
            identity
        ]
        sizes = getattr(properties, "shape", tensor.shape)
        strides = getattr(properties, "stride", tensor.stride())
        storage_offset = getattr(properties, "offset", tensor.storage_offset())
        sizes = tuple(int(size) for size in sizes)
        strides = tuple(int(stride) for stride in strides)
        storage_offset = int(storage_offset)
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
        if required_nbytes > serialized_nbytes:
            raise RuntimeError(
                f"AOTI view {fqn!r} requires {required_nbytes} bytes from a "
                f"{serialized_nbytes}-byte cloned storage"
            )
        entries.append(
            _FqnWeightEntry(
                fqn=fqn,
                storage_key=storage_key,
                storage_group=storage_group,
                storage_nbytes=serialized_nbytes,
                dtype=int(scalar_type_enum(tensor.dtype)),
                device_type=device_type,
                storage_offset=storage_offset,
                sizes=sizes,
                strides=strides,
                shareable=fqn not in mutated_fqns,
            )
        )

    # A mutable view makes its complete physical storage stateful.  The runtime
    # shares such storages by FQN (not by content hash), including aliases that
    # share the same backing buffer.
    local_storage_groups = {
        entry.storage_group for entry in entries if not entry.shareable
    }
    for entry in entries:
        if entry.storage_group in local_storage_groups:
            entry.shareable = False

    _trim_host_memory()
    return _FqnWeightArtifact(entries=entries, storages=storages)


def _encode_fqn_weight_manifest(
    so_blob_key: str, entries: List[_FqnWeightEntry]
) -> bytes:
    """Encode the CUDA per-storage manifest consumed by the runtime."""
    output = bytearray(_FQN_WEIGHTS_MAGIC)

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
                "<IQiiqI",
                entry.storage_group,
                entry.storage_nbytes,
                entry.dtype,
                entry.device_type,
                entry.storage_offset,
                len(entry.sizes),
            )
        )
        output.extend(struct.pack(f"<{len(entry.sizes)}q", *entry.sizes))
        output.extend(struct.pack(f"<{len(entry.strides)}q", *entry.strides))
        output.extend(struct.pack("<B", entry.shareable))
    return bytes(output)


@final
@experimental(
    "This API and all of cuda backend related functionality are experimental."
)
class CudaBackend(AotiBackend, BackendDetails):
    """
    CudaBackend is a backend that compiles a model to run on CUDA devices. It uses the AOTInductor compiler to generate
    optimized CUDA kernels for the model's operators with libtorch-free. The compiled model can be executed on CUDA devices
    using the Executorch runtime.
    """

    # AOTI calls materialize_weights_blob immediately before load_weights_blob
    # for a given output path. A new materialization overwrites any digest left
    # behind by an export that aborted before the consumer ran.
    _materialized_blob_hashes: Dict[str, bytes] = {}

    @classmethod
    def get_device_name(cls) -> str:
        return "cuda"

    @staticmethod
    def _find_ptxas_for_version(cuda_version: str) -> Optional[str]:  # noqa: C901
        """
        Find ptxas binary that matches the expected CUDA version.
        Returns the path to ptxas if found and version matches, None otherwise.
        """
        expected_version_marker = f"/cuda-{cuda_version}/"

        def _validate_ptxas_version(path: str) -> bool:
            """Check if ptxas at given path matches expected CUDA version."""
            if not os.path.exists(path):
                return False
            resolved = os.path.realpath(path)
            return expected_version_marker in resolved

        # 1. Try PyTorch's CUDA_HOME

        try:
            from torch.utils.cpp_extension import CUDA_HOME

            if CUDA_HOME:
                ptxas_path = os.path.join(CUDA_HOME, "bin", "ptxas")
                if _validate_ptxas_version(ptxas_path):
                    return ptxas_path
        except ImportError:
            pass
        # 2. Try CUDA_HOME / CUDA_PATH environment variables

        for env_var in ("CUDA_HOME", "CUDA_PATH", "CUDA_ROOT"):
            cuda_home = os.environ.get(env_var)
            if cuda_home:
                ptxas_path = os.path.join(cuda_home, "bin", "ptxas")
                if _validate_ptxas_version(ptxas_path):
                    return ptxas_path
        # 3. Try versioned path directly

        versioned_path = f"/usr/local/cuda-{cuda_version}/bin/ptxas"
        if os.path.exists(versioned_path):
            return versioned_path
        # 4. Try system PATH via shutil.which

        ptxas_in_path = shutil.which("ptxas")
        if ptxas_in_path and _validate_ptxas_version(ptxas_in_path):
            return ptxas_in_path
        # 5. Try default symlink path as last resort

        default_path = "/usr/local/cuda/bin/ptxas"
        if _validate_ptxas_version(default_path):
            return default_path
        return None

    @staticmethod
    def _setup_cuda_environment_for_fatbin() -> bool:
        """
        Configure CUDA environment variables based on detected CUDA version and GPU architecture.
        These are needed to compile fatbin kernels for more portable binaries on older CUDA versions.
        Returns True if setup succeeded or if setup was skipped (CUDA >= 12.9), false otherwise.
        """
        try:
            # Detect CUDA version from torch

            cuda_version = torch.version.cuda
            if cuda_version is None:
                return False
            major, minor = map(int, cuda_version.split(".")[:2])

            # Only set up environment variables for CUDA < 12.9

            if major > 12 or (major == 12 and minor >= 9):
                return True
            # Set TRITON_PTXAS_PATH for CUDA 12.6+

            if major == 12 and minor >= 6:
                ptxas_path = CudaBackend._find_ptxas_for_version(cuda_version)
                if ptxas_path is None:
                    return False
                os.environ["TRITON_PTXAS_PATH"] = ptxas_path
            if os.environ.get("TORCH_CUDA_ARCH_LIST") is not None:
                logging.warning(
                    f"TORCH_CUDA_ARCH_LIST is set to {os.environ.get('TORCH_CUDA_ARCH_LIST')}, skipping automatic architecture detection."
                )
                return True
            # Get compute capability of current CUDA device

            device = torch.cuda.current_device()
            capability = torch.cuda.get_device_capability(device)
            os.environ["TORCH_CUDA_ARCH_LIST"] = f"{capability[0]}.{capability[1]}"
            return True
        except Exception:
            return False

    @classmethod
    def save_data_externally(cls) -> bool:
        """
        CUDA backend saves weight storages (and, when configured, SO blobs) in
        external named data such as a .ptd file.
        This file must be provided at runtime via --data_path argument.
        """
        return True

    @classmethod
    def preprocess(  # noqa: C901
        cls, edge_program: Any, compile_specs: List[CompileSpec]
    ) -> PreprocessResult:
        """Compile CUDA weights as independently addressable AOTI storages."""
        # Keep every buffer model-instance-local while still sharing it by FQN
        # across methods in that model instance.
        mutated_fqns = _stateful_buffer_fqns(edge_program.graph_signature)
        previous_capture = getattr(_FQN_WEIGHTS_CAPTURE, "current", None)
        capture = _FqnWeightCapture(mutated_fqns=mutated_fqns)
        # AotiBackend packages weights synchronously on this thread. TLS keeps
        # nested or concurrent preprocess calls isolated while that callback
        # passes the structured artifact back to this invocation.
        _FQN_WEIGHTS_CAPTURE.current = capture
        try:
            result = super().preprocess(edge_program, compile_specs)
        finally:
            _FQN_WEIGHTS_CAPTURE.current = previous_capture

        artifact = capture.artifact
        if artifact is None:
            raise RuntimeError("CUDA AOTI did not return a structured Weights output")
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

        # Rebuild AotiBackend's store without the empty compatibility blob,
        # then add each physical weight storage as separately named external
        # data. This leaves a new PTD containing only real storages while the
        # legacy runtime path remains able to consume old dense blobs.
        parent_store = result.data_store_output
        named_data_store = NamedDataStore()
        keep_compatibility_blob = not artifact.storages
        for key, entry in parent_store.pte_data.items():
            if key != compatibility_blob_key or keep_compatibility_blob:
                named_data_store.add_named_data(
                    key,
                    parent_store.buffers[entry.buffer_index],
                    alignment=entry.alignment,
                    tensor_layout=entry.tensor_layout,
                )
        for tag, entries in parent_store.external_data.items():
            for key, entry in entries.items():
                if key != compatibility_blob_key or keep_compatibility_blob:
                    named_data_store.add_named_data(
                        key,
                        parent_store.buffers[entry.buffer_index],
                        alignment=entry.alignment,
                        external_tag=tag,
                        tensor_layout=entry.tensor_layout,
                    )

        external_tag = f"aoti_{cls.get_device_name()}_blob"
        for storage_key, data in artifact.storages.items():
            named_data_store.add_named_data(
                storage_key, data, alignment=1, external_tag=external_tag
            )

        result.processed_bytes = _encode_fqn_weight_manifest(
            so_blob_key, artifact.entries
        )
        result.data_store_output = named_data_store.get_named_data_store_output()
        return result

    @classmethod
    def load_weights_blob(
        cls, blob_path: str, compile_specs: List[CompileSpec]
    ) -> tuple[Any, str]:
        """Keep low-memory CUDA named data file-backed during serialization.

        New FQN artifacts use this path only for AotiBackend's empty
        compatibility placeholder. Legacy binary-blob handling remains
        unchanged for callers that still provide a real blob.
        """
        known_hash = cls._materialized_blob_hashes.pop(blob_path, None)
        if not cls._is_low_memory_mode(compile_specs):
            return super().load_weights_blob(blob_path, compile_specs)
        blob_data = FileBackedData.move_from(blob_path, sha256=known_hash)
        weights_blob_hash = known_hash or blob_data.sha256()
        return blob_data, weights_blob_hash.hex()

    @classmethod
    def materialize_weights_blob(
        cls, paths: Any, compile_specs: List[CompileSpec]
    ) -> Any:
        if not isinstance(paths, list):
            return paths

        from torch.export.pt2_archive._package_weights import Weights

        weights = [path for path in paths if isinstance(path, Weights)]
        if not weights:
            return paths
        if len(weights) != 1:
            raise RuntimeError(
                f"Expected one CUDA AOTI weights output, got {len(weights)}"
            )

        so_path = next(
            (
                path
                for path in paths
                if isinstance(path, str) and path.endswith(".wrapper.so")
            ),
            None,
        )
        if so_path is None:
            raise RuntimeError(f"Expected a CUDA AOTI .wrapper.so output, got {paths}")
        blob_path = os.path.splitext(so_path)[0] + "_weights.blob"
        capture = getattr(_FQN_WEIGHTS_CAPTURE, "current", None)
        if capture is None:
            raise RuntimeError(
                "CUDA structured weights must be materialized inside preprocess"
            )
        capture.artifact = _materialize_fqn_weights(
            weights[0], os.path.dirname(blob_path), capture.mutated_fqns
        )

        # Keep AotiBackend's existing path contract intact. The compatibility
        # blob is empty and ignored by the versioned CUDA runtime path; old
        # artifacts continue to carry and load their original dense blob.
        with open(blob_path, "wb"):
            pass
        cls._materialized_blob_hashes[blob_path] = hashlib.sha256(b"").digest()

        # Replace the structured Weights output with the compatibility path
        # expected by AotiBackend's existing named-data packaging contract.
        materialized = [path for path in paths if not isinstance(path, Weights)]
        if blob_path not in materialized:
            materialized.append(blob_path)
        return materialized

    @classmethod
    def copy_exported_program_for_preprocess(
        cls, edge_program, compile_specs: List[CompileSpec]
    ):
        """Copy graph structure while sharing immutable tensor storage.

        CUDA preprocessing replaces state-dict entries when moving them to the
        target device; it does not mutate the source tensors.  Memoizing those
        tensors therefore avoids a model-sized host copy for every delegated
        method while preserving an independent graph and state-dict mapping.
        """
        if not cls._is_low_memory_mode(compile_specs):
            return copy.deepcopy(edge_program)

        tensor_memo = {
            id(tensor): tensor
            for values in (edge_program.state_dict, edge_program.constants)
            for tensor in values.values()
            if isinstance(tensor, torch.Tensor)
        }
        return copy.deepcopy(edge_program, tensor_memo)

    @classmethod
    def get_supported_fallback_kernels(cls) -> Dict[str, Any]:
        # ROCm does not build the CUDA-only .cu fallback shims.
        if torch.version.hip is not None:
            return {}
        return {
            "at::_ops::_weight_int4pack_mm::call": None,
            "at::_ops::sort_stable::call": None,
            "aoti_torch_cuda_randint_low_out": None,
            "executorch_cuda::int4_plain_mm": None,
            "aoti_torch_cuda_int4_plain_mm": None,
            "executorch_cuda::int5_plain_mm": None,
            "aoti_torch_cuda_int5_plain_mm": None,
            "executorch_cuda::int6_plain_mm": None,
            "aoti_torch_cuda_int6_plain_mm": None,
            "executorch_cuda::int8_plain_mm": None,
            "aoti_torch_cuda_int8_plain_mm": None,
        }

    @staticmethod
    def _get_custom_ops_to_c_shim_options() -> Dict[str, Any]:
        if torch.version.hip is not None:
            return {}
        try:
            return {
                "aot_inductor.custom_ops_to_c_shims": {
                    torch.ops.executorch_cuda.int4_plain_mm.default: [
                        "AOTITorchError aoti_torch_cuda_int4_plain_mm("
                        "AtenTensorHandle, AtenTensorHandle, AtenTensorHandle, "
                        "AtenTensorHandle, AtenTensorHandle, AtenTensorHandle, "
                        "int64_t, AtenTensorHandle*)"
                    ],
                    torch.ops.executorch_cuda.int5_plain_mm.default: [
                        "AOTITorchError aoti_torch_cuda_int5_plain_mm("
                        "AtenTensorHandle, AtenTensorHandle, AtenTensorHandle, "
                        "AtenTensorHandle, AtenTensorHandle, AtenTensorHandle, "
                        "AtenTensorHandle, int64_t, AtenTensorHandle*)"
                    ],
                    torch.ops.executorch_cuda.int6_plain_mm.default: [
                        "AOTITorchError aoti_torch_cuda_int6_plain_mm("
                        "AtenTensorHandle, AtenTensorHandle, AtenTensorHandle, "
                        "AtenTensorHandle, AtenTensorHandle, int64_t, "
                        "AtenTensorHandle*)"
                    ],
                    torch.ops.executorch_cuda.int8_plain_mm.default: [
                        "AOTITorchError aoti_torch_cuda_int8_plain_mm("
                        "AtenTensorHandle, AtenTensorHandle, AtenTensorHandle, "
                        "AtenTensorHandle, int64_t, AtenTensorHandle*)"
                    ],
                }
            }
        except AttributeError:
            # Custom ops may not be registered in this process.
            return {}

    @classmethod
    def get_decomposition_table(cls) -> Dict[Any, Any]:
        return {
            torch.ops.aten.conv1d.default: conv1d_to_conv2d,
        }

    @classmethod
    def get_custom_passes(cls, compile_specs: List[CompileSpec]) -> List[typing.Any]:
        """
        Return CUDA-specific passes: ReplaceEdgeOpWithTritonOpPass.

        The Triton kernel replacement behavior can be controlled via compile_specs:
        - triton_kernel_mode="ON": Always use Triton kernels
        - triton_kernel_mode="OFF": Never use Triton kernels and fallback to other implementations like cuda or decomposed operator.
        """
        # Parse compile_specs for triton_kernel_mode

        triton_kernel_mode = "ON"  # Default mode
        for spec in compile_specs:
            if spec.key == "triton_kernel_mode":
                mode = spec.value.decode("utf-8").upper()
                if mode not in ["ON", "OFF"]:
                    raise ValueError(
                        f"Invalid triton_kernel_mode: {mode}. Expected 'ON' or 'OFF'."
                    )
                triton_kernel_mode = mode
        passes = [MoveCondPredicateToCpuPass(), ReplaceInt64FloorDivWithFloatPass()]
        if triton_kernel_mode == "ON":
            passes.append(ReplaceEdgeOpWithTritonOpPass())
        return passes

    @classmethod
    def get_aoti_compile_options(
        cls, compile_specs: List[CompileSpec]
    ) -> Dict[str, typing.Any]:
        """
        Get AOTI compile options for CUDA backend.
        Options may vary based on platform (Linux vs Windows).
        """

        # Configure CUDA environment variables based on detected version

        emit_multi_arch_kernel = CudaBackend._setup_cuda_environment_for_fatbin()
        # Base options for all platforms

        options: Dict[str, typing.Any] = {
            # Disable this to support sdpa decomposition
            # TODO(gasoonjia): remove it after pin bump to latest pytorch
            "loop_ordering_after_fusion": False,
            # Better model precision
            "emulate_precision_casts": True,
            # Embed CUDA kernel binaries directly into the compiled shared object
            "aot_inductor.embed_kernel_binary": True,
            # Do not link against the full PyTorch/libtorch library
            "aot_inductor.link_libtorch": False,
            # Separate weight constants from the .so file
            "aot_inductor.package": True,
            "aot_inductor.package_constants_in_so": False,
            # Ask AOTI for structured constants. CUDABackend converts these to
            # independently named physical storages plus an FQN view manifest.
            "aot_inductor.package_constants_on_disk_format": cls._weights_format(
                compile_specs
            ),
            # Enable maximum automatic tuning for optimal performance
            "max_autotune": True,
            # Use TRITON for GEMM (General Matrix Multiply) operations tuning only to avoid using operators in libtorch
            "max_autotune_gemm_backends": "TRITON",
            # Use TRITON backend for convolution operations tuning only to avoid using operators in libtorch
            "max_autotune_conv_backends": "TRITON",
            "aot_inductor.emit_multi_arch_kernel": emit_multi_arch_kernel,
        }

        options.update(cls._get_custom_ops_to_c_shim_options())

        # Parse compile_specs to check for platform

        platform = "linux"
        emulate_precision_casts = True
        max_autotune = True
        autotune_at_compile_time = None
        shim_library_path = None
        for spec in compile_specs:
            if spec.key == "platform":
                platform = spec.value.decode("utf-8")
            elif spec.key == "emulate_precision_casts":
                emulate_precision_casts = _on_off_compile_spec_value(spec)
            elif spec.key == "max_autotune":
                max_autotune = _on_off_compile_spec_value(spec)
            elif spec.key == "autotune_at_compile_time":
                autotune_at_compile_time = _on_off_compile_spec_value(spec)
            elif spec.key == "shim_library_path":
                shim_library_path = spec.value.decode("utf-8")
        options["emulate_precision_casts"] = emulate_precision_casts
        options["max_autotune"] = max_autotune
        if autotune_at_compile_time is not None:
            options["triton.autotune_at_compile_time"] = autotune_at_compile_time
        # Add platform-specific options

        if platform == "windows":
            # For Windows, get default shim library path if not provided

            if shim_library_path is None:
                lib_dir = resources.files("executorch").joinpath("data/lib")
                # Only a CUDA build ships the import library, and a package directory
                # that does not exist still reads back as an ordinary path rather than
                # raising, so without this the failure surfaces from the linker instead.
                if not lib_dir.joinpath("aoti_cuda_shims.lib").is_file():
                    raise RuntimeError(
                        "Lowering for Windows links against aoti_cuda_shims.lib, which "
                        "only a CUDA build of executorch ships. Install a CUDA build, "
                        "or pass a shim_library_path compile spec naming a directory "
                        "that holds the import library."
                    )
                shim_library_path = str(lib_dir)
            options.update(
                {
                    "aot_inductor.cross_target_platform": "windows",
                    "aot_inductor.aoti_shim_library": "aoti_cuda_shims",
                    "aot_inductor.aoti_shim_library_path": shim_library_path,
                    "aot_inductor.precompile_headers": False,
                }
            )
        else:
            # Linux platform

            assert shim_library_path is None, (
                "shim_library_path should not be set for Linux"
            )
        return options

    @classmethod
    def get_extra_aoti_compile_context_manager(
        cls, compile_specs: Optional[List[CompileSpec]] = None
    ):
        """
        Combine all extra context managers needed during AOTInductor
        compilation for the CUDA backend. Each manager is documented at
        its own `enter_context` call site below.

        The low-memory export monkey-patch (CPU clones for mutated buffers)
        is gated on the ``low_memory_mode`` compile spec — only models that
        explicitly opt in (currently Qwen3.5 MoE) get it. Other models go
        through the unmodified AOTI codepath, which avoids regressions in
        their cuda CI exports.
        """
        # Parse compile_specs for low_memory_mode (default OFF). compile_specs
        # may be None when called without specs (parity with base default).
        low_memory_mode = "OFF"
        for spec in compile_specs or []:
            if spec.key == "low_memory_mode":
                mode = spec.value.decode("utf-8").upper()
                if mode not in ["ON", "OFF"]:
                    raise ValueError(
                        f"Invalid low_memory_mode: {mode}. Expected 'ON' or 'OFF'."
                    )
                low_memory_mode = mode

        @contextlib.contextmanager
        def _combined():
            with contextlib.ExitStack() as stack:
                # Force any remaining PyTorch SDPA ops to use the MATH
                # backend during compilation so AOTI can lower / decompose
                # them. SDPA ops already replaced by Triton kernels via
                # `ReplaceEdgeOpWithTritonOpPass` are unaffected; this is
                # only the fallback for the `triton_kernel_mode="OFF"` path.
                stack.enter_context(torch.nn.attention.sdpa_kernel([SDPBackend.MATH]))
                if low_memory_mode == "ON":
                    # Force AOTI's mutated-buffer clones onto CPU during
                    # compile so we stay under tight GPU memory caps (e.g.
                    # 24 GB on a consumer 4090). See
                    # `_compile_time_cpu_clones` for details. Only enabled
                    # for models that explicitly opt in via the
                    # `low_memory_mode="ON"` compile spec, since the
                    # monkey-patch can interact poorly with other models'
                    # AOTI compile pipelines.
                    stack.enter_context(_keep_triton_reduction_loads_loop_scoped())
                    stack.enter_context(
                        _compile_time_cpu_clones(torch.device(cls.get_device_name()))
                    )
                    _trim_host_memory()
                yield

        return _combined()

    @staticmethod
    def _is_low_memory_mode(compile_specs: List[CompileSpec]) -> bool:
        """Return True if any compile spec opts into low-memory export."""
        for spec in compile_specs:
            if spec.key == "low_memory_mode":
                return spec.value.decode("utf-8").upper() == "ON"
        return False

    @classmethod
    def _weights_format(cls, compile_specs: List[CompileSpec]) -> str:
        # CUDA consumes the structured AOTI output directly and emits a
        # versioned per-storage manifest. This is backend-wide rather than a
        # model/export-script option.
        return "pickle_weights"

    @classmethod
    def move_program_to_device(
        cls,
        edge_program,
        device: str,
        compile_specs: List[CompileSpec],
    ):
        """Move the program to ``device`` for AOTI compile.

        On a low-memory export (``low_memory_mode="ON"``) the KV-cache buffers —
        which can be 10+ GiB at long context — are placed on-device but with their
        storage freed (``resize_(0)``), so they never occupy device memory during
        the autotune / cpp_wrapper compile while still satisfying the device-match
        check on the cache update. They are re-synthesized as zeros for the lifted
        graph and the serialized blob. This activates automatically with low-memory
        mode. Other (non-low-memory) exports use the stock pass.
        """
        from torch.export.passes import move_to_device_pass

        if not cls._is_low_memory_mode(compile_specs):
            return move_to_device_pass(edge_program, device)
        return _move_to_device_resize_kv(edge_program, device)

    @classmethod
    def release_moved_tensors(
        cls,
        device_edge_program,
        compile_specs: List[CompileSpec],
    ) -> None:
        """
        Free GPU memory held by tensors that ``move_to_device_pass`` placed
        on CUDA (params, buffers, and constants of ``device_edge_program``).

        Resizing the underlying storage to 0 returns those bytes to PyTorch's
        caching allocator, so the next ``preprocess`` call (e.g. for the
        next method in a multi-method export) can reuse them when its own
        ``move_to_device_pass`` runs.
        """
        if not torch.cuda.is_available():
            return

        pools = []
        state_dict = getattr(device_edge_program, "state_dict", None)
        if state_dict:
            pools.append(state_dict.values())
        constants = getattr(device_edge_program, "constants", None)
        if constants:
            pools.append(constants.values())

        for pool in pools:
            for tensor in pool:
                if isinstance(tensor, torch.Tensor) and tensor.is_cuda:
                    try:
                        tensor.untyped_storage().resize_(0)
                    except Exception:
                        # Some storages may be shared / non-resizable; skip
                        # them rather than failing the export.
                        pass
