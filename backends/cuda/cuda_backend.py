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
import logging
import os
import shutil
import threading
import typing
from importlib import resources
from typing import Any, Dict, final, List, Optional

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
from executorch.exir._warnings import experimental
from executorch.exir.backend.backend_details import BackendDetails
from executorch.exir.backend.compile_spec_schema import CompileSpec
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


def _is_cpu_clone_active() -> bool:
    return getattr(_CPU_CLONE_GUARD, "active", False)


def _trim_host_memory() -> None:
    gc.collect()
    try:
        ctypes.CDLL(None).malloc_trim(0)
    except AttributeError:
        pass
      
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

    def _force_external_weights_for_streaming(consts_size):
        # ``pickle_weights`` normally tells AOTI that no external binary blob
        # exists. We materialize that pickle output as a streamed blob below,
        # so the generated wrapper must use the matching external-weights ABI.
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
    _codecache.determine_aoti_mmap_flags = _force_external_weights_for_streaming
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


def _write_aoti_weights_blob(weights, blob_path: str) -> None:
    """Stream AOTI tensor storages without creating a model-sized bytes object."""
    _trim_host_memory()
    tensors = [tensor for tensor, _ in weights.values()]
    all_cuda = all(tensor.is_cuda for tensor in tensors)
    chunk_size = 8 * 1024 * 1024

    with open(blob_path, "wb") as output:
        for tensor in tensors:
            if tensor.is_mkldnn:
                raise RuntimeError("MKLDNN constants are not supported by CUDA AOTI")
            storage = tensor.untyped_storage()
            nbytes = storage.nbytes()
            if nbytes and tensor.is_cuda:
                byte_tensor = torch.empty(
                    0, dtype=torch.uint8, device=tensor.device
                ).set_(storage, 0, (nbytes,), (1,))
                for offset in range(0, nbytes, chunk_size):
                    cpu_chunk = byte_tensor[offset : offset + chunk_size].cpu()
                    output.write(memoryview(cpu_chunk.numpy()))
                del byte_tensor, cpu_chunk
            elif nbytes:
                raw_array = (ctypes.c_ubyte * nbytes).from_address(storage.data_ptr())
                raw_view = memoryview(raw_array).cast("B")
                for offset in range(0, nbytes, chunk_size):
                    output.write(raw_view[offset : offset + chunk_size])
                del raw_view, raw_array
            # Match AOTInductor's binary_blob layout: CUDA-only constants are
            # packed, while CPU/mixed constants are aligned to 64 bytes.
            if not all_cuda and (padding := (-nbytes) % 64):
                output.write(bytes(padding))
            del storage
    _trim_host_memory()


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
        CUDA backend saves SO blob and weights blob to an external .ptd file.
        This file must be provided at runtime via --data_path argument.
        """
        return True

    @classmethod
    def load_weights_blob(
        cls, blob_path: str, compile_specs: List[CompileSpec]
    ) -> tuple[Any, str]:
        """Keep low-memory CUDA weights file-backed during PTE serialization.

        The streamed file has the same layout as AOTInductor's ``binary_blob``.
        Keeping it file-backed avoids reading another model-sized copy into
        host memory without changing its bytes.
        """
        if not cls._is_low_memory_mode(compile_specs):
            return super().load_weights_blob(blob_path, compile_specs)
        blob_data = FileBackedData.move_from(blob_path)
        return blob_data, blob_data.sha256().hex()

    @classmethod
    def materialize_weights_blob(
        cls, paths: Any, compile_specs: List[CompileSpec]
    ) -> Any:
        if not cls._is_low_memory_mode(compile_specs) or not isinstance(paths, list):
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
        _write_aoti_weights_blob(weights[0], blob_path)

        # Forcing the external-weights ABI makes Inductor emit an empty blob
        # path alongside the Weights object. Replace that file in place and do
        # not add a duplicate path to the returned package outputs.
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
            # Store weight constants on disk in a binary blob. Low-memory mode
            # asks AOTI for a Weights object and streams the equivalent blob in
            # materialize_weights_blob; its context also forces the generated
            # wrapper to use the required external-weights ABI.
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

            assert (
                shim_library_path is None
            ), "shim_library_path should not be set for Linux"
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
        return (
            "pickle_weights"
            if cls._is_low_memory_mode(compile_specs)
            else "binary_blob"
        )

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
