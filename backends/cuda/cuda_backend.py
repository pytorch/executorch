# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


import contextlib
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


@contextlib.contextmanager
def _compile_time_cpu_clones(target_device: torch.device):
    """Force AOTI's mutated-buffer clones onto CPU while preserving the
    serialized constants' target device."""
    from torch._inductor import compile_fx as _cfx, graph as _graph
    from torch._inductor.codegen.cpp_wrapper_cpu import CppWrapperCpu as _Cpp
    from torch._inductor.graph import GraphLowering as _GL

    orig_clone = _cfx.clone_preserve_strides
    orig_codegen_device = _Cpp.codegen_device
    orig_get_const = _GL.get_original_value_of_constant
    orig_is_same = _graph.is_same_tensor

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
    def get_supported_fallback_kernels(cls) -> Dict[str, Any]:
        return {
            "at::_ops::_weight_int4pack_mm::call": None,
            "at::_ops::sort_stable::call": None,
            "aoti_torch_cuda_randint_low_out": None,
            "executorch_cuda::int4_plain_mm": None,
            "aoti_torch_cuda_int4_plain_mm": None,
            "executorch_cuda::int6_plain_mm": None,
            "aoti_torch_cuda_int6_plain_mm": None,
            "executorch_cuda::int8_plain_mm": None,
            "aoti_torch_cuda_int8_plain_mm": None,
        }

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
                        f"Invalid triton_kernel_mode: {mode}. "
                        f"Expected 'ON' or 'OFF'."
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
            # Store weight constants on disk in a binary blob
            "aot_inductor.package_constants_on_disk_format": "binary_blob",
            # Enable maximum automatic tuning for optimal performance
            "max_autotune": True,
            # Use TRITON for GEMM (General Matrix Multiply) operations tuning only to avoid using operators in libtorch
            "max_autotune_gemm_backends": "TRITON",
            # Use TRITON backend for convolution operations tuning only to avoid using operators in libtorch
            "max_autotune_conv_backends": "TRITON",
            "aot_inductor.emit_multi_arch_kernel": emit_multi_arch_kernel,
        }

        try:
            import torch

            options["aot_inductor.custom_ops_to_c_shims"] = {
                torch.ops.executorch_cuda.int4_plain_mm.default: [
                    "AOTITorchError aoti_torch_cuda_int4_plain_mm("
                    "AtenTensorHandle, AtenTensorHandle, AtenTensorHandle, "
                    "AtenTensorHandle, int64_t, AtenTensorHandle*)"
                ],
                torch.ops.executorch_cuda.int6_plain_mm.default: [
                    "AOTITorchError aoti_torch_cuda_int6_plain_mm("
                    "AtenTensorHandle, AtenTensorHandle, AtenTensorHandle, "
                    "AtenTensorHandle, int64_t, AtenTensorHandle*)"
                ],
                torch.ops.executorch_cuda.int8_plain_mm.default: [
                    "AOTITorchError aoti_torch_cuda_int8_plain_mm("
                    "AtenTensorHandle, AtenTensorHandle, AtenTensorHandle, "
                    "AtenTensorHandle, int64_t, AtenTensorHandle*)"
                ],
            }
        except AttributeError:
            # quantize_op_dispatch not imported — op not registered, skip C shim mapping
            pass

        # Parse compile_specs to check for platform

        platform = "linux"
        shim_library_path = None
        for spec in compile_specs:
            if spec.key == "platform":
                platform = spec.value.decode("utf-8")
            if spec.key == "shim_library_path":
                shim_library_path = spec.value.decode("utf-8")
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
                    stack.enter_context(
                        _compile_time_cpu_clones(torch.device(cls.get_device_name()))
                    )
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
