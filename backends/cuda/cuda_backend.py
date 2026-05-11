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


@contextlib.contextmanager
def _compile_time_cpu_clones(target_device: torch.device):
    """Force AOTI's mutated-buffer clones onto CPU while preserving the
    serialized constants' target device."""
    from torch._inductor import compile_fx as _cfx
    from torch._inductor.codegen.cpp_wrapper_cpu import CppWrapperCpu as _Cpp

    orig_clone = _cfx.clone_preserve_strides
    orig_codegen_device = _Cpp.codegen_device

    def _cpu_clone_preserve_strides(x: torch.Tensor) -> torch.Tensor:
        # `clone_preserve_strides` is shared by `_unlift_graph` (clones
        # lifted buffers — can be safely kept on CPU) and by autotuning code
        # in `triton_heuristics.py` (clones for benchmark — must stay on
        # GPU for Triton). Discriminate by caller frame so we only force
        # CPU clones for the buffer-lifting path.
        import sys

        caller = sys._getframe(1).f_code.co_name
        if caller == "_unlift_graph":
            return orig_clone(x).cpu()
        return orig_clone(x)

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
    prev_active = getattr(_CPU_CLONE_GUARD, "active", False)
    _CPU_CLONE_GUARD.active = True
    try:
        yield
    finally:
        _CPU_CLONE_GUARD.active = prev_active
        _cfx.clone_preserve_strides = orig_clone
        _Cpp.codegen_device = orig_codegen_device


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
        passes = [MoveCondPredicateToCpuPass()]
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
