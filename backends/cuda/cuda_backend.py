# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import os
import shutil
import typing
from importlib import resources
from typing import Any, Dict, final, List, Optional

import torch
from executorch.backends.aoti.aoti_backend import AotiBackend
from executorch.backends.cuda.triton.replacement_pass import (
    ReplaceEdgeOpWithTritonOpPass,
)
from executorch.exir._warnings import experimental
from executorch.exir.backend.backend_details import BackendDetails
from executorch.exir.backend.compile_spec_schema import CompileSpec
from torch._inductor.decomposition import conv1d_to_conv2d
from torch.nn.attention import SDPBackend


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

            # Get compute capability of current CUDA device
            device = torch.cuda.current_device()
            capability = torch.cuda.get_device_capability(device)
            os.environ["TORCH_CUDA_ARCH_LIST"] = f"{capability[0]}.{capability[1]}"
            return True
        except Exception:
            return False

    @classmethod
    def get_supported_fallback_kernels(cls) -> Dict[str, Any]:
        return {
            "at::_ops::_weight_int4pack_mm::call": None,
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

        # return [ReplaceEdgeOpWithTritonOpPass()] if triton_kernel_mode == "ON" else []
        return []

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
    def get_extra_aoti_compile_context_manager(cls):
        """
        Return SDPA MATH backend context manager for CUDA compilation.

        This context manager plays as a fallback solution for any remaining PyTorch SDPA
        operations to use the MATH backend (decomposed SDPA) during AOTInductor compilation.

        Note:
        - If SDPA ops are replaced with Triton kernels by ReplaceEdgeOpWithTritonOpPass,
          this context manager will have no effect on those ops (they are no longer
          PyTorch SDPA ops).
        - If SDPA ops are NOT replaced (e.g., when triton_kernel_mode="OFF"), this
          context manager will force them to use the MATH backend, causing them to
          be automatically decomposed during compilation.
        """
        return torch.nn.attention.sdpa_kernel([SDPBackend.MATH])
