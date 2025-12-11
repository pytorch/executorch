# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import typing
from importlib import resources
from typing import Any, Dict, final, List

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
