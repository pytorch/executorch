# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import contextlib
import os
import typing
from enum import Enum

from typing import Any, Dict, final, List, Optional, Set

import torch
from executorch.backends.cuda.replace_slice_copy_with_slice import (
    ReplaceSliceCopyWithSlicePass,
)
from executorch.exir._serialize._named_data_store import NamedDataStore
from executorch.exir._warnings import experimental
from executorch.exir.backend.backend_details import (
    BackendDetails,
    ExportedProgram,
    PreprocessResult,
)
from executorch.exir.backend.compile_spec_schema import CompileSpec
from torch._inductor.codegen.cpp_wrapper_cpu import CppWrapperCpu
from torch.export.passes import move_to_device_pass
from torch.nn.attention import SDPBackend

# exist fallback operators in et namespace;
supported_fallback_kernels: Dict[str, Any] = {}

# required fallback kernels but not supported
missing_fallback_kernels: Set[str] = set()


class COMPILE_SPEC_KEYS(Enum):
    METHOD_NAME = "method_name"


# context manager for non-fallback guarantee
# it will raise exception when generating fallback kernels during aoti compile
@contextlib.contextmanager
def collect_unsupported_fallback_kernels():
    original_generate_c_shim_extern_kernel_call = (
        CppWrapperCpu.generate_c_shim_extern_kernel_call
    )
    original_generate_fallback_kernel_with_runtime_lookup_aot = (
        CppWrapperCpu.generate_fallback_kernel_with_runtime_lookup_aot
    )

    def generate_c_shim_extern_kernel_call_and_collect_unsupported_kernels(
        self,
        kernel: str,
        args: list[str],
        device: str,
        *,
        debug_args: Optional[list[str]] = None,
    ):
        if kernel not in supported_fallback_kernels:
            missing_fallback_kernels.add(kernel)

        original_generate_c_shim_extern_kernel_call(
            self, kernel, args, device, debug_args=debug_args
        )

    def generate_fallback_kernel_with_runtime_lookup_aot_and_collect_unsupported_kernels(
        self,
        op_overload,
        raw_args,
        output_args,
        raw_outputs,
    ):
        # Extract kernel name for collection
        kernel_name = getattr(op_overload, "_name", str(op_overload))
        if kernel_name not in supported_fallback_kernels:
            missing_fallback_kernels.add(kernel_name)

        original_generate_fallback_kernel_with_runtime_lookup_aot(
            self, op_overload, raw_args, output_args, raw_outputs
        )

    CppWrapperCpu.generate_c_shim_extern_kernel_call = (
        generate_c_shim_extern_kernel_call_and_collect_unsupported_kernels
    )
    CppWrapperCpu.generate_fallback_kernel_with_runtime_lookup_aot = (
        generate_fallback_kernel_with_runtime_lookup_aot_and_collect_unsupported_kernels
    )
    try:
        yield
    finally:
        CppWrapperCpu.generate_c_shim_extern_kernel_call = (
            original_generate_c_shim_extern_kernel_call
        )
        CppWrapperCpu.generate_fallback_kernel_with_runtime_lookup_aot = (
            original_generate_fallback_kernel_with_runtime_lookup_aot
        )


@final
@experimental(
    "This API and all of cuda backend related functionality are experimental."
)
class CudaBackend(BackendDetails):
    """
    CudaBackend is a backend that compiles a model to run on CUDA devices. It uses the AOTInductor compiler to generate
    optimized CUDA kernels for the model's operators with libtorch-free. The compiled model can be executed on CUDA devices
    using the Executorch runtime.
    """

    @staticmethod
    def preprocess(
        edge_program: ExportedProgram,
        compile_specs: List[CompileSpec],
    ) -> PreprocessResult:
        # Move the edge_program from CPU to CUDA for aoti compile
        cuda_edge_program = move_to_device_pass(edge_program, "cuda")

        # replace slice_copy with slice
        ReplaceSliceCopyWithSlicePass()(cuda_edge_program.graph_module)

        edge_program_module = cuda_edge_program.module()

        # Grab all input placeholders from the graph
        user_input_names = cuda_edge_program.graph_signature.user_inputs
        user_input_placeholders = []
        for node in cuda_edge_program.graph.nodes:
            if node.op == "placeholder" and node.name in user_input_names:
                user_input_placeholders.append(node.meta["val"])

        options: dict[str, typing.Any] = {
            # Embed CUDA kernel binaries directly into the compiled shared object
            "aot_inductor.embed_kernel_binary": True,
            # Do not link against the full PyTorch/libtorch library
            "aot_inductor.link_libtorch": False,
            # Package model constants and other generated files directly in the shared object (.so) file
            "aot_inductor.package_constants_in_so": True,
            # Enable maximum automatic tuning for optimal performance
            "max_autotune": True,
            # Use TRITON for GEMM (General Matrix Multiply) operations tuning only to avoid using operators in libtorch
            "max_autotune_gemm_backends": "TRITON",
            # Use TRITON backend for convolution operations tuning only to avoid using operators in libtorch
            "max_autotune_conv_backends": "TRITON",
        }

        with collect_unsupported_fallback_kernels(), torch.nn.attention.sdpa_kernel(
            [SDPBackend.MATH]
        ), torch.no_grad():
            # torch._logging.set_logs(post_grad_graphs=True)
            so_path = torch._inductor.aot_compile(edge_program_module, tuple(user_input_placeholders), options=options)  # type: ignore[arg-type]
            if len(missing_fallback_kernels) > 0:
                formatted_kernels = "\n  - ".join(sorted(missing_fallback_kernels))
                raise RuntimeError(
                    f"Missing fallback kernels ({len(missing_fallback_kernels)} total):\n  - {formatted_kernels}\n"
                    "Please add them to the AOTI backend."
                )

        # pyre-ignorep[6]: Incompatible parameter type
        with open(so_path, "rb") as f:
            so_data = f.read()

        named_data_store = NamedDataStore()
        method_name = CudaBackend.method_name_from_compile_specs(compile_specs)
        named_data_store.add_named_data(
            method_name + "_so_blob", so_data, 1, "aoti_cuda_blob"
        )

        # Clean up the generated so file; it has been packaged into the NamdeDataStore
        # pyre-ignorep[6]: Incompatible parameter type
        os.remove(so_path)

        return PreprocessResult(
            processed_bytes=b"",
            debug_handle_map={},
            data_store_output=named_data_store.get_named_data_store_output(),
        )

    @staticmethod
    def generate_method_name_compile_spec(
        method_name: str,
    ) -> CompileSpec:
        """
        Returns the compile spec representing the model compute precision, for additional details
        please refer to the documentation for ``coremltools.precision``.
        """
        return CompileSpec(
            COMPILE_SPEC_KEYS.METHOD_NAME.value,
            method_name.encode("utf-8"),
        )

    @staticmethod
    def method_name_from_compile_specs(
        compile_specs: List[CompileSpec],
    ) -> str:
        """
        Returns the method name from the compile specs.
        """
        for spec in compile_specs:
            if spec.key == COMPILE_SPEC_KEYS.METHOD_NAME.value:
                return spec.value.decode("utf-8")
        raise RuntimeError(
            f"Could not find method name in compile specs: {compile_specs}"
        )
