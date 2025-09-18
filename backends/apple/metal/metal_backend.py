# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import contextlib
import copy
import os
import platform
import typing

from subprocess import check_call
from typing import Any, Dict, final, List, Optional, Set

import torch
from executorch.exir._serialize._named_data_store import NamedDataStore
from executorch.exir.backend.backend_details import (
    BackendDetails,
    ExportedProgram,
    PreprocessResult,
)
from executorch.exir.backend.compile_spec_schema import CompileSpec
from torch._inductor.codegen.cpp_wrapper_cpu import CppWrapperCpu
from torch.export.passes import move_to_device_pass


# exist fallback operators in et namespace;
supported_fallback_kernels: Dict[str, Any] = {
    "aoti_torch_mps_addmm_out": None,
    "aoti_torch_mps_convolution": None,
    "aoti_torch_mps_mm_out": None,
    "at::_ops::_scaled_dot_product_attention_math_for_mps::call": None,
}

# required fallback kernels but not supported
missing_fallback_kernels: Set[str] = set()


# context manager for non-fallback guarantee
# it will raise exception when generating fallback kernels during aoti compile
@contextlib.contextmanager
def collect_unsupported_fallback_kernels():
    original_generate_c_shim_extern_kernel_call = (
        CppWrapperCpu.generate_c_shim_extern_kernel_call
    )

    def generate_c_shim_extern_kernel_call_and_collect_unsupported_kernels(
        self,
        kernel: str,
        args: list[str],
        device: str,
        *,
        debug_args: Optional[list[str]] = None,
        debug_handle: Optional[int] = None,
    ):
        if kernel not in supported_fallback_kernels:
            missing_fallback_kernels.add(kernel)

        original_generate_c_shim_extern_kernel_call(
            self, kernel, args, device, debug_args=debug_args, debug_handle=debug_handle
        )

    CppWrapperCpu.generate_c_shim_extern_kernel_call = (
        generate_c_shim_extern_kernel_call_and_collect_unsupported_kernels
    )
    try:
        yield
    finally:
        CppWrapperCpu.generate_c_shim_extern_kernel_call = (
            original_generate_c_shim_extern_kernel_call
        )


@final
class MetalBackend(BackendDetails):
    @staticmethod
    def preprocess(
        edge_program: ExportedProgram,
        compile_specs: List[CompileSpec],
    ) -> PreprocessResult:
        print("entering the lowerable parts in MetalBackend.preprocess....")
        # Move the edge_program from CPU to MPS for aoti compile
        mps_edge_program = move_to_device_pass(edge_program, "mps")

        edge_program_module = mps_edge_program.module()

        # Grab all input placeholders from the graph
        user_input_names = mps_edge_program.graph_signature.user_inputs
        user_input_placeholders = []
        for node in mps_edge_program.graph.nodes:
            if node.op == "placeholder" and node.name in user_input_names:
                user_input_placeholders.append(node.meta["val"])

        output_path = os.path.join(os.getcwd(), "aoti.so")

        # Base options for all devices
        options: dict[str, typing.Any] = {
            "aot_inductor.package_constants_in_so": True,
            "aot_inductor.output_path": output_path,
            "aot_inductor.force_mmap_weights": False,
            "max_autotune": True,
            # "aot_inductor.embed_kernel_binary": True,
            "aot_inductor.link_libtorch": False,
            # "aot_inductor.debug_compile": True,
            # # Disable CPU threading/OpenMP to avoid libomp.dylib dependency
            # "cpp.enable_kernel_profile": False,
            # "cpp.threads": 1,  # Use single-threaded mode
        }

        with collect_unsupported_fallback_kernels():
            so_path = torch._inductor.aot_compile(edge_program_module, tuple(user_input_placeholders), options=options)  # type: ignore[arg-type]
            if len(missing_fallback_kernels) > 0:
                formatted_kernels = "\n  - ".join(sorted(missing_fallback_kernels))
                raise RuntimeError(
                    f"Missing fallback kernels ({len(missing_fallback_kernels)} total):\n  - {formatted_kernels}\n"
                    "Please add them to the AOTI backend."
                )

        assert so_path == output_path, f"Expected {output_path} but got {so_path}"

        print("so_path", so_path)

        with open(so_path, "rb") as f:
            so_data = f.read()

        # Use device-specific blob name
        named_data_store = NamedDataStore()
        named_data_store.add_named_data("so_blob", so_data, 1, f"aoti_metal_blob")

        # Clean up the generated so file; it has been packaged into the NamdeDataStore
        # pyre-ignorep[6]: Incompatible parameter type
        os.remove(so_path)

        return PreprocessResult(
            processed_bytes=b"",
            debug_handle_map={},
            data_store_output=named_data_store.get_named_data_store_output(),
        )
