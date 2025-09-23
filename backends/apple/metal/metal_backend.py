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
        named_data_store = NamedDataStore()

        target_device = "mps"
        blob_suffix = "metal"

        # Make a deep copy to avoid modifying the original
        copy_edge_program = copy.deepcopy(edge_program)

        # Move the edge_program to the appropriate device
        copy_edge_program = move_to_device_pass(copy_edge_program, target_device)
        edge_program_module = copy_edge_program.module()
        args, kwargs = copy_edge_program.example_inputs

        output_path = os.path.join(os.getcwd(), "aoti.so")

        # Base options for all devices
        options: dict[str, typing.Any] = {
            "aot_inductor.package_constants_in_so": True,
            "aot_inductor.output_path": output_path,
            "aot_inductor.force_mmap_weights": False,
            "max_autotune": True,
            # "aot_inductor.embed_kernel_binary": True,
            # "aot_inductor.link_libtorch": False,
            # "aot_inductor.debug_compile": True,
        }

        with collect_unsupported_fallback_kernels():
            so_path = torch._inductor.aot_compile(edge_program_module, args, kwargs, options=options)  # type: ignore[arg-type]
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
        named_data_store.add_named_data("so_blob", so_data, 1, f"aoti_{blob_suffix}_blob")


        return PreprocessResult(
            processed_bytes=b"",
            debug_handle_map={},
            data_store_output=named_data_store.get_named_data_store_output(),
        )
