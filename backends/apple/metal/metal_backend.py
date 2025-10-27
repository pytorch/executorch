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
from executorch.backends.aoti.passes.replace_view_copy_with_view import (
    ReplaceViewCopyWithViewPass,
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


# exist fallback operators in et namespace;
supported_fallback_kernels: Dict[str, Any] = {
    "aoti_torch_mps_convolution": None,
    "aoti_torch_mps_mm_out": None,
    "at::_ops::_scaled_dot_product_attention_math_for_mps::call": None,
}

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
@experimental(
    "This API and all of Metal backend related functionality are experimental."
)
class MetalBackend(BackendDetails):
    @staticmethod
    def preprocess(
        edge_program: ExportedProgram,
        compile_specs: List[CompileSpec],
    ) -> PreprocessResult:
        print("entering the lowerable parts in MetalBackend.preprocess....")
        # Move the edge_program from CPU to MPS for aoti compile
        mps_edge_program = move_to_device_pass(edge_program, "mps")

        # replace slice_copy with slice
        ReplaceViewCopyWithViewPass()(mps_edge_program.graph_module)

        edge_program_module = mps_edge_program.module()

        # Grab all input placeholders from the graph
        user_input_names = mps_edge_program.graph_signature.user_inputs
        user_input_placeholders = []
        for node in mps_edge_program.graph.nodes:
            if node.op == "placeholder" and node.name in user_input_names:
                user_input_placeholders.append(node.meta["val"])

        # Base options for all devices
        options: dict[str, typing.Any] = {
            # Do not link against the full PyTorch/libtorch library
            "aot_inductor.link_libtorch": False,
            # Separate weight constants from the .so file
            "aot_inductor.package": True,
            "aot_inductor.package_constants_in_so": False,
            # Store weight constants on disk in a binary blob
            "aot_inductor.package_constants_on_disk_format": "binary_blob",
            # Enable maximum automatic tuning for optimal performance
            "max_autotune": True,
            # "aot_inductor.debug_compile": True,
            # "aot_inductor.force_mmap_weights": False,
        }

        with collect_unsupported_fallback_kernels():
            paths = torch._inductor.aot_compile(edge_program_module, tuple(user_input_placeholders), options=options)  # type: ignore[arg-type]
            if len(missing_fallback_kernels) > 0:
                formatted_kernels = "\n  - ".join(sorted(missing_fallback_kernels))
                raise RuntimeError(
                    f"Missing fallback kernels ({len(missing_fallback_kernels)} total):\n  - {formatted_kernels}\n"
                    "Please add them to the AOTI backend."
                )

        # Extract the .so and .blob paths from the returned list
        so_path = None
        blob_path = None
        for path in paths:
            if path.endswith(".wrapper.so"):
                so_path = path
            elif path.endswith(".wrapper_weights.blob"):
                blob_path = path

        if so_path is None or blob_path is None:
            raise RuntimeError(
                f"Could not find required files in compiled paths, got {paths}"
            )

        # pyre-ignorep[6]: Incompatible parameter type
        with open(so_path, "rb") as f:
            so_data = f.read()

        named_data_store = NamedDataStore()
        method_name = MetalBackend.method_name_from_compile_specs(compile_specs)

        # Keep the so file in the NamedDataStore, so that it can be packaged into the .pte file.
        named_data_store.add_named_data(method_name + "_so_blob", so_data, 1, None)

        # Add weights blob to named data store
        with open(blob_path, "rb") as f:
            blob_data = f.read()

        named_data_store.add_named_data(
            method_name + "_weights_blob", blob_data, 1, "aoti_metal_blob"
        )

        # Clean up the weights blob file
        os.remove(blob_path)

        # Clean up the generated so file; it has been packaged into the NamedDataStore
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
        Generates a CompileSpec for the given method name.
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
