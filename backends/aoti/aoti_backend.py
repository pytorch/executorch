# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import contextlib
import copy
import os
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
supported_fallback_kernels: Dict[str, Any] = {}

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
    ):
        if kernel not in supported_fallback_kernels:
            missing_fallback_kernels.add(kernel)

        original_generate_c_shim_extern_kernel_call(
            self, kernel, args, device, debug_args=debug_args
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
class AotiBackend(BackendDetails):
    @staticmethod
    def preprocess(
        edge_program: ExportedProgram,
        compile_specs: List[CompileSpec],
    ) -> PreprocessResult:

        print("entering  the lowerable parts in AotiBackend.preprocess....")
        named_data_store = NamedDataStore()

        # print("here", edge_program.example_inputs)
        copy_edge_program = copy.deepcopy(edge_program)

        # Move the edge_program from CPU to CUDA using move_to_device_pass
        copy_edge_program = move_to_device_pass(copy_edge_program, "cuda")
        # graph_module = copy_edge_program.graph_module
        edge_program_module = copy_edge_program.module()
        args, kwargs = copy_edge_program.example_inputs

        # Deep copy args and move tensors to CUDA for aot_compile
        def move_to_cuda(obj):
            if isinstance(obj, torch.Tensor):
                return obj.cuda()
            elif isinstance(obj, (list, tuple)):
                return type(obj)(move_to_cuda(item) for item in obj)
            elif isinstance(obj, dict):
                return {key: move_to_cuda(value) for key, value in obj.items()}
            else:
                return obj

        args = move_to_cuda(copy.deepcopy(args))
        kwargs = move_to_cuda(copy.deepcopy(kwargs))

        # print("args, kwargs", args, kwargs)
        print("len(args)", len(args))
        print("args[0].shape", args[0].shape)
        print("len(kwargs)", len(kwargs))

        output_path = os.path.join(os.getcwd(), "aoti.so")

        options: dict[str, typing.Any] = {
            "aot_inductor.package_constants_in_so": True,
            "aot_inductor.output_path": output_path,
            "aot_inductor.force_mmap_weights": False,
            "max_autotune": True,
            "max_autotune_gemm_backends": "TRITON",
            "max_autotune_conv_backends": "TRITON",
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

        check_call(
            f"patchelf --remove-needed libtorch.so --remove-needed libc10.so --remove-needed libtorch_cuda.so --remove-needed libc10_cuda.so --remove-needed libtorch_cpu.so --add-needed libcudart.so {output_path}",
            shell=True,
        )

        print("so_path", so_path)

        with open(so_path, "rb") as f:
            so_data = f.read()

        named_data_store.add_named_data("so_blob", so_data, 1, "aoti_cuda_blob")

        return PreprocessResult(
            processed_bytes=b"",
            debug_handle_map={},
            data_store_output=named_data_store.get_named_data_store_output(),
        )
