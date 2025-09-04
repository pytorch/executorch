# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import contextlib
import copy
import os
import shutil
import typing

from subprocess import check_call
from typing import Any, Dict, final, List, Optional, Set

import torch
from executorch.exir.backend.backend_details import (
    BackendDetails,
    ExportedProgram,
    PreprocessResult,
)
from executorch.exir.backend.compile_spec_schema import CompileSpec
from torch._inductor.codegen.cpp_wrapper_cpu import CppWrapperCpu


# exist fallback operators in et namespace;
supported_fallback_kernels: Dict[str, Any] = {}

# required fallback kernels but not supported
missing_fallback_kernels: Set[str] = set()


# context manager for non-fallback guarantee
# it will raise exception when generating fallback kernels during aoti compile
@contextlib.contextmanager
def raise_on_generate_fall_back_call():
    original_generate_c_shim_extern_kernel_call = (
        CppWrapperCpu.generate_c_shim_extern_kernel_call
    )

    def generate_supported_c_shim_extern_kernel_call(
        self,
        kernel: str,
        args: list[str],
        device: str,
        *,
        debug_args: Optional[list[str]] = None,
    ):
        if kernel in supported_fallback_kernels:
            original_generate_c_shim_extern_kernel_call(
                self, kernel, args, device, debug_args=debug_args
            )
        else:
            missing_fallback_kernels.add(kernel)

    CppWrapperCpu.generate_c_shim_extern_kernel_call = (
        generate_supported_c_shim_extern_kernel_call
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

        # print("here", edge_program.example_inputs)
        copy_edge_program = copy.deepcopy(edge_program)
        # graph_module = copy_edge_program.graph_module
        edge_program_module = copy_edge_program.module()
        args, kwargs = copy_edge_program.example_inputs
        # print("args, kwargs", args, kwargs)
        print("len(args)", len(args))
        print("args[0].shape", args[0].shape)
        print("len(kwargs)", len(kwargs))

        output_path = os.path.join(os.getcwd(), "aoti.so")

        options: dict[str, typing.Any] = {
            "aot_inductor.package_constants_in_so": True,
            "aot_inductor.output_path": output_path,
            "max_autotune": True,
            "max_autotune_gemm_backends": "TRITON",
            "max_autotune_conv_backends": "TRITON",
        }

        with raise_on_generate_fall_back_call():
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

        return PreprocessResult(so_path.encode("utf-8"))
