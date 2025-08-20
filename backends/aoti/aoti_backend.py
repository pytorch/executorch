# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import copy
import os
import shutil
import typing

from subprocess import check_call
from typing import final, List

import torch
from executorch.exir.backend.backend_details import (
    BackendDetails,
    ExportedProgram,
    PreprocessResult,
)
from executorch.exir.backend.compile_spec_schema import CompileSpec


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
            "aot_inductor.debug_compile": True,
            "aot_inductor.repro_level": 3
        }
        so_path = torch._inductor.aot_compile(edge_program_module, args, kwargs, options=options)  # type: ignore[arg-type]

        assert so_path == output_path, f"Expected {output_path} but got {so_path}"

        check_call(
            f"patchelf --remove-needed libtorch.so --remove-needed libc10.so --remove-needed libtorch_cuda.so --remove-needed libc10_cuda.so --remove-needed libtorch_cpu.so --add-needed libcudart.so {output_path}",
            shell=True,
        )

        return PreprocessResult(so_path.encode("utf-8"))
