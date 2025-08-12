# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import copy
import os
import shutil

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

        print("here", edge_program.example_inputs)
        copy_edge_program = copy.deepcopy(edge_program)
        graph_module = copy_edge_program.graph_module
        args, kwargs = copy_edge_program.example_inputs
        temp_so_path = torch._inductor.aot_compile(graph_module, args, kwargs, options={})  # type: ignore[arg-type]
        so_path = os.path.join(os.getcwd(), "aoti.so")
        print("so_path after aot_compile: ", temp_so_path)
        print("so path we will using ", so_path)
        shutil.copyfile(temp_so_path, so_path)

        check_call(
            f"patchelf --remove-needed libtorch.so --remove-needed libc10.so --remove-needed libtorch_cuda.so --remove-needed libc10_cuda.so --remove-needed libtorch_cpu.so --add-needed libcudart.so {so_path}",
            shell=True,
        )

        return PreprocessResult(so_path.encode("utf-8"))
