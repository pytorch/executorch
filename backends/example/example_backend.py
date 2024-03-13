# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import copy
from typing import final, List

from executorch.backends.example.example_backend_delegate_passes.merge_to_dim_pass import (
    MergeToDimPass,
)
from executorch.backends.example.example_backend_delegate_passes.permute_memory_formats_pass import (
    PermuteMemoryFormatsPass,
)

from executorch.exir.backend.backend_details import (
    BackendDetails,
    ExportedProgram,
    PreprocessResult,
)
from executorch.exir.backend.compile_spec_schema import CompileSpec


@final
class ExampleBackend(BackendDetails):
    @staticmethod
    def preprocess(
        edge_program: ExportedProgram,
        compile_specs: List[CompileSpec],
    ) -> PreprocessResult:
        print("entering  the lowerable parts in ExampleBackend.preprocess....")

        copy_edge_program = copy.deepcopy(edge_program)
        graph_module = copy_edge_program.graph_module
        graph_module_res = PermuteMemoryFormatsPass()(graph_module)
        assert graph_module_res is not None
        graph_module_res = MergeToDimPass()(graph_module_res.graph_module)
        assert graph_module_res is not None
        processed_bytes = str(graph_module_res.graph_module.graph)
        return PreprocessResult(bytes(processed_bytes, encoding="utf8"))
