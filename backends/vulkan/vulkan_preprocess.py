# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import final, List

import executorch.backends.vulkan.serialization.vulkan_graph_schema as vk_graph_schema

from executorch.backends.vulkan.serialization.vulkan_graph_builder import VkGraphBuilder
from executorch.backends.vulkan.serialization.vulkan_graph_serialize import (
    serialize_vulkan_graph,
)

from executorch.exir.backend.backend_details import (
    BackendDetails,
    CompileSpec,
    ExportedProgram,
    PreprocessResult,
)

from executorch.exir.passes import MemoryPlanningPass, SpecPropPass

from executorch.exir.passes.sym_shape_eval_pass import ConstraintBasedSymShapeEvalPass

from executorch.exir.program._program import _copy_module
from torch import dtype, float32

DEFAULT_DEBUG_HANDLE = 65535


@final
class VulkanBackend(BackendDetails):
    @staticmethod
    def get_vk_datatype(torch_dtype: dtype) -> vk_graph_schema.VkDataType:
        if torch_dtype == float32:
            return vk_graph_schema.VkDataType.fp32
        else:
            raise AssertionError(f"Invalid dtype for vulkan_preprocess ({torch_dtype})")

    @classmethod
    # pyre-ignore
    def preprocess(  # noqa: C901
        cls,
        program: ExportedProgram,
        module_compile_spec: List[CompileSpec],
    ) -> PreprocessResult:
        passes = [
            SpecPropPass(),
            ConstraintBasedSymShapeEvalPass(),
            MemoryPlanningPass("greedy"),
        ]

        new_gm = program.graph_module

        for p in passes:
            # This is a workaround to allow the memory planning pass to work without
            # having to first apply ToOutVarPass(). See the `greedy()` function in
            # `exir.memory_planning`; if this attribute isn't set, assertions in
            # `collect_spec_from_nodes()` will fail.
            if isinstance(p, MemoryPlanningPass):
                new_gm.encounter_to_out_var_failure = True
            new_gm_res = p(new_gm)
            assert new_gm_res is not None
            new_gm = new_gm_res.graph_module

        _copy_module(program.graph_module, new_gm)

        graph_builder = VkGraphBuilder(program)
        vk_graph = graph_builder.build_graph()

        return PreprocessResult(
            processed_bytes=serialize_vulkan_graph(
                vk_graph, graph_builder.const_tensors, []
            ),
        )
