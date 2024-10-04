# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from typing import final, List

from executorch.backends.transforms.addmm_mm_to_linear import AddmmToLinearTransform
from executorch.backends.transforms.fuse_batch_norm_with_conv import (
    FuseBatchNormWithConvPass,
)
from executorch.backends.transforms.fuse_conv_with_clamp import FuseClampPass
from executorch.backends.transforms.fuse_view_copy import FuseViewCopyTransform
from executorch.backends.transforms.mean_to_sum_div import MeanToSumDiv
from executorch.backends.transforms.remove_clone_ops import RemoveCloneOpsTransform

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
from executorch.exir.backend.utils import DelegateMappingBuilder

from executorch.exir.passes import MemoryPlanningPass, SpecPropPass

from executorch.exir.passes.sym_shape_eval_pass import ConstraintBasedSymShapeEvalPass

from executorch.exir.program._program import _copy_module

DEFAULT_DEBUG_HANDLE = 65535


@final
class VulkanBackend(BackendDetails):
    @classmethod
    # pyre-ignore
    def preprocess(  # noqa: C901
        cls,
        program: ExportedProgram,
        module_compile_spec: List[CompileSpec],
    ) -> PreprocessResult:
        passes = [
            RemoveCloneOpsTransform(),
            AddmmToLinearTransform(),
            FuseViewCopyTransform(),
            FuseBatchNormWithConvPass(program),
            FuseClampPass(),
            MeanToSumDiv(),
            SpecPropPass(),
            ConstraintBasedSymShapeEvalPass(),
            MemoryPlanningPass(),
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

        graph_builder = VkGraphBuilder(
            program, DelegateMappingBuilder(generated_identifiers=True)
        )
        vk_graph = graph_builder.build_graph()

        return PreprocessResult(
            processed_bytes=serialize_vulkan_graph(
                vk_graph, graph_builder.const_tensors, []
            ),
            debug_handle_map=graph_builder.delegate_mapping_builder.get_delegate_mapping(),
        )
