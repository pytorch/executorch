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
from executorch.backends.transforms.fuse_dequant_linear import FuseDequantLinearPass
from executorch.backends.transforms.fuse_view_copy import FuseViewCopyTransform
from executorch.backends.transforms.remove_clone_ops import RemoveCloneOpsTransform

from executorch.backends.vulkan._passes import (
    insert_prepack_nodes,
    RemoveLocalScalarDenseOpsTransform,
)

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
from executorch.exir.pass_base import ExportPass, PassBase

from executorch.exir.passes import MemoryPlanningPass, SpecPropPass

from executorch.exir.passes.sym_shape_eval_pass import ConstraintBasedSymShapeEvalPass

from executorch.exir.program._program import _copy_module

from torch.export._remove_auto_functionalized_pass import (
    unsafe_remove_auto_functionalized_pass,
)

DEFAULT_DEBUG_HANDLE = 65535


# pyre-ignore
def apply_passes(program: ExportedProgram, passes) -> ExportedProgram:
    for p in passes:

        if issubclass(type(p), ExportPass) or issubclass(type(p), PassBase):
            new_gm = program.graph_module
            # This is a workaround to allow the memory planning pass to work without
            # having to first apply ToOutVarPass(). See the `greedy()` function in
            # `exir.memory_planning`; if this attribute isn't set, assertions in
            # `collect_spec_from_nodes()` will fail.
            if isinstance(p, MemoryPlanningPass):
                new_gm.encounter_to_out_var_failure = True

            new_gm_res = p(new_gm)
            assert new_gm_res is not None
            new_gm = new_gm_res.graph_module

            # See the application of this function in exir/program/_program.py for more
            # details on why this step is necessary.
            if isinstance(p, SpecPropPass):
                p.update_placeholder_tensor_specs(program, new_gm)

            _copy_module(program.graph_module, new_gm)
        else:
            program = p(program)

    return program


@final
class VulkanBackend(BackendDetails):
    @classmethod
    # pyre-ignore
    def preprocess(  # noqa: C901
        cls,
        program: ExportedProgram,
        module_compile_spec: List[CompileSpec],
    ) -> PreprocessResult:
        program = unsafe_remove_auto_functionalized_pass(program)

        # First, apply passes that fuse/remove operators to consolidate the graph
        # structure but still preserve an "ATen-compliant" graph structure (i.e. all
        # arguments to ATen operators must match the ATen function schema).
        program = apply_passes(
            program,
            [
                RemoveCloneOpsTransform(),
                AddmmToLinearTransform(),
                FuseDequantLinearPass(),
                FuseViewCopyTransform(),
                FuseBatchNormWithConvPass(program),
                FuseClampPass(),
            ],
        )

        # Next annotate tensor nodes with TensorSpec structs which is needed for dynamic
        # shapes and memory planning. Until this point, the graph must be ATen compliant
        # because SpecPropPass will be calling the underlying ATen operators during its
        # execution.
        program = apply_passes(program, [SpecPropPass()])

        # Apply graph transforms which either require `TensorSpec`s to have been created
        # or would create an non ATen compliant graph structure.
        program = apply_passes(
            program,
            [
                # Since this pass may replace a scalar argument with a tensor argument,
                # this pass may result in a non ATen compliant graph structure.
                RemoveLocalScalarDenseOpsTransform(),
                insert_prepack_nodes,
            ],
        )

        # Finally, apply dynamic shape passes and memory planning pass. These passes
        # must be applied only when the graph structure is finalized.
        program = apply_passes(
            program, [ConstraintBasedSymShapeEvalPass(), MemoryPlanningPass()]
        )

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
