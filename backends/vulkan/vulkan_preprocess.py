# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from functools import partial

from typing import Any, Dict, final, List

import executorch.backends.vulkan.utils as utils

from executorch.backends.transforms.addmm_mm_to_linear import AddmmToLinearTransform
from executorch.backends.transforms.fuse_clamp_with_binary_op import (
    FuseClampBinaryOpPass,
)
from executorch.backends.transforms.fuse_clamps import FuseClampsPass
from executorch.backends.transforms.fuse_conv_with_clamp import FuseConvClampPass
from executorch.backends.transforms.fuse_view_copy import FuseViewCopyTransform
from executorch.backends.transforms.view_copy_to_squeeze_unsqueeze import (
    ViewCopyToSqueezeUnsqueezePass,
)
from executorch.backends.vulkan._passes import (
    FoldQDQPass,
    FuseQuantizedOpsTransform,
    insert_prepack_nodes,
    RemoveLocalScalarDenseOpsTransform,
    RemoveRedundantOpsTransform,
    SqueezeUnsqueezeInputs,
    TagMemoryMetaPass,
)
from executorch.backends.vulkan._passes.fuse_patterns import FusePatternsPass
from executorch.backends.vulkan._passes.remove_asserts import RemoveAssertsTransform

from executorch.backends.vulkan.serialization.vulkan_graph_builder import VkGraphBuilder
from executorch.backends.vulkan.serialization.vulkan_graph_schema import (
    VkMemoryLayout,
    VkStorageType,
)
from executorch.backends.vulkan.serialization.vulkan_graph_serialize import (
    serialize_vulkan_graph,
)
from executorch.backends.xnnpack._passes import FuseBatchNormPass

from executorch.exir.backend.backend_details import (
    BackendDetails,
    CompileSpec,
    ExportedProgram,
    PreprocessResult,
)
from executorch.exir.backend.utils import DelegateMappingBuilder

from executorch.exir.memory_planning import greedy, MemoryPlanningAlgorithmSuite
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


def parse_compile_spec(compile_specs: List[CompileSpec]) -> Dict[str, Any]:
    options = {}
    for spec in compile_specs:
        if spec.key == "storage_type_override":
            options[spec.key] = VkStorageType(
                int.from_bytes(spec.value, byteorder="little")
            )
        if spec.key == "memory_layout_override":
            options[spec.key] = VkMemoryLayout(
                int.from_bytes(spec.value, byteorder="little")
            )
        if spec.key in {"texture_limits_x", "texture_limits_y", "texture_limits_z"}:
            options[spec.key] = int.from_bytes(spec.value, byteorder="little")

        if spec.key == "skip_tag_memory_metadata":
            options[spec.key] = bool.from_bytes(spec.value, byteorder="little")

        if spec.key == "downcast_64_bit":
            options[spec.key] = bool.from_bytes(spec.value, byteorder="little")

        if spec.key == "force_fp16":
            options[spec.key] = bool.from_bytes(spec.value, byteorder="little")

        # Unhandled options are ignored

    return options


@final
class VulkanBackend(BackendDetails):
    @classmethod
    # pyre-ignore
    def preprocess(  # noqa: C901
        cls,
        program: ExportedProgram,
        module_compile_spec: List[CompileSpec],
    ) -> PreprocessResult:
        compile_options = parse_compile_spec(module_compile_spec)
        limits_x = compile_options.get(
            "texture_limits_x", utils.DEFAULT_TEXTURE_LIMITS[0]
        )
        limits_y = compile_options.get(
            "texture_limits_y", utils.DEFAULT_TEXTURE_LIMITS[1]
        )
        limits_z = compile_options.get(
            "texture_limits_z", utils.DEFAULT_TEXTURE_LIMITS[2]
        )
        texture_limits = (limits_x, limits_y, limits_z)

        default_storage_type = compile_options.get(
            "storage_type_override", VkStorageType.TEXTURE_3D
        )
        default_memory_layout = compile_options.get(
            "memory_layout_override", VkMemoryLayout.TENSOR_WIDTH_PACKED
        )
        downcast_64_bit = compile_options.get("downcast_64_bit", True)
        force_fp16 = compile_options.get("force_fp16", False)

        program = unsafe_remove_auto_functionalized_pass(program)

        # First, apply passes that fuse/remove operators to consolidate the graph
        # structure but still preserve an "ATen-compliant" graph structure (i.e. all
        # arguments to ATen operators must match the ATen function schema).
        program = apply_passes(
            program,
            [
                FusePatternsPass(program),
                RemoveRedundantOpsTransform(),
                AddmmToLinearTransform(),
                FuseQuantizedOpsTransform(program),
                FoldQDQPass(program),
                SqueezeUnsqueezeInputs(),
                FuseViewCopyTransform(),
                ViewCopyToSqueezeUnsqueezePass(),
                FuseBatchNormPass(program),
                FuseClampsPass(),
                FuseConvClampPass(),
                FuseClampBinaryOpPass(),
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
                RemoveAssertsTransform(),
                # Since this pass may replace a scalar argument with a tensor argument,
                # this pass may result in a non ATen compliant graph structure.
                RemoveLocalScalarDenseOpsTransform(),
                insert_prepack_nodes,
            ],
        )

        # Optionally apply the memory metadata tagging pass, which will insert storage
        # type and memory layout transition nodes to ensure that all tensor arguments
        # to an operator is in a supported or optimal configuration. If this pass is not
        # applied, there will be a risk that some operators recieve arguments with
        # memory settings that are not supported by the implementation.
        if not compile_options.get("skip_tag_memory_metadata", False):
            program = apply_passes(
                program,
                [
                    TagMemoryMetaPass(
                        texture_limits,
                        default_storage_type=default_storage_type,
                        default_memory_layout=default_memory_layout,
                    ),
                ],
            )

        # Finally, apply dynamic shape passes and memory planning pass. These passes
        # must be applied only when the graph structure is finalized.
        greedy_memory_planning = partial(greedy, allow_overlapping_allocations=False)
        mem_planning_suite = MemoryPlanningAlgorithmSuite(
            algo_list=[greedy_memory_planning]
        )
        program = apply_passes(
            program,
            [
                ConstraintBasedSymShapeEvalPass(),
                MemoryPlanningPass(memory_planning_algo=mem_planning_suite),
            ],
        )

        graph_builder = VkGraphBuilder(
            program,
            DelegateMappingBuilder(generated_identifiers=True),
            downcast_64_bit=downcast_64_bit,
            force_fp16=force_fp16,
        )
        vk_graph = graph_builder.build_graph()

        return PreprocessResult(
            processed_bytes=serialize_vulkan_graph(
                vk_graph, graph_builder.const_tensors, []
            ),
            debug_handle_map=graph_builder.delegate_mapping_builder.get_delegate_mapping(),
            data_store_output=graph_builder.named_data_store.get_named_data_store_output(),
        )
