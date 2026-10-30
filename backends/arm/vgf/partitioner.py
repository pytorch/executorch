# Copyright 2025-2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import final, Optional, Sequence

import torch

from executorch.backends.arm.tosa.partitioner import TOSAPartitioner
from executorch.backends.arm.vgf import VgfBackend, VgfCompileSpec
from executorch.backends.arm.vgf.backend import MUTABLE_BUFFER_OWNERSHIP_KEY
from executorch.exir.backend.compile_spec_schema import CompileSpec
from executorch.exir.backend.partitioner import DelegationSpec, PartitionResult
from executorch.exir.backend.utils import tag_mutated_buffer
from executorch.exir.dialects._ops import ops as exir_ops
from torch._ops import OpOverload
from torch.export import ExportedProgram
from torch.fx.passes.operator_support import OperatorSupportBase


@final
class VgfPartitioner(TOSAPartitioner):
    """Partitions subgraphs supported by the Arm Vgf backend.

    Args:
        compile_spec: The Vgf compilation specification.
        additional_checks: Optional sequence of additional operator support checks.

    """

    def __init__(
        self,
        compile_spec: VgfCompileSpec,
        additional_checks: Optional[Sequence[OperatorSupportBase]] = None,
    ) -> None:
        # Override the delegation spec for Vgf
        self.delegation_spec = DelegationSpec(
            VgfBackend.__name__, compile_spec._to_list()
        )
        self.compile_spec = compile_spec
        self.additional_checks = additional_checks
        self.tosa_spec = compile_spec.tosa_spec
        self._custom_partition_ops: set[OpOverload] = set()
        self.intermediate_path = compile_spec._get_intermediate_path()
        self.alias_buffer_mutations = compile_spec.alias_buffer_mutations
        # Preserve grid_sampler_2d for the VGF custom-lowering path only.
        self.register_custom_partition_op(exir_ops.edge.aten.grid_sampler_2d.default)
        self._requires_resolved_tensor_shapes = False

    def _validate_mutable_buffers(self, exported_program: ExportedProgram) -> None:
        signature = exported_program.graph_signature
        mutations_by_buffer: dict[str, list[str]] = {}
        for mutation_name, buffer_name in signature.buffers_to_mutate.items():
            mutations_by_buffer.setdefault(buffer_name, []).append(mutation_name)

        nodes_by_name = {
            node.name: node for node in exported_program.graph_module.graph.nodes
        }
        for input_name, buffer_name in signature.inputs_to_buffers.items():
            mutation_names = mutations_by_buffer.get(buffer_name)
            if mutation_names is None:
                continue
            if len(mutation_names) != 1:
                raise ValueError(
                    f"VGF mutable buffer {buffer_name!r} must have exactly one "
                    f"mutation output, found {len(mutation_names)}."
                )

            buffer_node = nodes_by_name[input_name]
            buffer = (
                exported_program.constants.get(buffer_name)
                if buffer_name in signature.non_persistent_buffers
                else exported_program.state_dict.get(buffer_name)
            )
            if buffer is None:
                raise ValueError(f"VGF mutable buffer {buffer_name!r} must be present.")
            if buffer.dtype != torch.int8:
                raise ValueError(
                    f"VGF mutable buffer {buffer_name!r} must use torch.int8, "
                    f"got {buffer.dtype}."
                )
            if any(isinstance(dim, torch.SymInt) for dim in buffer.shape):
                raise ValueError(
                    f"VGF mutable buffer {buffer_name!r} must have a static shape."
                )
            mutation_node = nodes_by_name[mutation_names[0]]
            mutation_tag = mutation_node.meta.get("delegation_tag")
            user_tags = {
                user.meta.get("delegation_tag")
                for user in buffer_node.users
                if user.op != "output"
            }
            if mutation_tag is None or user_tags != {mutation_tag}:
                consumers = ", ".join(
                    f"{user.name}:{user.meta.get('delegation_tag')}"
                    for user in buffer_node.users
                    if user.op != "output"
                )
                raise ValueError(
                    f"VGF mutable buffer {buffer_name!r} must be consumed and "
                    "mutated entirely within one delegated partition; "
                    f"mutation tag={mutation_tag}, consumers=[{consumers}]."
                )

    def partition(self, exported_program: ExportedProgram) -> PartitionResult:
        """Partition the program and tag TOSA-compatible subgraphs.

        Run the FX capability-based partitioner to propose subgraphs, then
        refine tags by removing boundary-only quantize/dequantize nodes and by
        rejecting partitions that would lower to no-ops. Emit a detailed report
        of rejected nodes and their reasons. When VGF aliasing is enabled,
        validate and tag eligible mutable buffers for backend ownership.

        Args:
            exported_program (ExportedProgram): Program to analyze and
                partition.

        Returns:
            PartitionResult: The input program with nodes tagged for delegation
            and a mapping of partition tags to delegation specs.

        """
        result = super().partition(exported_program)
        if self.alias_buffer_mutations:
            result.partition_tags = {
                tag: DelegationSpec(
                    spec.backend_id,
                    list(spec.compile_specs)
                    + [CompileSpec(MUTABLE_BUFFER_OWNERSHIP_KEY, b"1")],
                )
                for tag, spec in result.partition_tags.items()
            }
            self._validate_mutable_buffers(result.tagged_exported_program)
            tag_mutated_buffer(result.tagged_exported_program)
        return result
