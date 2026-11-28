# Copyright 2025-2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Callable, final, List, Mapping, Optional, Sequence, Tuple

import torch

from executorch.backends.arm.operator_support.tosa_supported_operators import (
    CheckResolvedTensorShapes,
    tosa_support_factory,
)
from executorch.backends.arm.tosa.partitioner import CustomOpSupported, TOSAPartitioner
from executorch.backends.arm.vgf import VgfBackend, VgfCompileSpec
from executorch.backends.arm.vgf._passes.rewrite_custom_sdpa_to_tosa_custom import (
    RewriteCustomSDPAToTosaCustomPass,
)
from executorch.backends.arm.vgf.backend import MUTABLE_BUFFER_OWNERSHIP_KEY
from executorch.exir.backend.compile_spec_schema import CompileSpec
from executorch.exir.backend.partitioner import DelegationSpec, PartitionResult
from executorch.exir.backend.utils import tag_mutated_buffer, WhyNoPartitionReporter
from executorch.exir.dialects._ops import ops as exir_ops
from torch._ops import OpOverload
from torch.export import ExportedProgram
from torch.fx.passes.operator_support import OperatorSupportBase


class SupportedCustomSDPA(OperatorSupportBase):
    """Accept only fixed-position decode custom SDPA nodes handled by VGF."""

    def is_node_supported(
        self, submodules: Mapping[str, torch.nn.Module], node: torch.fx.Node
    ) -> bool:
        """Return True when the node matches the VGF shader-backed SDPA path."""
        del submodules
        return RewriteCustomSDPAToTosaCustomPass.supports_tosa_custom(node)


@final
class VgfPartitioner(TOSAPartitioner):
    """Partitions subgraphs supported by the Arm Vgf backend.

    Args:
        compile_spec: The Vgf compilation specification.
        additional_checks: Optional sequence of additional operator support
            checks.
        mutable_buffer_names: Exact names to internalize. None selects all
            mutable buffers; unselected buffers remain program-owned.

    """

    def __init__(
        self,
        compile_spec: VgfCompileSpec,
        additional_checks: Optional[Sequence[OperatorSupportBase]] = None,
        mutable_buffer_names: Optional[set[str]] = None,
    ) -> None:
        self.delegation_spec = DelegationSpec(
            VgfBackend.__name__, compile_spec._to_list()
        )
        self.compile_spec = compile_spec
        self.additional_checks = additional_checks
        self.tosa_spec = compile_spec.tosa_spec
        self._custom_partition_ops: set[OpOverload] = set()
        self.intermediate_path = compile_spec._get_intermediate_path()
        self.alias_buffer_mutations = compile_spec.alias_buffer_mutations
        self.mutable_buffer_names = mutable_buffer_names
        # Preserve grid_sampler_2d for the VGF custom-lowering path only.
        self.register_custom_partition_op(exir_ops.edge.aten.grid_sampler_2d.default)
        self._requires_resolved_tensor_shapes = False

    def _create_operator_support(
        self,
        containing_program: ExportedProgram,
        reporter: WhyNoPartitionReporter,
    ) -> OperatorSupportBase:
        additional_checks = list(self.additional_checks or ())
        additional_checks.append(
            CheckResolvedTensorShapes(
                reporter,
                targets=(
                    exir_ops.edge.aten.view_copy.default,
                    exir_ops.edge.aten.permute_copy.default,
                ),
                reject_symbolic_scalars=True,
            )
        )
        if self._requires_resolved_tensor_shapes:
            additional_checks.append(CheckResolvedTensorShapes(reporter))

        positive_overrides: list[OperatorSupportBase] = []
        if self.tosa_spec.support_float():
            positive_overrides.append(SupportedCustomSDPA())
        if self._custom_partition_ops:
            positive_overrides.append(CustomOpSupported(self._custom_partition_ops))
        return tosa_support_factory(
            self.tosa_spec,
            containing_program,
            reporter,
            additional_checks=additional_checks,
            additional_positive_overrides=positive_overrides,
        )

    def ops_to_not_decompose(
        self,
        ep: ExportedProgram,
    ) -> Tuple[List[OpOverload], Optional[Callable[[torch.fx.Node], bool]]]:
        """Preserve only the SDPA nodes that the VGF custom shader can lower."""
        ops_to_not_decompose, filter_fn = super().ops_to_not_decompose(ep)

        def vgf_filter(node: torch.fx.Node) -> bool:
            if RewriteCustomSDPAToTosaCustomPass.supports_tosa_custom(node):
                return True
            if filter_fn is None:
                return False
            return filter_fn(node)

        return ops_to_not_decompose, vgf_filter

    def _get_docgen_custom_partition_ops(self) -> set[OpOverload | str]:
        """Return custom partition ops that should appear in docgen."""
        ops: set[OpOverload | str] = set(self._custom_partition_ops)
        if self.tosa_spec.support_float():
            ops.add("llama.custom_sdpa.default")
        return ops

    def _validate_mutable_buffers(self, exported_program: ExportedProgram) -> None:
        signature = exported_program.graph_signature
        mutations_by_buffer: dict[str, list[str]] = {}
        for mutation_name, buffer_name in signature.buffers_to_mutate.items():
            mutations_by_buffer.setdefault(buffer_name, []).append(mutation_name)

        nodes_by_name = {
            node.name: node for node in exported_program.graph_module.graph.nodes
        }
        for input_name, buffer_name in signature.inputs_to_buffers.items():
            if (
                self.mutable_buffer_names is not None
                and buffer_name not in self.mutable_buffer_names
            ):
                continue
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
            if buffer.dtype not in (torch.int8, torch.float32):
                raise ValueError(
                    f"VGF mutable buffer {buffer_name!r} must use torch.int8 "
                    "or torch.float32, "
                    f"got {buffer.dtype}."
                )
            if any(isinstance(dim, torch.SymInt) for dim in buffer.shape):
                raise ValueError(
                    f"VGF mutable buffer {buffer_name!r} must have a static shape."
                )
            mutation_node = nodes_by_name[mutation_names[0]]
            mutation = mutation_node.meta["val"]
            if mutation.dtype != buffer.dtype or mutation.shape != buffer.shape:
                raise ValueError(
                    f"VGF mutable buffer {buffer_name!r} must preserve its "
                    "shape and dtype at the mutation output."
                )
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

    def _check_mutable_buffers_owned(self, exported_program: ExportedProgram) -> None:
        # tag_mutated_buffer also keeps a buffer out of the delegate when its
        # new value is returned or read outside it, which
        # _validate_mutable_buffers does not see. Aliasing needs the delegate
        # to own the buffer.
        signature = exported_program.graph_signature
        mutated = set(signature.buffers_to_mutate.values())
        if self.mutable_buffer_names is not None:
            missing = self.mutable_buffer_names - mutated
            if missing:
                raise ValueError(
                    f"VGF selected mutable buffers are absent: {sorted(missing)}"
                )
            mutated &= self.mutable_buffer_names
        for node in exported_program.graph.nodes:
            buffer_name = signature.inputs_to_buffers.get(node.name)
            if buffer_name in mutated and "delegation_tag" not in node.meta:
                raise ValueError(
                    f"VGF mutable buffer {buffer_name!r} cannot be aliased: its "
                    "new value is returned or read outside the delegated "
                    "partition."
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
            if self.mutable_buffer_names is not None:
                signature = result.tagged_exported_program.graph_signature
                for node in result.tagged_exported_program.graph.nodes:
                    buffer_name = signature.inputs_to_buffers.get(node.name)
                    if (
                        buffer_name in signature.buffers_to_mutate.values()
                        and buffer_name not in self.mutable_buffer_names
                    ):
                        node.meta.pop("delegation_tag", None)
            self._check_mutable_buffers_owned(result.tagged_exported_program)
        return result
