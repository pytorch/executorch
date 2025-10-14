# Copyright 2023-2025 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe
"""Provide a partitioner for delegating subgraphs to the TOSA backend.

Implement logic to identify and tag regions of an ``ExportedProgram`` that can
be delegated to the TOSA backend. Use this module to:

- Partition graphs based on operator support and additional checks.
- Prune trivial no-op partitions that would lower to empty TOSA graphs.
- Tag constant data and report reasons for rejected nodes.
"""

import logging
from typing import Callable, List, Optional, Sequence, Tuple

import torch
from executorch.backends.arm._passes.arm_pass_utils import get_first_fake_tensor
from executorch.backends.arm._passes.convert_expand_copy_to_repeat import (
    calculate_multiples,
)
from executorch.backends.arm.constants import DQ_OPS, Q_OPS
from executorch.backends.arm.operator_support.tosa_supported_operators import (
    tosa_support_factory,
)
from executorch.backends.arm.tosa.backend import TOSABackend
from executorch.backends.arm.tosa.compile_spec import TosaCompileSpec
from executorch.exir.backend.partitioner import (
    DelegationSpec,
    Partitioner,
    PartitionResult,
)
from executorch.exir.backend.utils import tag_constant_data, WhyNoPartitionReporter
from executorch.exir.dialects._ops import ops as exir_ops
from torch.export.exported_program import ExportedProgram
from torch.fx.passes.infra.partitioner import CapabilityBasedPartitioner
from torch.fx.passes.operator_support import OperatorSupportBase

logger = logging.getLogger(__name__)


def is_noop_clone(node: torch.fx.node.Node) -> bool:
    """Return True if the node is a no-op ``dim_order_ops._clone_dim_order``.

    Args:
        node (torch.fx.Node): FX node to inspect.

    Returns:
        bool: True if the node targets ``dim_order_ops._clone_dim_order.default``
        in the Edge dialect; otherwise, False.

    """
    return node.target == exir_ops.edge.dim_order_ops._clone_dim_order.default


def is_noop_alias_copy(node: torch.fx.Node) -> bool:
    """Return True if the node is a no-op ``aten.alias_copy``.

    Args:
        node (torch.fx.Node): FX node to inspect.

    Returns:
        bool: True if the node targets ``aten.alias_copy.default``; otherwise,
        False.

    """
    return node.target == exir_ops.edge.aten.alias_copy.default


def is_noop_to_dim_order_copy(node: torch.fx.node.Node) -> bool:
    """Return True if node is a no-op ``dim_order_ops._to_dim_order_copy``.

    Consider the op a no-op when the output dtype equals the input's dtype.

    Args:
        node (torch.fx.Node): FX node to inspect.

    Returns:
        bool: True if it targets ``_to_dim_order_copy.default`` and preserves
        dtype; otherwise, False.

    """
    if node.target != exir_ops.edge.dim_order_ops._to_dim_order_copy.default:
        return False
    else:
        return node.meta.get("dtype") == get_first_fake_tensor(node.args[0]).dtype  # type: ignore[arg-type]


def is_noop_expand(node: torch.fx.node.Node) -> bool:
    """Return True if the node is an ``expand_copy`` with all-ones multiples.

    This corresponds to a semantic no-op, since expanding by 1 along every
    dimension leaves the tensor unchanged.

    Args:
        node (torch.fx.Node): FX node to inspect.

    Returns:
        bool: True if the node targets ``aten.expand_copy.default`` and all
        computed multiples are 1; otherwise, False.

    """
    if node.target != exir_ops.edge.aten.expand_copy.default:
        return False
    else:
        multiples = calculate_multiples(node.args)
    return all(m == 1 for m in multiples)


class TOSAPartitioner(Partitioner):
    """Partition an exported program into TOSA-delegable subgraphs.

    Construct this partitioner for compile specs targeting TOSA. The partition
    algorithm uses capability checks and optional additional operator-support
    rules to tag nodes with a delegation tag per subgraph.
    """

    def __init__(
        self,
        compile_spec: TosaCompileSpec,
        additional_checks: Optional[Sequence[OperatorSupportBase]] = None,
    ) -> None:
        """Initialize the TOSAPartitioner.

        Args:
            compile_spec (TosaCompileSpec): Parsed compile specifications for
                TOSA containing the TOSA spec and original list.
            additional_checks (Optional[Sequence[OperatorSupportBase]]): Extra
                operator-support checks to apply when partitioning.

        Raises:
            RuntimeError: If the provided compile spec does not target TOSA.

        """
        self.delegation_spec = DelegationSpec(
            TOSABackend.__name__, compile_spec.to_list()
        )
        self.tosa_spec = compile_spec.tosa_spec
        self.additional_checks = additional_checks
        self.tosa_spec = compile_spec.tosa_spec

    def partition(self, exported_program: ExportedProgram) -> PartitionResult:  # noqa
        """Partition the program and tag TOSA-compatible subgraphs.

        Run the FX capability-based partitioner to propose subgraphs, then
        refine tags by removing boundary-only quantize/dequantize nodes and by
        rejecting partitions that would lower to no-ops. Emit a detailed report
        of rejected nodes and their reasons.

        Args:
            exported_program (ExportedProgram): Program to analyze and
                partition.

        Returns:
            PartitionResult: The input program with nodes tagged for delegation
            and a mapping of partition tags to delegation specs.

        """
        logger.info("TOSAPartitioner::partition")
        partition_tags: dict[str, DelegationSpec] = {}

        logger.info(
            f"Partitioning for {self.delegation_spec.backend_id}: {self.tosa_spec}"
        )

        reporter = WhyNoPartitionReporter()
        operator_support = tosa_support_factory(
            self.tosa_spec, exported_program, reporter, self.additional_checks
        )
        capability_partitioner = CapabilityBasedPartitioner(
            exported_program.graph_module,
            operator_support,
            allows_single_node_partition=True,
        )
        partition_list = capability_partitioner.propose_partitions()

        def reject_partition(reason: str, partition, tag) -> None:
            """Remove a proposed partition and record the rejection reason.

            Args:
                reason (str): Human-readable explanation for rejection.
                partition (object): Proposed partition object from the
                    capability partitioner.
                tag (str): Delegation tag associated with the partition.

            """
            for node in partition.nodes:
                if "delegation_tag" in node.meta:
                    del node.meta["delegation_tag"]
                    reporter.report_reject(
                        node,
                        reason,
                    )
            partition_tags.pop(tag, None)

        for partition in partition_list:
            tag = f"tag{partition.id}"

            def is_partitioned(node: torch.fx.Node, tag=tag) -> bool:
                """Return True if the node currently belongs to the partition ``tag``.

                Args:
                    node (torch.fx.Node): FX node to check.
                    tag (str): Delegation tag identifying the partition.

                Returns:
                    bool: True if the node carries the matching delegation tag.

                """
                return (
                    "delegation_tag" in node.meta and node.meta["delegation_tag"] == tag
                )

            for node in partition.nodes:
                node.meta["delegation_tag"] = tag
                partition_tags[tag] = self.delegation_spec

            # De-tag outermost q-nodes upwards and dq-nodes downwards.
            # De-tag if at least one input/output is not part of the partition.
            for node in exported_program.graph_module.graph.nodes:
                if not is_partitioned(node):
                    continue
                if node.target in Q_OPS:
                    for input in node.all_input_nodes:
                        if not is_partitioned(input):
                            del node.meta["delegation_tag"]
                            break
                    continue

                if node.target in DQ_OPS:
                    for user in node.users:
                        if not is_partitioned(user):
                            del node.meta["delegation_tag"]
                            break
                    continue

                if self.tosa_spec.support_float():
                    continue

                if is_partitioned(node):
                    for input in node.all_input_nodes:
                        if is_partitioned(input):
                            continue
                        if get_first_fake_tensor(input).dtype.is_floating_point:
                            reporter.report_reject(
                                node,
                                f"Was first node in partition and input {input.name} had fp dtype.",
                            )
                            del node.meta["delegation_tag"]
                            break

            is_noop_partition = all(
                is_noop_clone(node)
                or is_noop_alias_copy(node)
                or is_noop_expand(node)
                or is_noop_to_dim_order_copy(node)
                or node.target in Q_OPS
                or node.target in DQ_OPS
                for node in partition.nodes
            )
            if is_noop_partition:
                reject_partition(
                    "Partition contained only ops which are removed in the TOSA lowering, leading to an empty partition.",
                    partition,
                    tag,
                )

        tag_constant_data(exported_program)
        logger.info(f"The following nodes were rejected for {self.tosa_spec}:")
        logger.info("\n" + reporter.get_table_report())
        logger.info("(Placeholders and outputs are not included in this list)")
        return PartitionResult(
            tagged_exported_program=exported_program, partition_tags=partition_tags
        )

    def ops_to_not_decompose(
        self,
        ep: ExportedProgram,
    ) -> Tuple[List[torch._ops.OpOverload], Optional[Callable[[torch.fx.Node], bool]]]:
        """Return operators and a filter that should not be decomposed.

        Provide a base set of ops to preserve as-is and a predicate that keeps
        certain activations whole when surrounded by quantize/dequantize ops in
        a quantized graph. This helps downstream TOSA lowering and delegation.

        Args:
            ep (ExportedProgram): Program used to infer target-specific policy.

        Returns:
            Tuple[List[torch._ops.OpOverload], Optional[Callable[[torch.fx.Node], bool]]]:
                A list of op overloads to keep intact, and an optional filter
                function that returns True when an op should not be decomposed.

        """
        ops_to_not_decompose_if_quant_op = [
            torch.ops.aten.hardsigmoid.default,
            torch.ops.aten.hardswish.default,
        ]

        def filter_fn(node: torch.fx.Node) -> bool:
            """Return True to keep selected ops intact inside quantized regions.

            The predicate holds when the target is in
            ``ops_to_not_decompose_if_quant_op`` and all inputs/outputs are
            quantize/dequantize ops, indicating a quantized activation that
            should not be decomposed.

            Args:
                node (torch.fx.Node): FX node to evaluate.

            Returns:
                bool: True to keep the op intact; otherwise, False.

            """
            dq = torch.ops.quantized_decomposed.dequantize_per_tensor.default
            q = torch.ops.quantized_decomposed.quantize_per_tensor.default

            if node.target in ops_to_not_decompose_if_quant_op:
                # Assume we should not decompose the operator (it is quantized)
                should_not_decompose = True

                input_nodes = node.all_input_nodes
                ouput_nodes = node.users

                for inp in input_nodes:
                    if inp.target != dq:
                        should_not_decompose = False

                for out in ouput_nodes:
                    if out.target != q:
                        should_not_decompose = False

                return should_not_decompose

            # By default, do not decompose the operator
            return True

        ops_to_not_decompose = [
            torch.ops.aten.linear.default,
            torch.ops.aten.eye.default,
            torch.ops.aten.linspace.default,
            torch.ops.aten.logit.default,
        ] + ops_to_not_decompose_if_quant_op

        if not self.tosa_spec.is_U55_subset:
            # Tosa operator "RESIZE" is not supported on U55. Since upsample_bilinear2d
            # and upsample_nearest2d decompose into that it will not be possible to
            # delegate those operators on U55. If we have said here to not decompose
            # them there will be an error saying the operator was not decomposed. It
            # will not be possible for it to end up on either CPU or NPU.
            ops_to_not_decompose.append(torch.ops.aten.upsample_nearest2d.vec)
            ops_to_not_decompose.append(torch.ops.aten.upsample_bilinear2d.vec)

        return (ops_to_not_decompose, filter_fn)
