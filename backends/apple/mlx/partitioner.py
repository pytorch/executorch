#
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
#

"""
MLX Partitioner - decides which ops should run on the MLX delegate.

This module provides a Partitioner implementation that analyzes an EdgeIR
graph and marks supported operations for delegation to MLX.
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Dict, List, Tuple, Union

import torch
from executorch.backends.apple.mlx.preprocess import MLXBackend
from executorch.exir.backend.backend_details import CompileSpec
from executorch.exir.backend.canonical_partitioners.pattern_op_partitioner import (
    generate_partitions_from_list_of_nodes,
)
from executorch.exir.backend.partitioner import (
    DelegationSpec,
    Partitioner,
    PartitionResult,
)
from executorch.exir.backend.utils import tag_constant_data, tag_mutated_buffer
from torch.export.exported_program import ExportedProgram
from torch.fx.passes.infra.partitioner import Partition
from torch.fx.passes.operator_support import OperatorSupportBase

FORMAT = "[%(levelname)s %(asctime)s %(filename)s:%(lineno)s] %(message)s"
logging.basicConfig(level=logging.INFO, format=FORMAT)


class MLXOperatorSupport(OperatorSupportBase):
    """
    Determines which operators are supported by the MLX delegate.

    Uses MLXProgramBuilder to determine support - this ensures the partitioner
    uses the exact same logic as the actual compilation. A node is supported
    if the builder can handle it (either via direct handler or pattern match).
    """

    def __init__(
        self,
        edge_program: torch.export.ExportedProgram,
        compile_specs: List[CompileSpec],
    ):
        self.edge_program = edge_program
        self.compile_specs = compile_specs

        # Run the builder to determine which nodes are supported
        # The builder populates node_info with supported/unsupported status
        from executorch.backends.apple.mlx.program_builder import MLXProgramBuilder

        self._builder = MLXProgramBuilder(edge_program)
        try:
            # WARNING: build() calls _build_mlx_graph() which evaluates SymInts to
            # concrete values (via int(shape_dim)), corrupting the shape_env. This
            # is safe here because this class is only used during partitioning,
            # AFTER run_decompositions() has already been called. The shape_env
            # corruption only matters if run_decompositions() is called afterward.
            # For pre-decomposition support checking (e.g., ops_to_not_decompose()),
            # use check_support_only() instead.
            # See: backends/apple/mlx/docs/issues/dynamic_shapes_lost_during_delegate_lowering.md
            self._builder.build()
        except ValueError:
            # Build may fail if some nodes are unsupported, but node_info
            # will still be populated with support status for each node
            pass

    def is_node_supported(self, submodules, node: torch.fx.Node) -> bool:
        if node.op != "call_function":
            return False

        # Check if builder determined this node is supported
        info = self._builder.node_info.get(node)
        if info is not None and info.supported:
            logging.debug(f"[SUPPORTED] Node {node.target}")
            return True

        logging.debug(f"[UNSUPPORTED] Node {node.target}")
        return False


class MLXPartitioner(Partitioner):
    """
    Partitioner for the MLX delegate.

    Analyzes an EdgeIR graph and partitions supported operations
    for delegation to MLX.
    """

    def __init__(self, compile_specs: List[CompileSpec] | None = None) -> None:
        self.compile_specs = compile_specs or []
        self.delegation_spec = DelegationSpec(MLXBackend.__name__, self.compile_specs)
        self.partition_tags: Dict[str, DelegationSpec] = {}

    def ops_to_not_decompose(
        self, ep: ExportedProgram
    ) -> tuple[list[torch._ops.OpOverload], Callable[[torch.fx.Node], bool] | None]:
        """
        Return ops that should NOT be decomposed during edge lowering.

        This runs the MLXProgramBuilder to trace through the graph and determine
        which nodes are supported (either via direct handlers or patterns).
        Only ops for nodes that are actually supported should be preserved.

        This is called by to_edge_transform_and_lower to determine which
        ops to preserve before partitioning.

        NOTE: We use check_support_only() instead of build() to avoid corrupting
        the shape_env. build() calls _build_mlx_graph() which evaluates SymInts
        to concrete values when converting tensor shapes, which corrupts the
        shape_env and causes dynamic shapes to be lost during decomposition.
        """
        from executorch.backends.apple.mlx.program_builder import MLXProgramBuilder

        # Check if the graph already contains lowered modules (post-partitioning pass)
        # In this case, we should return empty since partitioning is already done
        for node in ep.graph.nodes:
            if node.op == "get_attr" and "lowered_module" in node.name:
                logging.debug(
                    "MLX ops_to_not_decompose: Graph already partitioned, returning empty"
                )
                return ([], None)

        # Run the builder to determine which nodes are supported
        # Use check_support_only() instead of build() to avoid corrupting shape_env
        # See: backends/apple/mlx/docs/issues/dynamic_shapes_lost_during_delegate_lowering.md
        builder = MLXProgramBuilder(ep)
        builder.check_support_only()

        # Collect ops for nodes that are actually supported
        do_not_decompose: list[torch._ops.OpOverload] = []

        for node in ep.graph.nodes:
            if node.op == "call_function" and isinstance(
                node.target, torch._ops.OpOverload
            ):
                info = builder.node_info.get(node)
                if info is not None and info.supported:
                    if node.target not in do_not_decompose:
                        do_not_decompose.append(node.target)

        logging.info(
            f"MLX ops_to_not_decompose: {[str(op) for op in do_not_decompose]}"
        )
        return (do_not_decompose, None)

    def generate_partitions(self, edge_program: ExportedProgram) -> List[Any]:
        """Generate partitions of supported nodes."""
        self.supported_ops = MLXOperatorSupport(
            edge_program=edge_program,
            compile_specs=self.delegation_spec.compile_specs,
        )

        # Collect unsupported ops, aggregated by target
        unsupported_by_target: Dict[str, Tuple[int, str]] = (
            {}
        )  # target -> (count, reason)
        for node in edge_program.graph.nodes:
            is_supported = self.supported_ops.is_node_supported({}, node)
            if not is_supported and node.op == "call_function":
                target_str = str(node.target)
                info = self.supported_ops._builder.node_info.get(node)
                reason = info.unsupported_reason if info else "No handler registered"
                if target_str in unsupported_by_target:
                    count, _ = unsupported_by_target[target_str]
                    unsupported_by_target[target_str] = (count + 1, reason)
                else:
                    unsupported_by_target[target_str] = (1, reason)

        logging.info("=" * 80)
        logging.info("MLX Partitioner: UNSUPPORTED OPS SUMMARY")
        logging.info("=" * 80)
        if unsupported_by_target:
            for target, (count, reason) in unsupported_by_target.items():
                logging.info(f"  [UNSUPPORTED x{count}] {target}")
                logging.info(f"      Reason: {reason}")
        else:
            logging.info("  (All call_function nodes are supported!)")
        logging.info("=" * 80)

        partitions = generate_partitions_from_list_of_nodes(
            edge_program.graph_module,
            op_support=self.supported_ops,
        )

        # WORKAROUND for dynamic shapes bug: Include sym_size nodes in partitions
        # when any of their users are in the partition. This prevents symbolic
        # shapes from being concretized during delegate lowering.
        # See: backends/apple/mlx/docs/issues/dynamic_shapes_lost_during_delegate_lowering.md
        partitions = self._include_sym_size_nodes_in_partitions(
            edge_program.graph_module, partitions
        )

        return partitions

    def _include_sym_size_nodes_in_partitions(
        self, gm: torch.fx.GraphModule, partitions: List[Partition]
    ) -> List[Partition]:
        """
        Include sym_size nodes in partitions when any of their users are in the partition.

        This is a workaround for the dynamic shapes bug where symbolic shapes are lost
        during delegate lowering if the sym_size node is not included in the partition.
        """
        from executorch.exir.dialects.edge._ops import EdgeOpOverload

        for partition in partitions:
            partition_nodes = set(partition.nodes)
            nodes_to_add = []

            for node in gm.graph.nodes:
                if node.op != "call_function":
                    continue

                # Check if this is a sym_size node
                target = node.target
                if isinstance(target, EdgeOpOverload):
                    target = target._op

                if target != torch.ops.aten.sym_size.int:
                    continue

                # Check if any user of this sym_size node is in the partition
                for user in node.users:
                    if user in partition_nodes:
                        # Add sym_size to partition if not already there
                        if node not in partition_nodes:
                            nodes_to_add.append(node)
                            logging.debug(
                                f"Adding sym_size node {node.name} to partition "
                                f"(used by {user.name})"
                            )
                        break

            # Add the sym_size nodes to the partition
            for node in nodes_to_add:
                partition.add_node(node)

        return partitions

    def tag_nodes(self, partitions: List[Partition]) -> None:
        """Tag nodes in each partition for delegation."""
        for partition in partitions:
            delegation_tag = f"mlx_{partition.id}"
            for node in partition.nodes:
                node.meta["delegation_tag"] = delegation_tag
                self.partition_tags[delegation_tag] = self.delegation_spec

    @staticmethod
    def check_partitions(partitions: Union[dict, list]) -> bool:
        """Check if any partitions were found."""
        pl = len(partitions)
        if pl == 0:
            logging.warning("MLX: Nothing can be partitioned!")
        else:
            logging.info(f"MLX: Found {pl} subgraphs to be partitioned.")
        return pl != 0

    def partition(self, edge_program: ExportedProgram) -> PartitionResult:
        """
        Partition the edge program for MLX delegation.

        Args:
            edge_program: The ExportedProgram to partition.

        Returns:
            PartitionResult with tagged nodes and partition specs.
        """
        partitions = self.generate_partitions(edge_program=edge_program)
        if self.check_partitions(partitions):
            self.tag_nodes(partitions)
            # Tag constant data that are used by the supported ops
            tag_constant_data(edge_program)
            # Tag mutated buffers so they are included in the partition
            # This ensures the partitioned subgraph has proper mutation tracking
            tag_mutated_buffer(edge_program)

        return PartitionResult(
            tagged_exported_program=edge_program,
            partition_tags=self.partition_tags,
        )


# =============================================================================
# Supported ops list (for reference/documentation)
# =============================================================================

# The following ops are supported by the MLX delegate:
#
# Basic tensor ops:
#   - aten.view, aten.reshape
#   - aten.permute, aten.transpose
#   - aten.slice
#   - aten.unsqueeze, aten.squeeze
#   - aten.clone, aten.alias
#   - aten.repeat (tile)
#   - aten.index (take_along_axis)
#
# Math ops:
#   - aten.add (tensor and scalar)
#   - aten.mul (tensor and scalar)
#   - aten.linear
#   - aten.embedding
#
# Activation functions:
#   - aten.silu
#   - aten.gelu
#
# Normalization:
#   - aten.layer_norm
#   - aten.rms_norm
#
# Attention:
#   - aten.scaled_dot_product_attention (via SDPA pattern)
#   - mlx.rope (custom op)
#
# Quantized ops (via patterns):
#   - Quantized linear (torchao.dequantize_affine + aten.linear)
#   - Quantized embedding (torchao.dequantize_affine + aten.embedding)
#
# Other:
#   - aten.arange
#   - aten.sym_size
#   - aten.item (for SymInt extraction)
#   - operator.getitem
#   - operator.add (scalar)
#
# Patterns (fused ops):
#   - SDPA: scaled_dot_product_attention with optional GQA repeat_interleave
#   - QUANTIZED_LINEAR: dequantize_affine + linear
#   - QUANTIZED_EMBEDDING: dequantize_affine + embedding
#   - SLICE_UPDATE: slice + copy + slice_scatter (for KV cache updates)
