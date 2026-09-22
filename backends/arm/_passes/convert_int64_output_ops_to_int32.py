# Copyright 2025-2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


import logging
from typing import Any, Dict, Literal, Set, Tuple, Type

import torch

from executorch.backends.arm._passes import ArmPass
from executorch.backends.arm._passes.arm_pass_utils import (
    create_node,
    get_first_fake_tensor,
    set_node_arg,
)
from executorch.backends.arm._passes.int32_range_analysis import (
    Int32RangeAnalysis,
    ValueRange,
)
from executorch.exir.dialects._ops import ops as exir_ops
from executorch.exir.pass_base import ExportPass, PassResult

logger = logging.getLogger(__name__)


class ConvertInt64OutputOpsToInt32Pass(ArmPass):
    """Rewrite or remove operations that produce int64 outputs.

    Convert them to int32 where possible.

    Currently, this pass handles casting, argmax, argmin and topk operators:
      1. int32 -> int64:
         removes the cast and redirects all uses to the original int32 value.
      2. other types -> int64:
         rewrites the cast to produce int32 instead of int64.
      3. torch.argmax() / torch.argmin()
         insert an int64->int32 cast only along downstream paths whose values
         are proven to remain within the int32 range. Other paths keep the
         original int64 value or receive an int32->int64 boundary cast.
      4. torch.topk()
         apply the same bounded-index handling to ``getitem(topk, 1)`` while
         leaving the values output, ``getitem(topk, 0)``, unchanged.

    Argmax, argmin and extracted TopK indices are the bounded-index sources.
    Range propagation from those sources recognizes a separate allowlist of
    safe shape and arithmetic operations.

    Future extensions may include other operators that return int64 outputs by
    default, rewriting them or inserting an int64 -> int32 cast to yield int32
    results.

    Args:
        convert_cast_ops (bool): Whether to convert general int64 cast
            operators. Defaults to True.
        on_overflow (Literal["raise", "warn", "skip"]): Action when an
            argmax/argmin/topk index cannot safely fit in int32 (i.e. the
            reduced dimension has more than INT32_MAX elements).
            ``"raise"`` (default) raises a ``RuntimeError`` at compile time.
            ``"warn"`` logs a warning and skips the cast for that node.
            ``"skip"`` silently skips the cast for that node.

    """

    _passes_required_after: Set[Type[ExportPass]] = set()

    _INT32_MAX = torch.iinfo(torch.int32).max

    def __init__(
        self,
        *args,
        convert_cast_ops: bool = True,
        on_overflow: Literal["raise", "warn", "skip"] = "raise",
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        if on_overflow not in ("raise", "warn", "skip"):
            raise ValueError(
                f"on_overflow must be 'raise', 'warn', or 'skip', got {on_overflow!r}"
            )
        self.convert_cast_ops = convert_cast_ops
        self.on_overflow = on_overflow

    aten_cast_ops = Int32RangeAnalysis.aten_cast_ops
    edge_cast_ops = Int32RangeAnalysis.edge_cast_ops
    aten_bounded_index_ops = Int32RangeAnalysis.aten_bounded_index_ops
    edge_bounded_index_ops = Int32RangeAnalysis.edge_bounded_index_ops
    aten_topk_ops = Int32RangeAnalysis.aten_topk_ops
    edge_topk_ops = Int32RangeAnalysis.edge_topk_ops

    aten_ops = aten_cast_ops + aten_bounded_index_ops + aten_topk_ops
    edge_ops = edge_cast_ops + edge_bounded_index_ops + edge_topk_ops

    # dtype is specified in args
    cast_ops_args = (
        torch.ops.aten.to.dtype,  # to_2: node.args: (gt, torch.int64) node.kwargs: {}
    )
    # dtype is specified in kwargs
    cast_ops_kwargs = (
        torch.ops.aten.to.dtype_layout,  # to_1: node.args: (unsqueeze,) node.kwargs: {'dtype': torch.int64, 'layout': torch.strided, 'device': device(type='cpu')}
        exir_ops.edge.dim_order_ops._to_dim_order_copy.default,  # node.args: (aten_gt_scalar,) node.kwargs: {'dtype': torch.int64, 'dim_order': [0, 1]}
    )

    def _get_decomposition(self, op):
        if op in self.edge_ops:
            return exir_ops.edge.dim_order_ops._to_dim_order_copy.default

        if op in self.aten_ops:
            return torch.ops.dim_order_ops._to_dim_order_copy.default

        raise RuntimeError(
            f"[{self.__class__.__name__}] Can't get decomposition for op {op}"
        )

    def _convert_casting_operators(self, node: torch.fx.Node):
        input_node = node.all_input_nodes[0]
        input_dtype = get_first_fake_tensor(input_node).dtype
        # Case 1: int32 -> int64 - removes the ops
        if input_dtype == torch.int32:
            users = [user for user in node.users if node != user]
            for user in users:
                user.replace_input_with(node, input_node)
            self._removed_casts += 1
        # Case 2: other types -> int64 - rewrites to cast to int32
        else:
            if node.target in self.cast_ops_kwargs:
                set_node_arg(node, "dtype", torch.int32)
            elif node.target in self.cast_ops_args:
                set_node_arg(node, 1, torch.int32)
            else:
                raise RuntimeError(f"Unexpected target {node.target} in {node.name}")
            self._converted_casts += 1

    @staticmethod
    def _insert_int64_boundary(
        graph: torch.fx.Graph,
        node: torch.fx.Node,
        to_copy_op,
        boundaries: Dict[torch.fx.Node, torch.fx.Node],
    ) -> torch.fx.Node:
        if node not in boundaries:
            with graph.inserting_after(node):
                boundaries[node] = create_node(
                    graph,
                    to_copy_op,
                    args=(node,),
                    kwargs={"dtype": torch.int64},
                )
        return boundaries[node]

    def _cast_safe_scalar_constants_to_int32(
        self,
        graph_module: torch.fx.GraphModule,
        analysis: Int32RangeAnalysis,
        safe_consumers: Set[torch.fx.Node],
        ranges: Dict[torch.fx.Node, ValueRange],
        to_copy_op,
    ) -> None:
        """Cast int64 scalar constants used by safe binary consumers.

        Args:
            graph_module (torch.fx.GraphModule): Graph being transformed.
            analysis (Int32RangeAnalysis): Range analysis for the graph.
            safe_consumers (set): Consumers on int32 paths.
            ranges (dict): Proven ranges.
            to_copy_op (Any): Dialect-specific operator used for casts.

        """
        graph = graph_module.graph
        constant_casts: Dict[torch.fx.Node, torch.fx.Node] = {}
        for consumer in safe_consumers:
            for input_node in consumer.all_input_nodes:
                if not analysis.is_safe_scalar_constant_input(
                    consumer, input_node, ranges
                ):
                    continue
                if input_node not in constant_casts:
                    with graph.inserting_after(input_node):
                        constant_casts[input_node] = create_node(
                            graph,
                            to_copy_op,
                            args=(input_node,),
                            kwargs={"dtype": torch.int32},
                        )
                consumer.replace_input_with(input_node, constant_casts[input_node])

    def _cast_safe_index_paths_to_int32(
        self,
        graph_module: torch.fx.GraphModule,
        analysis: Int32RangeAnalysis,
        source: torch.fx.Node,
        source_range: ValueRange,
        to_copy_op,
    ) -> bool:
        """Convert proven-safe paths from a bounded index source to int32.

        The caller identifies the bounded source and supplies its inclusive
        value range. Direct consumers that cannot be proven safe retain the
        original int64 source. An int64 boundary cast is inserted when an
        unproven consumer follows an intermediate converted to int32.

        Args:
            graph_module (torch.fx.GraphModule): Graph containing the source.
            analysis (Int32RangeAnalysis): Range analysis for the graph.
            source (torch.fx.Node): Int64 node with a statically known range.
            source_range (ValueRange): Inclusive minimum and maximum.
            to_copy_op (Any): Dialect-specific operator used for casts.

        Returns:
            bool: True when at least one path is converted to int32.

        """
        result = analysis.find_safe_index_consumers(source, source_range)
        ranges = result.ranges
        safe_consumers = result.safe_consumers
        if not safe_consumers:
            return False

        graph = graph_module.graph
        original_users = {node: list(node.users) for node in ranges}
        with graph.inserting_after(source):
            cast_to_int32 = create_node(
                graph,
                to_copy_op,
                args=(source,),
                kwargs={"dtype": torch.int32},
            )

        self._cast_safe_scalar_constants_to_int32(
            graph_module, analysis, safe_consumers, ranges, to_copy_op
        )

        boundaries: Dict[torch.fx.Node, torch.fx.Node] = {}
        for node, users in original_users.items():
            for user in users:
                if user in safe_consumers:
                    if node is source:
                        user.replace_input_with(source, cast_to_int32)
                elif node is not source:
                    boundary = self._insert_int64_boundary(
                        graph, node, to_copy_op, boundaries
                    )
                    user.replace_input_with(node, boundary)

        self._safe_index_casts += 1
        return True

    def _log_summary(self) -> None:
        if self._removed_casts or self._converted_casts or self._safe_index_casts:
            logger.warning(
                "ConvertInt64OutputOpsToInt32Pass: removed %d int32-to-int64 "
                "cast(s), rewrote %d cast(s) to int32, and inserted %d "
                "range-safe index cast(s).",
                self._removed_casts,
                self._converted_casts,
                self._safe_index_casts,
            )

    def _convert_topk_indices(
        self,
        graph_module: torch.fx.GraphModule,
        analysis: Int32RangeAnalysis,
        topk: torch.fx.Node,
    ) -> bool:
        """Convert safe paths from extracted TopK indices.

        Args:
            graph_module (torch.fx.GraphModule): Graph containing TopK.
            analysis (Int32RangeAnalysis): Range analysis for the graph.
            topk (torch.fx.Node): TopK tuple-producing node.

        Returns:
            bool: True when at least one index path is converted.

        """
        index_getitems = analysis.topk_index_getitems(topk)
        if not index_getitems:
            return False

        index_range = analysis.topk_index_range(topk)
        if index_range is None:
            return False
        if not analysis.index_size_fits_int32_policy(index_range):
            msg = (
                f"{topk.target} indexes a dimension with more than "
                f"{self._INT32_MAX} elements; the int64 index cannot be "
                "safely cast to int32."
            )
            if self.on_overflow == "raise":
                raise RuntimeError(msg)
            if self.on_overflow == "warn":
                logger.warning(msg)
            return False

        modified = False
        to_copy_op = self._get_decomposition(topk.target)
        for indices in index_getitems:
            if get_first_fake_tensor(indices).dtype == torch.int64:
                modified |= self._cast_safe_index_paths_to_int32(
                    graph_module,
                    analysis,
                    indices,
                    index_range,
                    to_copy_op,
                )
        return modified

    def _convert_bounded_index(
        self,
        graph_module: torch.fx.GraphModule,
        analysis: Int32RangeAnalysis,
        node: torch.fx.Node,
    ) -> bool:
        """Convert safe paths from an argmax or argmin index.

        Args:
            graph_module (torch.fx.GraphModule): Graph containing the node.
            analysis (Int32RangeAnalysis): Range analysis for the graph.
            node (torch.fx.Node): Bounded index-producing node.

        Returns:
            bool: True when at least one index path is converted.

        """
        index_range = analysis.arg_index_range(node)
        if index_range is None:
            return False
        if not analysis.index_size_fits_int32_policy(index_range):
            msg = (
                f"{node.target} reduces over more than {self._INT32_MAX} "
                "elements; the int64 index cannot be safely cast to int32."
            )
            if self.on_overflow == "raise":
                raise RuntimeError(msg)
            if self.on_overflow == "warn":
                logger.warning(msg)
            return False
        return self._cast_safe_index_paths_to_int32(
            graph_module,
            analysis,
            node,
            index_range,
            self._get_decomposition(node.target),
        )

    def _target_ops(self) -> Tuple[Any, ...]:
        ops: Tuple[Any, ...] = self.aten_bounded_index_ops + self.aten_topk_ops
        ops += self.edge_bounded_index_ops + self.edge_topk_ops
        if self.convert_cast_ops:
            ops += self.aten_cast_ops + self.edge_cast_ops
        return ops

    def should_run(self, graph_module: torch.fx.GraphModule) -> bool:
        target_ops = self._target_ops()
        return any(
            node.op == "call_function" and node.target in target_ops and node.users
            for node in graph_module.graph.nodes
        )

    def call(self, graph_module: torch.fx.GraphModule):
        modified = False
        self._removed_casts = 0
        self._converted_casts = 0
        self._safe_index_casts = 0
        graph = graph_module.graph
        analysis = Int32RangeAnalysis(graph_module)
        target_ops = self._target_ops()
        for node in list(graph.nodes):
            if node.op != "call_function":
                continue
            if node.target not in target_ops or not node.users:
                continue
            if node.target in self.aten_topk_ops + self.edge_topk_ops:
                modified |= self._convert_topk_indices(graph_module, analysis, node)
                continue

            output_dtype = get_first_fake_tensor(node).dtype
            if output_dtype != torch.int64:
                continue

            if node.target in self.aten_cast_ops + self.edge_cast_ops:
                self._convert_casting_operators(node)
                modified = True
            elif node.target in (
                self.aten_bounded_index_ops + self.edge_bounded_index_ops
            ):
                modified |= self._convert_bounded_index(graph_module, analysis, node)
            else:
                raise RuntimeError(f"Unexpected target {node.target} in {node.name}")

        if modified:
            graph_module.graph.eliminate_dead_code()
            graph_module.recompile()
            graph_module = super().call(graph_module).graph_module

        self._log_summary()

        return PassResult(graph_module, modified)
