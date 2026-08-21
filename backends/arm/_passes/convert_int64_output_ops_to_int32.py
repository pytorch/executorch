# Copyright 2025-2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


import logging
from typing import Any, cast, Dict, Literal, Optional, Set, Tuple, Type

import torch
from executorch.backends.arm._passes import ArmPass
from executorch.backends.arm._passes.arm_pass_utils import (
    create_node,
    get_first_fake_tensor,
    set_node_arg,
)
from executorch.exir.dialects._ops import ops as exir_ops
from executorch.exir.pass_base import ExportPass, PassResult
from torch._subclasses.fake_tensor import FakeTensor


logger = logging.getLogger(__name__)


class ConvertInt64OutputOpsToInt32Pass(ArmPass):
    """Rewrites or removes operations that produce int64 outputs, converting
    them to int32 where possible.

    Currently, this pass handles casting, argmax and argmin operators:
      1. int32 -> int64:
         removes the cast and redirects all uses to the original int32 value.
      2. other types -> int64:
         rewrites the cast to produce int32 instead of int64.
      3. torch.argmax() / torch.argmin()
         insert an int64->int32 cast only along downstream paths whose values
         are proven to remain within the int32 range. Other paths keep the
         original int64 value or receive an int32->int64 boundary cast.

    Argmax and argmin are currently the only bounded-index sources. Range
    propagation from those sources recognizes a separate allowlist of safe
    shape and arithmetic operations.

    Future extensions may include other operators that return int64 outputs by
    default, rewriting them or inserting an int64 -> int32 cast to yield int32
    results.

    Args:
        on_overflow: Action when an argmax/argmin index cannot safely fit in
            int32 (i.e. the reduced dimension has more than INT32_MAX elements).
            ``"raise"`` (default) raises a ``RuntimeError`` at compile time.
            ``"warn"`` logs a warning and skips the cast for that node.
            ``"skip"`` silently skips the cast for that node.

    """

    _passes_required_after: Set[Type[ExportPass]] = set()

    _INT32_MAX = torch.iinfo(torch.int32).max

    def __init__(
        self,
        *args,
        on_overflow: Literal["raise", "warn", "skip"] = "raise",
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        if on_overflow not in ("raise", "warn", "skip"):
            raise ValueError(
                f"on_overflow must be 'raise', 'warn', or 'skip', got {on_overflow!r}"
            )
        self.on_overflow = on_overflow

    def _index_range(self, node: torch.fx.Node) -> Tuple[int, int]:
        """Return the inclusive output range of an argmax/argmin node."""
        input_tensor = get_first_fake_tensor(cast(torch.fx.Node, node.args[0]))
        dim = node.args[1] if len(node.args) > 1 and node.args[1] is not None else None
        if dim is None:
            size = input_tensor.numel()
        else:
            size = input_tensor.shape[cast(int, dim)]
        return 0, int(size) - 1

    aten_cast_ops = (
        torch.ops.aten.to.dtype,
        torch.ops.aten.to.dtype_layout,
    )
    edge_cast_ops = (exir_ops.edge.dim_order_ops._to_dim_order_copy.default,)

    aten_argmax_ops = (torch.ops.aten.argmax.default,)
    edge_argmax_ops = (exir_ops.edge.aten.argmax.default,)

    aten_argmin_ops = (torch.ops.aten.argmin.default,)
    edge_argmin_ops = (exir_ops.edge.aten.argmin.default,)

    aten_bounded_index_ops = aten_argmax_ops + aten_argmin_ops
    edge_bounded_index_ops = edge_argmax_ops + edge_argmin_ops

    aten_index_relay_ops = (
        torch.ops.aten.unsqueeze.default,
        torch.ops.aten.expand.default,
        torch.ops.aten.view.default,
    )
    edge_index_relay_ops = (
        exir_ops.edge.aten.unsqueeze_copy.default,
        exir_ops.edge.aten.expand_copy.default,
        exir_ops.edge.aten.view_copy.default,
    )
    aten_index_add_ops = (torch.ops.aten.add.Tensor,)
    edge_index_add_ops = (exir_ops.edge.aten.add.Tensor,)
    aten_index_sub_ops = (torch.ops.aten.sub.Tensor,)
    edge_index_sub_ops = (exir_ops.edge.aten.sub.Tensor,)
    aten_index_mul_ops = (torch.ops.aten.mul.Tensor,)
    edge_index_mul_ops = (exir_ops.edge.aten.mul.Tensor,)

    aten_index_binary_ops = aten_index_add_ops + aten_index_sub_ops + aten_index_mul_ops
    edge_index_binary_ops = edge_index_add_ops + edge_index_sub_ops + edge_index_mul_ops

    aten_ops = aten_cast_ops + aten_bounded_index_ops
    edge_ops = edge_cast_ops + edge_bounded_index_ops

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
                logger.warning(
                    f"Removing int32->int64 casting node {node.name} defined in"
                    f" {node.meta.get('stack_trace','[no stack trace found]')}"
                )
                user.replace_input_with(node, input_node)
        # Case 2: other types -> int64 - rewrites to cast to int32
        else:
            if node.target in self.cast_ops_kwargs:
                set_node_arg(node, "dtype", torch.int32)
            elif node.target in self.cast_ops_args:
                set_node_arg(node, 1, torch.int32)
            else:
                raise RuntimeError(f"Unexpected target {node.target} in {node.name}")
            output_dtype = get_first_fake_tensor(node).dtype
            logger.warning(
                f"Converting casting node {node.name} from {input_dtype}->{output_dtype} to"
                f" {input_dtype}->torch.int32 defined in {node.meta.get('stack_trace','[no stack trace found]')}"
            )

    def _range_fits_int32(self, value_range: Tuple[int, int]) -> bool:
        return -self._INT32_MAX - 1 <= value_range[0] and (
            value_range[1] <= self._INT32_MAX
        )

    def _index_size_fits_int32_policy(self, index_range: Tuple[int, int]) -> bool:
        """Apply the established source-dimension overflow policy.

        Args:
            index_range (tuple): Inclusive index range.

        Returns:
            bool: True when the source dimension is accepted.

        """
        return index_range[1] < self._INT32_MAX

    @staticmethod
    def _scalar_int(value: Any) -> Optional[int]:
        if isinstance(value, int):
            return value
        if (
            isinstance(value, torch.Tensor)
            and not isinstance(value, FakeTensor)
            and value.numel() == 1
            and not value.dtype.is_floating_point
            and not value.dtype.is_complex
        ):
            return int(value.item())
        return None

    def _constant_range(
        self, value: Any, graph_module: torch.fx.GraphModule
    ) -> Optional[Tuple[int, int]]:
        scalar = self._scalar_int(value)
        if scalar is not None:
            return scalar, scalar
        if not isinstance(value, torch.fx.Node):
            return None

        constant = None
        if value.op == "get_attr" and isinstance(value.target, str):
            constant = getattr(graph_module, value.target, None)
        elif value.op == "placeholder" and isinstance(value.target, str):
            buffer_name = value.target.removeprefix("_lifted")
            if buffer_name != value.target:
                try:
                    constant = graph_module.get_buffer(buffer_name)
                except AttributeError:
                    pass

        scalar = self._scalar_int(constant)
        return None if scalar is None else (scalar, scalar)

    def _operand_range(
        self,
        value: Any,
        ranges: Dict[torch.fx.Node, Tuple[int, int]],
        graph_module: torch.fx.GraphModule,
    ) -> Optional[Tuple[int, int]]:
        if isinstance(value, torch.fx.Node) and value in ranges:
            return ranges[value]
        return self._constant_range(value, graph_module)

    @staticmethod
    def _scale_range(value_range: Tuple[int, int], scale: int) -> Tuple[int, int]:
        values = value_range[0] * scale, value_range[1] * scale
        return min(values), max(values)

    def _infer_binary_range(
        self,
        node: torch.fx.Node,
        ranges: Dict[torch.fx.Node, Tuple[int, int]],
        graph_module: torch.fx.GraphModule,
    ) -> Optional[Tuple[int, int]]:
        if get_first_fake_tensor(node).dtype != torch.int64 or len(node.args) < 2:
            return None
        lhs = self._operand_range(node.args[0], ranges, graph_module)
        rhs = self._operand_range(node.args[1], ranges, graph_module)
        if lhs is None or rhs is None:
            return None
        if not self._range_fits_int32(lhs) or not self._range_fits_int32(rhs):
            return None

        if node.target in self.aten_index_mul_ops + self.edge_index_mul_ops:
            products = (
                lhs[0] * rhs[0],
                lhs[0] * rhs[1],
                lhs[1] * rhs[0],
                lhs[1] * rhs[1],
            )
            result = min(products), max(products)
        else:
            alpha = self._scalar_int(node.kwargs.get("alpha", 1))
            if alpha is None:
                return None
            rhs = self._scale_range(rhs, alpha)
            if node.target in self.aten_index_add_ops + self.edge_index_add_ops:
                result = lhs[0] + rhs[0], lhs[1] + rhs[1]
            elif node.target in self.aten_index_sub_ops + self.edge_index_sub_ops:
                result = lhs[0] - rhs[1], lhs[1] - rhs[0]
            else:
                return None
        return result if self._range_fits_int32(result) else None

    def _infer_safe_int32_range(
        self,
        node: torch.fx.Node,
        ranges: Dict[torch.fx.Node, Tuple[int, int]],
        graph_module: torch.fx.GraphModule,
    ) -> Optional[Tuple[int, int]]:
        if node.target in self.aten_index_relay_ops + self.edge_index_relay_ops:
            if get_first_fake_tensor(node).dtype != torch.int64:
                return None
            return ranges.get(cast(torch.fx.Node, node.args[0]))
        return self._infer_binary_range(node, ranges, graph_module)

    def _is_safe_widening_consumer(
        self,
        node: torch.fx.Node,
        ranges: Dict[torch.fx.Node, Tuple[int, int]],
    ) -> bool:
        """Return whether a node safely ends integer range propagation.

        Args:
            node (torch.fx.Node): Candidate downstream consumer.
            ranges (dict): Proven ranges.

        Returns:
            bool: True when the node safely produces a non-int64 value.

        """
        output = node.meta.get("val")
        if not isinstance(output, torch.Tensor) or output.dtype == torch.int64:
            return False
        int64_inputs = [
            input_node
            for input_node in node.all_input_nodes
            if isinstance(input_node.meta.get("val"), torch.Tensor)
            and input_node.meta["val"].dtype == torch.int64
        ]
        if not int64_inputs or not all(node in ranges for node in int64_inputs):
            return False
        return node.target in (
            self.aten_cast_ops
            + self.edge_cast_ops
            + self.aten_index_add_ops
            + self.edge_index_add_ops
            + self.aten_index_sub_ops
            + self.edge_index_sub_ops
            + self.aten_index_mul_ops
            + self.edge_index_mul_ops
        )

    def _find_safe_index_consumers(
        self,
        graph_module: torch.fx.GraphModule,
        source: torch.fx.Node,
        source_range: Tuple[int, int],
    ) -> Tuple[Dict[torch.fx.Node, Tuple[int, int]], Set[torch.fx.Node]]:
        """Collect consumers proven safe for the int32 index path.

        Args:
            graph_module (torch.fx.GraphModule): Graph containing the source.
            source (torch.fx.Node): Bounded int64 index source.
            source_range (tuple): Inclusive source range.

        Returns:
            tuple: Proven ranges and safe consumers.

        """
        ranges = {source: source_range}
        safe_consumers: Set[torch.fx.Node] = set()
        for node in graph_module.graph.nodes:
            if node.op != "call_function":
                continue
            inferred_range = self._infer_safe_int32_range(node, ranges, graph_module)
            if inferred_range is not None:
                ranges[node] = inferred_range
                safe_consumers.add(node)
            elif self._is_safe_widening_consumer(node, ranges):
                safe_consumers.add(node)
        return ranges, safe_consumers

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
        safe_consumers: Set[torch.fx.Node],
        ranges: Dict[torch.fx.Node, Tuple[int, int]],
        to_copy_op,
    ) -> None:
        """Cast int64 scalar constants used by safe binary consumers.

        Args:
            graph_module (torch.fx.GraphModule): Graph being transformed.
            safe_consumers (set): Consumers on int32 paths.
            ranges (dict): Proven ranges.
            to_copy_op (Any): Dialect-specific operator used for casts.

        """
        graph = graph_module.graph
        constant_casts: Dict[torch.fx.Node, torch.fx.Node] = {}
        for consumer in safe_consumers:
            if consumer.target not in (
                self.aten_index_binary_ops + self.edge_index_binary_ops
            ):
                continue
            for input_node in consumer.all_input_nodes:
                if input_node in ranges:
                    continue
                input_value = input_node.meta.get("val")
                if (
                    not isinstance(input_value, torch.Tensor)
                    or input_value.dtype != torch.int64
                    or self._constant_range(input_node, graph_module) is None
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
        source: torch.fx.Node,
        source_range: Tuple[int, int],
        to_copy_op,
    ) -> bool:
        """Convert proven-safe paths from a bounded index source to int32.

        The caller identifies the bounded source and supplies its inclusive
        value range. Direct consumers that cannot be proven safe retain the
        original int64 source. An int64 boundary cast is inserted when an
        unproven consumer follows an intermediate converted to int32.

        Args:
            graph_module (torch.fx.GraphModule): Graph containing the source.
            source (torch.fx.Node): Int64 node with a statically known range.
            source_range (Tuple[int, int]): Inclusive minimum and maximum.
            to_copy_op (Any): Dialect-specific operator used for casts.

        Returns:
            bool: True when at least one path is converted to int32.

        """
        ranges, safe_consumers = self._find_safe_index_consumers(
            graph_module, source, source_range
        )
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
            graph_module, safe_consumers, ranges, to_copy_op
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

        logger.warning(
            f"Inserting a casting node {cast_to_int32.name} after "
            f"{source.name} for range-safe index consumers defined in "
            f"{source.meta.get('stack_trace','[no stack trace found]')}"
        )
        return True

    def call(self, graph_module: torch.fx.GraphModule):
        modified = False
        graph = graph_module.graph
        for node in list(graph.nodes):
            if node.op != "call_function":
                continue
            if node.target not in self.aten_ops + self.edge_ops:
                continue
            output_dtype = get_first_fake_tensor(node).dtype
            if output_dtype != torch.int64:
                continue

            if node.target in self.aten_cast_ops + self.edge_cast_ops:
                self._convert_casting_operators(node)
            elif node.target in (
                self.aten_bounded_index_ops + self.edge_bounded_index_ops
            ):
                index_range = self._index_range(node)
                if not self._index_size_fits_int32_policy(index_range):
                    msg = (
                        f"{node.target} reduces over more than {self._INT32_MAX} elements; "
                        f"the int64 index cannot be safely cast to int32."
                    )
                    if self.on_overflow == "raise":
                        raise RuntimeError(msg)
                    if self.on_overflow == "warn":
                        logger.warning(msg)
                    continue
                if not self._cast_safe_index_paths_to_int32(
                    graph_module,
                    node,
                    index_range,
                    self._get_decomposition(node.target),
                ):
                    continue
            else:
                raise RuntimeError(f"Unexpected target {node.target} in {node.name}")

            modified = True

        if modified:
            graph_module.graph.eliminate_dead_code()
            graph_module.recompile()
            graph_module = super().call(graph_module).graph_module

        return PassResult(graph_module, modified)
