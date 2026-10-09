# Copyright 2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import operator
from dataclasses import dataclass
from typing import Any, cast, Dict, List, Optional, Set, Tuple

import sympy  # type: ignore[import-untyped]
import torch

from executorch.backends.arm._passes.arm_pass_utils import get_first_fake_tensor
from executorch.exir.dialects._ops import ops as exir_ops
from torch._subclasses.fake_tensor import FakeTensor
from torch.utils._sympy.value_ranges import bound_sympy


ValueRange = Tuple[int, int]


@dataclass
class Int32RangeAnalysisResult:
    """Ranges and consumers proven safe to use int32."""

    ranges: Dict[torch.fx.Node, ValueRange]
    safe_consumers: Set[torch.fx.Node]


class Int32RangeAnalysis:
    """Analyze bounded int64 index paths that can safely use int32.

    Args:
        graph_module (torch.fx.GraphModule): Graph module to analyze.

    """

    _INT32_MAX = torch.iinfo(torch.int32).max

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

    aten_topk_ops = (torch.ops.aten.topk.default,)
    edge_topk_ops = (exir_ops.edge.aten.topk.default,)

    aten_index_consumer_ops = (torch.ops.aten.index.Tensor,)
    edge_index_consumer_ops = (exir_ops.edge.aten.index.Tensor,)
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
    aten_index_div_ops = (torch.ops.aten.div.Tensor_mode,)
    edge_index_div_ops = (exir_ops.edge.aten.div.Tensor_mode,)
    aten_index_remainder_ops = (torch.ops.aten.remainder.Tensor,)
    edge_index_remainder_ops = (exir_ops.edge.aten.remainder.Tensor,)
    aten_index_cat_ops = (torch.ops.aten.cat.default,)
    edge_index_cat_ops = (exir_ops.edge.aten.cat.default,)
    aten_index_full_like_ops = (torch.ops.aten.full_like.default,)
    edge_index_full_like_ops = (exir_ops.edge.aten.full_like.default,)
    aten_index_binary_ops = (
        aten_index_add_ops
        + aten_index_sub_ops
        + aten_index_mul_ops
        + aten_index_div_ops
        + aten_index_remainder_ops
    )
    edge_index_binary_ops = (
        edge_index_add_ops
        + edge_index_sub_ops
        + edge_index_mul_ops
        + edge_index_div_ops
        + edge_index_remainder_ops
    )

    def __init__(self, graph_module: torch.fx.GraphModule) -> None:
        self.graph_module = graph_module

    @staticmethod
    def _index_range_from_size(size: Any) -> Optional[ValueRange]:
        """Return an index range when the size has a finite upper bound.

        Args:
            size (Any): Static or symbolic dimension size.

        Returns:
            Optional[ValueRange]: Index range, or ``None`` when a finite upper
            bound cannot be proven.

        """
        if isinstance(size, int):
            return 0, size - 1
        if not isinstance(size, torch.SymInt):
            return None
        shape_env = size.node.shape_env
        if shape_env is None:
            return None

        try:
            value_range = bound_sympy(
                size.node.expr,
                shape_env.var_to_range,
            )
        except (AttributeError, KeyError, RecursionError, TypeError, ValueError):
            return None
        if not isinstance(value_range.upper, sympy.Integer):
            return None
        return 0, int(value_range.upper) - 1

    @classmethod
    def arg_index_range(cls, node: torch.fx.Node) -> Optional[ValueRange]:
        """Return the proven inclusive range of an argmax or argmin node."""
        input_tensor = get_first_fake_tensor(cast(torch.fx.Node, node.args[0]))
        dim = node.args[1] if len(node.args) > 1 and node.args[1] is not None else None
        if dim is None:
            size = input_tensor.numel()
        else:
            size = input_tensor.shape[cast(int, dim)]
        return cls._index_range_from_size(size)

    @staticmethod
    def topk_index_getitems(topk: torch.fx.Node) -> List[torch.fx.Node]:
        """Return live getitem nodes that extract indices from TopK."""
        return [
            user
            for user in topk.users
            if user.op == "call_function"
            and user.target is operator.getitem
            and len(user.args) > 1
            and user.args[1] == 1
            and user.users
        ]

    @classmethod
    def topk_index_range(cls, topk: torch.fx.Node) -> Optional[ValueRange]:
        """Return the proven inclusive range of TopK indices.

        Args:
            topk (torch.fx.Node): TopK tuple-producing node.

        Returns:
            Optional[ValueRange]: Index range, or ``None`` when a finite upper
            bound cannot be proven.

        """
        input_tensor = get_first_fake_tensor(cast(torch.fx.Node, topk.args[0]))
        dim = topk.args[2] if len(topk.args) > 2 else topk.kwargs.get("dim", -1)
        size = input_tensor.shape[cast(int, dim)]
        return cls._index_range_from_size(size)

    @classmethod
    def index_size_fits_int32_policy(cls, index_range: ValueRange) -> bool:
        """Return whether a source index range satisfies the overflow policy."""
        return index_range[1] < cls._INT32_MAX

    @classmethod
    def _range_fits_int32(cls, value_range: ValueRange) -> bool:
        return -cls._INT32_MAX - 1 <= value_range[0] and (
            value_range[1] <= cls._INT32_MAX
        )

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

    def _constant_range(self, value: Any) -> Optional[ValueRange]:
        scalar = self._scalar_int(value)
        if scalar is not None:
            return scalar, scalar
        if not isinstance(value, torch.fx.Node):
            return None

        constant = None
        if value.op == "get_attr" and isinstance(value.target, str):
            constant = getattr(self.graph_module, value.target, None)
        elif value.op == "placeholder" and isinstance(value.target, str):
            buffer_name = value.target.removeprefix("_lifted")
            if buffer_name != value.target:
                try:
                    constant = self.graph_module.get_buffer(buffer_name)
                except AttributeError:
                    pass

        scalar = self._scalar_int(constant)
        return None if scalar is None else (scalar, scalar)

    def _operand_range(
        self,
        value: Any,
        ranges: Dict[torch.fx.Node, ValueRange],
    ) -> Optional[ValueRange]:
        if isinstance(value, torch.fx.Node) and value in ranges:
            return ranges[value]
        return self._constant_range(value)

    @staticmethod
    def _scale_range(value_range: ValueRange, scale: int) -> ValueRange:
        values = value_range[0] * scale, value_range[1] * scale
        return min(values), max(values)

    @staticmethod
    def _trunc_div(value: int, divisor: int) -> int:
        quotient = abs(value) // abs(divisor)
        return -quotient if (value < 0) != (divisor < 0) else quotient

    def _infer_div_range(
        self,
        lhs: ValueRange,
        rhs: ValueRange,
        rounding_mode: Optional[str],
    ) -> Optional[ValueRange]:
        if rhs[0] != rhs[1] or rhs[0] == 0:
            return None
        divisor = rhs[0]
        if rounding_mode == "floor":
            values = lhs[0] // divisor, lhs[1] // divisor
        elif rounding_mode == "trunc":
            values = (
                self._trunc_div(lhs[0], divisor),
                self._trunc_div(lhs[1], divisor),
            )
        else:
            return None
        return min(values), max(values)

    @staticmethod
    def _infer_remainder_range(rhs: ValueRange) -> Optional[ValueRange]:
        if rhs[0] != rhs[1] or rhs[0] == 0:
            return None
        divisor = rhs[0]
        return (0, divisor - 1) if divisor > 0 else (divisor + 1, 0)

    def _infer_add_sub_range(
        self,
        node: torch.fx.Node,
        lhs: ValueRange,
        rhs: ValueRange,
    ) -> Optional[ValueRange]:
        alpha = self._scalar_int(node.kwargs.get("alpha", 1))
        if alpha is None or not self._range_fits_int32((alpha, alpha)):
            return None
        rhs = self._scale_range(rhs, alpha)
        if node.target in self.aten_index_add_ops + self.edge_index_add_ops:
            return lhs[0] + rhs[0], lhs[1] + rhs[1]
        if node.target in self.aten_index_sub_ops + self.edge_index_sub_ops:
            return lhs[0] - rhs[1], lhs[1] - rhs[0]
        return None

    def _infer_binary_range(
        self,
        node: torch.fx.Node,
        ranges: Dict[torch.fx.Node, ValueRange],
    ) -> Optional[ValueRange]:
        if node.target not in self.aten_index_binary_ops + self.edge_index_binary_ops:
            return None
        if get_first_fake_tensor(node).dtype != torch.int64 or len(node.args) < 2:
            return None
        lhs = self._operand_range(node.args[0], ranges)
        rhs = self._operand_range(node.args[1], ranges)
        if lhs is None or rhs is None:
            return None
        if not self._range_fits_int32(lhs) or not self._range_fits_int32(rhs):
            return None

        result: Optional[ValueRange]
        if node.target in self.aten_index_mul_ops + self.edge_index_mul_ops:
            products = (
                lhs[0] * rhs[0],
                lhs[0] * rhs[1],
                lhs[1] * rhs[0],
                lhs[1] * rhs[1],
            )
            result = min(products), max(products)
        elif node.target in self.aten_index_div_ops + self.edge_index_div_ops:
            rounding_mode = cast(Optional[str], node.kwargs.get("rounding_mode"))
            result = self._infer_div_range(lhs, rhs, rounding_mode)
        elif node.target in (
            self.aten_index_remainder_ops + self.edge_index_remainder_ops
        ):
            result = self._infer_remainder_range(rhs)
        else:
            result = self._infer_add_sub_range(node, lhs, rhs)
        if result is None:
            return None
        return result if self._range_fits_int32(result) else None

    def _infer_safe_int32_range(
        self,
        node: torch.fx.Node,
        ranges: Dict[torch.fx.Node, ValueRange],
    ) -> Optional[ValueRange]:
        if node.target in self.aten_index_relay_ops + self.edge_index_relay_ops:
            if get_first_fake_tensor(node).dtype != torch.int64:
                return None
            return ranges.get(cast(torch.fx.Node, node.args[0]))

        if node.target in self.aten_index_cat_ops + self.edge_index_cat_ops:
            inputs = self._tensor_inputs(node)
            if inputs and all(input_node in ranges for input_node in inputs):
                return (
                    min(ranges[input_node][0] for input_node in inputs),
                    max(ranges[input_node][1] for input_node in inputs),
                )
            return None

        if node.target in (
            self.aten_index_full_like_ops + self.edge_index_full_like_ops
        ):
            if (
                get_first_fake_tensor(node).dtype != torch.int64
                or node.kwargs.get("dtype") is not None
            ):
                return None
            if cast(torch.fx.Node, node.args[0]) not in ranges:
                return None
            fill_range = self._operand_range(node.args[1], ranges)
            if (
                fill_range is not None
                and fill_range[0] == fill_range[1]
                and self._range_fits_int32(fill_range)
            ):
                return fill_range
            return None

        return self._infer_binary_range(node, ranges)

    @staticmethod
    def _tensor_inputs(node: torch.fx.Node) -> List[torch.fx.Node]:
        return [
            input_node
            for input_node in node.all_input_nodes
            if isinstance(input_node.meta.get("val"), torch.Tensor)
        ]

    def _is_safe_widening_consumer(
        self,
        node: torch.fx.Node,
        ranges: Dict[torch.fx.Node, ValueRange],
    ) -> bool:
        output = node.meta.get("val")
        if not isinstance(output, torch.Tensor) or output.dtype == torch.int64:
            return False
        int64_inputs = [
            input_node
            for input_node in node.all_input_nodes
            if isinstance(input_node.meta.get("val"), torch.Tensor)
            and input_node.meta["val"].dtype == torch.int64
        ]
        if not int64_inputs or not all(
            input_node in ranges for input_node in int64_inputs
        ):
            return False
        return node.target in (
            self.aten_cast_ops
            + self.edge_cast_ops
            + self.aten_index_consumer_ops
            + self.edge_index_consumer_ops
            + self.aten_index_add_ops
            + self.edge_index_add_ops
            + self.aten_index_sub_ops
            + self.edge_index_sub_ops
            + self.aten_index_mul_ops
            + self.edge_index_mul_ops
            + self.aten_index_full_like_ops
            + self.edge_index_full_like_ops
        )

    def find_safe_index_consumers(
        self,
        source: torch.fx.Node,
        source_range: ValueRange,
    ) -> Int32RangeAnalysisResult:
        """Collect consumers proven safe for the int32 index path.

        Args:
            source (torch.fx.Node): Bounded int64 index source.
            source_range (ValueRange): Inclusive source range.

        Returns:
            Int32RangeAnalysisResult: Proven ranges and safe consumers.

        """
        ranges = {source: source_range}
        safe_consumers: Set[torch.fx.Node] = set()
        for node in self.graph_module.graph.nodes:
            if node.op != "call_function":
                continue
            inferred_range = self._infer_safe_int32_range(node, ranges)
            if inferred_range is not None:
                ranges[node] = inferred_range
                safe_consumers.add(node)
            elif self._is_safe_widening_consumer(node, ranges):
                safe_consumers.add(node)
        return Int32RangeAnalysisResult(ranges, safe_consumers)

    def is_safe_scalar_constant_input(
        self,
        consumer: torch.fx.Node,
        input_node: torch.fx.Node,
        ranges: Dict[torch.fx.Node, ValueRange],
    ) -> bool:
        """Return whether an int64 scalar input can be cast to int32.

        Args:
            consumer (torch.fx.Node): Consumer on a proven int32-safe path.
            input_node (torch.fx.Node): Candidate scalar constant input.
            ranges (dict): Proven ranges for the safe path.

        Returns:
            bool: True when the input is a safe int64 scalar constant.

        """
        if consumer.target not in (
            self.aten_index_binary_ops + self.edge_index_binary_ops
        ):
            return False
        if input_node in ranges:
            return False
        input_value = input_node.meta.get("val")
        return (
            isinstance(input_value, torch.Tensor)
            and input_value.dtype == torch.int64
            and self._constant_range(input_node) is not None
        )
