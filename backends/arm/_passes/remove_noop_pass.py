# Copyright 2024-2026 Arm Limited and/or its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


import logging
from typing import Any, Set, Type

from executorch.backends.arm._passes import ArmOpTargetedPass
from executorch.backends.arm.constants import DQ_OPS, Q_OPS

from executorch.exir.dialects._ops import ops as exir_ops
from executorch.exir.pass_base import ExportPass, PassResult, ProxyValue
from torch.fx import GraphModule, Node

logger = logging.getLogger(__name__)


class RemoveNoopPass(ArmOpTargetedPass):
    """Remove no-ops from graph_module."""

    _passes_required_after: Set[Type[ExportPass]] = set()
    _single_input_concat_ops = (
        exir_ops.edge.aten.cat.default,
        exir_ops.edge.aten.concatenate.default,
    )
    _resize_ops = (
        exir_ops.edge.aten.upsample_bilinear2d.vec,
        exir_ops.edge.aten.upsample_nearest2d.vec,
    )
    target_ops = (
        exir_ops.edge.dim_order_ops._clone_dim_order.default,
        exir_ops.edge.dim_order_ops._to_dim_order_copy.default,
        exir_ops.edge.aten.alias_copy.default,
        exir_ops.edge.aten.copy.default,
        exir_ops.edge.aten.detach_copy.default,
        *_single_input_concat_ops,
        *_resize_ops,
        exir_ops.backend.tosa.PAD.default,
        exir_ops.backend.tosa.SLICE.default,
    )

    @staticmethod
    def _get_static_shape(value: Any) -> tuple[int, ...] | None:
        if isinstance(value, ProxyValue):
            # Shape calculations may carry concrete example values without being
            # compile-time constants. Only accept explicit CONST_SHAPE nodes.
            if value.node.target != exir_ops.backend.tosa.CONST_SHAPE.default:
                return None
            value = value.data

        # Reject unresolved symbolic dimensions rather than relying on hints.
        if not isinstance(value, (list, tuple)) or not all(
            type(dim) is int for dim in value
        ):
            return None
        return tuple(value)

    @staticmethod
    def _is_identity_scale_factor(scale_factors: Any) -> bool:
        if type(scale_factors) in (int, float):
            return scale_factors == 1.0
        return (
            isinstance(scale_factors, (list, tuple))
            and len(scale_factors) == 2
            and all(
                type(scale) in (int, float) and scale == 1.0 for scale in scale_factors
            )
        )

    def _resize_scale_factors(self, op, args: tuple[Any, ...]) -> Any:
        return args[3] if op == exir_ops.edge.aten.upsample_bilinear2d.vec else args[2]

    def _is_identity_resize(self, op, args: tuple[Any, ...]) -> bool:
        scale_factors = self._resize_scale_factors(op, args)
        if self._is_identity_scale_factor(scale_factors):
            return True

        output_spatial_size = self._get_static_shape(args[1])
        input_spatial_size = self._get_static_shape(args[0].data.shape[2:])
        return (
            output_spatial_size is not None
            and input_spatial_size is not None
            and input_spatial_size == output_spatial_size
            and scale_factors is None
        )

    def _is_identity_resize_node(self, node: Node) -> bool:
        if node.target not in self._resize_ops:
            return False
        input_node = node.args[0]
        if not isinstance(input_node, Node):
            return False

        scale_factors = self._resize_scale_factors(node.target, node.args)
        if self._is_identity_scale_factor(scale_factors):
            return True

        output_spatial_size = self._get_static_shape(node.args[1])
        input_spatial_size = self._get_static_shape(input_node.meta["val"].shape[2:])
        return (
            output_spatial_size is not None
            and input_spatial_size is not None
            and input_spatial_size == output_spatial_size
            and scale_factors is None
        )

    def _is_removable_noop_node(self, node: Node) -> bool:
        if node.target in self._single_input_concat_ops:
            inputs = node.args[0]
            return isinstance(inputs, (list, tuple)) and len(inputs) == 1

        if node.target in (
            exir_ops.edge.dim_order_ops._clone_dim_order.default,
            exir_ops.edge.aten.alias_copy.default,
            exir_ops.edge.aten.detach_copy.default,
        ):
            return True

        if node.target == exir_ops.edge.aten.copy.default:
            return True

        if node.target == exir_ops.edge.dim_order_ops._to_dim_order_copy.default:
            input_node = node.args[0]
            if not isinstance(input_node, Node):
                return False
            input_dtype = input_node.meta["val"].dtype
            output_dtype = node.kwargs.get("dtype", input_dtype)
            return input_dtype == output_dtype

        return False

    def _partition_would_be_empty_without_identity_resize(
        self, graph_module: GraphModule
    ) -> bool:
        has_identity_resize = False
        for node in graph_module.graph.nodes:
            if (
                node.op != "call_function"
                or node.target in Q_OPS
                or node.target in DQ_OPS
            ):
                continue
            if self._is_identity_resize_node(node):
                has_identity_resize = True
                continue
            if self._is_removable_noop_node(node):
                continue
            return False
        return has_identity_resize

    def call(self, graph_module: GraphModule) -> PassResult:
        if self._partition_would_be_empty_without_identity_resize(graph_module):
            return PassResult(graph_module, False)

        result = super().call(graph_module)
        # Removing a no-op can leave its shape operands without users.
        removed_dead_code = result.graph_module.graph.eliminate_dead_code()
        if removed_dead_code:
            result.graph_module.graph.lint()
            result.graph_module.recompile()
        return PassResult(result.graph_module, result.modified or removed_dead_code)

    def call_operator(self, op, args, kwargs, meta, updated=False):  # noqa: C901
        if op not in self.target_ops:
            return super().call_operator(op, args, kwargs, meta, updated)

        if op in self._single_input_concat_ops:
            inputs = args[0]
            # Concatenating one tensor returns that tensor unchanged.
            if isinstance(inputs, (list, tuple)) and len(inputs) == 1:
                return inputs[0]
            return super().call_operator(op, args, kwargs, meta, updated)

        if op in self._resize_ops:
            if self._is_identity_resize(op, args):
                return args[0]
            return super().call_operator(op, args, kwargs, meta, updated)

        if op == exir_ops.backend.tosa.PAD.default:
            padding = self._get_static_shape(args[1])
            # PAD is an identity when every before/after padding value is zero.
            if (
                padding is not None
                and len(padding) == 2 * len(args[0].data.shape)
                and all(value == 0 for value in padding)
            ):
                return args[0]
            return super().call_operator(op, args, kwargs, meta, updated)

        if op == exir_ops.backend.tosa.SLICE.default:
            starts = self._get_static_shape(args[1])
            sizes = self._get_static_shape(args[2])
            input_shape = self._get_static_shape(args[0].data.shape)
            # Starting at zero and selecting the full input shape is an identity.
            if (
                starts is not None
                and sizes is not None
                and input_shape is not None
                and starts == (0,) * len(input_shape)
                and sizes == input_shape
            ):
                return args[0]
            return super().call_operator(op, args, kwargs, meta, updated)

        input_dtype = args[0].data.dtype
        output_dtype = kwargs.get("dtype", input_dtype)

        # A clone-like operation that changes dtype is not a no-op.
        if input_dtype != output_dtype:
            return super().call_operator(op, args, kwargs, meta, updated)

        # copy writes its source (args[1]); the remaining targets return args[0].
        if op == exir_ops.edge.aten.copy.default:
            return args[1]
        return args[0]
