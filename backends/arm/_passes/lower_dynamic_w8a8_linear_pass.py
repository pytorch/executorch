# Copyright 2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any, Set, Type

import torch
from executorch.backends.arm._passes.arm_pass import ArmPass
from executorch.backends.arm._passes.arm_pass_utils import (
    create_node,
    get_first_fake_tensor,
)
from executorch.backends.arm.common.annotation_meta import ArmAnnotationInfo
from executorch.backends.arm.constants import ARM_DYNAMIC_W8A8_LINEAR_META_KEY
from executorch.exir.dialects._ops import ops as exir_ops
from executorch.exir.pass_base import ExportPass, PassResult
from torch.fx import GraphModule, Node


class LowerDynamicW8A8LinearPass(ArmPass):
    """Lower detected dynamic-W8A8 Linear patterns to INT8 TOSA MATMUL.

    This pass consumes the metadata attached by DetectDynamicW8A8LinearPass.
    Linear nodes that are not marked with a validated dynamic-W8A8 match are
    left unchanged. The detector guarantees symmetric INT8 activation and
    weight quantization with zero points equal to zero, compatible shapes and
    dtypes, and supported static weight qparams.

    Conceptually, the matched graph:

        x_fp32
          -> dynamic quantization -> x_q:int8
          -> dequantization -----------+
                                        |
        w_q:int8 -> dequantization -----+-> Linear -> y_fp32
                                             |
                                           bias_fp32

    is lowered to:

        x_q:int8 ----+
                     |
        w_q:int8 ----+-> TOSA MATMUL
                           |
                           v
                       accumulator
                           |
                        cast FP32
                           |
                        * x_scale
                           |
                        * w_scale
                           |
                        + bias          # when present
                           |
                        reshape
                           |
                        y_fp32

    Activations are flattened to two dimensions for MATMUL and the weight is
    transposed to match the TOSA operand layout. The result is reshaped back to
    the original Linear output shape.

    Because both zero points are known to be zero, no zero-point correction is
    required. The MATMUL result is converted to FP32 and rescaled using the
    runtime activation scale and static weight scale. An optional FP32 bias is
    added after rescaling.

    The original Linear is replaced only when its detector metadata still
    satisfies the lowering contract.

    """

    _passes_required_after: Set[Type[ExportPass]] = set()

    @staticmethod
    def _clear_lowering_meta(node: Node) -> Node:
        node.meta.pop(ARM_DYNAMIC_W8A8_LINEAR_META_KEY, None)
        custom = node.meta.get("custom")
        if not isinstance(custom, Mapping):
            return node

        custom = dict(custom)
        custom.pop(ArmAnnotationInfo.CUSTOM_META_KEY, None)
        if custom:
            node.meta["custom"] = custom
        else:
            node.meta.pop("custom", None)
        return node

    @classmethod
    def _materialize_scale(cls, graph, value: Any, from_node: Node) -> Node:
        if isinstance(value, Node):
            return value
        if (
            not isinstance(value, (float, int))
            or isinstance(value, bool)
            or not math.isfinite(float(value))
            or float(value) <= 0.0
        ):
            raise TypeError(f"Unsupported static weight scale: {value!r}")
        return cls._clear_lowering_meta(
            create_node(
                graph,
                exir_ops.edge.aten.full.default,
                args=((1,), float(value)),
                kwargs={"dtype": torch.float32},
                from_node=from_node,
                inherit_qparams=False,
            )
        )

    @staticmethod
    def _node_dtype(node: Node) -> torch.dtype | None:
        return getattr(node.meta.get("val"), "dtype", None)

    @staticmethod
    def _matched_nodes(
        meta: dict[str, Any],
    ) -> tuple[Node, Node, Node] | None:
        x_q = meta.get("x_q")
        x_scale = meta.get("x_scale")
        w_q = meta.get("w_q")
        if not isinstance(x_q, Node):
            return None
        if not isinstance(x_scale, Node):
            return None
        if not isinstance(w_q, Node):
            return None
        return x_q, x_scale, w_q

    @classmethod
    def _matched_scales_are_supported(cls, meta: dict[str, Any], x_scale: Node) -> bool:
        w_scale = meta.get("w_scale")
        if not isinstance(w_scale, (Node, float, int)) or isinstance(w_scale, bool):
            return False
        if cls._node_dtype(x_scale) is not torch.float32:
            return False
        return not isinstance(w_scale, Node) or (
            cls._node_dtype(w_scale) is torch.float32
        )

    @classmethod
    def _matched_output_is_supported(cls, node: Node, meta: dict[str, Any]) -> bool:
        if cls._node_dtype(node) is not torch.float32:
            return False
        bias = meta.get("bias")
        if bias is not None and (
            not isinstance(bias, Node) or cls._node_dtype(bias) is not torch.float32
        ):
            return False
        return meta.get("output_dtype") in (None, torch.float32)

    @classmethod
    def _validate_match(cls, node: Node, meta: dict[str, Any]) -> bool:
        matched = cls._matched_nodes(meta)
        if matched is None:
            return False
        x_q, x_scale, w_q = matched
        return (
            cls._node_dtype(x_q) is torch.int8
            and cls._node_dtype(w_q) is torch.int8
            and cls._matched_scales_are_supported(meta, x_scale)
            and cls._matched_output_is_supported(node, meta)
        )

    @staticmethod
    def _flattened_shapes(
        x_q: Node, w_q: Node, node: Node
    ) -> tuple[list[Any], list[Any], tuple[Any, ...]] | None:
        x_shape = tuple(get_first_fake_tensor(x_q).shape)
        w_shape = tuple(get_first_fake_tensor(w_q).shape)
        out_shape = tuple(get_first_fake_tensor(node).shape)
        if len(x_shape) < 1 or len(w_shape) != 2:
            return None
        if x_shape[-1] != w_shape[1]:
            return None

        batches: int | torch.SymInt = 1
        for dim in x_shape[:-1]:
            batches = batches * dim
        return (
            [batches, x_shape[-1]],
            [batches, w_shape[0]],
            out_shape,
        )

    @classmethod
    def _new_node(
        cls,
        graph,
        target,
        *,
        args: tuple = (),
        kwargs: dict[str, Any] | None = None,
        from_node: Node,
    ) -> Node:
        return cls._clear_lowering_meta(
            create_node(
                graph,
                target,
                args=args,
                kwargs=kwargs,
                from_node=from_node,
                inherit_qparams=False,
            )
        )

    def _lower_one(self, graph_module: GraphModule, node: Node) -> bool:
        meta = node.meta.get(ARM_DYNAMIC_W8A8_LINEAR_META_KEY)
        if not isinstance(meta, dict):
            return False
        if meta.get("symmetric_zero_points") is not True:
            return False
        if not self._validate_match(node, meta):
            return False

        x_q = meta["x_q"]
        x_scale = meta["x_scale"]
        w_q = meta["w_q"]
        w_scale = meta.get("w_scale")
        bias = meta.get("bias")
        assert isinstance(x_q, Node)
        assert isinstance(x_scale, Node)
        assert isinstance(w_q, Node)

        shapes = self._flattened_shapes(x_q, w_q, node)
        if shapes is None:
            return False
        flat_x_shape, flat_y_shape, out_shape = shapes

        graph = graph_module.graph
        n_x = len(flat_x_shape)
        n_y = len(flat_y_shape)
        sizes = graph.materialize_symints([*flat_x_shape, *flat_y_shape, *out_shape])
        flat_x_shape = sizes[:n_x]
        flat_y_shape = sizes[n_x : n_x + n_y]
        output_size = sizes[n_x + n_y :]

        with graph.inserting_before(node):
            x_flat = self._new_node(
                graph,
                exir_ops.edge.aten.view_copy.default,
                args=(x_q, flat_x_shape),
                from_node=x_q,
            )
            w_t = self._new_node(
                graph,
                exir_ops.edge.aten.permute_copy.default,
                args=(w_q, (1, 0)),
                from_node=w_q,
            )
            x_b = self._new_node(
                graph,
                exir_ops.edge.aten.unsqueeze_copy.default,
                args=(x_flat, 0),
                from_node=x_flat,
            )
            w_b = self._new_node(
                graph,
                exir_ops.edge.aten.unsqueeze_copy.default,
                args=(w_t, 0),
                from_node=w_t,
            )
            acc_b = self._new_node(
                graph,
                exir_ops.edge.tosa.MATMUL.default,
                args=(x_b, w_b),
                from_node=node,
            )
            acc_b.meta[ARM_DYNAMIC_W8A8_LINEAR_META_KEY] = {
                "symmetric_zero_points": True,
            }
            acc = self._new_node(
                graph,
                exir_ops.edge.aten.view_copy.default,
                args=(acc_b, flat_y_shape),
                from_node=acc_b,
            )
            acc_fp = self._new_node(
                graph,
                exir_ops.edge.dim_order_ops._to_dim_order_copy.default,
                args=(acc,),
                kwargs={"dtype": torch.float32},
                from_node=acc,
            )
            x_scale_fp = self._new_node(
                graph,
                exir_ops.edge.dim_order_ops._to_dim_order_copy.default,
                args=(x_scale,),
                kwargs={"dtype": torch.float32},
                from_node=x_scale,
            )
            weight_scale = self._materialize_scale(graph, w_scale, node)
            weight_scale_fp = self._new_node(
                graph,
                exir_ops.edge.dim_order_ops._to_dim_order_copy.default,
                args=(weight_scale,),
                kwargs={"dtype": torch.float32},
                from_node=weight_scale,
            )
            scaled = self._new_node(
                graph,
                exir_ops.edge.aten.mul.Tensor,
                args=(acc_fp, x_scale_fp),
                from_node=acc_fp,
            )
            scaled = self._new_node(
                graph,
                exir_ops.edge.aten.mul.Tensor,
                args=(scaled, weight_scale_fp),
                from_node=scaled,
            )
            if bias is not None:
                assert isinstance(bias, Node)
                scaled = self._new_node(
                    graph,
                    exir_ops.edge.aten.add.Tensor,
                    args=(scaled, bias),
                    from_node=scaled,
                )
            output = self._new_node(
                graph,
                exir_ops.edge.aten.view_copy.default,
                args=(scaled, output_size),
                from_node=node,
            )

        node.replace_all_uses_with(output)
        graph.erase_node(node)
        return True

    def call(self, graph_module: GraphModule) -> PassResult:
        modified = False
        for node in list(graph_module.graph.nodes):
            if (
                node.op == "call_function"
                and node.target == exir_ops.edge.aten.linear.default
            ):
                modified |= self._lower_one(graph_module, node)

        if modified:
            graph_module.graph.eliminate_dead_code()
            graph_module.graph.lint()
            graph_module.recompile()
            graph_module = super().call(graph_module).graph_module

        return PassResult(graph_module, modified)
