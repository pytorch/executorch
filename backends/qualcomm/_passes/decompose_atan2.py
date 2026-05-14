# Copyright (c) Qualcomm Innovation Center, Inc.
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch
from executorch.exir.dialects._ops import ops as exir_ops
from executorch.exir.dialects.edge._ops import EdgeOpOverload
from executorch.exir.pass_base import ExportPass, PassResult

from .utils import copy_meta, create_node, get_const_node


class DecomposeAtan2(ExportPass):
    """
    Decompose atan2(y, x) with full piecewise definition:
        atan2(y, x) =
            atan(y/x)        if x > 0
            atan(y/x) + π    if x < 0, y >= 0
            atan(y/x) - π    if x < 0, y < 0
            +π/2             if x = 0, y > 0
            -π/2             if x = 0, y < 0
            0                if x = 0, y = 0
    """

    _OPS = {
        "eq": (exir_ops.edge.aten.eq.Tensor, torch.ops.aten.eq.Tensor),
        "lt": (exir_ops.edge.aten.lt.Tensor, torch.ops.aten.lt.Tensor),
        "gt": (exir_ops.edge.aten.gt.Tensor, torch.ops.aten.gt.Tensor),
        "ge": (exir_ops.edge.aten.ge.Tensor, torch.ops.aten.ge.Tensor),
        "where": (exir_ops.edge.aten.where.self, torch.ops.aten.where.self),
        "div": (exir_ops.edge.aten.div.Tensor, torch.ops.aten.div.Tensor),
        "atan": (exir_ops.edge.aten.atan.default, torch.ops.aten.atan.default),
        "add": (exir_ops.edge.aten.add.Tensor, torch.ops.aten.add.Tensor),
    }

    _TO_FLOAT_OP = (
        exir_ops.edge.aten._to_copy.default,
        torch.ops.aten._to_copy.default,
    )

    def __init__(self):
        super(DecomposeAtan2, self).__init__()
        self.atan2_targets = {
            torch.ops.aten.atan2.default,
            torch.ops.aten.atan2.out,
            exir_ops.edge.aten.atan2.default,
        }

    def _get_op(self, name, is_edge):
        return self._OPS[name][0] if is_edge else self._OPS[name][1]

    def _cast_to_float(self, graph, node, meta, is_edge):
        """Insert a cast from integer to float if the input is not floating-point."""
        node_val = node.meta.get("val")
        if node_val is not None and not node_val.is_floating_point():
            to_float_op = self._TO_FLOAT_OP[0] if is_edge else self._TO_FLOAT_OP[1]
            cast_node = graph.create_node(
                "call_function", to_float_op, (node,), {"dtype": torch.float32}
            )
            cast_node.meta = copy_meta(meta)
            return cast_node
        return node

    def _get_constants(self, graph, graph_module, node, is_edge, const_cache):
        if is_edge:

            def make_const(name, val):
                if name not in const_cache:
                    const_cache[name] = get_const_node(
                        graph, graph_module, name, val, node
                    )
                return const_cache[name]

            return {
                "zero": make_const("_atan2_zero", 0.0),
                "one": make_const("_atan2_one", 1.0),
                "pi": make_const("_atan2_pi", torch.pi),
                "neg_pi": make_const("_atan2_neg_pi", -torch.pi),
                "pi_half": make_const("_atan2_pi_half", torch.pi / 2),
                "neg_pi_half": make_const("_atan2_neg_pi_half", -torch.pi / 2),
            }
        return {
            "zero": 0.0,
            "one": 1.0,
            "pi": torch.pi,
            "neg_pi": -torch.pi,
            "pi_half": torch.pi / 2,
            "neg_pi_half": -torch.pi / 2,
        }

    def call(self, graph_module: torch.fx.GraphModule):
        graph = graph_module.graph
        const_cache = {}
        for node in list(graph.nodes):
            if node.op == "call_function" and node.target in self.atan2_targets:
                y_node, x_node = node.args[0], node.args[1]
                is_edge = isinstance(node.target, EdgeOpOverload)
                meta = node.meta

                with graph.inserting_before(node):
                    y_node = self._cast_to_float(graph, y_node, meta, is_edge)
                    x_node = self._cast_to_float(graph, x_node, meta, is_edge)

                    consts = self._get_constants(
                        graph, graph_module, node, is_edge, const_cache
                    )

                    x_eq_zero = create_node(
                        graph,
                        self._get_op("eq", is_edge),
                        (x_node, consts["zero"]),
                        meta,
                        callback=lambda m: {**m, "val": m["val"].to(torch.bool)},
                    )
                    safe_x = create_node(
                        graph,
                        self._get_op("where", is_edge),
                        (x_eq_zero, consts["one"], x_node),
                        meta,
                    )
                    ratio = create_node(
                        graph,
                        self._get_op("div", is_edge),
                        (y_node, safe_x),
                        meta,
                    )

                    base = create_node(
                        graph,
                        self._get_op("atan", is_edge),
                        (ratio,),
                        meta,
                    )

                    x_lt_zero = create_node(
                        graph,
                        self._get_op("lt", is_edge),
                        (x_node, consts["zero"]),
                        meta,
                        callback=lambda m: {**m, "val": m["val"].to(torch.bool)},
                    )
                    y_ge_zero = create_node(
                        graph,
                        self._get_op("ge", is_edge),
                        (y_node, consts["zero"]),
                        meta,
                        callback=lambda m: {**m, "val": m["val"].to(torch.bool)},
                    )
                    y_sign_pi = create_node(
                        graph,
                        self._get_op("where", is_edge),
                        (y_ge_zero, consts["pi"], consts["neg_pi"]),
                        meta,
                    )
                    adjustment = create_node(
                        graph,
                        self._get_op("where", is_edge),
                        (x_lt_zero, y_sign_pi, consts["zero"]),
                        meta,
                    )
                    adjusted = create_node(
                        graph,
                        self._get_op("add", is_edge),
                        (base, adjustment),
                        meta,
                    )

                    y_gt_zero = create_node(
                        graph,
                        self._get_op("gt", is_edge),
                        (y_node, consts["zero"]),
                        meta,
                        callback=lambda m: {**m, "val": m["val"].to(torch.bool)},
                    )
                    x_zero_result = create_node(
                        graph,
                        self._get_op("where", is_edge),
                        (y_gt_zero, consts["pi_half"], consts["neg_pi_half"]),
                        meta,
                    )

                    y_eq_zero = create_node(
                        graph,
                        self._get_op("eq", is_edge),
                        (y_node, consts["zero"]),
                        meta,
                        callback=lambda m: {**m, "val": m["val"].to(torch.bool)},
                    )
                    x_zero_final = create_node(
                        graph,
                        self._get_op("where", is_edge),
                        (y_eq_zero, consts["zero"], x_zero_result),
                        meta,
                    )

                    result = create_node(
                        graph,
                        self._get_op("where", is_edge),
                        (x_eq_zero, x_zero_final, adjusted),
                        meta,
                    )

                for user in node.users.copy():
                    user.replace_input_with(node, result)

        graph.eliminate_dead_code()
        graph_module.recompile()
        return PassResult(graph_module, True)
