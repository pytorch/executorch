# Copyright (c) Qualcomm Innovation Center, Inc.
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch
from executorch.exir.dialects._ops import ops as exir_ops
from executorch.exir.pass_base import ExportPass, PassResult

from .utils import create_const_node, create_node


class DecomposeHyperbolicVariants(ExportPass):
    """
    Decompose hyperbolic functions into supported primitives:
        sinh(x)  = 0.5 * (exp(x) - exp(-x))
        cosh(x)  = 0.5 * (exp(x) + exp(-x))
        asinh(x) = log(x + sqrt(x*x + 1))
        acosh(x) = log(x + sqrt(x*x - 1))
        atanh(x) = 0.5 * log((1 + x) / (1 - x))
    """

    _EDGE_OPS = {
        exir_ops.edge.aten.sinh.default,
        exir_ops.edge.aten.cosh.default,
        exir_ops.edge.aten.asinh.default,
        exir_ops.edge.aten.acosh.default,
        exir_ops.edge.aten.atanh.default,
    }

    def __init__(self):
        super(DecomposeHyperbolicVariants, self).__init__()
        self._dispatcher = {
            # ATen dialect
            torch.ops.aten.sinh.default: self._decompose_sinh,
            torch.ops.aten.cosh.default: self._decompose_cosh,
            torch.ops.aten.asinh.default: self._decompose_asinh,
            torch.ops.aten.acosh.default: self._decompose_acosh,
            torch.ops.aten.atanh.default: self._decompose_atanh,
            # Edge dialect
            exir_ops.edge.aten.sinh.default: self._decompose_sinh,
            exir_ops.edge.aten.cosh.default: self._decompose_cosh,
            exir_ops.edge.aten.asinh.default: self._decompose_asinh,
            exir_ops.edge.aten.acosh.default: self._decompose_acosh,
            exir_ops.edge.aten.atanh.default: self._decompose_atanh,
        }

    def _get_ops(self, is_edge):
        if is_edge:
            return {
                "exp": exir_ops.edge.aten.exp.default,
                "neg": exir_ops.edge.aten.neg.default,
                "add": exir_ops.edge.aten.add.Tensor,
                "sub": exir_ops.edge.aten.sub.Tensor,
                "mul": exir_ops.edge.aten.mul.Tensor,
                "div": exir_ops.edge.aten.div.Tensor,
                "log": exir_ops.edge.aten.log.default,
                "sqrt": exir_ops.edge.aten.sqrt.default,
            }
        return {
            "exp": torch.ops.aten.exp.default,
            "neg": torch.ops.aten.neg.default,
            "add": torch.ops.aten.add.Tensor,
            "sub": torch.ops.aten.sub.Tensor,
            "mul": torch.ops.aten.mul.Tensor,
            "div": torch.ops.aten.div.Tensor,
            "log": torch.ops.aten.log.default,
            "sqrt": torch.ops.aten.sqrt.default,
        }

    def _decompose_exp_symmetry(
        self, node, graph, graph_module, const_cache, combine_op_key
    ):
        """Shared helper for sinh and cosh: (exp(x) ± exp(-x)) / 2."""
        is_edge = node.target in self._EDGE_OPS
        ops = self._get_ops(is_edge)
        meta = node.meta
        input_node = node.args[0]

        if is_edge:
            half_name = "_half_constant"
            if half_name not in const_cache:
                const_cache[half_name] = create_const_node(
                    graph, graph_module, half_name, 0.5, node
                )
            half_arg = const_cache[half_name]
        else:
            half_arg = 0.5

        with graph.inserting_before(node):
            exp_pos = create_node(graph, ops["exp"], (input_node,), meta)
            neg_x = create_node(graph, ops["neg"], (input_node,), meta)
            exp_neg = create_node(graph, ops["exp"], (neg_x,), meta)
            combine = create_node(graph, ops[combine_op_key], (exp_pos, exp_neg), meta)
            result = create_node(graph, ops["mul"], (combine, half_arg), meta)

        for user in node.users.copy():
            user.replace_input_with(node, result)

    def _decompose_sinh(self, node, graph, graph_module, const_cache):
        self._decompose_exp_symmetry(node, graph, graph_module, const_cache, "sub")

    def _decompose_cosh(self, node, graph, graph_module, const_cache):
        self._decompose_exp_symmetry(node, graph, graph_module, const_cache, "add")

    def _decompose_asinh(self, node, graph, graph_module, const_cache):
        is_edge = node.target in self._EDGE_OPS
        ops = self._get_ops(is_edge)
        meta = node.meta
        input_node = node.args[0]

        if is_edge:
            one_name = "_one_constant"
            if one_name not in const_cache:
                const_cache[one_name] = create_const_node(
                    graph, graph_module, one_name, 1.0, node
                )
            one_arg = const_cache[one_name]
        else:
            one_arg = 1.0

        with graph.inserting_before(node):
            x_sq = create_node(graph, ops["mul"], (input_node, input_node), meta)
            x_sq_plus_1 = create_node(graph, ops["add"], (x_sq, one_arg), meta)
            sqrt_node = create_node(graph, ops["sqrt"], (x_sq_plus_1,), meta)
            sum_node = create_node(graph, ops["add"], (input_node, sqrt_node), meta)
            result = create_node(graph, ops["log"], (sum_node,), meta)

        for user in node.users.copy():
            user.replace_input_with(node, result)

    def _decompose_acosh(self, node, graph, graph_module, const_cache):
        is_edge = node.target in self._EDGE_OPS
        ops = self._get_ops(is_edge)
        meta = node.meta
        input_node = node.args[0]

        if is_edge:
            one_name = "_one_constant"
            if one_name not in const_cache:
                const_cache[one_name] = create_const_node(
                    graph, graph_module, one_name, 1.0, node
                )
            one_arg = const_cache[one_name]
        else:
            one_arg = 1.0

        with graph.inserting_before(node):
            x_sq = create_node(graph, ops["mul"], (input_node, input_node), meta)
            x_sq_minus_1 = create_node(graph, ops["sub"], (x_sq, one_arg), meta)
            sqrt_node = create_node(graph, ops["sqrt"], (x_sq_minus_1,), meta)
            sum_node = create_node(graph, ops["add"], (input_node, sqrt_node), meta)
            result = create_node(graph, ops["log"], (sum_node,), meta)

        for user in node.users.copy():
            user.replace_input_with(node, result)

    def _decompose_atanh(self, node, graph, graph_module, const_cache):
        is_edge = node.target in self._EDGE_OPS
        ops = self._get_ops(is_edge)
        meta = node.meta
        input_node = node.args[0]

        if is_edge:
            one_name = "_one_constant"
            if one_name not in const_cache:
                const_cache[one_name] = create_const_node(
                    graph, graph_module, one_name, 1.0, node
                )
            one_arg = const_cache[one_name]
            half_name = "_half_constant"
            if half_name not in const_cache:
                const_cache[half_name] = create_const_node(
                    graph, graph_module, half_name, 0.5, node
                )
            half_arg = const_cache[half_name]
        else:
            one_arg = 1.0
            half_arg = 0.5

        with graph.inserting_before(node):
            one_plus_x = create_node(graph, ops["add"], (one_arg, input_node), meta)
            one_minus_x = create_node(graph, ops["sub"], (one_arg, input_node), meta)
            ratio = create_node(graph, ops["div"], (one_plus_x, one_minus_x), meta)
            log_node = create_node(graph, ops["log"], (ratio,), meta)
            result = create_node(graph, ops["mul"], (half_arg, log_node), meta)

        for user in node.users.copy():
            user.replace_input_with(node, result)

    def call(self, graph_module: torch.fx.GraphModule) -> PassResult:
        graph = graph_module.graph
        const_cache = {}

        for node in list(graph.nodes):
            if node.target in self._dispatcher:
                self._dispatcher[node.target](node, graph, graph_module, const_cache)

        graph.eliminate_dead_code()
        graph_module.recompile()
        return PassResult(graph_module, True)
