# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Graph Transformation Pass for Integer Floor-Division Replacement.

Rewrites integer (int64/int32) floor-division into a float64-domain floor to
work around a torch-2.12 AOTInductor/Inductor CUDA miscompile:

    floor_divide(a, b)  ->  floor(a.to(float64) / b.to(float64)).to(orig_int_dtype)
"""

import logging

import torch
from executorch.exir.dialects._ops import ops as exir_ops

from torch.fx import GraphModule, Node
from torch.fx.passes.infra.pass_base import PassBase, PassResult

logger = logging.getLogger(__name__)

# Integer dtypes we rewrite. float64 (53-bit mantissa) is exact for
# |value| < 2**53, which covers these models' index ranges.
_INT_DTYPES = (torch.int64, torch.int32)

# Edge ops that perform a floor-rounded integer division.
_FLOOR_DIVIDE_OP = exir_ops.edge.aten.floor_divide.default
_DIV_MODE_OPS = (
    exir_ops.edge.aten.div.Tensor_mode,
    exir_ops.edge.aten.div.Scalar_mode,
)


class ReplaceInt64FloorDivWithFloatPass(PassBase):
    # Work around a torch-2.12 AOTInductor/Inductor CUDA miscompile of integer
    # (int64) floor-division: fused/broadcast int64 floor_divide is mis-lowered
    # (truncation instead of floor; cross-division term bleed under dynamic shapes).
    # Rewriting into a float64-domain floor lowers correctly. Upstream issue: TODO(link).
    """
    Pass to rewrite integer floor-division into a float64-domain floor.

    Matches ``floor_divide.default`` and the floor-mode ``div.Tensor_mode`` /
    ``div.Scalar_mode`` overloads on integer operands, and replaces each with
    ``floor(a.to(float64) / b.to(float64)).to(orig_int_dtype)`` built from edge
    dialect ops. Float floor-division and non-integer nodes are left untouched.
    """

    def __init__(self):
        super().__init__()
        self._replacement_count = 0

    def call(self, graph_module: GraphModule) -> PassResult:
        self._replacement_count = 0
        modified = False

        for node in graph_module.graph.nodes:
            if not self._should_replace_node(node):
                continue
            try:
                self._replace_node(graph_module, node)
                modified = True
                self._replacement_count += 1
            except Exception as e:
                logger.warning(f"Failed to rewrite floor-div node {node.name}: {e}")
                # Continue with other nodes even if one fails.

        if modified:
            graph_module.recompile()

        logger.info(
            f"Rewrote {self._replacement_count} integer floor-division nodes "
            f"into float64-domain floor"
        )

        return PassResult(graph_module, modified)

    @staticmethod
    def _node_dtype(node: Node):
        val = node.meta.get("val", None)
        if isinstance(val, torch.Tensor):
            return val.dtype
        return None

    @staticmethod
    def _rounding_mode(node: Node):
        if "rounding_mode" in node.kwargs:
            return node.kwargs["rounding_mode"]
        # Trailing positional arg: div(self, other, rounding_mode)
        if len(node.args) > 2:
            return node.args[2]
        return None

    def _should_replace_node(self, node: Node) -> bool:
        if node.op != "call_function":
            return False

        if node.target == _FLOOR_DIVIDE_OP:
            pass
        elif node.target in _DIV_MODE_OPS:
            if self._rounding_mode(node) != "floor":
                return False
        else:
            return False

        # Only rewrite when the result is an integer tensor. Guard meta access:
        # a node may lack meta["val"]; skip conservatively if so.
        out_dtype = self._node_dtype(node)
        if out_dtype not in _INT_DTYPES:
            return False

        return True

    def _replace_node(self, graph_module: GraphModule, node: Node) -> None:
        orig_dtype = self._node_dtype(node)
        a = node.args[0]
        b = node.args[1]

        graph = graph_module.graph
        with graph.inserting_before(node):
            a_f = graph.call_function(
                exir_ops.edge.aten._to_copy.default,
                args=(a,),
                kwargs={"dtype": torch.float64},
            )
            if isinstance(b, Node):
                b_f = graph.call_function(
                    exir_ops.edge.aten._to_copy.default,
                    args=(b,),
                    kwargs={"dtype": torch.float64},
                )
                q = graph.call_function(exir_ops.edge.aten.div.Tensor, args=(a_f, b_f))
            else:
                # Python-scalar divisor: stays bit-exact, no cast needed for b.
                q = graph.call_function(
                    exir_ops.edge.aten.div.Scalar, args=(a_f, float(b))
                )
            fl = graph.call_function(exir_ops.edge.aten.floor.default, args=(q,))
            new_node = graph.call_function(
                exir_ops.edge.aten._to_copy.default,
                args=(fl,),
                kwargs={"dtype": orig_dtype},
            )

            new_node.meta = node.meta.copy()

        node.replace_all_uses_with(new_node)
        graph.erase_node(node)
