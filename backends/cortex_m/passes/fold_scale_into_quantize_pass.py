# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Optional, Set, Type

from executorch.backends.arm._passes.arm_pass_utils import (
    get_param_tensor,
    is_param_node,
)
from executorch.exir import ExportedProgram
from executorch.exir.dialects._ops import ops as exir_ops
from executorch.exir.pass_base import ExportPass, PassResult
from torch.fx import GraphModule, Node

_QUANTIZE = exir_ops.edge.quantized_decomposed.quantize_per_tensor.default
_DIV = exir_ops.edge.aten.div.Tensor
_MUL = exir_ops.edge.aten.mul.Tensor


class FoldScaleIntoQuantizePass(ExportPass):
    """Fold a constant elementwise scale (``x / c`` or ``x * c``) into the scale
    of the per-tensor quantize that consumes it, then drop the div/mul.

    Because ``quantize(x / c, scale=S) == quantize(x, scale=S*c)`` and
    ``quantize(x * c, scale=S) == quantize(x, scale=S/c)`` produce identical int8
    values, the constant scale can be absorbed into the adjacent quantize with no
    numerical change. This erases the attention-score ``/sqrt(d)`` scale -- an
    fp32 div that otherwise stays between the QK^T bmm and softmax -- so the
    attention chain is int8 through softmax.

    Runs before ``FoldAndAnnotateQParamsPass`` while the softmax-input quantize is
    still an explicit ``quantized_decomposed.quantize_per_tensor`` node.
    """

    _passes_required_after: Set[Type[ExportPass]] = set()

    def __init__(self, exported_program: Optional[ExportedProgram] = None) -> None:
        super().__init__()
        self.exported_program = exported_program

    def call(self, graph_module: GraphModule) -> PassResult:
        ep = self.exported_program
        if ep is None:
            return PassResult(graph_module, False)

        modified = False
        for node in list(graph_module.graph.nodes):
            if node.op != "call_function" or node.target not in (_DIV, _MUL):
                continue

            scaled, const = node.args[0], node.args[1]
            if not (isinstance(scaled, Node) and isinstance(const, Node)):
                continue
            if not is_param_node(ep, const):
                continue
            const_t = get_param_tensor(ep, const)
            if const_t is None or const_t.numel() != 1:
                continue
            c = float(const_t.reshape(-1)[0])
            if c == 0.0:
                continue

            users = list(node.users)
            if len(users) != 1 or users[0].target != _QUANTIZE:
                continue

            quantize = users[0]
            scale = quantize.args[1]
            new_scale = scale * c if node.target is _DIV else scale / c
            quantize.update_arg(1, new_scale)
            quantize.replace_input_with(node, scaled)
            graph_module.graph.erase_node(node)
            modified = True

        if modified:
            graph_module.graph.eliminate_dead_code()
            graph_module.recompile()
        return PassResult(graph_module, modified)
