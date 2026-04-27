# Copyright (c) Qualcomm Innovation Center, Inc.
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch
from executorch.exir.dialects._ops import ops as exir_ops
from executorch.exir.dialects.edge._ops import EdgeOpOverload
from executorch.exir.pass_base import ExportPass, PassResult


class DecomposePad(ExportPass):
    """
    Convert aten.pad.default with non-constant modes to specific pad ops.
    After torch.export, nn.ReflectionPad2d becomes aten.pad.default with mode='reflect'.
    This pass converts it to aten.reflection_pad2d.default which the QNN pad builder handles directly.

    Supported:
    - mode='reflect', 4 padding values -> reflection_pad2d (QNN MIRROR_REFLECT, max rank 4).

    Not supported by QNN (max rank 4 for non-constant schemes):
    - mode='reflect', 6 padding values (3d) -> reflection_pad3d (QNN MIRROR_REFLECT max rank is 4)
    - mode='replicate' -> QNN EDGE scheme produces incorrect results for FP32 inputs for replication_pad2d

    Note: reflection_pad1d is handled by PyTorch's built-in decomposition of aten.pad.default (mode='reflect', 2 padding values)
    -> reflection_pad1d, combined with the skip decomp table entry for reflection_pad1d.
    """

    _PAD_TARGETS = {
        torch.ops.aten.pad.default,
        exir_ops.edge.aten.pad.default,
    }

    _PAD_OPS = {
        ("reflect", 4, False): torch.ops.aten.reflection_pad2d.default,
        ("reflect", 4, True): exir_ops.edge.aten.reflection_pad2d.default,
    }

    def call(self, graph_module: torch.fx.GraphModule):
        graph = graph_module.graph
        for node in list(graph.nodes):
            if node.op != "call_function" or node.target not in self._PAD_TARGETS:
                continue
            mode = node.args[2] if len(node.args) > 2 else "constant"

            padding = node.args[1]
            is_edge = isinstance(node.target, EdgeOpOverload)
            target_op = self._PAD_OPS.get((mode, len(padding), is_edge))
            if target_op is None:
                continue

            node.target = target_op
            node.args = (node.args[0], list(padding))

        graph_module.recompile()
        return PassResult(graph_module, True)
