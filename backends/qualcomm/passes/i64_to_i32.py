# Copyright (c) Qualcomm Innovation Center, Inc.
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
import torch
from executorch.backends.qualcomm.builders.utils import get_parameter, is_constant
from executorch.exir.pass_base import ExportPass, PassResult


class I64toI32(ExportPass):
    """
    Cast unsupported int64 datatype into int32.
    """

    def __init__(self, edge_program: torch.export.ExportedProgram):
        super(I64toI32, self).__init__()
        self.edge_program = edge_program

    def _update_meta(self, node: torch.fx.node) -> None:
        meta_val = node.meta["val"]
        if isinstance(meta_val, tuple):
            node.meta["val"] = (
                (
                    fake_tensor.to(torch.int32)
                    if fake_tensor.dtype == torch.int64
                    else fake_tensor
                )
                for fake_tensor in meta_val
            )
        else:
            if meta_val.dtype == torch.int64:
                node.meta["val"] = meta_val.to(torch.float)

    def _cast_to_int32(self, graph_module: torch.fx.GraphModule):
        for n in graph_module.graph.nodes:
            if is_constant(n, self.edge_program):
                param = get_parameter(n, self.edge_program)
                if param.dtype == torch.int64:
                    # QNN does not support int64
                    self._update_meta(n)

    def call(self, graph_module: torch.fx.GraphModule):
        self._cast_to_int32(graph_module)
        graph_module.recompile()
        graph_module = super().call(graph_module).graph_module
        return PassResult(graph_module, True)
