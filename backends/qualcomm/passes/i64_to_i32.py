# Copyright (c) Qualcomm Innovation Center, Inc.
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
import torch
from executorch.backends.qualcomm.builders.utils import get_parameter, is_constant
from executorch.exir.dialects._ops import ops as exir_ops
from executorch.exir.pass_base import ExportPass, PassResult
from torch._subclasses.fake_tensor import FakeTensor


class I64toI32(ExportPass):
    """
    Cast unsupported int64 datatype into int32.
    """

    def __init__(self, edge_program: torch.export.ExportedProgram):
        super(I64toI32, self).__init__()
        self.edge_program = edge_program
        # pyre-ignore[4]
        self.copy_op = exir_ops.edge.aten._to_copy.default

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

    # pyre-ignore[2]
    def _is_tensor_of_dtype(self, node_val, dtype: torch.dtype) -> bool:
        return isinstance(node_val, FakeTensor) and node_val.dtype == dtype

    def _cast_to_int32(self, graph_module: torch.fx.GraphModule):
        for n in graph_module.graph.nodes:
            if is_constant(n, self.edge_program):
                param = get_parameter(n, self.edge_program)
                if param.dtype == torch.int64:
                    # QNN does not support int64
                    self._update_meta(n)
            elif n.op == "placeholder":
                node_val = n.meta["val"]
                if self._is_tensor_of_dtype(node_val, torch.int64):
                    with graph_module.graph.inserting_after(n):
                        args = (n,)
                        to_dst_node = graph_module.graph.create_node(
                            "call_function",
                            self.copy_op,
                            args,
                            {"dtype": torch.int32},
                        )
                        to_dst_node.meta["val"] = node_val.to(torch.int32)

                        # Replace usage of the src dtype result with the dst dtype result.
                        n.replace_all_uses_with(to_dst_node)
                        to_dst_node.args = (n,)

    def call(self, graph_module: torch.fx.GraphModule):
        self._cast_to_int32(graph_module)
        graph_module.recompile()
        graph_module = super().call(graph_module).graph_module
        return PassResult(graph_module, True)
