# Copyright (c) Qualcomm Innovation Center, Inc.
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import logging
from typing import FrozenSet

import torch
from executorch.backends.qualcomm.builders.utils import (
    get_parameter,
    is_constant,
    is_graph_output,
)
from executorch.backends.qualcomm.utils.constants import QCOM_ORIG_DTYPE
from executorch.exir.dialects._ops import ops as exir_ops
from executorch.exir.pass_base import ExportPass, PassResult
from torch._subclasses.fake_tensor import FakeTensor


class I64toI32(ExportPass):
    """
    Insert a cast node to cast dtype from int64 to int32.
    This will be applied on operator and constant nodes such as weights.
    """

    I64_OPS = {
        exir_ops.edge.aten.argmax.default,
        exir_ops.edge.aten.argmin.default,
        exir_ops.edge.aten.arange.start_step,
        exir_ops.edge.aten.cumsum.default,
        exir_ops.edge.aten.full.default,
        exir_ops.edge.aten.scalar_tensor.default,
        exir_ops.edge.dim_order_ops._to_dim_order_copy.default,
    }
    # This dict is to ensure that the input of the OPs are int64 due to Pytorch restrictions.
    # For example, scatter op can only accept args[2], the index, as int64.
    # Key: Ops to cast input to i64
    # Value: The args' indices to add casting op
    I64_IN_OPS = {
        exir_ops.edge.aten.gather.default: [2],
        exir_ops.edge.aten.scatter.src: [2],
    }
    copy_op = exir_ops.edge.aten._to_copy.default

    def __init__(
        self,
        edge_program,
        skip_node: FrozenSet[str] = frozenset(),
    ):
        super(I64toI32, self).__init__()
        self.edge_program = edge_program
        self.skip_node = skip_node

    # pyre-ignore[2]
    def _is_tensor_of_dtype(self, node_val, dtype: torch.dtype) -> bool:
        return isinstance(node_val, FakeTensor) and node_val.dtype == dtype

    def call_operator(self, op, args, kwargs, meta):
        if op in self.I64_OPS and self._is_tensor_of_dtype(meta["val"], torch.int64):
            res = super().call_operator(op, args, kwargs, meta)
            return super().call_operator(
                self.copy_op, (res,), {"dtype": torch.int32}, meta
            )
        return super().call_operator(op, args, kwargs, meta)

    def _record_original_output_dtype(self, graph_module: torch.fx.GraphModule):
        for n in graph_module.graph.nodes:
            # Keep track of original output dtype so we ensure the dtype of the graph is consistent with nn.Module
            if is_graph_output(n):
                if isinstance(n.meta["val"], (tuple, list)):
                    dtype_list = [tensor.dtype for tensor in n.meta["val"]]
                    n.meta[QCOM_ORIG_DTYPE] = dtype_list
                else:
                    n.meta[QCOM_ORIG_DTYPE] = n.meta["val"].dtype

    def _preserve_output_dtype(self, graph_module: torch.fx.GraphModule):
        for n in graph_module.graph.nodes:
            if is_graph_output(n) and QCOM_ORIG_DTYPE in n.meta:
                if isinstance(n.meta["val"], (tuple, list)):
                    for i, dtype in enumerate(n.meta[QCOM_ORIG_DTYPE]):
                        # TODO: Enable this in future to support OP such as topK
                        if n.meta["val"][i].dtype != dtype:
                            raise AssertionError(
                                "Multi output nodes currently don't support casting dtype back."
                            )
                elif n.meta["val"].dtype != n.meta[QCOM_ORIG_DTYPE]:
                    if n.meta[QCOM_ORIG_DTYPE] != torch.int64:
                        logging.warning(
                            "This pass is intended to maintain output as int64 when nn.Module outputs int64. Other dtype modification is detected. Please ensure this is desired."
                        )
                    with graph_module.graph.inserting_after(n):
                        orig_dtype = n.meta[QCOM_ORIG_DTYPE]
                        node_val = n.meta["val"]
                        args = (n,)
                        users = list(n.users.keys())
                        output_users = [
                            user for user in users if user.target == "output"
                        ]
                        cast_node = graph_module.graph.create_node(
                            "call_function",
                            self.copy_op,
                            args,
                            {"dtype": orig_dtype},
                        )
                        cast_node.meta["val"] = node_val.to(orig_dtype)
                        cast_node.args = args
                        for user in output_users:
                            user.replace_input_with(n, cast_node)

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
                # TODO This trick seems to use in mobilebert.
                # It would be better to convert to torch.int32
                node.meta["val"] = meta_val.to(torch.float)

    def _cast_constant_to_int32(self, graph_module: torch.fx.GraphModule):
        for n in graph_module.graph.nodes:
            if n.target in self.skip_node:
                continue
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

    def _cast_op_args_to_i64(self, graph_module: torch.fx.GraphModule):
        # input will be cast to i32 during call_operator dtype propogation
        # insert i64 cast node to prevent PyTorch's operator validation failure
        for node in graph_module.graph.nodes:
            if node.target in self.I64_IN_OPS:
                with graph_module.graph.inserting_before(node):
                    arg_indices = self.I64_IN_OPS[node.target]
                    for arg_index in arg_indices:
                        input_node = node.args[arg_index]
                        cast_i64_node = graph_module.graph.create_node(
                            "call_function",
                            self.copy_op,
                            (input_node,),
                            {"dtype": torch.int64},
                        )
                        cast_i64_node.meta["val"] = node.meta["val"].to(torch.int64)
                        args_list = list(node.args)
                        args_list[arg_index] = cast_i64_node
                        node.args = tuple(args_list)

    def call(self, graph_module: torch.fx.GraphModule) -> PassResult:
        # Record original output dtype to ensure that if user expects int64 as output,
        # convert the output back to int64 if it is casted from int64->int32.
        self._record_original_output_dtype(graph_module)
        self._cast_constant_to_int32(graph_module)
        self._cast_op_args_to_i64(graph_module)
        graph_module = super().call(graph_module).graph_module
        self._preserve_output_dtype(graph_module)
        graph_module.recompile()
        return PassResult(graph_module, True)
