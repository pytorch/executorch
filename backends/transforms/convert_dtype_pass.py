# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import operator
from typing import Any, Dict

import torch
from executorch.exir.dialects._ops import ops as exir_ops
from executorch.exir.pass_base import ExportPass, PassResult
from torch._subclasses.fake_tensor import FakeTensor


class ConvertDtypePass(ExportPass):

    def __init__(
        self,
        src_dtype: torch.dtype,
        dst_dtype: torch.dtype,
        _skip_dim_order: bool = False,
    ) -> None:
        super(ConvertDtypePass, self).__init__()
        # pyre-ignore[4]
        self.copy_op = (
            exir_ops.edge.aten._to_copy.default
            if _skip_dim_order
            else exir_ops.edge.dim_order_ops._to_dim_order_copy.default
        )
        self.src_dtype = src_dtype
        self.dst_dtype = dst_dtype

    # pyre-ignore[2]
    def _is_tensor_of_dtype(self, node_val, dtype: torch.dtype) -> bool:
        return isinstance(node_val, FakeTensor) and node_val.dtype == dtype

    def _kwargs_to_dst_dtype(self, kwargs: Dict[str, Any]) -> Dict[str, Any]:
        new_kwargs = {"dtype": self.dst_dtype}
        for k, v in kwargs.items():
            if k != "dtype":
                new_kwargs[k] = v
        return new_kwargs

    def _convert_dtype(self, graph: torch.fx.Graph) -> None:
        for node in graph.nodes:
            # For each src dtype input, append a _to_dim_order_copy node to convert to dst dtype.
            if node.op == "placeholder":
                node_val = node.meta["val"]
                if self._is_tensor_of_dtype(node_val, self.src_dtype):
                    with graph.inserting_after(node):
                        args = (node,)
                        to_dst_node = graph.create_node(
                            "call_function",
                            self.copy_op,
                            args,
                            {"dtype": self.dst_dtype},
                        )
                        to_dst_node.meta["val"] = node_val.to(self.dst_dtype)

                        # Replace usage of the src dtype result with the dst dtype result.
                        node.replace_all_uses_with(to_dst_node)
                        to_dst_node.args = (node,)

            # For each operator yielding src dtype, replace it with dst dtype.
            if node.op == "call_function" and node.target != operator.getitem:
                node_val = node.meta["val"]
                if self._is_tensor_of_dtype(node_val, self.src_dtype):
                    node.meta["val"] = node.meta["val"].to(self.dst_dtype)
                    for schema_arg in node.target._schema.arguments:
                        if schema_arg.name == "dtype":
                            node.kwargs = self._kwargs_to_dst_dtype(node.kwargs)
                            break

            # For each src dtype output, prepend a _to_dim_order_copy node to convert from dst dtype.
            if node.op == "output":
                for i, node_val in enumerate(node.meta["val"]):
                    if self._is_tensor_of_dtype(node_val, self.src_dtype):
                        with graph.inserting_before(node):
                            args = (node.args[0][i],)
                            to_src_node = graph.create_node(
                                "call_function",
                                self.copy_op,
                                args,
                                {"dtype": self.src_dtype},
                            )
                            to_src_node.meta["val"] = node_val

                            # Replace usage of the dst dtype result with the src dtype result.
                            node_args_list = list(node.args[0])
                            node_args_list[i] = to_src_node
                            node.args = (tuple(node_args_list),)

    def call(self, graph_module: torch.fx.GraphModule) -> PassResult:
        self._convert_dtype(graph_module.graph)
        return PassResult(graph_module, True)


class I64toI32(ConvertDtypePass):
    def __init__(self, _skip_dim_order=False) -> None:
        super().__init__(torch.int64, torch.int32, _skip_dim_order)
