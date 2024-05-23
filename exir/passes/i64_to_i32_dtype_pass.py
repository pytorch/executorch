# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch
from executorch.exir.pass_base import ExportPass, PassResult

from torch._subclasses.fake_tensor import FakeTensor


#
# For node metadata
#


def get_converted_val(v):
    if isinstance(v, FakeTensor) and v.dtype == torch.int64:
        return v.to(torch.int32)

    if isinstance(v, list):
        return [get_converted_val(item) for item in v]

    if isinstance(v, tuple):
        return tuple(get_converted_val(item) for item in v)

    return v


def traverse_meta(node_meta):
    node_meta["val"] = get_converted_val(node_meta.get("val"))


#
# For schema arguments
#


def get_function_arg(node, i: int, schema_arg):
    if not schema_arg.kwarg_only and i < len(node.args):
        return node.args[i]
    elif schema_arg.name in node.kwargs:
        return node.kwargs[schema_arg.name]
    else:
        return schema_arg.default_value


def should_convert_kwargs(schema_arg_name, function_arg) -> bool:
    return schema_arg_name == "dtype" and (
        function_arg == torch.int64 or function_arg is None
    )


def get_converted_kwargs(kwargs):
    new_kwargs = {"dtype": torch.int32}
    for k, v in kwargs.items():
        if k != "dtype":
            new_kwargs[k] = v
    return new_kwargs


#
# Main
#


def traverse_graph(graph: torch.fx.Graph):
    for node in graph.nodes:
        if node.op == "placeholder" or node.op == "output":
            traverse_meta(node.meta)
        elif node.op == "call_function":
            traverse_meta(node.meta)
            if not hasattr(node.target, "_schema"):
                continue
            for i, schema_arg in enumerate(node.target._schema.arguments):
                function_arg = get_function_arg(node, i, schema_arg)

                if should_convert_kwargs(schema_arg.name, function_arg):
                    node.kwargs = get_converted_kwargs(node.kwargs)
                elif isinstance(function_arg, torch.fx.Node):
                    traverse_meta(function_arg.meta)


class I64ToI32DtypePass(ExportPass):
    def call(self, graph_module: torch.fx.GraphModule) -> PassResult:
        traverse_graph(graph_module.graph)
        return PassResult(graph_module, True)
