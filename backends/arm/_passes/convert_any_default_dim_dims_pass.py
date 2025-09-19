# Copyright 2025 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Set, Type

import torch
from executorch.exir.dialects._ops import (  # type: ignore[import-not-found]
    ops as exir_ops,
)
from executorch.exir.pass_base import (  # type: ignore[import-not-found]
    ExportPass,
    PassResult,
)


class ConvertAnyDefaultDimDimsPass(ExportPass):
    """
    Converts any.default, any.dim and any.dims to a sequence of any.dim by unrolling multi-dimensional reduction.
    Please refer to KeepDimsFalseToSqueezePass for an explanation of this coversion.

    Example 1
    Original:
        any()  # x.shape: [dim1, dim2, ..., dimn]
    After pass:
        any.dim(dim1, keepdim = True)
        any.dim(dim2, keepdim = True)
        ...
        any.dim(dimn, keepdim = True)
        squeeze(dim = [dim1, dim2, ...., dimn])

    Example 2
    Original:
        any.dim(dim1, keepdim = False)
    After pass:
        any.dim(dim1, keepdim = True)
        squeeze(dim = [dim1])

    Example 3
    Original:
        any.dims([dim1, dim2], keepdim = False)
    After pass:
        any.dim(dim1, keepdim = True)
        any.dim(dim2, keepdim = True)
        squeeze(dim = [dim1, dim2])
    """

    _passes_required_after: Set[Type[ExportPass]] = set()

    def call(self, graph_module: torch.fx.GraphModule):
        modified = False
        for node in graph_module.graph.nodes:
            if node.op != "call_function":
                continue
            if node.target not in [
                exir_ops.edge.aten.any.default,
                exir_ops.edge.aten.any.dim,
                exir_ops.edge.aten.any.dims,
            ]:
                continue

            if len(node.args) == 1:
                # any.default(input)
                input_node = (node.args)[0]
                dims = range(len(input_node.meta["val"].shape))
                keepdim = False
            elif len(node.args) == 2:
                # any.dim/dims(input, dims=dims)
                input_node, dims = node.args
                keepdim = False
            elif len(node.args) == 3:
                # any.dim/dims(input, dims=dims, keepdim=keepdim)
                input_node, dims, keepdim = node.args
            else:
                raise RuntimeError(
                    f"Unexpected arg size {len(node.args)} in {node.name}"
                )
            try:
                iter(dims)
            except:
                dims = [dims]  # type: ignore[assignment]
            else:
                dims = list(dims)  # type: ignore[assignment]

            # Unroll multi-dimensional reduction and keep-dims arg
            with graph_module.graph.inserting_before(node):
                for dim in dims:
                    args = (input_node, dim, True)
                    input_node = graph_module.graph.create_node(
                        "call_function", exir_ops.edge.aten.any.dim, args, node.kwargs
                    )

                if not keepdim:
                    args = (input_node, dims)  # type: ignore[assignment]
                    input_node = graph_module.graph.create_node(
                        "call_function",
                        exir_ops.edge.aten.squeeze_copy.dims,
                        args,
                    )

            node.replace_all_uses_with(input_node)
            modified = True

        if modified:
            graph_module.graph.eliminate_dead_code()
            graph_module.recompile()
            graph_module = super().call(graph_module).graph_module

        return PassResult(graph_module, modified)
