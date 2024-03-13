# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import copy

import torch
from executorch.exir.dialects._ops import ops as exir_ops
from executorch.exir.pass_base import ExportPass, PassResult

"""
The passes below were taking fron bolt/nn/executorch/passes/quant_fusion.py
"""

T_DQuantPerTensor = exir_ops.edge.quantized_decomposed.dequantize_per_tensor.default


class DequantDuplicator:
    def __init__(self, pass_obj):
        self.pass_obj = pass_obj
        self.duplicated_dequant = False

    def __call__(self, arg):
        if (
            not isinstance(arg, torch.fx.Node)
            or arg.op != "call_function"
            or arg.target != T_DQuantPerTensor  # TODO handle per channel case
        ):
            return arg

        if arg not in self.pass_obj.dequant_map:
            self.pass_obj.dequant_map[arg] = (arg.args, arg.kwargs, arg.meta)
            return arg
        else:
            args, kwargs, meta = self.pass_obj.dequant_map[arg]
            with self.pass_obj.mod.graph.inserting_before(self.pass_obj.current_node):
                dup_dequant = self.pass_obj.mod.graph.call_function(
                    T_DQuantPerTensor,
                    args=args,
                    kwargs=kwargs,
                )
                dup_dequant.meta = meta
                dup_dequant.meta["val"] = copy.copy(meta["val"])
            self.duplicated_dequant = True
            return dup_dequant


class DuplicateDequantNodePass(ExportPass):
    """
    Duplicates all of the dequantize. This is such that all
    quantized ops have their own unique dequant nodes. Since
    quantized ops are represented as dq -> op -> q. Sharing dq nodes
    for quantized ops makes it impossibl to partition against quantized
    ops. As a result we need to duplicate dq nodes for ops which
    share a dq node

    In this example, the graph below:

                            --> op --> q
                          /
      dq -> op -> q -> dq
                          \
                            --> op --> q

    is transformed into:

                      --> dq --> op --> q
                    /
      dq -> op -> q
                    \
                      --> dq --> op --> q


    """

    def __init__(self):
        super().__init__()
        self.dequant_map = {}  # Map of dequant results to its node's arguments

    def call(self, mod):
        self.mod = mod
        duplicator = DequantDuplicator(self)
        for node in list(mod.graph.nodes):
            self.current_node = node

            if node.op != "call_function":
                continue

            new_args = []
            duplicator.duplicated_dequant = False
            for arg in node.args:
                if isinstance(arg, list):
                    new_args.append(list(map(duplicator, arg)))
                else:
                    new_args.append(duplicator(arg))

            if duplicator.duplicated_dequant:
                with mod.graph.inserting_before(node):
                    new_node = mod.graph.call_function(
                        node.target,
                        args=tuple(new_args),
                        kwargs=node.kwargs,
                    )
                    new_node.meta = node.meta
                node.replace_all_uses_with(new_node)
        mod.graph.eliminate_dead_code()
        mod.recompile()

        return PassResult(mod, True)
