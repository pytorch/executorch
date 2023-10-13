# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch
from executorch.exir.dialects._ops import ops as exir_ops
from executorch.exir.pass_base import ExportPass
from torch.fx import GraphModule, subgraph_rewriter
from torch.fx.passes.infra.pass_base import PassResult
from torch.utils import _pytree as pytree

from ._quant_patterns_and_replacements import get_quant_patterns_and_replacements


def _fuse_quantized_cat(model: GraphModule) -> None:
    """fuse "dequantize -> cat -> quantize" pattern to cat operator, only happens if the quantization
    parameters for dequantize for all the inputs matches, and it also matches the quantization
    parameters for the quantize node after cat
    """

    # get quantization parameters for the node, either for quantize or dequantize node
    def _get_qparams(node):
        assert node.target in (
            exir_ops.edge.quantized_decomposed.quantize_per_tensor.default,
            exir_ops.edge.quantized_decomposed.dequantize_per_tensor.default,
        )
        args = list(node.args)
        # skip input
        qparams = args[1:]
        return qparams

    for n in model.graph.nodes:
        if (
            n.op != "call_function"
            or n.target
            != exir_ops.edge.quantized_decomposed.quantize_per_tensor.default
        ):

            continue
        qnode = n
        maybe_cat = qnode.args[0]
        if (
            maybe_cat.op != "call_function"
            or maybe_cat.target != exir_ops.edge.aten.cat.default
        ):

            continue
        tensor_args = maybe_cat.args[0]
        if not isinstance(tensor_args, (tuple, list)):
            continue

        matched_quantized_cat = True
        output_qparams = _get_qparams(qnode)
        for tensor_arg in tensor_args:
            if (
                tensor_arg.op != "call_function"
                or tensor_arg.target
                != exir_ops.edge.quantized_decomposed.dequantize_per_tensor.default
            ):

                matched_quantized_cat = False
                break

            # make sure the input qparams for each input tensor in the concat list
            # matches the output qparams
            current_input_qparams = _get_qparams(tensor_arg)
            if not current_input_qparams == output_qparams:
                matched_quantized_cat = False
                break

        if not matched_quantized_cat:
            continue

        # now we matched a pattern for quantized cat, e.g.
        # input1 (int8) -> dq1 -> cat -> q -> following_op
        # input2 (int8) -> dq2 -/

        # remove dq for inputs and q for output and run cat on the int8 input directly
        # input1 (int8) -> cat -> following_op
        # input2 (int8) -/

        # reroute the input of dq to the cat node
        for tensor_arg in tensor_args:
            maybe_cat.replace_input_with(tensor_arg, tensor_arg.args[0])

        # remove q for output
        qnode.replace_all_uses_with(maybe_cat)
        model.graph.erase_node(qnode)


class QuantFusionPass(ExportPass):
    def __init__(self, _fix_node_meta_val=False):
        super().__init__()
        # TODO This pass violate IR spec because it produces a graph missing node.meta['val']
        self._fix_node_meta_val = _fix_node_meta_val

    def call(self, graph_module: GraphModule) -> PassResult:
        """Lower a quantized reference model (with reference quantized operator patterns)
        to executorch backend, that has a canonical set of quantized operators. This pass
        is a backend pass and should be applied on top of Edge dialect, ideally in
        `ExecutorchBackendConfig.passes`. See `test_quant_fusion_pass.py` for an example.
        """
        # linear, conv2d
        # dynamic_linear
        # add
        # batchnorm2d, relu, adaptive_avg_pool2d, reshape, squeeze, permute
        for (
            pattern,
            replacement,
            match_filters,
        ) in get_quant_patterns_and_replacements():
            subgraph_rewriter.replace_pattern_with_filters(
                graph_module, pattern, replacement, match_filters
            )

        _fuse_quantized_cat(graph_module)
        if self._fix_node_meta_val:
            for n in graph_module.graph.nodes:
                if n.op == "call_function" and "val" not in n.meta:
                    args, kwargs = pytree.tree_map_only(
                        torch.fx.Node, lambda x: x.meta["val"], (n.args, n.kwargs)
                    )
                    n.meta["val"] = n.target(*args, **kwargs)
        graph_module.graph.lint()
        graph_module.graph.eliminate_dead_code()
        return PassResult(graph_module, True)
