# Copyright 2024 Arm Limited and/or its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import copy

from typing import Callable, cast, Iterable

from executorch.backends.arm.tosa_quant_utils import QuantArgs

from executorch.exir.dialects._ops import ops as exir_ops

from executorch.exir.pass_base import ExportPass, PassResult
from torch.fx import GraphModule, Node


class FoldAndAnnotateQParamsPass(ExportPass):
    """
    A pass that walks the graph and removes any DQ and Q nodes before and after the target
     node in the supplied list of operators.
     The quantization parameters from the DQ/Q nodes are stored as meta values to be
     accessible for later lowering and serialization passes.
     The assumption is that the quantization annotatation adds DQ nodes for all tensor
     inputs to the target one Q node to the output.

     Example ('executorch_exir_dialects_edge__ops_' prefix removed from operators for readability):

        x_q: "i8[5]" = quantized_decomposed_quantize_per_tensor_default(x, 0.05487706884741783, -128, -128, 127, torch.int8)

        x_dq: "f32[5]" = quantized_decomposed_dequantize_per_tensor_default(x_q, 0.05487706884741783, -128, -128, 127, torch.int8)
        aten_add_tensor: "f32[5]" = ops_aten_add_Tensor(x_dq, x_dq)
        aten_add_tensor_q: "i8[5]" = quantized_decomposed_quantize_per_tensor_default(aten_add_tensor, 0.05487706884741783, -128, -128, 127, torch.int8)

        output_dq: "f32[5]" = quantized_decomposed_dequantize_per_tensor_default(aten_add_tensor_q, 0.05487706884741783, -128, -128, 127, torch.int8)

     Becomes:
        x_q: "i8[5]" = quantized_decomposed_quantize_per_tensor_default(x, 0.05487706884741783, -128, -128, 127, torch.int8)

        aten_add_tensor: "i8[5]" = aten_add_Tensor(x_q, x_q)

        output_dq: "f32[5]" = quantized_decomposed_dequantize_per_tensor_default(aten_add_tensor_q, 0.05487706884741783, -128, -128, 127, torch.int8)

    The quantization parameters for x_dq and aten_add_tensor_q are store in meta for the aten_add_tensor node.

    """

    def __init__(self, targeted_ops: Iterable[Callable]):
        super().__init__()
        self.targeted_ops = targeted_ops

    def call(self, graph_module: GraphModule) -> PassResult:
        q_op = exir_ops.edge.quantized_decomposed.quantize_per_tensor.default
        dq_op = exir_ops.edge.quantized_decomposed.dequantize_per_tensor.default

        # Loop over the graph nodes and find any node in the 'targeted_ops' list.
        for n in graph_module.graph.nodes:
            n = cast(Node, n)
            if n.op != "call_function" or n.target not in self.targeted_ops:
                continue

            # Make sure we haven't already set qparams meta information on the node
            assert "input_qparams" not in n.meta.keys()
            assert "output_qparams" not in n.meta.keys()

            # for the inputs and outputs search the graph for quantization info and
            # store the information in a dict with order of the _tensor_ inputs as key,
            # ignoring any other arguments to the target node.
            n.meta["input_qparams"] = {}
            n.meta["output_qparams"] = {}
            for i, arg in enumerate(n.args):
                if not isinstance(arg, Node):
                    continue
                if arg.target != dq_op:
                    continue

                # arg.target for argument i is a dequant node, extract the information
                n.meta["input_qparams"][i] = QuantArgs.from_operator(
                    arg.target, arg.args
                )

                # arg.args[0] is the tensor input, replace the input usage
                n.replace_input_with(arg, arg.args[0])
                graph_module.graph.erase_node(arg)

            # Copy the users, since we are modifying it.
            users_copy = copy.copy(n.users)
            for i, user in enumerate(users_copy):
                if user.target != q_op:
                    continue

                # quantization node found here, store the quantization parameters in meta value
                n.meta["output_qparams"][i] = QuantArgs.from_operator(
                    user.target, user.args
                )

                user.replace_all_uses_with(n)
                graph_module.graph.erase_node(user)

        # retrace the graph to update the fake tensor types
        graph_module = super().call(graph_module).graph_module

        graph_module.recompile()
        return PassResult(graph_module, True)
