# Copyright 2024 Arm Limited and/or its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import copy

from typing import cast, Iterable

from executorch.backends.arm.tosa_quant_utils import QuantArgs

from executorch.exir.dialects._ops import ops as exir_ops
from executorch.exir.dialects.edge._ops import EdgeOpOverload

from executorch.exir.pass_base import ExportPass, PassResult
from torch.fx import GraphModule, Node

q_op: EdgeOpOverload = exir_ops.edge.quantized_decomposed.quantize_per_tensor.default
dq_op: EdgeOpOverload = exir_ops.edge.quantized_decomposed.dequantize_per_tensor.default


def get_input_qparams(node: Node) -> dict[int, QuantArgs]:
    """
    Get the input quantization parameters from a node, set by the 'FoldAndAnnotateQParamsPass'.
    Raises a ValueError if the node doesn't have any parameters set.
    """
    if "input_qparams" not in node.meta.keys():
        raise ValueError(f"No input quantization parameter found in node {node}")
    input_qparams = cast(dict[int, QuantArgs], node.meta["input_qparams"])
    if len(input_qparams) == 0:
        raise ValueError(f"No input quantization parameter found in node {node}")
    return input_qparams


def get_output_qparams(node: Node) -> dict[int, QuantArgs]:
    """
    Get the output quantization parameters from a node, set by the 'FoldAndAnnotateQParamsPass'.
    Raises a ValueError if the node doesn't have any parameters set.
    """
    if "output_qparams" not in node.meta.keys():
        raise ValueError(f"No output quantization parameter found in node {node}")
    input_qparams = cast(dict[int, QuantArgs], node.meta["output_qparams"])
    if len(input_qparams) == 0:
        raise ValueError(f"No output quantization parameter found in node {node}")
    return input_qparams


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

    def __init__(self, targeted_ops: Iterable[EdgeOpOverload]) -> None:
        super().__init__()
        self.targeted_ops = targeted_ops

    def call(self, graph_module: GraphModule) -> PassResult:

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

                # Make sure arg has requires_grad set to False
                # For parameters that are not quantized, sometimes (i.e. convolution)
                # the Parameter(FakeTensor(...)) has requires_grad set to True, which
                # causes the retracing of the graph to fail with:
                #
                # E       RuntimeError: isDifferentiableType(variable.scalar_type()) INTERNAL ASSERT FAILED at "/Users/runner/work/pytorch/pytorch/pytorch/torch/csrc/autograd/functions/utils.h":74, please report a bug to PyTorch.
                # E
                # E       While executing %aten_convolution_default : [num_users=1] = call_function[target=executorch.exir.dialects.edge._ops.aten.convolution.default](args = (%quantized_decomposed_quantize_per_tensor_default, %b__frozen_param0, %p__param_constant1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
                # E       Original traceback:
                # E         File "/Users/perast01/src/executorch/backends/arm/test/ops/test_conv2d.py", line 110, in forward
                # E           x = conv(x)
                #
                if arg.op == "placeholder":
                    arg.meta["val"].requires_grad = False

                if arg.target != dq_op:
                    continue

                # arg.target for argument i is a dequant node, extract the information
                n.meta["input_qparams"][i] = QuantArgs.from_operator(
                    arg.target, arg.args
                )

                # arg.args[0] is the tensor input, replace the input usage
                tensor_input = cast(Node, arg.args[0])
                n.replace_input_with(arg, tensor_input)
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


class QuantizeFullArgument(ExportPass):
    """
    Make sure the fill_value for full.default is quantized. This pass needs to be run before
    the folding pass above to make sure that the retraced output of the full.default op is
    the right dtype.
    """

    def call(self, graph_module: GraphModule) -> PassResult:
        modified = False
        # Loop over the graph nodes and find any node in the 'targeted_ops' list.
        for n in graph_module.graph.nodes:
            n = cast(Node, n)
            if n.target != exir_ops.edge.aten.full.default:
                continue

            # Make sure we have a quantized operator
            user = list(n.users)[0]
            if user.target != q_op:
                continue

            qargs = QuantArgs.from_operator(user.target, user.args)
            if "dtype" not in n.kwargs.keys() or n.kwargs["dtype"] != qargs.dtype:
                # replace the node arg with a quantized dito and also set dtype
                # to get the right output according to the Edge IR specification:
                # exir/dialects/edge/edge.yaml:3596
                quantized_full_value = qargs.quantize_value(n.args[1]).item()
                n.update_arg(1, quantized_full_value)
                n.update_kwarg("dtype", qargs.dtype)
                modified = True

        return PassResult(graph_module, modified)
