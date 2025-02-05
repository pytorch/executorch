# Copyright 2024-2025 Arm Limited and/or its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import copy

from typing import cast, Dict, Set, Tuple

from executorch.backends.arm.tosa_quant_utils import QuantArgs

from executorch.exir.dialects._ops import ops as exir_ops
from executorch.exir.dialects.edge._ops import EdgeOpOverload

from executorch.exir.pass_base import (
    Argument,
    ExportPass,
    NodeMetadata,
    PassResult,
    ProxyValue,
)
from torch.fx import GraphModule, Node

q_op: EdgeOpOverload = exir_ops.edge.quantized_decomposed.quantize_per_tensor.default
dq_op: EdgeOpOverload = exir_ops.edge.quantized_decomposed.dequantize_per_tensor.default


def get_input_qparams(node: Node) -> dict[int, QuantArgs]:
    """
    Get the input quantization parameters from a node, set by the 'FoldAndAnnotateQParamsPass'.
    Raises a ValueError if the node doesn't have any parameters set.
    """
    if "input_qparams" not in node.meta.keys():
        raise ValueError(
            f"No input quantization parameter found in node {node}\n"
            f"original_aten={node.meta.get('original_aten', 'None')}"
        )
    input_qparams = cast(dict[int, QuantArgs], node.meta["input_qparams"])
    if len(input_qparams) == 0:
        raise ValueError(
            f"No input quantization parameter found in node {node}\n"
            f"original_aten={node.meta.get('original_aten', 'None')}"
        )
    return input_qparams


def get_output_qparams(node: Node) -> dict[int, QuantArgs]:
    """
    Get the output quantization parameters from a node, set by the 'FoldAndAnnotateQParamsPass'.
    Raises a ValueError if the node doesn't have any parameters set.
    """
    if "output_qparams" not in node.meta.keys():
        raise ValueError(
            f"No output quantization parameter found in node {node}\n"
            f"original_aten={node.meta.get('original_aten', 'None')}"
        )
    output_qparams = cast(dict[int, QuantArgs], node.meta["output_qparams"])
    if len(output_qparams) == 0:
        raise ValueError(
            f"No output quantization parameter found in node {node}\n"
            f"original_aten={node.meta.get('original_aten', 'None')}"
        )
    return output_qparams


class FoldAndAnnotateQParamsPass(ExportPass):
    """
    A pass that walks the graph and removes any DQ and Q nodes before and after the target
     node.
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

    def __init__(self) -> None:
        super().__init__()

    def fold_and_annotate_arg(
        self, graph_module: GraphModule, node: Node, arg_list: list[Node], i: int
    ) -> None:
        input_qparams = None
        nodes_to_remove = set()
        for arg in arg_list:
            if not isinstance(arg, Node):
                return

            arg_quant_params = None
            if arg.target == dq_op:
                arg_quant_params = QuantArgs.from_operator(arg.target, arg.args)
                # add arg to nodes_to_remove to fold the dq-node
                nodes_to_remove.add(arg)
            if input_qparams is not None and input_qparams != arg_quant_params:
                # Two args are quantized differently
                raise RuntimeError("Input qparams does not match!")
            input_qparams = arg_quant_params
        if input_qparams is not None:
            node.meta["input_qparams"][i] = input_qparams
            for n in nodes_to_remove:
                assert n.target == dq_op
                n.replace_all_uses_with(n.args[0])  # type: ignore[arg-type]
                graph_module.graph.erase_node(n)

    def call(self, graph_module: GraphModule) -> PassResult:

        # Loop over the graph nodes and find any node in the 'targeted_ops' list.
        for n in graph_module.graph.nodes:
            n = cast(Node, n)
            if n.op != "call_function":
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
                if isinstance(arg, list):
                    self.fold_and_annotate_arg(graph_module, n, arg, i)

                elif isinstance(arg, Node):
                    self.fold_and_annotate_arg(graph_module, n, [arg], i)

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


class QuantizeOperatorArguments(ExportPass):
    """
    This pass makes sure that the arguments to full.default and clamp.default are quantized correctly.
    More specifically, this pass:
        - Makes sure the fill_value for full.default is quantized. This pass needs to be run before
        the folding pass above to make sure that the retraced output of the full.default op is
        the right dtype.
        - Makes sure the min and max values to clamp.default are quantized, if it's a quantized operator.
    """

    def call(self, graph_module: GraphModule) -> PassResult:
        modified = False
        # Loop over the graph nodes and find full.default nodes.
        for n in graph_module.graph.nodes:
            n = cast(Node, n)
            if n.target not in {
                exir_ops.edge.aten.clamp.default,
                exir_ops.edge.aten.full.default,
            }:
                continue

            # Make sure we have a quantized operator
            user = list(n.users)[0]
            if user.target != q_op:
                continue

            qargs = QuantArgs.from_operator(user.target, user.args)

            if n.target == exir_ops.edge.aten.full.default:
                if "dtype" not in n.kwargs.keys() or n.kwargs["dtype"] != qargs.dtype:
                    # replace the node arg with a quantized dito and also set dtype
                    # to get the right output according to the Edge IR specification:
                    # exir/dialects/edge/edge.yaml:3596
                    quantized_full_value = qargs.quantize_value(n.args[1]).item()
                    n.update_arg(1, quantized_full_value)
                    n.update_kwarg("dtype", qargs.dtype)
                    modified = True
            elif n.target == exir_ops.edge.aten.clamp.default:
                # Quantize the min and max arguments of clamp, if they are not None
                min_val = n.args[1]
                max_val = None if len(n.args) <= 2 else n.args[2]

                if min_val is not None:
                    quantized_min_val = qargs.quantize_value(min_val).item()
                    n.update_arg(1, quantized_min_val)

                if max_val is not None:
                    quantized_max_val = qargs.quantize_value(max_val).item()
                    n.update_arg(2, quantized_max_val)

                modified = True

        return PassResult(graph_module, modified)


class RetraceFoldedDtypesPass(ExportPass):
    """
    FoldAndAnnotateQParamsPass folds dq and q nodes. When the graph is retraced
    some operators are retraced to types that cannot be handled by TOSA. One
    such example is sum.dim_IntList:
        q (int8) -> dq (fp32) -> sum (fp32) -> q (int8) ...
    After folding it becomes:
        q (int8)              -> sum (int64) ->         ...
    This pass changes types of ops in self.targeted_ops, such as sum, so that
    the output type of that matches the type of the output_qparams.
    """

    targeted_ops: Set[EdgeOpOverload] = {
        exir_ops.edge.aten.sum.dim_IntList,
    }

    def call_operator(
        self,
        op,  # pyre-ignore
        args: Tuple[Argument, ...],
        kwargs: Dict[str, Argument],
        meta: NodeMetadata,
    ) -> ProxyValue:
        if op not in self.targeted_ops:
            return super().call_operator(op, args, kwargs, meta)

        node_kwargs = kwargs.copy()
        output_qparams = meta["output_qparams"]
        if len(output_qparams) == 0:
            return super().call_operator(op, args, kwargs, meta)

        output_dtype = output_qparams[0].dtype
        node_kwargs["dtype"] = output_dtype
        return super().call_operator(op, args, node_kwargs, meta)
