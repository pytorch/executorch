# Copyright 2024-2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


import copy

from typing import cast, Optional, Set, Type

import torch
from executorch.backends.arm._passes import ArmPass
from executorch.backends.arm._passes.arm_pass_utils import (
    get_param_tensor,
    is_param_node,
    set_node_arg,
)
from executorch.backends.arm._passes.fuse_constant_ops_pass import (
    ComputeConstantOpsAOTPass,
)
from executorch.backends.arm._passes.insert_table_ops import InsertTableOpsPass

from executorch.backends.arm._passes.quant_args import QuantArgs
from executorch.backends.arm._passes.remove_noop_pass import RemoveNoopPass
from executorch.backends.arm.common.annotation_meta import ArmAnnotationInfo
from executorch.backends.arm.constants import DQ_OPS, Q_OPS
from executorch.backends.arm.tosa.mapping import TosaSpecialDtype
from executorch.exir import ExportedProgram

from executorch.exir.dialects._ops import ops as exir_ops

from executorch.exir.pass_base import ExportPass, PassResult
from torch.fx import GraphModule, Node


def _get_special_dtype(qspec: QuantArgs) -> TosaSpecialDtype | None:
    if qspec.dtype == torch.int8:
        if qspec.qmax == 7 and qspec.qmin == -7:
            return TosaSpecialDtype.INT4
    return None


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


class FoldAndAnnotateQParamsPass(ArmPass):
    """
    A pass that walks the graph and removes any DQ and Q nodes before and after the target
     node.
     The quantization parameters from the DQ/Q nodes are stored as meta values to be
     accessible for later lowering and serialization passes.
     The assumption is that the quantization annotation adds DQ nodes for all tensor
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

    The quantization parameters for x_dq and aten_add_tensor_q are stored in meta for the aten_add_tensor node.

    """

    _passes_required_after: Set[Type[ExportPass]] = {
        InsertTableOpsPass,
        RemoveNoopPass,
    }

    def __init__(
        self, exported_program: Optional[ExportedProgram] = None, *args, **kwargs
    ) -> None:
        super().__init__(*args, **kwargs)
        self.exported_program = exported_program

    def fold_and_annotate_arg(
        self, graph_module: GraphModule, node: Node, arg_list: list[Node], i: int
    ) -> None:
        input_qparams = None
        nodes_to_remove = set()
        for arg in arg_list:
            if not isinstance(arg, Node):
                return

            arg_quant_params = None
            if arg.target in DQ_OPS:
                args = arg.args
                scales = args[1]
                if (
                    isinstance(args[1], Node)
                    and self.exported_program is not None
                    and is_param_node(self.exported_program, args[1])
                ):
                    scales = get_param_tensor(self.exported_program, args[1])
                zps = args[2]
                if (
                    isinstance(args[2], Node)
                    and self.exported_program is not None
                    and is_param_node(self.exported_program, args[2])
                ):
                    zps = get_param_tensor(self.exported_program, args[2])
                arg_quant_params = QuantArgs.from_operator(
                    arg.target, (args[0], scales, zps, *args[3:])
                )
                # add arg to nodes_to_remove to fold the dq-node
                nodes_to_remove.add(arg)
            if input_qparams is not None and input_qparams != arg_quant_params:
                # Two args are quantized differently
                raise RuntimeError("Input qparams do not match")
            input_qparams = arg_quant_params
        if input_qparams is not None:
            node.meta["input_qparams"][i] = input_qparams
            for n in nodes_to_remove:
                if n.target not in DQ_OPS:
                    raise RuntimeError(
                        f"Expected one of {DQ_OPS} dq_op, got {n.target}"
                    )

                node.replace_input_with(n, cast(Node, n.args[0]))
                if len(n.users) == 0:
                    graph_module.graph.erase_node(n)
            special_dtype = _get_special_dtype(input_qparams)
            if special_dtype:
                node.all_input_nodes[i].meta[
                    TosaSpecialDtype.meta_key()
                ] = special_dtype

    def _handle_control_flow_node(self, node: Node, graph_module: GraphModule):
        """Fold outmost quant nodes inside submodule.
        placeholders => qs => dqs => ... => qs => dqs => output
        becomes
        placeholders => dqs => ... => qs => output,
        With output_qparams meta in the placeholders, and input_qparams meta in the output node.
        """
        match node.target:
            case torch.ops.higher_order.cond:
                submodule_nodes = cast(list[Node], node.args[1:3])
                args = cast(list[Node], node.args[-1])
            case torch.ops.higher_order.while_loop:
                submodule_nodes = cast(list[Node], node.args[0:2])
                args = cast(list[Node], node.args[-2])
            case _:
                raise ValueError(f"Unhandled target {node.target}")
        submodules = (
            graph_module.get_submodule(str(submodule_node.target))
            for submodule_node in submodule_nodes
        )
        for submodule in submodules:
            submodule = cast(GraphModule, submodule)
            output_node = submodule.graph.output_node()
            output_node.meta["input_qparams"] = {}
            nodes_to_remove = []
            arg_id = 0
            for submodule_node in submodule.graph.nodes:
                # Remove initial q nodes and ending dq nodes in the module.
                submodule_node = cast(Node, submodule_node)
                if (
                    submodule_node.target in Q_OPS
                    and list(submodule_node.all_input_nodes)[0].op == "placeholder"
                ):
                    input_node = cast(Node, submodule_node.args[0])
                    input_node.meta["val"] = submodule_node.meta["val"]
                    quant_args = QuantArgs.from_operator(
                        submodule_node.target, submodule_node.args
                    )
                    input_node.meta["output_qparams"] = {0: quant_args}

                    submodule_node.replace_all_uses_with(input_node)
                    nodes_to_remove.append(submodule_node)
                if submodule_node.target in DQ_OPS:
                    has_non_output_user = False
                    for user in copy.copy(submodule_node.users):
                        if user.op != "output":
                            has_non_output_user = True
                        else:
                            input_node = cast(Node, submodule_node.args[0])
                            submodule_node.replace_all_uses_with(input_node)
                            arg_index = cast(list[Node], output_node.args[0]).index(
                                input_node
                            )
                            quant_args = QuantArgs.from_operator(
                                submodule_node.target, submodule_node.args
                            )
                            output_node.meta["input_qparams"][arg_index] = quant_args

                    # Remove dq node if it only has the output node as its user.
                    if not has_non_output_user:
                        nodes_to_remove.append(submodule_node)
                # Placeholders without users won't be retraced with correct dtype, do it manually.
                # Control flow node input is matched to placeholder nodes in the submodule by index.
                # This means it will break if another pass inserts a placeholder before this pass.
                if submodule_node.op == "placeholder":
                    if len(submodule_node.users) == 0:
                        submodule_node.meta["val"] = args[arg_id].meta["val"]
                    arg_id += 1
                    if arg_id > len(args):
                        raise RuntimeError(
                            "Submodule had more placeholders than calling node had inputs."
                            " This is probably due to a placeholder being inserted in a pass."
                        )
            for node_to_remove in nodes_to_remove:
                submodule.graph.erase_node(node_to_remove)
        return

    @staticmethod
    def is_foldable(node: Node) -> bool:
        if node.op != "call_function":
            return False
        # Don't fold chains of quant-ops into each other.
        if node.target in (*Q_OPS, *DQ_OPS):
            return False

        # Always fold q-dq into constant ops.
        if node.target in (
            exir_ops.edge.aten.full_like.default,
            *ComputeConstantOpsAOTPass.targeted_ops,
        ):
            return True

        # We should not fold q-dq nodes into non-quantized nodes.
        if not (
            ArmAnnotationInfo.CUSTOM_META_KEY in node.meta.get("custom", {})
            and ArmAnnotationInfo(
                node.meta["custom"][ArmAnnotationInfo.CUSTOM_META_KEY]
            ).quantized
        ):
            return False
        return True

    def call(self, graph_module: GraphModule) -> PassResult:  # noqa: C901

        # Loop over the graph nodes and find any node in the 'targeted_ops' list.
        for n in graph_module.graph.nodes:
            n = cast(Node, n)
            if not FoldAndAnnotateQParamsPass.is_foldable(n):
                continue

            # Make sure we haven't already set qparams meta information on the node
            if "input_qparams" in n.meta:
                raise RuntimeError(
                    f'Unexpected key "input_qparams" found in meta for node {n}. '
                    "input_qparams should not have been set at this point"
                )
            if "output_qparams" in n.meta:
                raise RuntimeError(
                    f'Unexpected key "output_qparams" found in meta for node {n}. '
                    "output_qparams should not have been set at this point"
                )

            # for the inputs and outputs search the graph for quantization info and
            # store the information in a dict with order of the _tensor_ inputs as key,
            # ignoring any other arguments to the target node.
            n.meta["input_qparams"] = {}
            n.meta["output_qparams"] = {}
            for i, arg in enumerate(n.args):
                if isinstance(arg, (list, tuple)):
                    self.fold_and_annotate_arg(graph_module, n, arg, i)  # type: ignore

                elif isinstance(arg, Node):
                    self.fold_and_annotate_arg(graph_module, n, [arg], i)

            # Copy the users, since we are modifying it.
            users_copy = copy.copy(n.users)
            for i, user in enumerate(users_copy):
                if user.target not in Q_OPS:
                    continue

                # quantization node found here, store the quantization parameters in meta value
                n.meta["output_qparams"][i] = QuantArgs.from_operator(
                    user.target, user.args
                )

                user.replace_all_uses_with(n)
                graph_module.graph.erase_node(user)

            # Some op(s) contain a "dtype" key in their node kwargs. Set this
            # to the type of output qparams.
            output_qparams = n.meta["output_qparams"]
            if (
                n.target in {exir_ops.edge.aten.sum.dim_IntList}
                and len(output_qparams) > 0
            ):
                output_dtype = output_qparams[0].dtype
                set_node_arg(n, "dtype", output_dtype)

            if n.target in (
                torch.ops.higher_order.cond,
                torch.ops.higher_order.while_loop,
            ):
                self._handle_control_flow_node(n, graph_module)

        # retrace the graph to update the fake tensor types
        graph_module = super().call(graph_module).graph_module

        graph_module.recompile()
        return PassResult(graph_module, True)


class QuantizeClampArgumentsPass(ArmPass):
    """
    This pass makes sure that the arguments to clamp.default are quantized correctly.
    More specifically, this pass:
        - Makes sure the min and max values to clamp.default are quantized, if it's a quantized operator.
    """

    _passes_required_after: Set[Type[ExportPass]] = set()

    def call(self, graph_module: GraphModule) -> PassResult:
        modified = False
        # Loop over the graph nodes and find full.default nodes.
        for n in graph_module.graph.nodes:
            n = cast(Node, n)
            if n.target not in {
                exir_ops.edge.aten.clamp.default,
            }:
                continue

            try:
                output_qparams = get_output_qparams(n)
            except ValueError:
                continue
            if len(output_qparams) == 0:
                continue

            # Qparams are stored per user index; use the first entry.
            qargs = next(iter(output_qparams.values()))

            if n.target == exir_ops.edge.aten.clamp.default:
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

        if modified:
            # Retrace to refresh fake tensor metadata after updating clamp min/max.
            graph_module = super().call(graph_module).graph_module
            graph_module.recompile()

        return PassResult(graph_module, modified)
