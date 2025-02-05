# Copyright 2025 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe

import torch
from executorch.backends.arm._passes.arm_pass_utils import (
    create_constant_placeholder,
    delete_constant_placeholder,
)
from executorch.exir import ExportedProgram
from executorch.exir.dialects._ops import ops as exir_ops
from executorch.exir.pass_base import ExportPass, PassResult
from torch._export.utils import get_buffer, get_param
from torch.export.graph_signature import InputKind
from torch.fx import Node
from torch.nn.utils.fusion import fuse_conv_bn_weights


class FuseBatchnorm2DPass(ExportPass):
    """Fuses the pattern convolution -> batchnorm by updating
    the weights and bias of the convolution and removing the batchnorm.
    """

    def __init__(self, exported_program: ExportedProgram):
        self.exported_program = exported_program
        super().__init__()

    def is_fuseable_conv_bn(self, node: Node) -> bool:
        """Returns True if node is a batchnorm that can be fused into
        a parent convolution."""
        if node.op != "call_function":
            return False
        if node.target not in (
            exir_ops.edge.aten._native_batch_norm_legit,
            exir_ops.edge.aten._native_batch_norm_legit_no_training.default,
        ):
            return False
        conv = node.all_input_nodes[0]
        if conv.target != exir_ops.edge.aten.convolution.default:
            return False
        # Batchnorm users are getitem, we can only handle those that get first element.
        for user in node.users:
            get_index = user.args[1]
            if get_index != 0:
                return False
        # Since we change the output of the conv, fuse only if it has single user.
        if len(conv.users) > 1:
            return False
        return True

    def get_bias_name(self, conv_weight_node: Node, conv_bias_node: Node) -> str:
        if conv_bias_node:
            return conv_bias_node.name + "_fused_bn"
        elif "weight" in conv_weight_node.name:
            return conv_weight_node.name.replace("weight", "bias") + "_fused_bn"
        else:
            return conv_weight_node.name + "_bias_fused_bn"

    def call(self, graph_module: torch.fx.GraphModule) -> PassResult:  # noqa: C901
        modified = False
        constant_placeholders_to_delete = set()
        for node in graph_module.graph.nodes:
            if not self.is_fuseable_conv_bn(node):
                continue

            def get_param_or_none(arg) -> torch.nn.Parameter | None:
                """get_param but check if arg is none first."""
                return (
                    get_param(self.exported_program, arg) if arg is not None else None
                )

            # Get weight, bias, mean, var and epsilon from the batchnorm
            bn_node = node
            conv, bn_weight_node, bn_bias_node, bn_mean_node, bn_var_node = (
                bn_node.args[0:5]
            )
            bn_weight_tensor = get_param_or_none(bn_weight_node)
            bn_bias_tensor = get_param_or_none(bn_bias_node)
            bn_mean_tensor = get_buffer(self.exported_program, bn_mean_node)
            bn_var_tensor = get_buffer(self.exported_program, bn_var_node)
            if bn_mean_tensor is None or bn_var_tensor is None:
                raise ValueError(
                    "Parameters running_mean and running_var of batchnorm can't be None."
                )
            epsilon = bn_node.args[-1]

            # Get weight and bias from conv
            conv_weight_node, conv_bias_node = conv.args[1:3]
            conv_weight_tensor = get_param(self.exported_program, conv_weight_node)
            conv_bias_tensor = get_param_or_none(conv_bias_node)
            if conv_weight_tensor is None:
                raise ValueError("Parameter weight of convolution can't be None.")

            # Compute conv parameters folded with batchnorm
            fused_conv_weight, fused_conv_bias = fuse_conv_bn_weights(
                conv_weight_tensor,
                conv_bias_tensor,
                bn_mean_tensor,
                bn_var_tensor,
                epsilon,
                bn_weight_tensor,
                bn_bias_tensor,
            )

            # Create fused weights and bias to conv and replace conv args
            with graph_module.graph.inserting_before(conv_weight_node):
                fused_conv_weight_node = create_constant_placeholder(
                    exp_program=self.exported_program,
                    graph=graph_module.graph,
                    kind=InputKind.PARAMETER,
                    name=conv_weight_node.name + "_fused_bn",
                    data=fused_conv_weight,
                )

                if fused_conv_bias is not None:
                    fused_conv_bias_node = create_constant_placeholder(
                        exp_program=self.exported_program,
                        graph=graph_module.graph,
                        kind=InputKind.PARAMETER,
                        name=self.get_bias_name(conv_weight_node, conv_bias_node),
                        data=fused_conv_bias,
                    )
                else:
                    fused_conv_bias_node = None

                conv.args = (
                    conv.args[0],
                    fused_conv_weight_node,
                    fused_conv_bias_node,
                    *conv.args[3:],
                )

            # Erasing batch-norm nodes is handled by dead-code elimination. After that we may remove their constant placeholder inputs
            for user in bn_node.users:
                user.replace_all_uses_with(conv)

            constant_placeholders_to_delete.update(
                [
                    bn_weight_node,
                    bn_bias_node,
                    bn_mean_node,
                    bn_var_node,
                    conv_weight_node,
                    conv_bias_node,
                ]
            )
            modified = True

        if modified:
            graph_module.graph.eliminate_dead_code()
            for constant_placeholder in constant_placeholders_to_delete:
                if (constant_placeholder is not None) and (
                    len(constant_placeholder.users) == 0
                ):
                    delete_constant_placeholder(
                        self.exported_program, constant_placeholder
                    )

            graph_module.recompile()
            graph_module = super().call(graph_module).graph_module

        return PassResult(graph_module=graph_module, modified=modified)
