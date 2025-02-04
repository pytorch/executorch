# Copyright 2025 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch
from executorch.exir import ExportedProgram
from executorch.exir.dialects._ops import ops as exir_ops
from executorch.exir.pass_base import ExportPass, PassResult
from torch._export.utils import get_buffer, get_param
from torch.fx import Node
from torch.nn.utils.fusion import fuse_conv_bn_weights


class FuseBatchnorm2DPass(ExportPass):
    """Fuses the pattern convolution -> batchnorm by updating
    the weights and bias of the convolution and removing the batchnorm.
    """

    def __init__(self, exported_program: ExportedProgram):
        self.exported_program = exported_program
        super().__init__()

    def is_fuseable_conv_bn(self, node: Node):
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
        # For similar reasons, only fuse if conv parameters have single user.
        if len(conv.all_input_nodes[1].users) > 1:
            return False
        if len(conv.all_input_nodes) > 2 and len(conv.all_input_nodes[2].users) > 1:
            return False
        return True

    def call(self, graph_module: torch.fx.GraphModule) -> PassResult:  # noqa: C901
        modified = False
        for node in graph_module.graph.nodes:
            if not self.is_fuseable_conv_bn(node):
                continue

            def get_param_or_none(arg) -> torch.nn.Parameter | None:
                """get_param but check if arg is none first."""
                return (
                    get_param(self.exported_program, arg) if arg is not None else None
                )

            # Get weight, bias, mean, var and epsilon from the batchnorm
            bn = node
            conv, bn_weight_node, bn_bias_node, bn_mean_node, bn_var_node = bn.args[0:5]
            bn_weight = get_param_or_none(bn_weight_node)
            bn_bias = get_param_or_none(bn_bias_node)

            running_mean = get_buffer(self.exported_program, bn_mean_node)
            running_var = get_buffer(self.exported_program, bn_var_node)
            if running_mean is None or running_var is None:
                raise ValueError(
                    "Parameters running_mean and running_var of batchnorm can't be None."
                )
            epsilon = bn.args[-1]

            # Get weight and bias from conv
            conv_weight_node, conv_bias_node = conv.args[1:3]
            conv_weight = get_param(self.exported_program, conv_weight_node)
            conv_bias = get_param_or_none(conv_bias_node)
            if conv_weight is None:
                raise ValueError("Parameter weight of convolution can't be None.")

            # Compute conv parameters folded with batchnorm
            fused_conv_weight, fused_conv_bias = fuse_conv_bn_weights(
                conv_weight,
                conv_bias,
                running_mean,
                running_var,
                epsilon,
                bn_weight,
                bn_bias,
            )

            # Set the conv parameters to fused value
            def try_set_param(
                param_node: Node | None, param_value: torch.nn.Parameter
            ) -> bool:
                """set_param but check if param_node is None first. Return True if param was set successfully, otherwise False."""
                if param_node is not None:
                    param_name = (
                        self.exported_program.graph_signature.inputs_to_parameters[
                            param_node.name
                        ]
                    )
                    self.exported_program.state_dict[param_name] = param_value
                    return True
                return False

            try_set_param(conv_weight_node, fused_conv_weight)
            if not try_set_param(conv_bias_node, fused_conv_bias) and try_set_param(
                bn_bias_node, fused_conv_bias
            ):
                # Conv didn't have bias but batchnorm did, steal bias from batchnorm.
                conv_args = (*conv.args[0:2], bn_bias_node, *conv.args[3:])
                conv.args = conv_args

            # Erasing nodes is handled by dead-code elimination.
            for user in bn.users:
                user.replace_all_uses_with(conv)
            modified = True

        if modified:
            graph_module.graph.eliminate_dead_code()
            graph_module.recompile()
            graph_module = super().call(graph_module).graph_module
        return PassResult(graph_module=graph_module, modified=modified)
