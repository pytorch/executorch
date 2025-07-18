# Copyright 2025 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe

from typing import Set, Type

import torch
from executorch.backends.arm._passes.arm_pass_utils import (
    create_node,
    get_first_fake_tensor,
)
from executorch.backends.transforms.utils import (
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

    _passes_required_after: Set[Type[ExportPass]] = set()

    def __init__(self, exported_program: ExportedProgram):
        self.exported_program = exported_program
        super().__init__()

    def get_bias_name(self, weight_node: Node, bias_node: Node | None) -> str:
        if bias_node:
            return bias_node.name + "_fused_bn"
        elif "weight" in weight_node.name:
            return weight_node.name.replace("weight", "bias") + "_fused_bn"
        else:
            return weight_node.name + "_bias_fused_bn"

    def call(self, graph_module: torch.fx.GraphModule) -> PassResult:  # noqa: C901
        modified = False
        constant_placeholders_to_delete = set()
        for node in graph_module.graph.nodes:
            if node.op != "call_function":
                continue
            if (
                node.target
                != exir_ops.edge.aten._native_batch_norm_legit_no_training.default
            ):
                continue

            # Get data from batchnorm
            input_node = node.all_input_nodes[0]
            is_single_user = len(input_node.users) == 1
            bn_weight_node, bn_bias_node, bn_mean_node, bn_var_node = node.args[1:5]
            assert bn_mean_node is not None, "Batchnorm mean node cannot be None."
            assert bn_var_node is not None, "Batchnorm var node cannot be None."

            epsilon = node.args[-1]

            bn_weight_tensor = (
                get_param(self.exported_program, bn_weight_node)
                if bn_weight_node is not None
                else None
            )
            bn_bias_tensor = (
                get_param(self.exported_program, bn_bias_node)
                if bn_bias_node is not None
                else None
            )

            bn_mean_tensor = torch.Tensor(
                get_buffer(self.exported_program, bn_mean_node)
            )
            bn_var_tensor = torch.Tensor(get_buffer(self.exported_program, bn_var_node))

            if (
                input_node.target != exir_ops.edge.aten.convolution.default
                or not is_single_user
            ):
                # Insert a transparent conv2d before bn to fuse with if none is present.
                shape = get_first_fake_tensor(node)
                if len(shape.size()) == 3:
                    input_weight_tensor = torch.ones((1, 1, 1))
                    stride = [1]
                    padding = [0]
                    dilation = [1]
                    output_padding = [0]
                else:
                    input_weight_tensor = torch.ones((1, 1, 1, 1))
                    stride = [1, 1]
                    padding = [0, 0]
                    dilation = [1, 1]
                    output_padding = [0, 0]

                with graph_module.graph.inserting_before(bn_weight_node):
                    input_weight_node = create_constant_placeholder(
                        exp_program=self.exported_program,
                        graph=graph_module.graph,
                        kind=InputKind.PARAMETER,
                        name=node.name + "_conv_weight",
                        data=input_weight_tensor,
                    )

                    input_bias_tensor = input_bias_node = None

                with graph_module.graph.inserting_before(node):
                    channels = bn_mean_tensor.size(0)
                    conv_args = (
                        input_node,
                        input_weight_node,
                        input_bias_node,
                        stride,
                        padding,
                        dilation,
                        False,  # Transposed
                        output_padding,
                        channels,
                    )
                    new_input_node = create_node(
                        graph_module.graph,
                        exir_ops.edge.aten.convolution.default,
                        conv_args,
                    )
                    node.replace_input_with(input_node, new_input_node)
                    input_node = new_input_node
            else:
                input_weight_node, input_bias_node = input_node.args[1:3]
                assert (
                    isinstance(input_weight_node, Node)
                    and input_weight_node.op == "placeholder"
                ), "Parameter weight of convolution must be a placeholder"
                assert (input_bias_node is None) or (
                    isinstance(input_weight_node, Node)
                    and input_weight_node.op == "placeholder"
                ), "Parameter bias of convolution must be a placeholder or None"

                input_weight_tensor = torch.Tensor(
                    get_param(self.exported_program, input_weight_node)
                )

                input_bias_tensor = (
                    get_param(self.exported_program, input_bias_node)
                    if input_bias_node is not None
                    else None
                )

            # Fuse bn weights/bias with input weights/bias
            fused_weight, fused_bias = fuse_conv_bn_weights(
                input_weight_tensor,
                input_bias_tensor,
                bn_mean_tensor,
                bn_var_tensor,
                epsilon,
                bn_weight_tensor,
                bn_bias_tensor,
            )

            # Create fused weights and bias to conv and replace conv args
            with graph_module.graph.inserting_before(input_weight_node):
                fused_conv_weight_node = create_constant_placeholder(
                    exp_program=self.exported_program,
                    graph=graph_module.graph,
                    kind=InputKind.PARAMETER,
                    name=input_weight_node.name + "_fused_bn",
                    data=fused_weight,
                )

                if fused_bias is not None:
                    fused_input_bias_node = create_constant_placeholder(
                        exp_program=self.exported_program,
                        graph=graph_module.graph,
                        kind=InputKind.PARAMETER,
                        name=self.get_bias_name(input_weight_node, input_bias_node),
                        data=fused_bias,
                    )
                else:
                    fused_input_bias_node = None

                input_node.args = (
                    input_node.args[0],
                    fused_conv_weight_node,
                    fused_input_bias_node,
                    *input_node.args[3:],
                )

            # Erasing batch-norm nodes is handled by dead-code elimination. After that we may remove their constant placeholder inputs
            for user in node.users:
                user.replace_all_uses_with(input_node)

            constant_placeholders_to_delete.update(
                [
                    bn_weight_node,
                    bn_bias_node,
                    bn_mean_node,
                    bn_var_node,
                    input_weight_node,
                    input_bias_node,
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
