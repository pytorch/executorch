# Copyright 2025-2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


from typing import Set, Type

import torch
from executorch.backends.arm._passes import ArmPass
from executorch.backends.arm._passes.arm_pass_utils import (
    create_node,
    get_first_fake_tensor,
)
from executorch.backends.arm._passes.decompose_grouped_conv_pass import (
    DecomposeGroupedConvPass,
)
from executorch.backends.arm.common.debug import get_node_debug_info
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


class FuseBatchNorm2dPass(ArmPass):
    """Fuse convolution followed by BatchNorm.

    Update the convolution weights and bias and remove the BatchNorm operation.

    """

    _passes_required_after: Set[Type[ExportPass]] = {DecomposeGroupedConvPass}

    def __init__(self, exported_program: ExportedProgram, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.exported_program = exported_program

    def get_bias_name(self, weight_node: Node, bias_node: Node | None) -> str:
        if bias_node:
            return bias_node.name + "_fused_bn"
        elif "weight" in weight_node.name:
            return weight_node.name.replace("weight", "bias") + "_fused_bn"
        else:
            return weight_node.name + "_bias_fused_bn"

    @staticmethod
    def _fuse_grouped_transposed_conv_bn_weights(
        conv_weight: torch.Tensor,
        conv_bias: torch.Tensor | None,
        bn_mean: torch.Tensor,
        bn_var: torch.Tensor,
        bn_epsilon: float,
        bn_weight: torch.Tensor | None,
        bn_bias: torch.Tensor | None,
        groups: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Fuse BatchNorm into grouped transposed-convolution parameters.

        This helper runs before ``DecomposeGroupedConvPass`` and transforms::

            grouped ConvTranspose -> BatchNorm

        into a grouped ConvTranspose with fused weights and bias. A transposed
        convolution weight has layout ``[Cin, Cout/groups, ...]``. The weight
        is split on its input-channel dimension, while the bias and BatchNorm
        parameters are split on their output-channel dimension. Each group is
        fused independently before the original grouped layout is restored.

        Args:
            conv_weight (torch.Tensor): Grouped transposed-convolution weight.
            conv_bias (torch.Tensor | None): Convolution bias.
            bn_mean (torch.Tensor): BatchNorm running mean.
            bn_var (torch.Tensor): BatchNorm running variance.
            bn_epsilon (float): BatchNorm numerical-stability constant.
            bn_weight (torch.Tensor | None): BatchNorm weight.
            bn_bias (torch.Tensor | None): BatchNorm bias.
            groups (int): Number of convolution groups.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: Fused weight and bias in the
                original grouped layout.

        Raises:
            RuntimeError: If the grouped channel dimensions are inconsistent.

        """
        if conv_weight.size(0) % groups != 0 or bn_mean.numel() % groups != 0:
            raise RuntimeError("Grouped transposed convolution has invalid channels")

        input_channels_per_group = conv_weight.size(0) // groups
        output_channels_per_group = bn_mean.numel() // groups
        if conv_weight.size(1) != output_channels_per_group:
            raise RuntimeError("BatchNorm channels do not match convolution output")

        fused_weights: list[torch.Tensor] = []
        fused_biases: list[torch.Tensor] = []
        for group in range(groups):
            input_start = group * input_channels_per_group
            input_end = input_start + input_channels_per_group
            output_start = group * output_channels_per_group
            output_end = output_start + output_channels_per_group
            output_slice = slice(output_start, output_end)

            fused_weight, fused_bias = fuse_conv_bn_weights(
                conv_weight[input_start:input_end],
                conv_bias[output_slice] if conv_bias is not None else None,
                bn_mean[output_slice],
                bn_var[output_slice],
                bn_epsilon,
                bn_weight[output_slice] if bn_weight is not None else None,
                bn_bias[output_slice] if bn_bias is not None else None,
                transpose=True,
            )
            fused_weights.append(fused_weight)
            fused_biases.append(fused_bias)

        return torch.cat(fused_weights, dim=0), torch.cat(fused_biases, dim=0)

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
            if get_first_fake_tensor(node).dtype == torch.bfloat16:
                # Don't fuse if the data type is bfloat16, as the fused weights may
                # not be accurate enough and cause significant accuracy drop.
                continue

            # Get data from batchnorm
            input_node = node.all_input_nodes[0]
            is_single_user = len(input_node.users) == 1
            bn_weight_node, bn_bias_node, bn_mean_node, bn_var_node = node.args[1:5]
            if bn_mean_node is None:
                raise RuntimeError(
                    "BatchNorm mean buffer missing for node: "
                    f"{get_node_debug_info(node, graph_module)}"
                )
            if bn_var_node is None:
                raise RuntimeError(
                    "BatchNorm variance buffer missing for node: "
                    f"{get_node_debug_info(node, graph_module)}"
                )

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
                input_dtype = get_first_fake_tensor(input_node).dtype
                if len(shape.size()) == 3:
                    input_weight_tensor = torch.ones((1, 1, 1), dtype=input_dtype)
                    stride = [1]
                    padding = [0]
                    dilation = [1]
                    output_padding = [0]
                else:
                    input_weight_tensor = torch.ones((1, 1, 1, 1), dtype=input_dtype)
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
                if not (
                    isinstance(input_weight_node, Node)
                    and input_weight_node.op == "placeholder"
                ):
                    raise RuntimeError(
                        "Parameter weight of convolution must be a placeholder"
                    )
                if not (
                    (input_bias_node is None)
                    or (
                        isinstance(input_bias_node, Node)
                        and input_bias_node.op == "placeholder"
                    )
                ):
                    raise RuntimeError(
                        "Parameter bias of convolution must be a placeholder or None"
                    )

                input_weight_tensor = torch.Tensor(
                    get_param(self.exported_program, input_weight_node)
                )

                input_bias_tensor = (
                    get_param(self.exported_program, input_bias_node)
                    if input_bias_node is not None
                    else None
                )

            # Fuse bn weights/bias with input weights/bias
            transposed = bool(input_node.args[6])
            groups = int(input_node.args[8])
            if transposed and groups > 1:
                fused_weight, fused_bias = (
                    self._fuse_grouped_transposed_conv_bn_weights(
                        input_weight_tensor,
                        input_bias_tensor,
                        bn_mean_tensor,
                        bn_var_tensor,
                        epsilon,
                        bn_weight_tensor,
                        bn_bias_tensor,
                        groups,
                    )
                )
            else:
                fused_weight, fused_bias = fuse_conv_bn_weights(
                    input_weight_tensor,
                    input_bias_tensor,
                    bn_mean_tensor,
                    bn_var_tensor,
                    epsilon,
                    bn_weight_tensor,
                    bn_bias_tensor,
                    transpose=transposed,
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
