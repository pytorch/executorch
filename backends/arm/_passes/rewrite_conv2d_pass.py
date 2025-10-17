# Copyright 2025 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


from typing import Set, Type

import torch
from executorch.backends.arm._passes import ArmPass

from executorch.backends.arm._passes.arm_pass_utils import (
    create_node,
    get_first_fake_tensor,
    get_param_tensor,
    is_buffer,
    is_param,
)
from executorch.backends.arm.constants import HWCM_ORDER, NHWC_INVERSE_ORDER
from executorch.backends.arm.tosa.mapping import TosaSpecialDtype
from executorch.backends.transforms.utils import create_constant_placeholder
from executorch.exir.dialects._ops import ops as exir_ops
from executorch.exir.pass_base import ExportPass, PassResult
from torch.export.graph_signature import InputKind


class RewriteConv2dPass(ArmPass):
    """Rewrites aten.convolution to tosa.CONV2D or tosa.DEPTHWISE_CONV2D."""

    def __init__(self, exported_program: torch.export.ExportedProgram):
        super().__init__()
        self.exported_program = exported_program

    _passes_required_after: Set[Type[ExportPass]] = set()

    # torch.nn.Conv2d does not require the result of
    # `(input + 2 * pad - dilation * (weight - 1) - 1) / stride`
    # to be an integer, but tosa currently strictly require this property.
    # This function adjusts the pad value to meet the requirement.
    def _adjust_pad_if_needed(
        self, input_len: int, input_weight: int, stride: int, pad: int, dilation: int
    ) -> int:
        """Adjust padding to satisfy TOSA's integer output-size requirement.

        Torch ``Conv2d`` does not require the result of
        ``(input + 2 * pad - dilation * (weight - 1) - 1) / stride`` to be an
        integer, but TOSA does. This helper reduces the provided padding so
        that the expression becomes divisible by ``stride``.

        Args:
            input_size (int): Spatial input size along the dimension (H or W).
            input_weight (int): Kernel size along the same dimension.
            stride (int): Stride along the same dimension.
            pad (int): Padding value to adjust (bottom or right after duplication).
            dilation (int): Dilation along the same dimension.

        Returns:
            int: Adjusted padding value that yields an integer output size.

        Raises:
            RuntimeError: If the required adjustment exceeds the provided
                padding, which should be handled by the ``SizeAdjustInputPass``
                pass instead.

        """
        mod_remainder = (
            input_len + 2 * pad - dilation * (input_weight - 1) - 1
        ) % stride

        # No need to adjust
        if mod_remainder == 0:
            return pad

        if mod_remainder > pad:
            raise RuntimeError(
                "This case should be handled by the SizeAdjustInputPass, is it enabled?"
            )
        return pad - mod_remainder

    def _is_depthwise_conv2d(self, node: torch.fx.Node) -> bool:
        if (
            node.op != "call_function"
            or node.target != exir_ops.edge.aten.convolution.default
        ):
            return False
        groups = node.args[-1]
        in_channels = get_first_fake_tensor(node.all_input_nodes[0]).shape[1]
        out_channels = get_first_fake_tensor(node.all_input_nodes[1]).shape[0]
        return (in_channels == groups) and (out_channels % in_channels) == 0

    def _reshape_weights(self, weight_node: torch.fx.Node, in_channels: int) -> None:
        """Reshape the weights for depthwise convolution such that when serialized to TOSA,
        the weights are in the format [H, W, in_channels, m_length] where
        m_length is the number of output channels per input channel.
        """
        weight_tensor = get_param_tensor(self.exported_program, weight_node)  # type: ignore[arg-type]
        if weight_tensor is None:
            raise RuntimeError(
                f"Weight node {weight_node.name} is not a parameter or buffer"
            )
        reshaped_weight_tensor = (
            weight_tensor.permute(HWCM_ORDER)
            .reshape(
                weight_tensor.shape[2],
                weight_tensor.shape[3],
                in_channels,
                weight_tensor.shape[0] // in_channels,
            )
            .permute(NHWC_INVERSE_ORDER)
        )

        if is_buffer(self.exported_program, weight_node):
            param_name = self.exported_program.graph_signature.inputs_to_buffers[
                weight_node.name
            ]
        elif is_param(self.exported_program, weight_node):
            param_name = self.exported_program.graph_signature.inputs_to_parameters[
                weight_node.name
            ]
        else:
            raise RuntimeError(
                f"Weight node {weight_node.name} is neither a parameter nor a buffer"
            )
        self.exported_program.state_dict[param_name] = reshaped_weight_tensor
        weight_node.meta["val"] = weight_node.meta["val"].reshape(
            weight_tensor.shape[2],
            weight_tensor.shape[0] // in_channels,
            weight_tensor.shape[3],
            in_channels,
        )

    def _add_bias(
        self,
        graph_module: torch.fx.GraphModule,
        node: torch.fx.Node,
        weight_node: torch.fx.Node,
    ) -> torch.fx.Node:
        output_channels = get_first_fake_tensor(node).shape[1]
        # add a node containging zeros if quantized, use int32, otherwise use float32
        if "output_qparams" in node.meta and len(node.meta["output_qparams"]) > 0:
            bias_data = torch.zeros(size=(output_channels,), dtype=torch.int32)
        else:
            bias_data = torch.zeros(size=(output_channels,), dtype=torch.float32)

        with graph_module.graph.inserting_after(weight_node):
            bias_node = create_constant_placeholder(
                self.exported_program,
                graph=graph_module.graph,
                kind=InputKind.PARAMETER,
                data=bias_data,
                persistent_buffer=True,
                name=f"{node.name}_bias",
            )
            if node.all_input_nodes[0].meta["val"].dtype == torch.int16:
                bias_node.meta[TosaSpecialDtype.meta_key()] = TosaSpecialDtype.INT48
        node.update_arg(2, bias_node)
        return bias_node

    def call(self, graph_module: torch.fx.GraphModule) -> PassResult:
        modified = False
        for node in graph_module.graph.nodes:
            if (
                node.op != "call_function"
                or node.target != exir_ops.edge.aten.convolution.default
            ):
                continue

            modified = True

            (
                x,
                weight,
                bias,
                stride,
                pad,
                dilation,
                transposed,
                output_pad,
                group,
            ) = node.args

            pad = [val for val in pad for _ in (0, 1)]
            input_shape = get_first_fake_tensor(x).shape
            weight_shape = get_first_fake_tensor(weight).shape
            # Adjust the pad value if needed to meet the
            # strict convolution output shape calculation.
            pad[1] = self._adjust_pad_if_needed(
                input_shape[2],
                weight_shape[2],
                stride[0],
                pad[1],
                dilation[0],
            )
            pad[3] = self._adjust_pad_if_needed(
                input_shape[3],
                weight_shape[3],
                stride[1],
                pad[3],
                dilation[1],
            )

            if bias is None:
                bias = self._add_bias(graph_module, node, weight)

            if self._is_depthwise_conv2d(node):
                target_op = exir_ops.backend.tosa.DEPTHWISE_CONV2D.default
                self._reshape_weights(weight, input_shape[1])
            else:
                target_op = exir_ops.backend.tosa.CONV2D.default

            conv2d_args = (
                x,
                weight,
                bias,
                stride,
                pad,
                dilation,
                transposed,
                output_pad,
                group,
            )

            with graph_module.graph.inserting_after(node):
                tosa_op = create_node(
                    graph=graph_module.graph,
                    op_target=target_op,
                    args=conv2d_args,
                    from_node=node,
                )

                node.replace_all_uses_with(tosa_op)
                graph_module.graph.erase_node(node)

        if modified:
            graph_module.recompile()
            graph_module = super().call(graph_module).graph_module
        return PassResult(graph_module, modified)
