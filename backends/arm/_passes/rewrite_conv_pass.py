# Copyright 2025 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


import itertools
from typing import Set, Type

import torch
from executorch.backends.arm._passes import ArmPass

from executorch.backends.arm._passes.arm_pass_utils import (
    create_node,
    expand_around_channel,
    get_first_fake_tensor,
    get_param_tensor,
    is_buffer,
    is_param,
)
from executorch.backends.arm._passes.fold_qdq_with_annotated_qparams_pass import (
    get_input_qparams,
    get_output_qparams,
)
from executorch.backends.arm.constants import HWCM_ORDER, NHWC_INVERSE_ORDER
from executorch.backends.arm.tosa.mapping import TosaSpecialDtype
from executorch.backends.transforms.utils import create_constant_placeholder
from executorch.exir.dialects._ops import ops as exir_ops
from executorch.exir.pass_base import ExportPass, PassResult
from torch.export.graph_signature import InputKind


class RewriteConvPass(ArmPass):
    """Rewrites aten.convolution to tosa.CONV2D or tosa.DEPTHWISE_CONV2D."""

    def __init__(self, exported_program: torch.export.ExportedProgram, *args, **kwargs):
        super().__init__(*args, **kwargs)
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
        input_tensor = get_first_fake_tensor(node.all_input_nodes[0])
        if len(input_tensor.shape) != 4:
            return False
        groups = node.args[-1]
        in_channels = input_tensor.shape[1]
        out_channels = get_first_fake_tensor(node).shape[1]
        return (in_channels == groups) and (out_channels % in_channels) == 0

    def _is_conv3d(self, rank, groups) -> bool:
        if rank == 5:
            # A Conv3D is considered depthwise if Group == InChannels and
            # Group * N == OutChannels, where N is a possitive integer.
            # Currently we do not support depthwise or grouped conv3d.
            # @TODO Add grouped/depthwise conv3d support or reject in partitioner.
            if groups != 1:
                raise RuntimeError(
                    "CONV3D with groups != 1 is not supported in the Arm backend."
                )
            return True
        return False

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
            reshaped_weight_tensor = torch.nn.Buffer(reshaped_weight_tensor)
        elif is_param(self.exported_program, weight_node):
            param_name = self.exported_program.graph_signature.inputs_to_parameters[
                weight_node.name
            ]
            reshaped_weight_tensor = torch.nn.Parameter(
                reshaped_weight_tensor, requires_grad=False
            )
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

    def insert_output_rescale(self, graph_module, node):
        input_qparams = get_input_qparams(node)
        output_qparams = get_output_qparams(node)[0]
        weight_qparams = input_qparams[1]
        input_qparams = input_qparams[0]
        is_per_channel = weight_qparams.per_channel
        if is_per_channel:
            weight_scale = weight_qparams.get_scale_per_channel()
        else:
            weight_scale = [weight_qparams.get_scale_per_tensor()]
        input_scale = input_qparams.get_scale_per_tensor()
        post_conv2d_scale = [
            (inp * w) / out
            for inp, w, out in zip(
                itertools.cycle([input_scale]),
                weight_scale,
                itertools.cycle([output_qparams.get_scale_per_tensor()]),
            )
        ]
        with graph_module.graph.inserting_after(node):
            rescale_node = create_node(
                graph=graph_module.graph,
                op_target=exir_ops.backend.tosa.RESCALE.default,
                args=(
                    node,
                    output_qparams.dtype,
                    post_conv2d_scale,
                    0,
                    output_qparams.get_zp_per_tensor(),
                ),
                from_node=node,
            )
        return rescale_node

    def call(self, graph_module: torch.fx.GraphModule) -> PassResult:  # noqa: C901
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

            input_fake_tensor = get_first_fake_tensor(x)
            weight_fake_tensor = get_first_fake_tensor(weight)
            input_shape = input_fake_tensor.shape
            weight_shape = weight_fake_tensor.shape
            spatial_rank = len(input_shape) - 2
            stride_list = expand_around_channel(stride, spatial_rank)
            dilation_list = expand_around_channel(dilation, spatial_rank)
            pad_list = expand_around_channel(pad, spatial_rank)

            pad_attr: list[int] = []
            for value in pad_list:
                pad_attr.extend([value, value])  # duplicate pad before/after per axis

            for axis_index in range(spatial_rank):
                pad_index = axis_index * 2 + 1  # adjust trailing pad entry
                pad_attr[pad_index] = self._adjust_pad_if_needed(
                    input_shape[axis_index + 2],
                    weight_shape[axis_index + 2],
                    stride_list[axis_index],
                    pad_attr[pad_index],
                    dilation_list[axis_index],
                )

            stride = tuple(stride_list)
            dilation = tuple(dilation_list)
            pad = pad_attr

            has_bias = bias is not None
            if not has_bias:
                bias = self._add_bias(graph_module, node, weight)

            if self._is_conv3d(len(input_shape), group):
                target_op = exir_ops.backend.tosa.CONV3D.default
            elif self._is_depthwise_conv2d(node):
                target_op = exir_ops.backend.tosa.DEPTHWISE_CONV2D.default
                # If there are any TOSA.DEPTHWISE_CONV2D nodes using the weights, we've already reshaped them.
                if all(user.target != target_op for user in weight.users):
                    self._reshape_weights(weight, input_fake_tensor.shape[1])
                weight_fake_tensor = get_first_fake_tensor(weight)
            else:
                target_op = exir_ops.backend.tosa.CONV2D.default

            conv_args = (
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
                    args=conv_args,
                    from_node=node,
                    inherit_qparams=True,
                )
            bias_fake_tensor = get_first_fake_tensor(bias) if bias else None
            tosa_node_fake_tensor = target_op(
                input_fake_tensor,
                weight_fake_tensor,
                bias_fake_tensor,
                *conv_args[3:],
            )

            if (
                tosa_node_fake_tensor.dtype == torch.int32
                and input_fake_tensor.dtype == torch.int8
            ):
                output_rescale = self.insert_output_rescale(graph_module, tosa_op)
                node.replace_all_uses_with(output_rescale)
            elif (
                tosa_node_fake_tensor.dtype == torch.int32
                and input_fake_tensor.dtype == torch.int16
            ):
                has_bias = len(node.meta["input_qparams"]) > 2
                if not has_bias:
                    output_rescale = self.insert_output_rescale(graph_module, tosa_op)
                    node.replace_all_uses_with(output_rescale)
                else:
                    node.replace_all_uses_with(tosa_op)
                tosa_op.meta[TosaSpecialDtype.meta_key()] = TosaSpecialDtype.INT48
            else:
                node.replace_all_uses_with(tosa_op)

            graph_module.graph.erase_node(node)

        if modified:
            graph_module.recompile()
            graph_module = super().call(graph_module).graph_module
        return PassResult(graph_module, modified)
