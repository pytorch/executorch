# Copyright 2024 NXP
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import torch

from executorch.backends.nxp.backend.edge_helper import (
    input_tensor,
    input_tensor_safe,
    node_is_effectively_static_tensor,
)
from executorch.backends.nxp.backend.ir.converter.conversion import (
    aten_translator,
    common,
)
from executorch.backends.nxp.backend.ir.converter.conversion.common import try_get_input
from executorch.backends.nxp.backend.ir.converter.node_converter import (
    NodeConverter,
    Target,
)
from executorch.backends.nxp.backend.ir.converter.node_converters.shared import (
    conv_utils,
)
from executorch.backends.nxp.backend.ir.converter.node_converters.shared.conv_utils import (
    ConvConversionResult,
    ConvParameters,
)
from executorch.backends.nxp.backend.ir.converter.quantization_utils import (
    set_quantization_parameters_to_tensor,
)
from executorch.backends.nxp.backend.ir.converter.tensor_utils import tensor_has_data
from executorch.backends.nxp.backend.ir.lib.tflite.TensorType import TensorType
from executorch.backends.nxp.backend.ir.tflite_generator import tflite_model
from executorch.backends.nxp.backend.ir.tflite_generator.builtin_options import (
    conv_2d_options,
    depthwise_conv_2d_options,
)
from torch.fx import Node
from torch.nn import Parameter


class ConvolutionConverter(NodeConverter):
    supported_targets = [Target.RT700]

    @staticmethod
    def _is_supported_in_IR(
        node: Node, parameters_mapping: dict[str, Parameter]
    ) -> bool:
        is_transposed = node.args[6]
        output_padding = node.args[7]
        groups = node.args[8]

        if is_transposed:
            return False

        if output_padding != [0, 0]:
            return False

        if groups == 1:
            # Regular convolution.
            pass

        elif conv_utils.group_conv_convertible_as_depthwise(
            node, groups
        ) and node_is_effectively_static_tensor(node.args[1], parameters_mapping):
            # Depthwise convolution.
            # Only supported if the weights are static, because TFLite `DepthwiseConv2D` uses permuted weights. In case
            #  the weights are dynamic, a Transpose operator would have to be added, which is not supported on Neutron.
            pass

        elif conv_utils.group_conv_convertible_into_multiple_convolutions(node, groups):
            # Separable convolution. Currently not supported.
            return False

        else:
            # All conversion options related to the `group` attribute have been checked and none of them can be used.
            return False

        if input_tensor_safe(node, 2) is None:
            # No bias tensor.
            weight_tensor = input_tensor(node, 1)
            if weight_tensor.dtype not in [torch.float32, torch.int8, torch.uint8]:
                return False

        return True

    Stride = Padding = Dilation = OutPadding = list[int]
    Transposed = bool
    Groups = int

    @staticmethod
    def _get_convolution_arguments(
        conv_node: Node,
    ) -> (Stride, Padding, Dilation, Transposed, OutPadding, Groups):
        # The arguments of the conv are:
        # [x, w, b, stride, padding, dilation, transposed, output padding, groups]
        # https://github.com/pytorch/pytorch/blob/v2.6.0/aten/src/ATen/native/Convolution.cpp#L286-L291
        _, _, _, stride, padding, dilation, transposed, out_padding, groups = (
            conv_node.args
        )
        return stride, padding, dilation, transposed, out_padding, groups

    # noinspection PyPep8Naming
    def _convert_unpadded_2D(
        self, t_op: tflite_model.Operator, conv_params: ConvParameters
    ) -> conv_utils.ConvConversionResult:
        """Convert the `aten.convolution` into TFLite. The `padding` and `builtin_options` must be converter by the
        caller.
        """
        common.assign_2d_strides(t_op.builtin_options, conv_params.stride)
        common.assign_2d_dilations(t_op.builtin_options, conv_params.dilation)

        x: tflite_model.Tensor = t_op.tmp_inputs[0]
        w: tflite_model.Tensor = t_op.tmp_inputs[1]
        y: tflite_model.Tensor = t_op.tmp_outputs[0]

        if (b := try_get_input(t_op, 2)) is None:
            # Operator has no bias. Convolution aten op can omit it, TFLite can't.
            output_channels = w.shape.vector[0]

            if w.type == TensorType.FLOAT32:
                bias_type = np.dtype(np.float32)
            elif w.type in [TensorType.INT8, TensorType.UINT8]:
                bias_type = np.dtype(np.int32)
            else:
                # Should never happen.
                raise NotImplementedError(
                    f"Convolution node with unsupported weight type: {w.type}"
                )

            b = self.builder.create_zeros_tensor(
                [output_channels], "zero_bias", bias_type, True
            )

            # Compute scale and zero point for bias tensor
            input_scale = np.array(x.quantization.scale.vector)
            weight_scale = np.array(w.quantization.scale.vector)
            bias_scale = input_scale * weight_scale
            bias_zero_point = np.zeros(weight_scale.shape, dtype=np.int64)

            set_quantization_parameters_to_tensor(
                b, bias_scale, bias_zero_point, quantized_dimension=0
            )

        # Assign the operator its TFLite inputs and outputs
        t_op.tmp_inputs = [x, w, b]
        t_op.tmp_outputs = [y]

        conversion_result = ConvConversionResult(x, w, b, y)
        conversion_result.ops_list.middle_op = t_op

        return conversion_result

    def _convert_2d_conv(
        self, t_op: tflite_model.Operator, conv_params: ConvParameters
    ) -> list[tflite_model.Operator]:
        if conv_utils.group_conv_convertible_as_depthwise(
            t_op, conv_params.groups
        ):  # Convert to `DepthwiseConv2D`.
            t_op.builtin_options = depthwise_conv_2d_options.DepthwiseConv2D()

            conversion_result = self._convert_unpadded_2D(t_op, conv_params)
            t_op.builtin_options.padding, explicit_padding = (
                aten_translator.convert_padding(conv_params.padding)
            )
            if explicit_padding is not None:
                # Need to prepend a 'Pad' operator, which adds 0s.
                conversion_result.ops_list.add_pre(
                    self.builder.create_pad_operator_before(t_op, 0, explicit_padding)
                )

            # DepthwiseConv2D expects weights in format [kernel_channels, kernel_height, kernel_width, output_channels]
            perm = [3, 1, 2, 0]
            weight_tensor = conversion_result.conv_weight_tensor
            if tensor_has_data(weight_tensor):
                # Transpose cloned tensor statically
                t_op.tmp_inputs[1] = self.builder.create_transposed_tensor(
                    weight_tensor, perm
                )
            else:
                raise NotImplementedError("Dynamic Depthwise Conv weights.")

        elif conv_utils.group_conv_convertible_into_multiple_convolutions(
            t_op, conv_params.groups
        ):
            t_op.builtin_options = conv_2d_options.Conv2D()

            return conv_utils.create_separated_convolutions_based_on_group(
                t_op,
                conv_params,
                self.builder,
                self._convert_unpadded_2D,
                conv_utils.conv_op_factory,
            )

        else:
            # Convert to regular `Conv2D`.
            t_op.builtin_options = conv_2d_options.Conv2D()
            conversion_result = self._convert_unpadded_2D(t_op, conv_params)
            t_op.builtin_options.padding, explicit_padding = (
                aten_translator.convert_padding(conv_params.padding)
            )
            if explicit_padding is not None:
                # Need to prepend a 'Pad' operator, which adds 0s.
                conversion_result.ops_list.add_pre(
                    self.builder.create_pad_operator_before(t_op, 0, explicit_padding)
                )

        return conversion_result.ops_list.flatten()

    def convert(self, node: Node):
        self.assert_convertible(node)

        stride, padding, dilation, _, _, groups = self._get_convolution_arguments(node)

        t_op = self._create_tflite_op_with_io_tensors(node)
        conv_params = ConvParameters(stride, padding, dilation, groups)

        rank = t_op.tmp_inputs[1].shape.len()
        if rank == 4:  # Conv2D
            ops_to_add = self._convert_2d_conv(t_op, conv_params)
        else:
            raise NotImplementedError(
                f"{rank - 2}D convolution is not supported."
            )  # Should never get here.

        self.builder.append_operators(ops_to_add)
