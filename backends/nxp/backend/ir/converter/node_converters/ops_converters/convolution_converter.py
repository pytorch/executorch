# Copyright 2024 NXP
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import torch

from executorch.backends.nxp.backend.edge_helper import input_tensor, input_tensor_safe
from executorch.backends.nxp.backend.ir.converter.conversion import (
    aten_translator,
    common,
)
from executorch.backends.nxp.backend.ir.converter.conversion.common import (
    OpsList,
    try_get_input,
)
from executorch.backends.nxp.backend.ir.converter.node_converter import (
    NodeConverter,
    Target,
)
from executorch.backends.nxp.backend.ir.converter.quantization_utils import (
    set_quantization_parameters_to_tensor,
)
from executorch.backends.nxp.backend.ir.lib.tflite.TensorType import TensorType
from executorch.backends.nxp.backend.ir.tflite_generator import tflite_model
from executorch.backends.nxp.backend.ir.tflite_generator.builtin_options import (
    conv_2d_options,
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

        if groups != 1:
            return False

        if input_tensor_safe(node, 2) is None:
            # No bias tensor.
            weight_tensor = input_tensor(node, 1)
            if weight_tensor.dtype not in [torch.float32, torch.int8, torch.uint8]:
                return False

        return True

    def _convert_2d_conv(
        self, stride, padding, dilation, t_op: tflite_model.Operator
    ) -> list[tflite_model.Operator]:
        ops = OpsList(middle_op=t_op)
        t_op.builtin_options = conv_2d_options.Conv2D()
        common.assign_2d_strides(t_op.builtin_options, stride)
        common.assign_2d_dilations(t_op.builtin_options, dilation)
        t_op.builtin_options.padding, explicit_padding = (
            aten_translator.convert_padding(padding)
        )

        if explicit_padding is not None:
            # Need to prepend a 'Pad' operator, which adds 0s. But these will be included in the computation!
            ops.add_pre(
                self.builder.create_pad_operator_before(t_op, 0, explicit_padding)
            )

        input_tensor: tflite_model.Tensor = t_op.tmp_inputs[0]
        weight_tensor: tflite_model.Tensor = t_op.tmp_inputs[1]
        output_tensor: tflite_model.Tensor = t_op.tmp_outputs[0]

        if (bias_tensor := try_get_input(t_op, 2)) is None:
            # Operator has no bias. Convolution aten op can omit it, TFLite can't.
            output_channels = weight_tensor.shape.vector[0]

            if weight_tensor.type == TensorType.FLOAT32:
                bias_type = np.dtype(np.float32)
            elif weight_tensor.type in [TensorType.INT8, TensorType.UINT8]:
                bias_type = np.dtype(np.int32)
            else:
                # Should never happen.
                raise NotImplementedError(
                    f"Convolution node with unsupported weight type: {weight_tensor.type}"
                )

            bias_tensor = self.builder.create_zeros_tensor(
                [output_channels], "zero_bias", bias_type, True
            )

            # Compute scale and zero point for bias tensor
            input_scale = np.array(input_tensor.quantization.scale.vector)
            weight_scale = np.array(weight_tensor.quantization.scale.vector)
            bias_scale = input_scale * weight_scale
            bias_zero_point = np.zeros(weight_scale.shape, dtype=np.int64)

            set_quantization_parameters_to_tensor(
                bias_tensor, bias_scale, bias_zero_point, quantized_dimension=0
            )

        # Assign the operator its TFLite inputs and outputs
        t_op.tmp_inputs = [input_tensor, weight_tensor, bias_tensor]
        t_op.tmp_outputs = [output_tensor]

        return ops.flatten()

    def convert(self, node: Node):
        self.assert_convertible(node)

        stride = node.args[3]
        padding = node.args[4]
        dilation = node.args[5]

        t_op = self._create_tflite_op_with_io_tensors(node)
        ops_to_add = self._convert_2d_conv(stride, padding, dilation, t_op)

        self.builder.append_operators(ops_to_add)
