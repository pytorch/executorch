# Copyright 2024-2025 NXP
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import logging

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
    translator,
)
from executorch.backends.nxp.backend.ir.converter.conversion.common import try_get_input
from executorch.backends.nxp.backend.ir.converter.conversion.translator import (
    tf_lite_type_to_numpy,
)
from executorch.backends.nxp.backend.ir.converter.node_converter import (
    CustomDelegationOptions,
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
    reshape_options,
)
from torch.fx import Node
from torch.nn import Parameter


class ConvolutionConverter(NodeConverter):
    @staticmethod
    def _is_supported_on_target(
        node: Node,
        target: Target,
        parameters_mapping: dict[str, Parameter],
        custom_delegation_options: CustomDelegationOptions,
    ) -> bool:
        match target:
            case Target.RT700:
                activations = node.args[0]
                weights = node.args[1]
                groups = node.args[8]

                if activations.meta["val"].shape[0] != 1:
                    # Only batch size 1 is supported on neutron.
                    return False

                if groups == 1:  # Regular convolution.
                    pass
                elif conv_utils.group_conv_convertible_as_depthwise(
                    node, groups
                ):  # Depthwise convolution.
                    # Only supported if the weights are static, because TFLite `DepthwiseConv2D` uses permuted
                    #  weights. In case the weights are dynamic, a Transpose operator would have to be added, which
                    #  is not supported on Neutron.
                    if not node_is_effectively_static_tensor(
                        weights, parameters_mapping
                    ):
                        return False
                elif conv_utils.group_conv_convertible_into_multiple_convolutions(
                    node, groups
                ):  # Separable conv. This should never be reached, as the node should have been decomposed into
                    #  multiple parallel convolutions by the `SplitGroupConvolution` pre-processing pass.
                    logging.warning("Group convolution was not decomposed.")
                    return False
                else:  # Unexpected case (should never happen).
                    return False

                return True

            case _:
                return False

    @staticmethod
    def _is_supported_in_IR(
        node: Node,
        parameters_mapping: dict[str, Parameter],
        custom_delegation_options: CustomDelegationOptions,
    ) -> bool:
        input_tensor_rank = len(node.meta["val"].shape)
        dimensions = input_tensor_rank - 2
        is_transposed = node.args[6]
        output_padding = node.args[7]

        if is_transposed:
            return False

        if output_padding != [0] * dimensions:
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
        return (
            list(stride),
            list(padding),
            list(dilation),
            transposed,
            out_padding,
            groups,
        )

    def _convert_1d_conv(
        self, t_op: tflite_model.Operator, conv_params: ConvParameters
    ) -> list[tflite_model.Operator]:
        """Convert the 'Conv' operator with a 1D kernel to TFLite 'Conv2D'.
        TFLite doesn't support 1D convolution, but this behaviour can be represented using
               Reshape -> Conv2D -> Reshape.
        The first reshape introduces a 4th dimension with size 1. The second Reshape removes the temporary dimension.
        """
        # -- Calculate the shapes for equivalent 2D convolution --
        conv_2d_input_shape = translator.nhc_dimensions_to_nhwc(
            t_op.tmp_inputs[0].shape.vector
        )
        conv_2d_weight_shape = translator.nhc_dimensions_to_nhwc(
            t_op.tmp_inputs[1].shape.vector
        )
        conv_2d_output_shape = translator.nhc_dimensions_to_nhwc(
            t_op.tmp_outputs[0].shape.vector
        )

        # -- Generate tensors taking part in the conversion --
        reshape1_input = t_op.tmp_inputs[0]

        reshape1_output = self.builder.duplicate_tensor(
            reshape1_input, name_suffix="_4D_"
        )
        reshape1_output.shape = tflite_model.Shape(conv_2d_input_shape)

        reshape2_input = self.builder.duplicate_tensor(
            t_op.tmp_outputs[0], name_suffix="_4D_"
        )
        reshape2_input.shape = tflite_model.Shape(conv_2d_output_shape)

        reshape2_output = t_op.tmp_outputs[0]

        pre_reshapes = []

        # Extend the weights tensor to 4D
        weights_tensor = t_op.tmp_inputs[1]
        if tensor_has_data(weights_tensor):
            # Do it statically
            weights_tensor.shape = tflite_model.Shape(conv_2d_weight_shape)
            weights_tensor.tmp_buffer.data = weights_tensor.tmp_buffer.data.reshape(
                conv_2d_weight_shape
            )

        else:
            # Add a Reshape before the weights tensor
            new_weights_tensor = self.builder.duplicate_tensor(
                weights_tensor, name_suffix="_4D_"
            )
            new_weights_tensor.shape = tflite_model.Shape(conv_2d_weight_shape)

            weight_reshape = tflite_model.Operator(
                builtin_options=reshape_options.Reshape(conv_2d_weight_shape)
            )
            weight_reshape.tmp_inputs = [weights_tensor]
            weight_reshape.tmp_outputs = [new_weights_tensor]

            pre_reshapes.append(weight_reshape)

            # Save the new weights tensor, to assign it later.
            weights_tensor = new_weights_tensor

        # -- Create the new operators --
        reshape1 = tflite_model.Operator(
            builtin_options=reshape_options.Reshape(conv_2d_input_shape)
        )
        reshape1.tmp_inputs = [reshape1_input]
        reshape1.tmp_outputs = [reshape1_output]
        pre_reshapes.append(reshape1)

        reshape2 = tflite_model.Operator(
            builtin_options=reshape_options.Reshape(reshape2_output.shape.vector)
        )
        reshape2.tmp_inputs = [reshape2_input]
        reshape2.tmp_outputs = [reshape2_output]

        # Assign the new input and output of the Conv2D
        t_op.tmp_inputs = [reshape1_output, weights_tensor] + t_op.tmp_inputs[
            2:
        ]  # Add bias as well, if present
        t_op.tmp_outputs = [reshape2_input]

        # Extend all Conv attributes to 2D
        common.extend_1d_stride_to_2d(conv_params.stride)
        common.extend_1d_dilation_to_2d(conv_params.dilation)
        common.extend_1d_padding_to_2d(conv_params.padding)

        # Convert the now 2D Conv
        converted_conv_ops = self._convert_2d_conv(t_op, conv_params)

        return pre_reshapes + converted_conv_ops + [reshape2]

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
                [output_channels], "zero_bias", bias_type, False
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
                # Need to prepend a 'Pad' operator, which adds 0s (or `zero_point` for the quantized case).
                input_quantization = t_op.tmp_inputs[0].quantization
                pad_value = (
                    None
                    if input_quantization is None
                    else np.array(input_quantization.zero_point[0]).astype(
                        tf_lite_type_to_numpy(t_op.tmp_inputs[0].type)
                    )
                )
                conversion_result.ops_list.add_pre(
                    self.builder.create_pad_operator_before(
                        t_op, 0, explicit_padding, constant_value=pad_value
                    )
                )

            # DepthwiseConv2D expects weights in format [kernel_channels, kernel_height, kernel_width, output_channels]
            perm = [3, 1, 2, 0]
            weight_tensor = conversion_result.conv_weight_tensor
            if tensor_has_data(weight_tensor):
                # Transpose cloned tensor statically
                t_op.tmp_inputs[1] = self.builder.create_transposed_tensor(
                    weight_tensor, perm
                )

                if t_op.tmp_inputs[1].quantization is not None:
                    # Model is quantized
                    t_op.tmp_inputs[1].quantization.quantized_dimension = 3
            else:
                raise NotImplementedError("Dynamic Depthwise Conv weights.")

        elif conv_utils.group_conv_convertible_into_multiple_convolutions(
            t_op, conv_params.groups
        ):
            # This case should have been rejected in the `is_supported_on_target()` method.
            raise RuntimeError("Group convolution was not decomposed.")

        else:
            # Convert to regular `Conv2D`.
            t_op.builtin_options = conv_2d_options.Conv2D()
            conversion_result = self._convert_unpadded_2D(t_op, conv_params)
            t_op.builtin_options.padding, explicit_padding = (
                aten_translator.convert_padding(conv_params.padding)
            )
            if explicit_padding is not None:
                # Need to prepend a 'Pad' operator, which adds 0s (or `zero_point` for the quantized case).
                input_quantization = t_op.tmp_inputs[0].quantization
                pad_value = (
                    None
                    if input_quantization is None
                    else np.array(input_quantization.zero_point[0]).astype(
                        tf_lite_type_to_numpy(t_op.tmp_inputs[0].type)
                    )
                )
                conversion_result.ops_list.add_pre(
                    self.builder.create_pad_operator_before(
                        t_op, 0, explicit_padding, constant_value=pad_value
                    )
                )

        return conversion_result.ops_list.flatten()

    def convert(self, node: Node):
        self.assert_convertible(node)

        stride, padding, dilation, _, _, groups = self._get_convolution_arguments(node)

        t_op = self._create_tflite_op_with_io_tensors(node)
        conv_params = ConvParameters(stride, padding, dilation, groups)

        rank = t_op.tmp_inputs[1].shape.len()
        if rank == 3:  # Conv1D
            ops_to_add = self._convert_1d_conv(t_op, conv_params)
        elif rank == 4:  # Conv2D
            ops_to_add = self._convert_2d_conv(t_op, conv_params)
        else:
            raise NotImplementedError(
                f"{rank - 2}D convolution is not supported."
            )  # Should never get here.

        self.builder.append_operators(ops_to_add)
