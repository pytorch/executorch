# Copyright 2024-2026 NXP
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import torch
from executorch.backends.nxp.backend.data_format import DataFormat

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
from executorch.backends.nxp.backend.ir.converter.conversion.translator import (
    tf_lite_type_to_numpy,
)
from executorch.backends.nxp.backend.ir.converter.node_converter import (
    CustomDelegationOptions,
    NodeConverter,
)
from executorch.backends.nxp.backend.ir.converter.node_converters.shared import (
    conv_utils,
)
from executorch.backends.nxp.backend.ir.converter.node_converters.shared.conv_utils import (
    ConvConversionResult,
    ConvParameters,
    get_node_tensor_params,
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
    transpose_conv_options,
)
from executorch.backends.nxp.backend.neutron_target_spec import NeutronTargetSpec
from torch.fx import Node
from torch.nn import Parameter


class ConvolutionConverter(NodeConverter):
    @staticmethod
    def _is_supported_on_target(
        node: Node,
        neutron_target_spec: NeutronTargetSpec,
        parameters_mapping: dict[str, Parameter],
        custom_delegation_options: CustomDelegationOptions,
    ) -> bool:
        num_macs = neutron_target_spec.get_num_macs()
        node_t_params = get_node_tensor_params(node)
        weights = node.args[1]
        conv_params = ConvParameters(
            *ConvolutionConverter._get_convolution_arguments(node)
        )

        if node_t_params["batch_size"] != 1:
            # Only batch size 1 is supported on neutron.
            return False

        if conv_params.transposed:
            # TransposeConv2d with groups > 1 is not supported
            # TODO: split into multiple convs with groups = 1
            if conv_params.groups > 1:
                return False
            if not node_is_effectively_static_tensor(weights, parameters_mapping):
                # Only supported if the weights are static, because TFLite `TransposeConv` uses permuted
                #  weights. In case the weights are dynamic, a Transpose operator would have to be added, which
                #  is not supported on Neutron.
                return False
            # neutron-library/src/utils/NeutronLibraryInterrogation.cpp#876 TransposeConv2DKernelKind
            if (
                conv_params.dilation != [1, 1]
                or conv_params.padding[0] != 0
                or conv_params.padding[1] >= node_t_params["kernel_width"]
                or (
                    conv_params.padding[1] != 0 and node_t_params["inp_height"] != 1
                )  # Slice added by explicit padding
                or conv_params.stride[0] != 1
                or (
                    (
                        conv_params.stride[1] != node_t_params["kernel_width"] / 2
                        or node_t_params["out_height"] != 1
                    )
                    and conv_params.stride[1] != node_t_params["kernel_width"]
                )
                or conv_params.stride[1] % 2 != 0
                or node_t_params["inp_channels"] % num_macs != 0
                or node_t_params["out_channels"] % num_macs != 0
                or node_t_params["kernel_width"] % 2 != 0
                or node_t_params["kernel_height"] != 1
            ):
                return False
        elif conv_params.groups == 1:  # Regular convolution.
            pass
        elif conv_utils.group_conv_convertible_as_depthwise(
            node, conv_params.groups
        ):  # Depthwise convolution.
            # Only supported if the weights are static, because TFLite `DepthwiseConv2D` uses permuted
            #  weights. In case the weights are dynamic, a Transpose operator would have to be added, which
            #  is not supported on Neutron.
            if not node_is_effectively_static_tensor(weights, parameters_mapping):
                return False
        elif conv_utils.group_conv_convertible_into_multiple_convolutions(
            node, conv_params.groups
        ):  # Separable conv.
            # Requires addition of `Split` and `Concatenation` operators, which are not supported on Neutron.
            return False
        else:  # Unexpected case (should never happen).
            return False

        return True

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
        groups = node.args[8]

        if is_transposed and conv_utils.group_conv_convertible_as_depthwise(
            node, groups
        ):
            # TFLite does not support transposed depthwise convolution
            return False

        if not is_transposed and output_padding != [0] * dimensions:
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

    def _compute_slicing_params(
        self, output_shape, explicit_padding
    ) -> tuple[list[int], list[int]]:
        begins = []
        sizes = []

        for axis in range(len(output_shape)):
            (start, end) = explicit_padding[axis]

            begins.append(start)
            sizes.append(output_shape[axis] - start - end)

        return begins, sizes

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
            list(out_padding),
            groups,
        )

    # noinspection PyPep8Naming
    def _convert_unpadded_2D(
        self, t_op: tflite_model.Operator, conv_params: ConvParameters
    ) -> conv_utils.ConvConversionResult:
        """Convert the `aten.convolution` into TFLite. The `padding` and `builtin_options` must be converted by the
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

            if w.type in [TensorType.INT8, TensorType.UINT8]:
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

    def _convert_transpose_conv(
        self, t_op: tflite_model.Operator, conv_params: ConvParameters
    ) -> conv_utils.ConvConversionResult:
        """Convert the `aten.convolution` into TFLite TransposeConv. The `builtin_options` must be
        converted by the caller.
        """
        common.assign_2d_strides(t_op.builtin_options, conv_params.stride)

        x: tflite_model.Tensor = t_op.tmp_inputs[0]
        w: tflite_model.Tensor = t_op.tmp_inputs[1]
        y: tflite_model.Tensor = t_op.tmp_outputs[0]

        if (b := try_get_input(t_op, 2)) is None:
            # Operator has no bias. Convolution aten op can omit it, TFLite can't.
            # Weight tensor format in TFLite: [C, kH, kW, O]
            # (C = input channels, O = output channels, kW = kernel width, kH = kernel height)
            output_channels = w.shape.vector[-1]

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

            if w.type in [TensorType.INT8, TensorType.UINT8]:
                # Compute scale and zero point for bias tensor
                input_scale = np.array(x.quantization.scale.vector)
                weight_scale = np.array(w.quantization.scale.vector)
                bias_scale = input_scale * weight_scale
                bias_zero_point = np.zeros(weight_scale.shape, dtype=np.int64)

                set_quantization_parameters_to_tensor(
                    b, bias_scale, bias_zero_point, quantized_dimension=0
                )

        # TransposeConv weight tensor format in TFLite: [O, kH, kW, C]
        # (C = input channels, O = output channels, kW = kernel width, kH = kernel height)
        if tensor_has_data(w):
            # Transpose cloned tensor statically
            w = self.builder.create_transposed_tensor(w, [3, 1, 2, 0])

            if w.quantization is not None:
                # Model is quantized
                w.quantization.quantized_dimension = 0
        else:
            raise NotImplementedError("Dynamic Transpose Conv weights.")
        w.tensor_format = DataFormat.TRANSPOSE_CONV_2D_WEIGHT_FORMAT

        output_shape_tensor_data = np.asarray(y.shape.vector, dtype=np.int32)
        o = self.builder.create_tensor_for_data(
            output_shape_tensor_data, "output_shape"
        )

        # Assign the operator its TFLite inputs and outputs
        t_op.tmp_inputs = [o, w, x, b]
        t_op.tmp_outputs = [y]
        conversion_result = ConvConversionResult(x, w, b, y, o)
        t_op.builtin_options.padding, explicit_padding = (
            aten_translator.convert_padding(conv_params.padding)
        )
        if explicit_padding is not None:
            # Add padding to output shape to make sure we have computed all the data we need
            for idx, padding in enumerate(explicit_padding):
                output_shape_tensor_data[idx] += padding[0] + padding[1]
            y.shape = tflite_model.Shape(output_shape_tensor_data.tolist())

            # We need to "cut" produced tensor by size of explicit padding
            begins, sizes = self._compute_slicing_params(
                output_shape_tensor_data.tolist(), explicit_padding
            )
            slice_op = self.builder.create_slice_after(t_op, 0, begins, sizes)
            conversion_result.ops_list.add_post(slice_op)

        conversion_result.ops_list.middle_op = t_op

        return conversion_result

    def _convert_2d_conv(
        self, t_op: tflite_model.Operator, conv_params: ConvParameters
    ) -> list[tflite_model.Operator]:
        if conv_params.transposed:
            t_op.builtin_options = transpose_conv_options.TransposeConv()
            if conv_utils.group_conv_convertible_into_multiple_convolutions(
                t_op, conv_params.groups
            ):
                # Convert to separated `TransposeConv`.
                raise NotImplementedError("Separated TransposeConv not implemented.")
            else:
                # Convert to `TransposeConv`.
                conversion_result = self._convert_transpose_conv(t_op, conv_params)

        else:
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
            ):  # Convert to separated `Conv2D`.
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

        stride, padding, dilation, transposed, out_padding, groups = (
            self._get_convolution_arguments(node)
        )

        t_op = self._create_tflite_op_with_io_tensors(node)
        conv_params = ConvParameters(
            stride, padding, dilation, transposed, out_padding, groups
        )

        rank = t_op.tmp_inputs[1].shape.len()
        if rank == 4:  # Conv2D
            ops_to_add = self._convert_2d_conv(t_op, conv_params)
        else:
            raise NotImplementedError(
                f"{rank - 2}D convolution is not supported."
            )  # Should never get here.

        self.builder.append_operators(ops_to_add)
