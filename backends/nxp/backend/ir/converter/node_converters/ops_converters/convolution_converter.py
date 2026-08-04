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
    requires_channels_first_format,
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


# The arguments of the conv are:
# convolution(
#   Tensor input, Tensor weight, Tensor? bias,
#   SymInt[] stride, SymInt[] padding, SymInt[] dilation,
#   bool transposed, SymInt[] output_padding, SymInt groups
# ) -> Tensor
Stride = Padding = Dilation = OutPadding = list[int]
Transposed = bool
Groups = int


@requires_channels_first_format
class ConvolutionConverter(NodeConverter):
    @staticmethod
    def _is_conv_quant_supported(
        node: Node, parameters_mapping: dict[str, Parameter]
    ) -> bool:
        conv_params = ConvolutionConverter._get_conv_params(node)

        # Input must be INT8/UINT8
        # Output must be INT8/UINT8
        inp_out_supported_types = [torch.int8, torch.uint8]
        if not NodeConverter.uses_quantization_type_for_io(
            node, inp_out_supported_types, [0], [0]
        ):
            return False

        # Weights must be INT8
        w_supported_types = [torch.int8]
        if not NodeConverter.uses_quantization_type_for_io(
            node, w_supported_types, [1], []
        ):
            return False

        # Bias must be INT32
        if conv_params.bias_node is not None:
            b_supported_types = [torch.int32]
            if not NodeConverter.uses_quantization_type_for_io(
                node, b_supported_types, [2], []
            ):
                return False

        # Weights must be constant
        if not node_is_effectively_static_tensor(
            conv_params.weight_node, parameters_mapping
        ):
            return False

        # Bias must be constant (if present)
        if conv_params.bias_node is not None and not node_is_effectively_static_tensor(
            conv_params.bias_node, parameters_mapping
        ):
            return False

        return True

    @staticmethod
    def _is_supported_on_target_regular_conv(
        node: Node,
        neutron_target_spec: NeutronTargetSpec,
        parameters_mapping: dict[str, Parameter],
    ) -> bool:
        # Check the quantization of inputs is supported
        if not ConvolutionConverter._is_conv_quant_supported(node, parameters_mapping):
            return False

        op_args = ConvolutionConverter._get_conv_params(node)
        node_params = get_node_tensor_params(node)

        # kernelH <= 4096, kernelW <= 4096
        # strideH <= 4096, strideW <= 4096
        # dilationH <= 4096, dilationW <= 4096
        kernel_h = node_params["kernel_height"]
        kernel_w = node_params["kernel_width"]
        stride_h = op_args.stride[0]
        stride_w = op_args.stride[1]
        dilation_h = op_args.dilation[0]
        dilation_w = op_args.dilation[1]

        dim_sizes = [kernel_h, kernel_w, stride_h, stride_w, dilation_h, dilation_w]

        if any(dim > 4096 for dim in dim_sizes):
            return False

        # The following checks are mentioned in Neutron docs, however they cause some models
        # to be non-delegable. Discussion with Neutron team is pending, and because the convolutions seem
        # to work even without this constraint, this code is commented out for now
        # to boost the models' performance.
        # padT < kernelH, padB < kernelH, padL < kernelW, padR < kernelW

        # padding_h = op_args.padding[0]
        # padding_w = op_args.padding[1]
        # if padding_h >= kernel_h or padding_w >= kernel_w:
        #     return False

        # kernelH * kernelW * ROUND_CEIL(inpC, NUM_MACS) <= 65535
        inp_channels = node_params["inp_channels"]
        num_macs = neutron_target_spec.get_num_macs()

        if (
            kernel_h
            * kernel_w
            * ConvolutionConverter._round_ceil(inp_channels, num_macs)
            > 65535
        ):
            return False

        return True

    @staticmethod
    def _is_supported_on_target_transp_conv(
        node: Node,
        neutron_target_spec: NeutronTargetSpec,
        parameters_mapping: dict[str, Parameter],
    ) -> bool:
        # Check the quantization of inputs is supported
        if not ConvolutionConverter._is_conv_quant_supported(node, parameters_mapping):
            return False

        op_args = ConvolutionConverter._get_conv_params(node)
        node_params = get_node_tensor_params(node)

        # TransposeConv2d with groups > 1 is not supported
        # TODO: split into multiple convs with groups = 1
        if op_args.groups > 1:
            return False

        # kernelH <= 4096, kernelW <= 4096
        kernel_h = node_params["kernel_height"]
        kernel_w = node_params["kernel_width"]

        dim_sizes = [kernel_h, kernel_w]
        if any(dim > 4096 for dim in dim_sizes):
            return False

        # strideH <= kernelH, strideW <= kernelW
        stride_h = op_args.stride[0]
        stride_w = op_args.stride[1]
        if stride_h > kernel_h or stride_w > kernel_w:
            return False

        # strideH <= 2, strideW <= 2
        if stride_h > 2 or stride_w > 2:
            return False

        # padT < kernelH, padB < kernelH, padL < kernelW, padR < kernelW
        padding_h = op_args.padding[0]
        padding_w = op_args.padding[1]
        if padding_h >= kernel_h or padding_w >= kernel_w:
            return False

        # kernelH * kernelW * ceil(inpC, NUM_MACS) <= 65535
        inp_channels = node_params["inp_channels"]
        num_macs = neutron_target_spec.get_num_macs()

        if (
            kernel_h
            * kernel_w
            * ConvolutionConverter._round_ceil(inp_channels, num_macs)
            > 65535
        ):
            return False

        return True

    @staticmethod
    def _is_supported_on_target(
        node: Node,
        neutron_target_spec: NeutronTargetSpec,
        parameters_mapping: dict[str, Parameter],
        custom_delegation_options: CustomDelegationOptions,
    ) -> bool:
        conv_params = ConvolutionConverter._get_conv_params(node)

        if conv_params.transposed:
            return ConvolutionConverter._is_supported_on_target_transp_conv(
                node, neutron_target_spec, parameters_mapping
            )

        else:
            return ConvolutionConverter._is_supported_on_target_regular_conv(
                node, neutron_target_spec, parameters_mapping
            )

    @staticmethod
    def _is_supported_in_IR(
        node: Node,
        parameters_mapping: dict[str, Parameter],
        custom_delegation_options: CustomDelegationOptions,
    ) -> bool:
        input_tensor_rank = len(node.meta["val"].shape)
        dimensions = input_tensor_rank - 2
        conv_params = ConvolutionConverter._get_conv_params(node)

        if conv_params.transposed and conv_utils.group_conv_convertible_as_depthwise(
            node, conv_params.groups
        ):
            # TFLite does not support transposed depthwise convolution
            return False

        if not conv_params.transposed and conv_params.out_padding != [0] * dimensions:
            return False

        if input_tensor_safe(node, 2) is None:
            # No bias tensor.
            weight_tensor = input_tensor(node, 1)
            if weight_tensor.dtype not in [torch.float32, torch.int8, torch.uint8]:
                return False

        return True

    @staticmethod
    def _round_ceil(x, n):
        return ((x + n - 1) // n) * n

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
    def _get_conv_params(
        conv_node: Node,
    ) -> ConvParameters:
        def _normalize_ls_arg(ls):
            # sometimes, `conv2d` args can be a list of one element. In such case, convert it to 2d arg
            # example: padding = [0] => [0, 0]
            return [ls[0], ls[0]] if len(ls) == 1 else ls

        x, w, b, stride, padding, dilation, transposed, out_padding, groups = (
            conv_node.args
        )

        stride = _normalize_ls_arg(list(stride))
        padding = _normalize_ls_arg(list(padding))
        dilation = _normalize_ls_arg(list(dilation))
        out_padding = (
            None if out_padding is None else _normalize_ls_arg(list(out_padding))
        )

        return ConvParameters(
            x, w, b, stride, padding, dilation, transposed, out_padding, groups
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
        self, torch_node: Node, t_op: tflite_model.Operator
    ) -> list[tflite_model.Operator]:
        conv_params = self._get_conv_params(torch_node)

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
            ):
                raise RuntimeError("NXP backend: Group convolution was not decomposed.")

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

        t_op = self._create_tflite_op_with_io_tensors(node)

        rank = t_op.tmp_inputs[1].shape.len()
        if rank == 4:  # Conv2D
            ops_to_add = self._convert_2d_conv(node, t_op)
        else:
            raise NotImplementedError(
                f"{rank - 2}D convolution is not supported."
            )  # Should never get here.

        self.builder.append_operators(ops_to_add)
