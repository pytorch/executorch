# Copyright 2023-2025 NXP
#
# License: LA_OPT_NXP_Software_License
# See the LICENSE_LA_OPT_NXP_Software_License for more details.
#

from copy import copy
from dataclasses import dataclass
from typing import Callable, cast

import numpy as np

from executorch.backends.nxp.backend.ir.converter.builder.model_builder import (
    ModelBuilder,
)
from executorch.backends.nxp.backend.ir.converter.conversion import aten_translator
from executorch.backends.nxp.backend.ir.converter.conversion.common import OpsList
from executorch.backends.nxp.backend.ir.converter.tensor_utils import tensor_has_data
from executorch.backends.nxp.backend.ir.lib.tflite.Padding import Padding
from executorch.backends.nxp.backend.ir.tflite_generator import tflite_model
from executorch.backends.nxp.backend.ir.tflite_generator.builtin_options import (
    concatenation_options,
    conv_2d_options,
    split_options,
)
from torch.fx import Node


@dataclass
class ConvParameters:
    stride: list[int]
    padding: list[int]
    dilation: list[int]
    groups: int


# noinspection PyPep8Naming
def _get_IO_channels(node: Node | tflite_model.Operator) -> (int, int):
    if isinstance(node, Node):
        input_channels = (
            node.args[0].meta["val"].shape[1]
        )  # Channels of the main input.
        output_channels = (
            node.args[1].meta["val"].shape[0]
        )  # Output channels of the weights.
    else:
        input_channels = node.tmp_inputs[0].shape[-1]  # Channels of the main input.
        output_channels = node.tmp_inputs[1].shape[0]  # Output channels of the weights.

    return input_channels, output_channels


def group_conv_convertible_as_depthwise(node: Node | tflite_model.Operator, group: int):
    input_channels, output_channels = _get_IO_channels(node)

    return input_channels == output_channels == group


def group_conv_convertible_into_multiple_convolutions(
    node: Node | tflite_model.Operator, group: int
) -> bool:
    if group == 1:
        return False

    _, output_channels = _get_IO_channels(node)
    if output_channels % group != 0:
        return False  # Unable to split group Conv into separated convolutions because out_channels % group != 0.

    # 10 is an empirical value. The `group` directly dictates how many branches will be created.
    return 2 <= group <= 10


class ConvConversionResult:
    """
    Holds references to the direct I/O tensors of the Conv operator
    and list of surrounding operators (Quantize, Transpose, etc.).
    """

    def __init__(
        self,
        input_tensor: tflite_model.Tensor,
        weight_tensor: tflite_model.Tensor,
        bias_tensor: tflite_model.Tensor,
        output_tensor: tflite_model.Tensor,
    ):
        self.conv_input_tensor = input_tensor
        self.conv_weight_tensor = weight_tensor
        self.conv_bias_tensor = bias_tensor
        self.conv_output_tensor = output_tensor
        self.ops_list = OpsList()


ConvBuiltinOptions = conv_2d_options.Conv2D
ConvOpFactory = Callable[
    [
        ConvParameters,
        tflite_model.Tensor,
        tflite_model.Tensor,
        tflite_model.Tensor,
        tflite_model.Tensor,
        ModelBuilder,
        ConvBuiltinOptions,
    ],
    OpsList,
]
ConvConversionFn = Callable[
    [tflite_model.Operator, ConvParameters], ConvConversionResult
]


class _InputTensorsSplitter:
    """Splits the tensors of a `Conv2D` operator. Static tensors are split statically, and for dynamic tensors, a
    TFLite `Split` operator is added.
    """

    input_tensors: list[tflite_model.Tensor]
    weight_tensors: list[tflite_model.Tensor]
    bias_tensors: list[tflite_model.Tensor]
    split_ops: list[tflite_model.Operator]

    def __init__(
        self,
        input_tensor: tflite_model.Tensor,
        weight_tensor: tflite_model.Tensor,
        bias_tensor: tflite_model.Tensor,
        groups: int,
        builder: ModelBuilder,
    ):
        self.input_tensors = []
        self.weight_tensors = []
        self.bias_tensors = []
        self.split_ops = []

        inputs = [
            # input tensor, split by axis, output tensors container
            (input_tensor, -1, self.input_tensors),
            (weight_tensor, 0, self.weight_tensors),
            (bias_tensor, 0, self.bias_tensors),
        ]

        for i in inputs:
            if tensor_has_data(i[0]):
                self._generate_static_tensors(builder, groups, i[0], i[1], i[2])
            else:
                self._generate_dynamic_tensors(builder, groups, i[0], i[1], i[2])

    def _generate_dynamic_tensors(
        self, builder, groups, split_tensor, axis, target_list
    ):
        quantization = None
        if split_tensor.quantization is not None:
            if split_tensor.quantization.is_per_channel():
                scale = np.split(
                    np.array(split_tensor.quantization.scale.vector, "float32"), groups
                )
                zero_point = np.split(
                    np.array(split_tensor.quantization.zero_point.vector, "int32"),
                    groups,
                )
                quantization = [
                    tflite_model.Quantization(
                        scale=tflite_model.Scale(s),
                        zero_point=tflite_model.ZeroPoint(zp),
                    )
                    for s, zp in zip(scale, zero_point)
                ]
            else:
                quantization = [split_tensor.quantization] * groups

        split_op = self._create_split_op(builder, groups, split_tensor, axis)

        new_tensor_shape = split_tensor.shape.vector.copy()
        new_tensor_shape[axis] = new_tensor_shape[axis] // groups

        for i in range(groups):
            conv_split_tensor = builder.duplicate_tensor(
                split_tensor, name_suffix="_group_" + str(i)
            )
            conv_split_tensor.shape = tflite_model.Shape(new_tensor_shape)
            if quantization is not None:
                conv_split_tensor.quantization = copy(quantization[i])

            split_op.tmp_outputs.append(conv_split_tensor)
            target_list.append(conv_split_tensor)
        self.split_ops.append(split_op)

    # noinspection PyMethodMayBeStatic
    def _generate_static_tensors(
        self, builder, groups, split_tensor, axis, target_list
    ):
        quantization = None
        if split_tensor.quantization is not None:
            if split_tensor.quantization.is_per_channel():
                scale = np.split(
                    np.array(split_tensor.quantization.scale.vector, "float32"), groups
                )
                zero_point = np.split(
                    np.array(split_tensor.quantization.zero_point.vector, "int32"),
                    groups,
                )
                quantization = [
                    tflite_model.Quantization(
                        scale=tflite_model.Scale(s),
                        zero_point=tflite_model.ZeroPoint(zp),
                    )
                    for s, zp in zip(scale, zero_point)
                ]
            else:
                quantization = [split_tensor.quantization] * groups

        input_data = np.split(split_tensor.tmp_buffer.data, groups, axis)

        for i in range(len(input_data)):
            tensor_name = split_tensor.name + "_group_" + str(i)
            conv_input_tensor = builder.create_tensor_for_data(
                input_data[i], tensor_name
            )
            if quantization is not None:
                conv_input_tensor.quantization = copy(quantization[i])

            target_list.append(conv_input_tensor)

    # noinspection PyMethodMayBeStatic
    def _create_split_op(self, builder, groups, input_tensor, axis):
        axis_tensor = builder.create_tensor_for_data(
            np.asarray([axis], np.int32), "split_dim_"
        )
        input_split_op = tflite_model.Operator(
            builtin_options=split_options.Split(groups)
        )
        input_split_op.tmp_inputs = [axis_tensor, input_tensor]

        return input_split_op

    def get_input_tensor(self, idx) -> tflite_model.Tensor:
        return self.input_tensors[idx]

    def get_weight_tensor(self, idx) -> tflite_model.Tensor:
        return self.weight_tensors[idx]

    def get_bias_tensor(self, idx) -> tflite_model.Tensor:
        return self.bias_tensors[idx]

    def get_ops(self) -> list[tflite_model.Operator]:
        return self.split_ops


class _OutputTensorsCombiner:
    """Handles creation and aggregation of the TFLite Conv2D output tensors.
    Aggregation is done with `Concatenation` op.
    """

    output_tensors: list[tflite_model.Tensor]
    concat_op: tflite_model.Operator

    def __init__(self, output_tensor, groups, builder):
        self.output_tensors = []
        combine_axis = -1

        new_conv_output_shape = output_tensor.shape.vector.copy()
        new_conv_output_shape[combine_axis] = (
            new_conv_output_shape[combine_axis] // groups
        )
        conv_output_shape = tflite_model.Shape(new_conv_output_shape)

        self.concat_op = tflite_model.Operator(
            builtin_options=concatenation_options.Concatenation(combine_axis)
        )
        self.concat_op.tmp_outputs = [output_tensor]

        for i in range(groups):
            tensor_name = output_tensor.name + "_group_" + str(i)
            output_tensor = builder.duplicate_tensor(output_tensor, tensor_name)
            output_tensor.shape = conv_output_shape

            self.output_tensors.append(output_tensor)
            self.concat_op.tmp_inputs.append(output_tensor)

    def get_output_tensor(self, idx):
        return self.output_tensors[idx]

    def get_ops(self):
        return [self.concat_op]


def build_input_tensor_padding(
    t_op, conv_params: ConvParameters, builder, input_idx=0
) -> (Padding, tflite_model.Operator | None):
    """Build padding for input tensor of Conv2D op 't_op'."""

    tfl_padding, explicit_padding = aten_translator.convert_padding(conv_params.padding)
    if explicit_padding is not None:
        # Must add extra 'Pad' operator
        return tfl_padding, builder.create_pad_operator_before(
            t_op, input_idx, explicit_padding
        )

    return tfl_padding, None


def conv_op_factory(
    conv_params: ConvParameters,
    input_tensor: tflite_model.Tensor,
    weight_tensor: tflite_model.Tensor,
    bias_tensor: tflite_model.Tensor,
    output_tensor: tflite_model.Tensor,
    builder,
    builtin_options,
) -> OpsList:
    """Build padded 'Conv2D' TFLite operator. Padding is realized by 'builtin_options.padding' definition and by
    optional prepended 'Pad' operator.
    """

    conv_op = tflite_model.Operator(builtin_options=copy(builtin_options))
    conv_op.tmp_inputs = [input_tensor, weight_tensor, bias_tensor]
    conv_op.tmp_outputs = [output_tensor]

    padding, pad_op = build_input_tensor_padding(conv_op, conv_params, builder)
    conv_op.builtin_options.padding = padding

    if pad_op is not None:
        return OpsList(pre_ops=[pad_op], middle_op=conv_op)
    else:
        return OpsList(middle_op=conv_op)


# noinspection GrazieInspection
def create_separated_convolutions_based_on_group(
    t_op: tflite_model.Operator,
    conv_params: ConvParameters,
    builder: ModelBuilder,
    conv_conversion_fn: ConvConversionFn,
    conv_op_factory_fn: ConvOpFactory,
) -> list[tflite_model.Operator]:
    """Build a subgraph with multiple TFLite Conv2D operators that replace an `aten.convolution` operator with 'group'
     attribute higher than one. The number of new Conv2D operators corresponds to the number of groups. Input
     tensors of the Aten operator are split and distributed into related convolution operators. Outputs are then
     concatenated back together.

    Example: 'aten.convolution' operator with group=2 converted into TFLite subgraph will have
     the following structure (tensor dimensions are just for illustrative purposes):

                                                  │ (1,4,4,48)
                                              ┌───▼──┐
                                              │Split │
                                              └┬────┬┘
                                    (1,4,4,24) │    │ (1,4,4,24)
                                         ┌─────▼┐  ┌▼─────┐
                                         │Conv2D│  │Conv2D│
                                         └────┬─┘  └─┬────┘
                                    (1,4,4,18)│      │(1,4,4,18)
                                            ┌─▼──────▼──┐
                                            │Concatenate│
                                            └─────┬─────┘
                                                  │ (1,4,4,36)
                                                  ▼
    """

    conversion_result = conv_conversion_fn(t_op, conv_params)

    splitter = _InputTensorsSplitter(
        conversion_result.conv_input_tensor,
        conversion_result.conv_weight_tensor,
        conversion_result.conv_bias_tensor,
        conv_params.groups,
        builder,
    )
    combiner = _OutputTensorsCombiner(
        conversion_result.conv_output_tensor, conv_params.groups, builder
    )

    conv_ops = []
    for i in range(conv_params.groups):
        input_tensor = splitter.get_input_tensor(i)
        weight_tensor = splitter.get_weight_tensor(i)
        bias_tensor = splitter.get_bias_tensor(i)
        output_tensor = combiner.get_output_tensor(i)

        conv_builtin_options = cast(
            ConvBuiltinOptions, conversion_result.ops_list.middle_op.builtin_options
        )
        conv_ops_list = conv_op_factory_fn(
            conv_params,
            input_tensor,
            weight_tensor,
            bias_tensor,
            output_tensor,
            builder,
            conv_builtin_options,
        )

        conv_ops.extend(conv_ops_list.flatten())

    return (
        conversion_result.ops_list.pre_ops  # `Pad` operator
        + splitter.get_ops()
        + conv_ops
        + combiner.get_ops()  # Split, Conv2D, Concatenate ops
        + conversion_result.ops_list.post_ops
    )  # Currently not used
