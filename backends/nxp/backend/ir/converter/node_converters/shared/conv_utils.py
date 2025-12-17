# Copyright 2023-2025 NXP
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass

from executorch.backends.nxp.backend.ir.converter.conversion.common import OpsList
from executorch.backends.nxp.backend.ir.tflite_generator import tflite_model

from torch.fx import Node


@dataclass
class ConvParameters:
    stride: list[int]
    padding: list[int]
    dilation: list[int]
    transposed: bool
    out_padding: list[int]
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


def get_node_tensor_params(node: Node) -> dict:
    node_tensor_params = {}

    input_tensor = node.args[0]
    assert len(input_tensor.meta["val"].shape) in [3, 4], "Supports only Conv 1D, 2D."
    node_tensor_params["batch_size"] = input_tensor.meta["val"].shape[0]
    node_tensor_params["inp_channels"] = input_tensor.meta["val"].shape[1]
    node_tensor_params["inp_height"] = input_tensor.meta["val"].shape[2]
    if len(input_tensor.meta["val"].shape) == 4:
        node_tensor_params["inp_width"] = input_tensor.meta["val"].shape[3]

    weights = node.args[1]
    node_tensor_params["out_channels"] = node.meta["val"].shape[1]
    node_tensor_params["out_height"] = node.meta["val"].shape[2]
    if len(node.meta["val"].shape) == 4:
        node_tensor_params["out_width"] = node.meta["val"].shape[3]
    node_tensor_params["kernel_height"] = weights.meta["val"].shape[2]
    if len(weights.meta["val"].shape) == 4:
        node_tensor_params["kernel_width"] = weights.meta["val"].shape[3]

    return node_tensor_params


def group_conv_convertible_as_depthwise(node: Node | tflite_model.Operator, group: int):
    input_channels, output_channels = _get_IO_channels(node)

    return input_channels == output_channels == group


def group_conv_convertible_into_multiple_convolutions(
    node: Node | tflite_model.Operator, group: int
) -> bool:
    if group == 1:
        return False

    if group_conv_convertible_as_depthwise(node, group):
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
        output_shape_tensor: tflite_model.Tensor | None = None,
    ):
        self.conv_input_tensor = input_tensor
        self.conv_weight_tensor = weight_tensor
        self.conv_bias_tensor = bias_tensor
        self.conv_output_tensor = output_tensor
        self.output_shape_tensor = output_shape_tensor
        self.ops_list = OpsList()
