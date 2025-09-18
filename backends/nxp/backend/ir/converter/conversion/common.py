#
# Copyright 2023 Martin Pavella
# Copyright 2023-2025 NXP
#
# License: MIT
# See the LICENSE_MIT for more details.
#
"""
    common

This file contains functions shared by the various files in the 
'conversion/builtin/' directory.
"""

from typing import List, MutableSequence, Optional

import executorch.backends.nxp.backend.ir.logger as logger
from executorch.backends.nxp.backend.ir.tflite_generator import tflite_model
from executorch.backends.nxp.backend.ir.tflite_generator.builtin_options import (
    average_pool_2d_options,
    conv_2d_options,
    max_pool_2d_options,
    transpose_conv_options,
)

from torch.fx import Node


def try_get_input(t_op: tflite_model.Operator, idx: int) -> tflite_model.Tensor | None:
    """Return the input tensors of 't_op' at index 'idx', or None if the operator doesn't have that input.

        This function should ALWAYS be used to get optional input tensors.

    :param t_op: TFLite operator to get the input tensor from.
    :param idx: Index of the input tensor to get.
    :return: The input tensor at index 'idx', or None.
    """

    if len(t_op.tmp_inputs) < idx + 1:
        # The operator doesn't have that many inputs.
        return None

    tensor = t_op.tmp_inputs[idx]

    return tensor


def extend_1d_padding_to_2d(tflite_1d_padding: MutableSequence):
    """Extend the PyTorch 'padding' operator attribute that represents padding for a 1D kernel to 2D, by adding '0's."""
    if tflite_1d_padding is not None:
        tflite_1d_padding.append(0)


def extend_1d_stride_to_2d(tflite_1d_stride: MutableSequence):
    """Extend the PyTorch 'stride' operator attribute that represents stride for a 1D kernel to 2D, by adding '1'."""
    if tflite_1d_stride is not None:
        tflite_1d_stride.append(1)


def extend_1d_dilation_to_2d(tflite_1d_dilation: MutableSequence):
    """Extend the PyTorch 'dilation' operator attribute that represents dilation for a 1D kernel to 2D, by adding '1'."""
    if tflite_1d_dilation is not None:
        tflite_1d_dilation.append(1)


StridedOptions = (
    average_pool_2d_options.AveragePool2D
    | conv_2d_options.Conv2D
    | max_pool_2d_options.MaxPool2D
    | transpose_conv_options.TransposeConv
)


def assign_2d_strides(options: StridedOptions, strides: Optional[List[int]]):
    """Assign to 'obj' the attributes 'stride_h' and 'stride_w' from 'strides'.
         If 'strides' is None, assign 1s.

    :param options: TFLite AveragePool2D, Conv2D, MaxPool2D or TransposeConv options object.
    :param strides: An optional list of ONNX strides attribute.
    """

    if strides is None:
        # Default values are [1, 1]
        options.stride_h = 1
        options.stride_w = 1

    elif len(strides) == 2:
        options.stride_h = strides[0]
        options.stride_w = strides[1]

    else:
        logger.e(
            logger.Code.INVALID_ONNX_OPERATOR_ATTRIBUTE,
            f"ONNX operator has invalid 'strides' attribute! ('{strides}')",
        )


def assign_2d_dilations(conv_2d: conv_2d_options, dilations: Optional[List[int]]):
    """Assign the 'conv_2d' attributes 'dilations_h' and 'dilations_2' from 'dilations'."""

    if dilations is None:
        return

    if len(dilations) == 2:
        conv_2d.dilation_h_factor = dilations[0]
        conv_2d.dilation_w_factor = dilations[1]
    else:
        logger.d(f"Expected 2D dilations, got '{dilations}'. Leaving default values.")


def uses_shape_broadcasting(t_op: tflite_model.Operator) -> bool:
    """Determine if given TFLite operator uses shape broadcasting for it's input tensors or not.

    :param t_op: TFLite operator with 'tmp_inputs' initialized.
    :return: True, if the operator uses shape broadcasting for it's input tensors.
             False otherwise.
    """

    if t_op.tmp_inputs is None:
        logger.e(
            logger.Code.INTERNAL_ERROR,
            "common.uses_shape_broadcasting(): 'tmp_inputs' are None!",
        )

    if len(t_op.tmp_inputs) == 0:
        logger.e(
            logger.Code.INTERNAL_ERROR,
            "common.uses_shape_broadcasting(): Operator has no inputs!",
        )

    first_input_shape = t_op.tmp_inputs[0].shape

    return any(
        input_tensor.shape != first_input_shape for input_tensor in t_op.tmp_inputs[1:]
    )


def node_uses_shape_broadcasting(node: Node) -> bool:
    """Determine if given PyTorch fx Node uses shape broadcasting for it's input nodes or not.

    :param node: PyTorch fx Node with 'all_input_nodes' initialized.
    :return: True, if the node uses shape broadcasting for it's input nodes.
             False otherwise.
    """

    if node.all_input_nodes is None:
        logger.e(
            logger.Code.INTERNAL_ERROR,
            "common.node_uses_shape_broadcasting(): 'all_input_nodes' are None!",
        )

    if len(node.all_input_nodes) == 0:
        logger.e(
            logger.Code.INTERNAL_ERROR,
            "common.node_uses_shape_broadcasting(): Operator has no inputs!",
        )

    first_input_shape = node.all_input_nodes[0].meta["val"].shape

    return any(
        input_tensor.meta["val"].shape != first_input_shape
        for input_tensor in node.all_input_nodes[1:]
    )


class OpsList:
    """
    Holder of TFLite operator (middle_op) that can be prefixed (pre_ops) of suffixed (post_ops)
    by other operators. When flattened, order of the operators is preserved.
    """

    pre_ops: List[tflite_model.Operator]
    middle_op: tflite_model.Operator
    post_ops: List[tflite_model.Operator]

    def __init__(
        self,
        pre_ops: List[tflite_model.Operator] | None = None,
        middle_op=None,
        post_ops: List[tflite_model.Operator] | None = None,
    ):
        self.pre_ops = pre_ops or []
        self.middle_op = middle_op
        self.post_ops = post_ops or []

    def flatten(self):
        return self.pre_ops + [self.middle_op] + self.post_ops

    def add_pre(self, ops: tflite_model.Operator | list[tflite_model.Operator]):
        if isinstance(ops, tflite_model.Operator):
            ops = [ops]

        logger.internal_assert(
            isinstance(ops, list), "OpsList: add_pre() called with invalid value."
        )

        self.pre_ops.extend(ops)

    def add_post(self, ops: tflite_model.Operator | list[tflite_model.Operator]):
        if isinstance(ops, tflite_model.Operator):
            ops = [ops]

        logger.internal_assert(
            isinstance(ops, list), "OpsList: add_post() called with invalid value."
        )

        self.post_ops.extend(ops)
