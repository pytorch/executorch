#
# Copyright 2023 Martin Pavella
# Copyright 2023-2024 NXP
#
# License: MIT
# See the LICENSE_MIT for more details.
#
"""
    common

This file contains functions shared by the various files in the 
'conversion/builtin/' directory.
"""

from typing import Any, List, MutableSequence, Optional

import executorch.backends.nxp.backend.ir.logger as logger
from executorch.backends.nxp.backend.ir.tflite_generator import tflite_model
from executorch.backends.nxp.backend.ir.tflite_generator.builtin_options import (
    average_pool_2d_options,
    conv_2d_options,
    max_pool_2d_options,
    transpose_conv_options,
)


def exactly_one_is_none(obj1: Optional, obj2: Optional) -> bool:
    """Determine if exactly 1 of the arguments is None, or not."""
    return (obj1 is None and obj2 is not None) or (obj1 is not None and obj2 is None)


def contains_duplicates(list_to_check: List[Any]) -> bool:
    """Determine if given list has duplicate elements or not."""
    return len(list_to_check) != len(set(list_to_check))


def clamp(val: int, start: int, end: int) -> int:
    """Clamp an int value between start and end (inclusive) and return it."""
    if val < start:
        return start

    elif val > end:
        return end

    return val


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

    if tensor.name == "":
        # ONNX allows the name "" for optional tensors. It indicates that the tensor should be ignored, and a default
        #  value should be used. Just like if the tensor was omitted altogether.
        return None

    return tensor


def extend_1d_pads_to_2d(onnx_1d_pads: MutableSequence):
    """Extend the onnx 'pads' operator attribute that represents padding for a 1D kernel to 2D, by adding '0's."""
    if onnx_1d_pads is not None:
        onnx_1d_pads.insert(1, 0)
        onnx_1d_pads.append(0)


def extend_1d_strides_to_2d(onnx_1d_strides: MutableSequence):
    """Extend the onnx 'strides' operator attribute that represents strides for a 1D kernel to 2D, by adding '1'."""
    if onnx_1d_strides is not None:
        onnx_1d_strides.append(1)


def extend_1d_dilations_to_2d(onnx_1d_dilations: MutableSequence):
    """Extend the onnx 'dilations' operator attribute that represents dilations for a 1D kernel to 2D, by adding '1'."""
    if onnx_1d_dilations is not None:
        onnx_1d_dilations.append(1)


def extend_1d_kernel_shape_to_2d(onnx_1d_kernel_shape: MutableSequence):
    """Extend the onnx 1D 'kernel_shape' operator attribute to 2D, by adding '1'."""
    if onnx_1d_kernel_shape is not None:
        onnx_1d_kernel_shape.append(1)


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


def uses_multiple_input_types(t_op: tflite_model.Operator) -> bool:
    """Determine if the input tensors of given TFLite operator use different data types or not.

    :param t_op: TFLite operator with 'tmp_inputs' initialized.
    :return: True, if any two input tensors have a different data type.
             False, if all input tensors use the same data type.
    """

    if t_op.tmp_inputs is None:
        logger.e(
            logger.Code.INTERNAL_ERROR,
            "common.uses_multiple_input_types(): 'tmp_inputs' are None!",
        )

    if len(t_op.tmp_inputs) == 0:
        logger.e(
            logger.Code.INTERNAL_ERROR,
            "common.uses_multiple_input_types(): Operator has no inputs!",
        )

    first_input_type = t_op.tmp_inputs[0].type
    return any(
        input_tensor.type != first_input_type for input_tensor in t_op.tmp_inputs[1:]
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
