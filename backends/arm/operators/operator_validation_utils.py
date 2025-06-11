# Copyright 2025 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any, List, Optional


def validate_num_inputs(op_name: str, inputs: List[Any], expected: int | List[int]):
    """
    Validates the number of inputs provided to an operation against expected values.

    This function checks whether the length of the input list matches the expected
    number(s) of inputs.

    Parameters:
    -----------
    op_name : str
        The name of the operation for which the inputs are being validated.
        Used in the error message to provide context.

    inputs : List[TosaArg]
        A list of inputs to be validated, where each input is assumed to be an
        instance of `TosaArg`.

    expected : int or List[int]
        The expected number of inputs. Can be either an integer or a list of integers.

    Raises:
    -------
    ValueError
        If the number of inputs does not match the expected value(s), a `ValueError` is
        raised with a message indicating the operation name and the mismatch in expected
        versus provided number of inputs.

    Example:
    --------
    # Example usage:
    from executorch.backends.arm.operators.operator_validation_utils import (
        validate_num_inputs,
    )

    validate_num_inputs(self.target, inputs, [3, 4])
    """
    if isinstance(expected, int):
        expected = [expected]
    if len(inputs) not in expected:
        expected_str = ", ".join(map(str, expected))
        raise ValueError(
            f"{op_name}: Expected number of input(s) to be "
            f"[{expected_str}], got {len(inputs)}"
        )


def validate_same_dtype(op_name: str, tensors: List[Any], ts: Optional[Any] = None):
    """
    Validates that all given tensors have the same dtype attribute.

    This function checks whether all items in the `tensors` list have the same
    `dtype` as the first item.

    Parameters:
    -----------
    op_name : str
        The name of the operation for which the dtype validation is being performed.
        Used in the error message to provide context.

    tensors : List[Any]
        A list of tensors to be validated, each is assumed to have a `dtype` attribute.

    ts: Optional[Any]
        TOSA serializer. Not required but only to get clearer error messages.

    Raises:
    -------
    ValueError
        If the dtype of any item in the list does not match the dtype of the first item,
        a `ValueError` is raised with a message indicating the operation name and the
        mismatch in dtypes.

    Example:
    --------
    # Example usage:
    from executorch.backends.arm.operators.operator_validation_utils import (
        validate_same_dtype,
    )

    validate_same_dtype(self.target, [input1, input2, output])

    """
    if not tensors:
        raise ValueError(
            f"{op_name}: Input tensor list is empty, cannot validate dtypes"
        )

    # Get dtype of the first tensor to reference for comparison
    reference_dtype = tensors[0].dtype

    for tensor in tensors:
        ref_dtype_name = (
            ts.DTypeNames[reference_dtype] if ts is not None else str(reference_dtype)
        )
        inconsistent_dtype_name = (
            ts.DTypeNames[tensor.dtype] if ts is not None else str(tensor.dtype)
        )
        if tensor.dtype != reference_dtype:
            raise ValueError(
                f"{op_name}: Expected all tensors to have dtype {ref_dtype_name}, but "
                f"found inconsistent dtype {inconsistent_dtype_name}."
            )


def adjust_pooling_pad_if_needed(
    input_size: int, kernel_size: int, stride: int, pad: int
) -> int:
    """
    Calculates the padding that needs to be removed to a pooling window to make it
    divisible by the kernels stride. All inputs should correspond to the same dimension.

    Parameters:
    -----------
    input_size : int
        The size of the input to the operator.

    kernel_size : int
        The size of the kernel.

    stride : int
        The size of the stride.

    pad : int
        The amount of padding.

    Output:
    -------
    An int, representing the padding to remove to make the window divisible.
    """
    if pad == 0:
        return pad

    mod_remainder = (input_size + 2 * pad - kernel_size) % stride

    # No need to adjust
    if mod_remainder == 0:
        return pad

    return pad - mod_remainder
