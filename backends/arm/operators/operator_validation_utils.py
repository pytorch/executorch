# Copyright 2025 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from math import ceil, floor
from typing import Any, List, Optional

import serializer.tosa_serializer as ts


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


def validate_valid_dtype(
    op_name: str, tensors: Any | List[Any], valid_dtypes: Any | List[Any], tosa_spec
):
    """
    Validates that one or more tensors have dtypes within a set of allowed dtypes.

    This function checks whether the `dtype` attribute of the provided tensor(s) is one
    of the valid dtype values. It supports checking a single tensor or a list of
    tensors.

    Parameters:
    -----------
    op_name : str
        The name of the operation performing the validation.
    tensors : Any or List[Any]
        A tensor or list of tensors (each assumed to have `dtype` and `name` attributes)
        whose dtype will be validated.
    valid_dtypes : Any or List[Any]
        A dtype enum or list of dtype enums representing allowed dtype values.
    tosa_spec : Any
        A TosaSpecification instance indicating which TOSA version is targeted. This
        determines which serializer to use for dtype name resolution.

    Raises:
    -------
    ValueError
        If no tensors are provided, or if any tensor has a dtype not in `valid_dtypes`.

    Example:
    --------
    # Example usage:
    from executorch.backends.arm.operators.operator_validation_utils import (
        validate_valid_dtype,
    )

    import serializer.tosa_serializer as ts

    validate_valid_dtype(
        self.target,
        [*inputs, output],
        [ts.DType.INT8, ts.DType.INT32],
        output.tosa_spec,
    )

    """

    if not tensors:
        raise ValueError(
            f"{op_name}: Input tensor list is empty, cannot validate dtypes"
        )

    if not isinstance(valid_dtypes, List):
        valid_dtypes = [valid_dtypes]

    if not isinstance(tensors, List):
        tensors = [tensors]

    for tensor in tensors:
        if tensor.dtype not in valid_dtypes:
            raise ValueError(
                f"Expected tensor {tensor.name} in {op_name} to have one of the "
                f"following dtypes: {[ts.DTypeNames[i] for i in valid_dtypes]}, "
                f"got: {ts.DTypeNames[tensor.dtype]}"
            )


def adjust_pooling_pad_if_needed(
    input_size: int, kernel_size: int, stride: int, pad: int, ceil_mode: bool
) -> int:
    """
    The Aten pooling ops has one value 'pad' per dimension to specify padding, but they
    do not require input and output sizes to match up perfectly. Instead, the output
    size is rounded up or down depending on ceil_mode, and padding at the end of the
    input is automatically added or removed. TOSA on the other hand specifies two
    padding values, one for pre-padding and one for post-padding, and these must satisfy

        output_size = (input_size + pre_pad + post_pad - kernel_size) / stride + 1

    This function returns the post_pad value required to satisfy the above condition.

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
    An int, giving the post-padding to use for the
    """

    if ceil_mode:
        output_size = ceil((input_size - kernel_size + 2 * pad) / stride) + 1
    else:
        output_size = floor((input_size - kernel_size + 2 * pad) / stride) + 1

    # Solve for post_pad from
    # output_size = (input_size + pre_pad + post_pad - kernel_size) / stride + 1
    adjusted_post_pad = (output_size - 1) * stride - input_size + kernel_size - pad

    return adjusted_post_pad
