# Copyright 2025-2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
"""Provide validation helpers for operator inputs and dtypes.

Use these utilities to validate input counts, ensure dtype consistency, check
allowed dtypes, and compute pooling padding adjustments.

"""

from math import ceil, floor
from typing import Any, List, Optional

from executorch.backends.arm.tosa.specification import Tosa_1_00, TosaSpecification


def validate_num_inputs(op_name: str, inputs: List[Any], expected: int | List[int]):
    """Validate the number of inputs against expected values.

    This function checks whether the length of the input list matches the
    expected number(s) of inputs.

    Args:
        op_name (str): The name of the operation for which the inputs are being
            validated. Used in the error message to provide context.
        inputs (List[TosaArg]): A list of inputs to be validated, where each
            input is assumed to be an instance of ``TosaArg``.
        expected (int | List[int]): The expected number of inputs. Can be either
            an integer or a list of integers.

    Raises:
        ValueError: If the number of inputs does not match the expected
            value(s); the message indicates the operation name and the mismatch
            in expected versus provided counts.

    Example:
        from executorch.backends.arm.operators.operator_validation_utils import \
            validate_num_inputs

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
    """Validate that all given tensors have the same dtype.

    This function checks whether all items in the ``tensors`` list have the
    same ``dtype`` as the first item.

    Args:
        op_name (str): The name of the operation for which the dtype validation
            is being performed. Used in the error message to provide context.
        tensors (List[Any]): A list of tensors to be validated, each assumed to
            have a ``dtype`` attribute.
        ts (Optional[Any]): TOSA serializer (optional) to improve readability of
            dtype names in error messages.

    Raises:
        ValueError: If the dtype of any item in the list does not match the
            dtype of the first item, or if the list is empty.

    Example:
        from executorch.backends.arm.operators.operator_validation_utils import \
            validate_same_dtype

        validate_same_dtype(self.target, [input1, input2, output])

    """
    if not tensors:
        raise ValueError(
            f"{op_name}: Input tensor list is empty, cannot validate dtypes"
        )

    # Get dtype of the first tensor to reference for comparison
    reference_dtype = tensors[0].dtype
    reference_dtype_name = str(reference_dtype)

    for tensor in tensors:
        if tensor.dtype != reference_dtype:
            inconsistent_dtype_name = str(tensor.dtype)
            raise ValueError(
                f"{op_name}: Expected all tensors to have dtype {reference_dtype_name}, "
                f"but found inconsistent dtype {inconsistent_dtype_name}."
            )


def validate_valid_dtype(
    op_name: str, tensors: Any | List[Any], valid_dtypes: Any | List[Any], tosa_spec
):
    """Validate that one or more tensors have allowed dtypes.

    This function checks whether the ``dtype`` attribute of the provided
    tensor(s) is one of the valid dtype values. It supports checking a single
    tensor or a list of tensors.

    Args:
        op_name (str): The name of the operation performing the validation.
        tensors (Any | List[Any]): A tensor or list of tensors (each assumed to
            have ``dtype`` and ``name`` attributes) whose dtype will be
            validated.
        valid_dtypes (Any | List[Any]): A dtype enum or list of dtype enums
            representing allowed dtype values.
        tosa_spec (Any): A TosaSpecification instance indicating which TOSA
            version is targeted. This determines which serializer to use for
            dtype name resolution.

    Raises:
        ValueError: If no tensors are provided, or if any tensor has a dtype not
            in ``valid_dtypes``.

    Example:
        from executorch.backends.arm.operators.operator_validation_utils import \
            validate_valid_dtype
        import serializer.tosa_serializer as ts

        validate_valid_dtype(
            self.target,
            [*inputs, output],
            [ts.DType.INT8, ts.DType.INT32],
            self.tosa_spec,
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
            valid_names = [str(dtype) for dtype in valid_dtypes]
            got_name = str(tensor.dtype)
            raise ValueError(
                f"Expected tensor {tensor.name} in {op_name} to have one of the "
                f"following dtypes: {valid_names}, got: {got_name}"
            )


def validate_cf_extension(op_name: str, tosa_spec: TosaSpecification) -> None:
    """Ensure that the requested control-flow operator is supported by the
    active TOSA spec.
    """
    if not isinstance(tosa_spec, Tosa_1_00):
        raise ValueError(
            f"Got TOSA version {tosa_spec.version}, that does not support extensions."
        )
    if not tosa_spec.support_extension("cf"):
        raise ValueError(
            f"Trying to lower {op_name}, but TOSA specification {tosa_spec} does not "
            "support the cf extension."
        )


def adjust_pooling_pad_if_needed(
    input_size: int, kernel_size: int, stride: int, pad: int, ceil_mode: bool
) -> int:
    """Compute the post padding needed for pooling.

    ATen pooling uses a single symmetric ``pad`` per dimension and rounds the
    output size up or down depending on ``ceil_mode``. TOSA requires distinct
    pre- and post-padding values that satisfy:

        output_size == (input_size + pre_pad + post_pad - kernel_size) / stride + 1

    This function returns the required ``post_pad`` given a symmetric ``pad``.

    Args:
        input_size (int): Input size.
        kernel_size (int): Kernel size.
        stride (int): Stride size.
        pad (int): Symmetric padding specified by ATen.
        ceil_mode (bool): Use ceil when computing output size.

    Returns:
        int: Post-padding to satisfy the TOSA formula.

    """
    if ceil_mode:
        output_size = ceil((input_size - kernel_size + 2 * pad) / stride) + 1
    else:
        output_size = floor((input_size - kernel_size + 2 * pad) / stride) + 1

    # Solve for post_pad from
    # output_size = (input_size + pre_pad + post_pad - kernel_size) / stride + 1
    adjusted_post_pad = (output_size - 1) * stride - input_size + kernel_size - pad

    return adjusted_post_pad
