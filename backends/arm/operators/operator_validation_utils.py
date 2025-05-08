# Copyright 2025 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any, List


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
