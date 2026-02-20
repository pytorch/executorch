# Copyright 2025 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
"""Type checking utilities."""

from typing import TypeVar

T = TypeVar("T")


def ensure_type(expected_type: type[T], arg: object) -> T:
    """Ensure that the argument is of the expected type.

    Args:
        expected_type (type[T]): The expected type.
        arg (object): The argument to check.

    Returns:
        T: The argument, if it is of the expected type.

    """
    if isinstance(arg, expected_type):
        return arg

    expected_name = getattr(expected_type, "__name__", str(expected_type))
    actual_name = type(arg).__name__
    raise TypeError(f"Expected value of type {expected_name}, got {actual_name!r}")
