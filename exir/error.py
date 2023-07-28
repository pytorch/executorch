# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict
from enum import Enum


class ExportErrorType(Enum):
    INVALID_INPUT_TYPE = (
        1  # User providing invalid inputs to either tracer, or other public facing APIs
    )
    INVALID_OUTPUT_TYPE = (
        2  # User returning values from their models that we don’t support.
    )
    VIOLATION_OF_SPEC = 3  # User doing something against our assumptions.
    NOT_SUPPORTED = 4  # User’s code contains types and functionalities we don’t support
    MISSING_PROPERTY = 5  # User's code didn't provide necessary details for us to successfully trace and export. For example, we use a lot of decorators and ask users to annotate their model.
    UNINITIALIZED = 6  # User is usingan API without proper initialization step.


def internal_assert(pred: bool, assert_msg: str) -> None:
    """
    This is exir's custom assert method. It internally just throws InternalError.
    Note that the sole purpose is to throw our own error while maintaining similar syntax
    as python assert.
    """

    if not pred:
        raise InternalError(assert_msg)


class InternalError(Exception):
    """
    Raised when an internal invariance is violated in EXIR stack.
    Should hint users to report a bug to dev and expose the original
    error message.
    """

    def __init__(self, message: str) -> None:
        super().__init__(message)


class ExportError(Exception):
    """
    This type of exception is raised for errors that are directly caused by the user
    code. In general, user errors happen during model authoring, tracing, using our public
    facing APIs, and writing graph passes.
    """

    def __init__(self, error_code: ExportErrorType, message: str) -> None:
        prefix = f"[{error_code}]: "
        super().__init__(prefix + message)
