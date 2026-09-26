# Copyright (c) Qualcomm Innovation Center, Inc.
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Optional


class PipelineError(Exception):
    """Base exception for all GenAI pipeline errors."""


class StageError(PipelineError):
    """Error during pipeline stage execution.

    Attributes:
        stage_name: The name of the stage that failed.
        original_exception: The underlying exception that caused the failure.
    """

    def __init__(
        self,
        stage_name: str,
        message: str,
        original_exception: Optional[Exception] = None,
    ) -> None:
        self.stage_name = stage_name
        self.original_exception = original_exception
        full_message = f"[{stage_name}] {message}"
        if original_exception:
            full_message += f" (caused by {type(original_exception).__name__}: {original_exception})"
        super().__init__(full_message)
        # Set __cause__ explicitly so exception chaining works even without
        # 'raise StageError(...) from e' syntax.
        if original_exception:
            self.__cause__ = original_exception


class ConfigValidationError(PipelineError):
    """Error during config validation at build time (before any stage runs)."""


class EngineNotAvailableError(PipelineError):
    """Raised when a requested engine is not installed or available."""

    def __init__(self, engine_name: str, message: Optional[str] = None) -> None:
        self.engine_name = engine_name
        full_message = f"Engine '{engine_name}' is not available"
        if message:
            full_message += f": {message}"
        super().__init__(full_message)
