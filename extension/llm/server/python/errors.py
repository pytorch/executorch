# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""OpenAI-shaped API errors.

Raising these lets the server return a structured `{"error": {...}}` body with
the right HTTP status instead of dropping the connection.
"""

from typing import Optional


class APIError(Exception):
    def __init__(
        self, status: int, message: str, err_type: str, code: Optional[str] = None
    ):
        super().__init__(message)
        self.status = status
        self.message = message
        self.err_type = err_type
        self.code = code

    def body(self) -> dict:
        return {
            "error": {"message": self.message, "type": self.err_type, "code": self.code}
        }


class ContextLengthExceeded(APIError):
    def __init__(self, num_tokens: int, max_context: int):
        super().__init__(
            status=400,
            message=(
                f"This model's maximum context length is {max_context} tokens, "
                f"but the request has {num_tokens} prompt tokens."
            ),
            err_type="invalid_request_error",
            code="context_length_exceeded",
        )


class GenerationError(APIError):
    def __init__(self, detail: str):
        super().__init__(
            status=500, message=f"Generation failed: {detail}", err_type="server_error"
        )
