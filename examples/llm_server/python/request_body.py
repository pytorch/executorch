# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Bound chat request aggregation before FastAPI JSON/Pydantic parsing."""

from fastapi.responses import JSONResponse

from .errors import APIError

MAX_HTTP_BODY_BYTES = 1024 * 1024


def _body_error(status, message):
    return APIError(
        status,
        message,
        "invalid_request_error",
        "request_too_large" if status == 413 else "invalid_request",
    )


def _content_length(headers, max_bytes):
    lengths = [value for key, value in headers if key.lower() == b"content-length"]
    if not lengths:
        return None
    value = lengths[0]
    if len(lengths) != 1 or not value or any(c < 48 or c > 57 for c in value):
        raise _body_error(400, "Invalid Content-Length.")
    # Compare decimal lengths without constructing unbounded integers.
    value = value.lstrip(b"0") or b"0"
    if len(value) > len(str(max_bytes)):
        raise _body_error(413, "Request body exceeds the byte limit.")
    length = int(value)
    if length > max_bytes:
        raise _body_error(413, "Request body exceeds the byte limit.")
    return length


class BoundedChatBody:
    """Bound chat body aggregation and replay the unchanged ASGI disconnect stream."""

    def __init__(self, app, max_bytes=MAX_HTTP_BODY_BYTES, enabled=None):
        if type(max_bytes) is not int or max_bytes <= 0:
            raise ValueError("max_bytes must be a positive integer")
        self.app = app
        self.max_bytes = max_bytes
        self.enabled = enabled

    async def __call__(self, scope, receive, send):
        """Validate length and chunks before invoking the application parser."""
        if (
            scope["type"] != "http"
            or scope.get("path", "").rstrip("/") != "/v1/chat/completions"
            or (self.enabled is not None and not self.enabled())
        ):
            return await self.app(scope, receive, send)

        async def reject(status, message):
            error = _body_error(status, message)
            await JSONResponse(error.body(), status_code=status)(scope, receive, send)

        try:
            length = _content_length(scope.get("headers", []), self.max_bytes)
        except APIError as error:
            return await JSONResponse(error.body(), status_code=error.status)(
                scope, receive, send
            )
        body = bytearray()
        while True:
            message = await receive()
            if message["type"] == "http.disconnect":
                return
            chunk = message.get("body", b"")
            if len(chunk) > self.max_bytes - len(body):
                return await reject(413, "Request body exceeds the byte limit.")
            body.extend(chunk)
            if not message.get("more_body", False):
                break
        if length is not None and len(body) != length:
            return await reject(400, "Content-Length does not match the request body.")

        replayed = False

        async def bounded_receive():
            nonlocal replayed
            if not replayed:
                replayed = True
                payload = bytes(body)
                body.clear()
                return {"type": "http.request", "body": payload, "more_body": False}
            # Preserve the original disconnect stream after replaying the body.
            return await receive()

        await self.app(scope, bounded_receive, send)
