# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Bounded inline image validation and template-independent prompt bindings.

Only container metadata is inspected here: no pixels are allocated or decoded.
The native CPU image-preparation hook must fully decode and validate the image
under its own allocation limits before model preparation.
"""

import base64
import struct
import uuid
import zlib
from dataclasses import dataclass

from .errors import APIError

MAX_IMAGE_BYTES = 512 * 1024
MAX_IMAGE_PIXELS = 4 * 1024 * 1024
MAX_IMAGE_DIMENSION = 4096
MAX_IMAGES = 1


def _invalid(message):
    return APIError(400, message, "invalid_request_error", "invalid_image")


@dataclass(frozen=True)
class ImageLimits:
    """Positive native capability limits, further bounded by local hard caps."""

    max_images: int
    max_image_bytes: int
    max_image_pixels: int
    max_image_dimension: int = MAX_IMAGE_DIMENSION

    def __post_init__(self):
        """Reject ambiguous capability values, including bool-as-int limits."""
        for name in (
            "max_images",
            "max_image_bytes",
            "max_image_pixels",
            "max_image_dimension",
        ):
            value = getattr(self, name)
            if type(value) is not int or value <= 0:
                raise ValueError(f"{name} must be a positive integer")


def has_images(messages):
    """Whether an OpenAI history includes inline-image content parts."""
    return any(
        isinstance(message.content, list)
        and any(part.get("type") == "image_url" for part in message.content)
        for message in messages
    )


def _png_dimensions(data):
    if not data.startswith(b"\x89PNG\r\n\x1a\n"):
        raise _invalid("Image MIME type does not match PNG signature.")
    pos = 8
    dimensions = None
    saw_data = False
    while pos + 12 <= len(data):
        size, kind = struct.unpack_from(">I4s", data, pos)
        end = pos + 12 + size
        if end > len(data):
            raise _invalid("Truncated PNG chunk.")
        chunk = memoryview(data)[pos + 4 : end - 4]
        crc = struct.unpack_from(">I", data, end - 4)[0]
        if zlib.crc32(chunk) != crc:
            raise _invalid("Invalid PNG checksum.")
        if dimensions is None and kind != b"IHDR":
            raise _invalid("PNG must begin with IHDR.")
        if kind == b"IHDR":
            if dimensions is not None or size != 13:
                raise _invalid("Invalid PNG IHDR.")
            width, height, depth, color, compression, filtering, interlace = (
                struct.unpack_from(">IIBBBBB", data, pos + 8)
            )
            depths = {
                0: (1, 2, 4, 8, 16),
                2: (8, 16),
                3: (1, 2, 4, 8),
                4: (8, 16),
                6: (8, 16),
            }
            if (
                depth not in depths.get(color, ())
                or compression
                or filtering
                or interlace > 1
            ):
                raise _invalid("Invalid PNG header.")
            dimensions = width, height
        elif kind == b"IDAT":
            saw_data = True
        elif kind == b"IEND":
            if size or end != len(data) or not saw_data:
                raise _invalid("Invalid PNG end marker.")
            return dimensions
        pos = end
    raise _invalid("Incomplete PNG image.")


def _jpeg_segment(data, pos):
    if data[pos] != 0xFF:
        raise _invalid("Invalid JPEG marker.")
    while pos < len(data) and data[pos] == 0xFF:
        pos += 1
    if pos >= len(data):
        raise _invalid("Incomplete JPEG image.")
    marker = data[pos]
    pos += 1
    if marker in (0x00, 0xD8, 0xD9) or 0xD0 <= marker <= 0xD7:
        raise _invalid("Unexpected JPEG marker.")
    if pos + 2 > len(data):
        raise _invalid("Incomplete JPEG image.")
    size = struct.unpack_from(">H", data, pos)[0]
    if size < 2 or pos + size > len(data):
        raise _invalid("Truncated JPEG segment.")
    return marker, pos, size


def _jpeg_dimensions(data):
    if not data.startswith(b"\xff\xd8") or not data.endswith(b"\xff\xd9"):
        raise _invalid("Image MIME type does not match a complete JPEG.")
    pos = 2
    dimensions = None
    # Length-delimited JPEG marker segments can be inspected without decoding
    # entropy data. Reject deferred height (DNL) and unsupported frame formats.
    while pos < len(data) - 2:
        marker, pos, size = _jpeg_segment(data, pos)
        if marker in (0xC0, 0xC1, 0xC2):
            if dimensions is not None or size < 8:
                raise _invalid("Invalid JPEG frame.")
            depth, height, width, components = struct.unpack_from(
                ">BHHB", data, pos + 2
            )
            if depth != 8 or components not in (1, 3, 4) or size != 8 + 3 * components:
                raise _invalid("Unsupported JPEG frame.")
            dimensions = width, height
        elif marker == 0xDA:
            if dimensions is None or size < 6:
                raise _invalid("JPEG scan has no valid frame.")
            return dimensions
        pos += size
    raise _invalid("Incomplete JPEG image.")


def validate_image(mime_type, encoded, limits):
    """Validate canonical base64 and container bounds without allocating pixels."""
    if mime_type not in ("image/png", "image/jpeg"):
        raise _invalid("Only inline image/png and image/jpeg are supported.")
    maximum = min(limits.max_image_bytes, MAX_IMAGE_BYTES)
    if (
        not isinstance(encoded, str)
        or not encoded
        or len(encoded) > 4 * ((maximum + 2) // 3)
    ):
        raise _invalid("Image exceeds the encoded-byte limit or is empty.")
    padding = 2 if encoded.endswith("==") else 1 if encoded.endswith("=") else 0
    if len(encoded) // 4 * 3 - padding > maximum:
        raise _invalid("Image exceeds the encoded-byte limit.")
    try:
        data = base64.b64decode(encoded, validate=True)
    except ValueError as error:
        raise _invalid("Image data must be strict base64.") from error
    if len(data) > maximum:
        raise _invalid("Image exceeds the encoded-byte limit.")
    if base64.b64encode(data).decode("ascii") != encoded:
        raise _invalid("Image base64 must use canonical padding.")
    width, height = (
        _png_dimensions(data) if mime_type == "image/png" else _jpeg_dimensions(data)
    )
    if (
        width <= 0
        or height <= 0
        or max(width, height) > min(limits.max_image_dimension, MAX_IMAGE_DIMENSION)
        or width * height > min(limits.max_image_pixels, MAX_IMAGE_PIXELS)
    ):
        raise _invalid("Image dimensions exceed the pixel limit or are empty.")
    return {"image": {"mime_type": mime_type, "data": encoded}}


def _parse_uri(part, limits):
    image = part.get("image_url")
    url = image.get("url") if isinstance(image, dict) else None
    if not isinstance(url, str):
        raise _invalid("image_url must contain a string url.")
    for mime in ("image/png", "image/jpeg"):
        prefix = f"data:{mime};base64,"
        if url.startswith(prefix):
            return validate_image(mime, url[len(prefix) :], limits)
    raise _invalid(
        "Images require an inline JPEG/PNG base64 data URI; "
        "URLs and files are not fetched."
    )


def prepare_image_bindings(messages, limits):
    """Replace image parts with unique text sentinels for existing chat templates."""
    if not has_images(messages):
        return messages, {}
    if limits is None:
        raise APIError(
            400,
            "This worker does not support image input.",
            "invalid_request_error",
            "unsupported_image",
        )
    count = sum(
        part.get("type") == "image_url"
        for message in messages
        if isinstance(message.content, list)
        for part in message.content
    )
    if count > min(limits.max_images, MAX_IMAGES):
        raise _invalid("At most one image is supported across the submitted history.")
    bindings = {}
    modified = []
    for message in messages:
        if not isinstance(message.content, list):
            modified.append(message)
            continue
        parts = []
        for part in message.content:
            if part.get("type") == "image_url":
                marker = f"ETIMAGE_{uuid.uuid4().hex}_END"
                bindings[marker] = _parse_uri(part, limits)
                parts.append({"type": "text", "text": marker})
            else:
                parts.append(part)
        modified.append(message.model_copy(update={"content": parts}))
    return modified, bindings


def bind_image_segments(segments, bindings):
    """Bind exact ordered sentinels after rendering and any assistant-ID splicing.

    Missing, changed, repeated, or reordered markers are errors, never a text
    fallback. Token segments pass through unchanged.
    """
    if not bindings:
        return segments
    text = "".join(segment.get("text", "") for segment in segments)
    positions = [text.find(marker) for marker in bindings]
    if any(text.count(marker) != 1 for marker in bindings) or positions != sorted(
        positions
    ):
        raise _invalid(
            "Chat template must preserve each image binding exactly once and in order."
        )
    result = []
    for segment in segments:
        if "text" not in segment:
            result.append(segment)
            continue
        remaining = segment["text"]
        while remaining:
            found = [
                (remaining.find(marker), marker)
                for marker in bindings
                if marker in remaining
            ]
            if not found:
                result.append({"text": remaining})
                break
            index, marker = min(found)
            if index:
                result.append({"text": remaining[:index]})
            result.append(dict(bindings[marker]))
            remaining = remaining[index + len(marker) :]
    if sum("image" in segment for segment in result) != len(bindings):
        raise _invalid("Image binding was split by assistant token splicing.")
    return result
