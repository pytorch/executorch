# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Relay native diagnostics through logging while the terminal HUD is enabled."""

import asyncio
import logging
import os
import re
import sys
from contextlib import asynccontextmanager, suppress

from .terminal import terminal_available

logger = logging.getLogger(__name__)
# Raw stderr must reach the HUD even when the application's log level is WARNING.
logger.setLevel(logging.DEBUG)
_ANSI = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")
_LEVEL = re.compile(
    r"^(?:(?:DEBUG|INFO|WARN(?:ING)?|ERROR|FATAL)\b|[DIWEF](?=\d|\s))", re.IGNORECASE
)
_LEVELS = {
    "D": logging.DEBUG,
    "I": logging.INFO,
    "W": logging.WARNING,
    "E": logging.ERROR,
    "F": logging.CRITICAL,
}


async def _relay(reader):
    head = b""
    while True:
        try:
            line = await reader.readuntil(b"\n")
        except asyncio.LimitOverrunError as error:
            chunk = await reader.readexactly(error.consumed)
            if not head:
                head = chunk[: 64 * 1024]
            continue
        except asyncio.IncompleteReadError as error:
            line = error.partial
            if not line and not head:
                break
        truncated = bool(head)
        message = _ANSI.sub(
            "", (head or line).decode("utf-8", errors="replace")
        ).rstrip()
        head = b""
        if truncated:
            message += " [truncated]"
        if not message:
            continue
        severity = _LEVEL.match(message)
        level = _LEVELS[severity[0][0].upper()] if severity else logging.INFO
        logger.log(level, "%s", message)


@asynccontextmanager
async def capture_worker_logs():
    if not terminal_available(sys.stderr):
        yield None
        return
    read_fd, write_fd = os.pipe()
    pipe = os.fdopen(read_fd, "rb", buffering=0)
    reader = asyncio.StreamReader()
    try:
        transport, _ = await asyncio.get_running_loop().connect_read_pipe(
            lambda: asyncio.StreamReaderProtocol(reader), pipe
        )
    except BaseException:
        pipe.close()
        os.close(write_fd)
        raise
    relay = asyncio.create_task(_relay(reader), name="hud-worker-logs")
    try:
        yield write_fd
    finally:
        os.close(write_fd)
        try:
            with suppress(asyncio.TimeoutError):
                await asyncio.wait_for(relay, timeout=1)
        finally:
            relay.cancel()
            transport.close()
            await asyncio.gather(relay, return_exceptions=True)
