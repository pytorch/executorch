# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""ASCII serving dashboard; terminal I/O happens at most twice per second."""

import asyncio
import atexit
import codecs
import logging
import os
import re
import shutil
import sys
import tempfile
import time
from collections import deque
from contextlib import suppress

from .metrics import Metrics

_STATE_COLORS = {
    "ACTIVE": "1;36",
    "IDLE": "2",
    "generating": "1;36",
    "waiting": "33",
    "done": "32",
    "error": "1;31",
    "cancelled": "33",
}
_LINE_COLORS = {
    "+--": "36",
    "  Request": "1",
    "Context limit:": "2",
    "Last error:": "2",
    "Rates/counts": "2",
    "Last 3": "2",
    "Waiting for": "2",
}


def _style(text, code):
    return f"\x1b[{code}m{text}\x1b[0m"


def _colorize(line):
    if line.startswith("  #"):
        match = re.match(r"(  #\d+\s+)(\w+)(.*)", line)
        if match and match[2] in _STATE_COLORS:
            return (
                _style(match[1], "2")
                + _style(match[2], _STATE_COLORS[match[2]])
                + re.sub(r"--", lambda m: _style(m[0], "2"), match[3])
            )
    elif line.startswith("+ "):
        match = re.match(r"(\+ .+?)(  )(ACTIVE|IDLE)(.*)", line)
        if match:
            return (
                _style(match[1], "1")
                + match[2]
                + _style(match[3], _STATE_COLORS[match[3]])
                + match[4]
            )
        return _style(line, "2")
    elif line.startswith("Sessions:"):
        line = re.sub(
            r"\b[1-9]\d* (?:active|running)\b",
            lambda m: _style(m[0], "1;36"),
            line,
        )
        return re.sub(
            r"\b(Done|Errors|Cancelled): (\d+)",
            lambda m: _style(
                m[0], "2" if m[2] == "0" else "32" if m[1] == "Done" else "33"
            ),
            line,
        )
    elif line.startswith("Avg since ready:"):
        return re.sub(
            r"(?:Decode|Prefill) [\d.,k]+ tok/s|Output [\d.,k]+",
            lambda m: _style(m[0], "1"),
            line,
        )
    for prefix, code in _LINE_COLORS.items():
        if line.startswith(prefix):
            return _style(line, code)
    return line


def terminal_available(stream) -> bool:
    try:
        return (
            stream is not None and stream.isatty() and os.environ.get("TERM") != "dumb"
        )
    except (OSError, ValueError):
        return False


def _text(value) -> str:
    # Model paths and session IDs are external input, never terminal commands.
    return "".join(c if " " <= c <= "~" else "?" for c in str(value))


def _number(value) -> str:
    if value is None:
        return "--"
    if value >= 10000:
        return f"{value / 1000:.1f}k"
    return f"{value:,.1f}" if isinstance(value, float) else f"{value:,}"


def _request_line(row, now):
    context = (
        row.prompt_tokens + row.output_tokens
        if row.prompt_tokens is not None and row.output_tokens is not None
        else None
    )
    reused = (
        f"{100 * row.reused_tokens / row.prompt_tokens:.0f}%"
        if row.reused_tokens is not None and row.prompt_tokens
        else "--"
    )
    ttft = (
        f"{1000 * (row.first_output - row.started):.0f}ms"
        if row.first_output is not None
        else "--"
    )
    elapsed = (row.ended if row.ended is not None else now) - row.started
    return (
        f"  #{row.number:<6} {row.state:<10} {_number(context):>8} "
        f"{_number(row.output_tokens):>7} {_number(row.prefill_rate):>9} "
        f"{_number(row.decode_rate):>8} {ttft:>7} {elapsed:>6.1f}s {reused:>6}"
    )


def _session_lines(session_id, session, active):
    if session_id is None:
        return ["+ Ungrouped requests"]
    totals = session.totals
    prefill = (
        totals.timed_prefill_tokens * 1000 / totals.prefill_ms
        if totals.prefill_ms
        else None
    )
    decode = (
        totals.timed_decode_tokens * 1000 / totals.decode_ms
        if totals.decode_ms
        else None
    )
    ttft = (
        f"{totals.ttft_seconds * 1000 / totals.ttft_count:.0f}ms"
        if totals.ttft_count
        else "--"
    )
    reused = (
        f"{100 * totals.reused_tokens / totals.prompt_tokens:.0f}%"
        if totals.prompt_tokens
        else "--"
    )
    return [
        f"+ {_text(session_id)[:24]}  {'ACTIVE' if active else 'IDLE'}"
        f" | {totals.requests} requests | Context: {_number(session.context)}"
        f" | Errors: {totals.errors} | Cancelled: {totals.cancelled}",
        f"  Output: {_number(totals.output_tokens)} | Prefill/s: {_number(prefill)}"
        f" | Decode/s: {_number(decode)} | Avg TTFT: {ttft} | Cache: {reused}",
    ]


def render(
    metrics: Metrics, model: str, context, capacity, width=100, height=24
) -> str:
    width, height = max(1, width), max(1, height)
    now = time.monotonic()
    uptime = max(0, now - metrics.started)
    totals = metrics.totals
    reported_time = uptime if totals.reported else 0
    decode = totals.output_tokens / reported_time if reported_time else None
    prefill = totals.prefilled_tokens / reported_time if reported_time else None
    rows = list(metrics.turns.values())
    active = {row.session_id for row in rows if row.ended is None}
    named = set(metrics.sessions) - {None}
    lines = [
        f"+-- LLM server {'-' * max(0, width - 16)}+",
        _text(model),
        f"Sessions: {len(active & named)} active / {len(named)} tracked"
        f" | Requests: {sum(row.ended is None for row in rows)} running"
        f" | Done: {metrics.completed} | Errors: {metrics.errors}"
        f" | Cancelled: {metrics.cancelled}",
        f"Avg since ready: Decode {_number(decode)} tok/s"
        f" | Prefill {_number(prefill)} tok/s | Output {_number(totals.output_tokens)}",
        f"Context limit: {_number(context)} | Concurrency: {capacity or '--'}"
        f" | Uptime: {int(uptime)}s",
        f"  {'Request':<7} {'State':<10} {'Context':>8} {'Output':>7} "
        f"{'Prefill/s':>9} {'Decode/s':>8} {'TTFT':>7} {'Elapsed':>7} {'Cache':>6}",
    ]
    footer = [
        "Rates/counts update at turn end; averages include idle time.",
        "Last 3 completed/session + active. TTFT = first text. -- = unavailable. Ctrl+C.",
    ]
    errors = [row for row in rows if row.error is not None]
    if errors:
        last = max(errors, key=lambda row: row.number)
        footer.insert(
            0, f"Last error: {_text(last.error)} (redirect stderr for full logs)"
        )
    groups = sorted(
        metrics.sessions.items(),
        key=lambda item: (item[0] not in active, -item[1].last_request),
    )
    available = max(0, height - len(lines) - len(footer))
    blocks = []
    for index, (session_id, session) in enumerate(groups):
        heading = _session_lines(session_id, session, session_id in active)
        reserve = int(index < len(groups) - 1)
        if len(heading) + 1 + reserve > available:
            break
        requests = sorted(
            (row for row in rows if row.session_id == session_id),
            key=lambda row: (
                row.ended is not None,
                -(row.ended if row.ended is not None else row.started),
            ),
        )
        count = min(
            max(1, sum(row.ended is None for row in requests)),
            available - len(heading) - reserve,
        )
        shown = [_request_line(row, now) for row in requests[:count]]
        blocks.append((heading, requests[count:], shown))
        available -= len(heading) + len(shown)
    hidden = len(groups) - len(blocks)
    available -= int(hidden > 0)
    # Share remaining rows across visible sessions, with active requests first.
    while available > 0:
        candidates = [block for block in blocks if block[1]]
        if not candidates:
            break
        block = min(
            candidates,
            key=lambda item: (item[1][0].ended is not None, len(item[2])),
        )
        block[2].append(_request_line(block[1].pop(0), now))
        available -= 1
    for heading, _, requests in blocks:
        lines.extend(heading + requests)
    if hidden:
        lines.append(f"+ {hidden} more groups (enlarge terminal)")
    elif not rows:
        lines.append("Waiting for requests...")
    lines.extend(footer)
    return "\n".join(line[:width] for line in lines[:height])


class TerminalHUD(logging.Handler):
    def __init__(
        self, metrics: Metrics, model: str, context=None, capacity=None, stream=None
    ):
        super().__init__()
        self.metrics = metrics
        self.model = model
        self.context = context
        self.capacity = capacity
        self.stream = stream if stream is not None else sys.stderr
        self._color = terminal_available(self.stream) and not os.environ.get("NO_COLOR")
        self._messages = deque(maxlen=3)
        self._handlers = []
        self._task = None
        self._diagnostics = None
        self._active = False

    def emit(self, record):
        message = None
        try:
            message = self.format(record)
            message = "".join(c for c in message if c.isprintable() or c in "\n\t")
            self._messages.append((record.levelno, _text(message)))
            native = record.name == f"{__package__}.worker_logs"
            if not self._active:
                self.stream.write(message + "\n")
                self.stream.flush()
            elif self._diagnostics is not None and (
                native or record.levelno >= logging.WARNING
            ):
                remaining = memoryview((message + "\n").encode("utf-8"))
                while remaining:
                    written = self._diagnostics.write(remaining)
                    if not written:
                        raise OSError("diagnostics write made no progress")
                    remaining = remaining[written:]
        except Exception:
            # Log storage must never stop the worker stderr pipe being drained.
            self._restore()
            if message is not None:
                with suppress(OSError, ValueError):
                    self.stream.write(message + "\n")
                    self.stream.flush()
            else:
                self.handleError(record)

    def draw(self):
        self.acquire()
        try:
            if self._active:
                self._draw()
        finally:
            self.release()

    def _draw(self):
        try:
            size = os.get_terminal_size(self.stream.fileno())
        except (AttributeError, OSError, ValueError):
            size = shutil.get_terminal_size(fallback=(100, 24))
        width = max(1, size.columns - 1)
        height = max(1, size.lines - 1)
        recent = list(self._messages)
        frame = render(
            self.metrics,
            self.model,
            self.context,
            self.capacity,
            width,
            max(1, height - len(recent)),
        )
        frame_lines = frame.splitlines()
        messages = [message[:width] for _, message in recent]
        if self._color:
            frame_lines = [_colorize(line) for line in frame_lines]
            messages = [
                _style(
                    message,
                    (
                        "1;31"
                        if level >= logging.ERROR
                        else "33" if level >= logging.WARNING else "2"
                    ),
                )
                for (level, _), message in zip(recent, messages)
            ]
        lines = (frame_lines + messages)[:height]
        # Address rows directly: newlines and full-screen clears can scroll.
        self.stream.write(
            "".join(f"\x1b[{row};1H\x1b[2K{line}" for row, line in enumerate(lines, 1))
            + "\x1b[J"
        )
        self.stream.flush()

    async def _refresh(self):
        try:
            while self._active:
                self.draw()
                await asyncio.sleep(0.5)
        except Exception:  # The dashboard must not fail the serving lifespan.
            self._restore()
            logging.getLogger(__name__).exception(
                "HUD stopped; continuing with plain logs"
            )

    async def __aenter__(self):
        try:
            # Unbuffered storage stays readable after a failed write.
            self._diagnostics = tempfile.TemporaryFile(mode="w+b", buffering=0)
            self._active = True
            self.stream.write("\x1b[?1049h\x1b[?25l")
            atexit.register(self._restore)
            for logger in [
                logging.getLogger(),
                logging.getLogger("uvicorn"),
                logging.getLogger("uvicorn.error"),
                logging.getLogger("uvicorn.access"),
            ]:
                previous = logger.handlers[:]
                self._handlers.append((logger, previous))
                logger.handlers = [
                    handler
                    for handler in previous
                    if not (
                        isinstance(handler, logging.StreamHandler)
                        and terminal_available(handler.stream)
                    )
                ]
                if logger is logging.getLogger() or len(logger.handlers) != len(
                    previous
                ):
                    logger.addHandler(self)
            self._task = asyncio.create_task(self._refresh())
        except (OSError, ValueError):
            self._restore()
            logging.getLogger(__name__).exception(
                "HUD unavailable; continuing with plain logs"
            )
        return self

    def _restore(self):
        self.acquire()
        try:
            if not self._active:
                return
            self._active = False
            atexit.unregister(self._restore)
            for logger, handlers in self._handlers:
                logger.handlers = handlers
            diagnostics, self._diagnostics = self._diagnostics, None
            try:
                with suppress(OSError, ValueError):
                    if self._color:
                        self.stream.write("\x1b[0m")
                    self.stream.write("\x1b[?25h\x1b[?1049l")
                    self.stream.flush()
                    if diagnostics is not None:
                        diagnostics.seek(0)
                        reader = codecs.getreader("utf-8")(
                            diagnostics, errors="replace"
                        )
                        shutil.copyfileobj(reader, self.stream)
                        self.stream.flush()
            finally:
                if diagnostics is not None:
                    with suppress(OSError, ValueError):
                        diagnostics.close()
        finally:
            self.release()

    async def __aexit__(self, *args):
        try:
            if self._task is not None:
                self._task.cancel()
                with suppress(asyncio.CancelledError):
                    await self._task
        finally:
            self._restore()
            with suppress(OSError, ValueError):
                self.stream.write(
                    f"Served {self.metrics.completed} requests; {self.metrics.errors} errors, "
                    f"{self.metrics.cancelled} cancelled.\n"
                )
                self.stream.flush()
