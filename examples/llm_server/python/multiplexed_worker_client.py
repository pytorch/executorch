# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Async request-scoped JSONL transport for explicitly multiplexed workers.

One reader task dispatches stdout into bounded token mailboxes; one writer task
serializes stdin. Neither invokes consumer callbacks. Local consumption and wire
completion have separate lifetimes: abandoning a stream cannot release a request
that the worker still owns. All operations run on the client's startup loop.
"""

import asyncio
import json
import logging
import math
from collections import deque
from dataclasses import dataclass, field
from typing import Optional, Sequence

from .worker_client import (
    _decode_worker_json,
    _PROCESS_WAIT_TIMEOUT_SECONDS,
    _UINT64_MAX,
    WorkerError,
    WorkerStats,
)


logger = logging.getLogger(__name__)
_MAX_REQUEST_BYTES = 1024 * 1024
_MAX_MESSAGE_BYTES = 1024 * 1024
_INT32_MAX = 2**31 - 1
_INT64_MIN = -(2**63)


def _validate_wire_values(value):
    # JSON permits escaped surrogates and arbitrary integers that the native
    # parser cannot represent. Check keys and nested values, not just prompts.
    if isinstance(value, str):
        value.encode("utf-8")
    elif isinstance(value, int):
        if not _INT64_MIN <= value <= _UINT64_MAX:
            raise ValueError("worker JSON integer is outside [INT64_MIN, UINT64_MAX]")
    elif isinstance(value, dict):
        for key, item in value.items():
            _validate_wire_values(key)
            _validate_wire_values(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            _validate_wire_values(item)


def _validate_integer(value, name, low, high):
    if type(value) is not int or not low <= value <= high:
        raise ValueError(f"{name} must be an integer in [{low}, {high}]")


def _validate_limits(**limits):
    for name, value in limits.items():
        if type(value) is not int or value < (0 if name == "max_named_sessions" else 1):
            raise WorkerError(f"invalid {name}: {value!r}")


async def _read_message(stdout, max_message_bytes):
    try:
        line = await stdout.readline()
    except (ValueError, asyncio.LimitOverrunError) as error:
        raise WorkerError("worker response is oversized or incomplete") from error
    if not line:
        raise WorkerError("worker exited mid-request")
    if len(line) > max_message_bytes or not line.endswith(b"\n"):
        raise WorkerError("worker response is oversized or incomplete")
    try:
        return _decode_worker_json(line.decode("utf-8"))
    except UnicodeDecodeError as error:
        raise WorkerError("invalid worker UTF-8") from error


async def _drain_and_wait(proc):
    # Drain in bounded pieces so a paused stdout pipe cannot prevent reaping.
    if proc.stdout is not None:
        try:
            while await proc.stdout.read(64 * 1024):
                pass
        except (OSError, ValueError):
            pass
    await proc.wait()


async def _shutdown_async_process(proc):
    if proc.stdin is not None:
        proc.stdin.close()
    for signal in (proc.terminate, proc.kill):
        if proc.returncode is None:
            try:
                signal()
            except ProcessLookupError:
                pass
        try:
            await asyncio.wait_for(
                _drain_and_wait(proc), timeout=_PROCESS_WAIT_TIMEOUT_SECONDS
            )
        except asyncio.TimeoutError:
            continue
        if proc.stdin is not None:
            try:
                await asyncio.wait_for(
                    proc.stdin.wait_closed(), timeout=_PROCESS_WAIT_TIMEOUT_SECONDS
                )
            except (OSError, asyncio.TimeoutError):
                pass
        return
    raise WorkerError("worker could not be reaped after termination")


@dataclass
class _Request:
    request_id: int
    op: str
    completion: asyncio.Future
    ack: str = "done"
    changed: asyncio.Event = field(default_factory=asyncio.Event)
    submitted: bool = False
    sending: bool = False
    consumed: bool = False
    cancel_requested: bool = False
    cancel_pending: bool = False
    tokens: deque[str] = field(default_factory=deque)
    token_chars: int = 0
    local_error: Optional[WorkerError] = None


@dataclass
class _Cancellation:
    target_request_id: int
    sending: bool = False


class WorkerGeneration:
    """A single-consumer token stream with independently awaitable wire completion.

    ``generate`` reserves and enqueues immediately. Iterate to consume tokens and
    use ``wait`` for terminal statistics. ``aclose`` abandons buffered output and
    requests cancellation, but does not wait for native settlement. Callers that
    stop consuming must close the stream and retain their session lease until
    ``wait`` finishes, including when that wait reports a transport failure.
    """

    def __init__(self, client, state):
        self._client = client
        self._state = state
        self._reading = False

    @property
    def request_id(self) -> int:
        return self._state.request_id

    def __aiter__(self):
        return self

    async def __anext__(self) -> str:
        self._client._check_loop()
        if self._reading:
            raise WorkerError("generation already has an active token reader")
        self._reading = True
        state = self._state
        try:
            while not state.consumed:
                if state.tokens:
                    token = state.tokens.popleft()
                    state.token_chars -= len(token)
                    return token
                if state.local_error is not None:
                    self._client._abandon(state)
                    raise state.local_error
                if state.completion.done():
                    result = state.completion.result()
                    self._client._abandon(state)
                    if isinstance(result, WorkerError):
                        raise result
                    break
                state.changed.clear()
                await state.changed.wait()
            raise StopAsyncIteration
        except asyncio.CancelledError:
            self._client._abandon(state)
            raise
        finally:
            self._reading = False

    def cancel(self) -> bool:
        """Request cancellation without abandoning tokens or waiting for an ACK."""
        return self._client.cancel(self.request_id)

    async def aclose(self) -> None:
        self._client._check_loop()
        self._client._abandon(self._state)

    async def wait(self) -> WorkerStats:
        """Wait for wire completion, unaffected by cancellation of another waiter.

        Local overflow/consumer errors are reported by iteration, not substituted
        for the worker's actual terminal result here.
        """
        self._client._check_loop()
        result = await asyncio.shield(self._state.completion)
        if isinstance(result, WorkerError):
            raise result
        return result


class MultiplexedWorkerClient:
    """Event-loop-owned, bounded native transport; use spawn_multiplexed_worker.

    Reservations, queued work, and unconsumed completions share the request
    budget. Cancellation operations have an independent, equally bounded budget.
    A supplied process must have binary pipes and a stdout stream limit at least
    max_message_bytes. The factory configures those pipes before readiness.
    """

    supports_multiplexing = True
    supports_cancel = True

    def __init__(
        self,
        proc: asyncio.subprocess.Process,
        max_named_sessions: int = 0,
        max_inflight_requests: int = 64,
        mailbox_capacity: int = 64,
        max_buffered_chars: int = 1024 * 1024,
        max_message_bytes: int = _MAX_MESSAGE_BYTES,
        max_request_bytes: int = _MAX_REQUEST_BYTES,
    ):
        _validate_limits(
            max_named_sessions=max_named_sessions,
            max_inflight_requests=max_inflight_requests,
            mailbox_capacity=mailbox_capacity,
            max_buffered_chars=max_buffered_chars,
            max_message_bytes=max_message_bytes,
            max_request_bytes=max_request_bytes,
        )
        if proc.stdin is None or proc.stdout is None:
            raise WorkerError("worker requires stdin and stdout pipes")
        self._loop = asyncio.get_running_loop()
        self.max_named_sessions = max_named_sessions
        self.max_inflight_requests = max_inflight_requests
        self._mailbox_capacity = mailbox_capacity
        self._max_buffered_chars = max_buffered_chars
        self._max_message_bytes = max_message_bytes
        self._max_request_bytes = max_request_bytes
        self._proc = proc
        self._write_ready = asyncio.Event()
        self._requests: dict[int, _Request] = {}
        self._controls: dict[int, _Cancellation] = {}
        self._writes: deque[tuple[int, bytes]] = deque()
        self._next_request_id = 1
        self._terminal_error: Optional[WorkerError] = None
        self._failed = False
        self._closed = False
        self._cleanup_task: Optional[asyncio.Task] = None
        self._writer = self._loop.create_task(self._write_loop(), name="worker-writer")
        self._reader = self._loop.create_task(self._read_loop(), name="worker-reader")

    @property
    def healthy(self) -> bool:
        return self._terminal_error is None and self._proc.returncode is None

    @property
    def failed(self) -> bool:
        return self._failed

    @property
    def closed(self) -> bool:
        return self._closed

    def _check_loop(self):
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None
        if loop is not self._loop:
            raise WorkerError("worker client must be used on its startup event loop")

    def _ensure_usable(self):
        self._check_loop()
        if self._terminal_error is not None:
            raise self._terminal_error

    def _fail(self, error, failed=True):
        if self._terminal_error is None:
            self._terminal_error = error
            self._failed = failed
        for state in self._requests.values():
            if not state.completion.done():
                state.tokens.clear()
                state.token_chars = 0
                # Store errors as values: abandoned requests need not retrieve a
                # future exception merely to let the transport settle them.
                state.completion.set_result(self._terminal_error)
            state.changed.set()
        self._requests.clear()
        self._controls.clear()
        self._writes.clear()
        self._write_ready.set()
        if self._cleanup_task is None or (
            self._cleanup_task.done()
            and not self._cleanup_task.cancelled()
            and not self._cleanup_task.result()
        ):
            self._cleanup_task = self._loop.create_task(
                self._cleanup(), name="worker-cleanup"
            )

    def _allocate_id(self):
        self._ensure_usable()
        if self._next_request_id > _UINT64_MAX:
            self._fail(WorkerError("worker request ids exhausted"))
            raise self._terminal_error
        request_id = self._next_request_id
        self._next_request_id += 1
        return request_id

    def _reserve(self, op="generate", ack="done"):
        self._ensure_usable()
        if len(self._requests) >= self.max_inflight_requests:
            raise WorkerError(
                "worker request capacity exhausted", code="capacity_exhausted"
            )
        state = _Request(self._allocate_id(), op, self._loop.create_future(), ack)
        self._requests[state.request_id] = state
        return state

    @staticmethod
    def _validate_id(request_id):
        if type(request_id) is not int or not 1 <= request_id <= _UINT64_MAX:
            raise WorkerError("request_id must be in [1, UINT64_MAX]")

    def reserve_request(self) -> int:
        """Reserve a cancellable ID before constructing/submitting a generation."""
        return self._reserve().request_id

    def release_request(self, request_id: int) -> bool:
        """Release a reservation that was never submitted."""
        self._ensure_usable()
        self._validate_id(request_id)
        state = self._requests.get(request_id)
        if state is None or state.submitted:
            return False
        state.consumed = True
        self._complete(state, WorkerError("worker request reservation released"))
        return True

    async def wait_for_request(self, request_id: int) -> None:
        """Wait for settlement, including after local failure or stream abandonment.

        Retired IDs are already settled. No completed-request history is kept.
        """
        self._check_loop()
        self._validate_id(request_id)
        state = self._requests.get(request_id)
        if state is not None:
            await asyncio.shield(state.completion)

    def _retire(self, state):
        if state.consumed and state.completion.done():
            self._requests.pop(state.request_id, None)

    def _complete(self, state, result):
        if not state.completion.done():
            state.completion.set_result(result)
        state.changed.set()
        self._retire(state)

    def _abandon(self, state):
        if state.consumed:
            return
        state.tokens.clear()
        state.token_chars = 0
        state.consumed = True
        if state.op == "generate" and not state.completion.done():
            self._cancel(state)
        state.changed.set()
        self._retire(state)

    def _cancel(self, state):
        if state.cancel_requested:
            return True
        if state.completion.done():
            return False
        if not state.sending:
            self._writes = deque(
                item for item in self._writes if item[0] != state.request_id
            )
            state.cancel_requested = True
            self._complete(state, WorkerStats(finish_reason="stop", cancelled=True))
            return True
        if len(self._controls) >= self.max_inflight_requests:
            state.cancel_pending = True
            return True
        cancel_id = self._allocate_id()
        payload = self._encode_request(
            {
                "op": "cancel",
                "request_id": cancel_id,
                "target_request_id": state.request_id,
            },
            self._max_request_bytes,
        )
        state.cancel_pending = False
        self._controls[cancel_id] = _Cancellation(state.request_id)
        state.cancel_requested = True
        self._writes.append((cancel_id, payload))
        self._write_ready.set()
        return True

    def cancel(self, request_id: int) -> bool:
        """Latch cancellation, without waiting for pipe I/O or a worker ACK.

        A full control budget retains per-request intent until an ACK frees a
        slot. Once sending starts, only the generation terminal (not cancel ACK)
        settles the generation. Unsent cancellation completes locally.
        """
        self._check_loop()
        self._validate_id(request_id)
        if self._terminal_error is not None:
            return False
        state = self._requests.get(request_id)
        if state is None or state.op != "generate":
            return False
        return self._cancel(state)

    def stop(self) -> bool:
        """Cancel the sole active generation, refusing ambiguous cancellation."""
        self._check_loop()
        active = [
            s
            for s in self._requests.values()
            if s.op == "generate" and not s.completion.done()
        ]
        if self._terminal_error is not None or len(active) != 1:
            return False
        return self._cancel(active[0])

    def reset(self) -> None:
        """Legacy no-op; reset_session performs persistent-state replacement."""
        self._ensure_usable()

    @staticmethod
    def _encode_request(request, max_request_bytes=_MAX_REQUEST_BYTES):
        try:
            _validate_wire_values(request)
            for name, low, high in (
                ("max_new_tokens", -1, _INT32_MAX),
                ("top_k", 0, _INT32_MAX),
                ("seed", 0, _UINT64_MAX),
            ):
                if name in request:
                    _validate_integer(request[name], name, low, high)
            for segment in request.get("prompt_segments", ()):
                if isinstance(segment, dict) and isinstance(
                    segment.get("ids"), (list, tuple)
                ):
                    for token in segment["ids"]:
                        _validate_integer(token, "token ID", 0, _UINT64_MAX)
            payload = (
                json.dumps(request, allow_nan=False, ensure_ascii=True) + "\n"
            ).encode("utf-8")
        except (TypeError, ValueError, RecursionError) as error:
            raise WorkerError(
                f"invalid worker request: {error}", code="invalid_argument"
            ) from error
        if len(payload) > max_request_bytes:
            raise WorkerError(
                f"worker request exceeds the {max_request_bytes} byte frame limit",
                code="invalid_argument",
            )
        return payload

    def _submit(self, request, request_id=None, op="generate", ack="done"):
        self._ensure_usable()
        if request_id is None:
            state = self._reserve(op, ack)
        else:
            self._validate_id(request_id)
            state = self._requests.get(request_id)
            if state is None or state.op != op or state.submitted:
                raise WorkerError(f"request id {request_id} is not reserved")
        state.submitted = True
        try:
            payload = self._encode_request(
                dict(request, request_id=state.request_id), self._max_request_bytes
            )
            if not state.completion.done():
                self._writes.append((state.request_id, payload))
                self._write_ready.set()
        except BaseException:
            state.consumed = True
            self._complete(state, WorkerError("worker request encoding failed"))
            raise
        return state

    async def _write_loop(self):
        try:
            dispatched = 0
            while self._terminal_error is None:
                await self._write_ready.wait()
                while self._writes and self._terminal_error is None:
                    request_id, payload = self._writes.popleft()
                    state = self._requests.get(request_id)
                    if state is not None:
                        if state.completion.done():
                            continue
                        state.sending = True
                    else:
                        control = self._controls.get(request_id)
                        if control is None:
                            continue
                        control.sending = True
                    written = self._proc.stdin.write(payload)
                    if written is not None and written != len(payload):
                        raise WorkerError("short write to worker stdin")
                    await self._proc.stdin.drain()
                    dispatched += 1
                    if dispatched == 32:
                        dispatched = 0
                        await asyncio.sleep(0)
                self._write_ready.clear()
        except asyncio.CancelledError:
            if self._terminal_error is None:
                self._fail(WorkerError("worker writer interrupted"))
            raise
        except BaseException as error:  # noqa: B036 - settle transport failures
            self._fail(WorkerError(f"worker write failed: {error}"))

    @staticmethod
    def _validate_message(msg):
        MultiplexedWorkerClient._validate_id(msg.get("request_id"))
        kinds = [
            key
            for key in ("token", "done", "error", "opened", "closed", "reset")
            if key in msg
        ]
        if "cancelled" in msg and "done" not in msg:
            kinds.append("cancelled")
        if len(kinds) != 1:
            raise WorkerError(f"invalid worker response shape: {msg}")
        kind = kinds[0]
        value = msg[kind]
        if kind in ("token", "error"):
            if not isinstance(value, str):
                raise WorkerError(f"invalid worker {kind}")
        elif value is not True:
            raise WorkerError(f"invalid worker {kind} acknowledgement")
        if "code" in msg and not isinstance(msg["code"], str):
            raise WorkerError("invalid worker error code")
        if kind == "done":
            MultiplexedWorkerClient._validate_generation_terminal(msg)
        return kind

    @staticmethod
    def _validate_generation_terminal(msg):
        for key in (
            "prompt_tokens",
            "completion_tokens",
            "reused_prompt_tokens",
            "prefilled_prompt_tokens",
        ):
            if key in msg and (type(msg[key]) is not int or msg[key] < 0):
                raise WorkerError(f"invalid worker statistic: {key}")
        for key in (
            "prefill_ms",
            "decode_ms",
            "total_ms",
            "prefill_tok_s",
            "decode_tok_s",
            "vision_encoder_ms",
        ):
            if key in msg and (
                type(msg[key]) not in (int, float)
                or not math.isfinite(msg[key])
                or msg[key] < 0
            ):
                raise WorkerError(f"invalid worker statistic: {key}")
        if "finish_reason" in msg and msg["finish_reason"] not in ("stop", "length"):
            raise WorkerError("invalid worker finish_reason")
        if "cancelled" in msg and type(msg["cancelled"]) is not bool:
            raise WorkerError("invalid worker cancelled flag")
        if msg.get("cancelled") and msg.get("finish_reason") != "stop":
            raise WorkerError("cancelled completion must have finish_reason stop")
        if "session_reset_reason" in msg and not isinstance(
            msg["session_reset_reason"], str
        ):
            raise WorkerError("invalid worker session_reset_reason")
        if "generated_token_ids" in msg and (
            not isinstance(msg["generated_token_ids"], list)
            or any(type(t) is not int or t < 0 for t in msg["generated_token_ids"])
        ):
            raise WorkerError("invalid worker generated_token_ids")

    def _dispatch_cancel(self, msg, kind):
        control = self._controls[msg["request_id"]]
        if not control.sending or kind not in ("cancelled", "error"):
            raise WorkerError("unexpected cancellation response")
        del self._controls[msg["request_id"]]
        target = self._requests.get(control.target_request_id)
        if (
            kind == "error"
            and target is not None
            and not target.completion.done()
            and target.local_error is None
        ):
            target.tokens.clear()
            target.token_chars = 0
            target.local_error = WorkerError(msg["error"], code=msg.get("code"))
            target.changed.set()
        for pending in self._requests.values():
            if pending.cancel_pending and not pending.completion.done():
                self._cancel(pending)
                if len(self._controls) >= self.max_inflight_requests:
                    break

    def _mailbox_full(self, state, token):
        return (
            len(state.tokens) >= self._mailbox_capacity
            or state.token_chars + len(token) > self._max_buffered_chars
        )

    def _dispatch(self, msg, kind):
        request_id = msg["request_id"]
        if request_id in self._controls:
            self._dispatch_cancel(msg, kind)
            return
        state = self._requests.get(request_id)
        if state is None or not state.sending or state.completion.done():
            raise WorkerError(f"response for inactive request {request_id}")
        if kind == "token" and state.op == "generate":
            if state.local_error is not None or state.consumed:
                return
            token = msg["token"]
            if self._mailbox_full(state, token):
                state.tokens.clear()
                state.token_chars = 0
                state.local_error = WorkerError(
                    "worker token mailbox overflow", code="slow_consumer"
                )
                self._cancel(state)
            else:
                state.tokens.append(token)
                state.token_chars += len(token)
            state.changed.set()
        elif kind in (state.ack, "error"):
            result = (
                WorkerError(msg["error"], code=msg.get("code"))
                if kind == "error"
                else WorkerStats.from_message(msg) if state.op == "generate" else None
            )
            self._complete(state, result)
        else:
            raise WorkerError(f"unexpected response for {state.op}: {kind}")

    async def _read_loop(self):
        try:
            dispatched = 0
            while self._terminal_error is None:
                msg = await _read_message(self._proc.stdout, self._max_message_bytes)
                kind = self._validate_message(msg)
                state = self._requests.get(msg["request_id"])
                if (
                    kind == "token"
                    and state is not None
                    and state.tokens
                    and not state.completion.done()
                    and self._mailbox_full(state, msg["token"])
                ):
                    # Buffered reads need not yield. Give an already-runnable
                    # consumer one turn before declaring it slow; never wait
                    # for mailbox space or for that consumer to finish.
                    await asyncio.sleep(0)
                    if self._terminal_error is not None:
                        return
                self._dispatch(msg, kind)
                dispatched += 1
                if dispatched == min(32, self._mailbox_capacity):
                    dispatched = 0
                    await asyncio.sleep(0)
        except asyncio.CancelledError:
            if self._terminal_error is None:
                self._fail(WorkerError("worker reader interrupted"))
            raise
        except BaseException as error:  # noqa: B036 - settle transport failures
            self._fail(
                error
                if isinstance(error, WorkerError)
                else WorkerError(f"worker read failed: {error}")
            )

    def generate(self, prompt, config, request_id=None) -> WorkerGeneration:
        """Enqueue a generation and return its async token stream, without I/O waits."""
        self._ensure_usable()
        if request_id is not None:
            self._validate_id(request_id)
        try:
            request = {
                "op": "generate",
                "max_new_tokens": getattr(config, "max_new_tokens", -1),
                "temperature": getattr(config, "temperature", 0.0),
                "top_p": getattr(config, "top_p", 1.0),
                "top_k": getattr(config, "top_k", 0),
                "seed": getattr(config, "seed", 0),
                "stop": list(getattr(config, "stop", []) or []),
            }
            segments = getattr(config, "prompt_segments", None)
            request["prompt_segments" if segments is not None else "prompt"] = (
                segments if segments is not None else prompt
            )
            session_id = getattr(config, "session_id", None)
            if session_id:
                request["session_id"] = session_id
            return WorkerGeneration(self, self._submit(request, request_id=request_id))
        except BaseException:
            if request_id is not None and self._terminal_error is None:
                self.release_request(request_id)
            raise

    async def _op(self, op, session_id, ack):
        state = self._submit({"op": op, "session_id": session_id}, op=op, ack=ack)
        try:
            result = await asyncio.shield(state.completion)
            if isinstance(result, WorkerError):
                raise result
        finally:
            self._abandon(state)

    async def open_session(self, session_id: str) -> None:
        """Wait for a named session's admission acknowledgement."""
        await self._op("open", session_id, "opened")

    async def reset_session(self, session_id: str) -> None:
        """Wait for the named session's replacement acknowledgement."""
        await self._op("reset", session_id, "reset")

    async def close_session(self, session_id: str) -> None:
        """Wait for the named session's logical close acknowledgement."""
        await self._op("close", session_id, "closed")

    async def _cleanup(self):
        self._reader.cancel()
        self._writer.cancel()
        await asyncio.gather(self._reader, self._writer, return_exceptions=True)
        try:
            await _shutdown_async_process(self._proc)
        except Exception:  # noqa: BLE001 - observe background cleanup failures
            logger.exception("Model worker could not be reaped after termination")
            return False
        return True

    async def abort(self) -> None:
        """Fail outstanding operations and terminate/reap the worker."""
        self._check_loop()
        self._fail(WorkerError("worker client aborted"))
        if not await asyncio.shield(self._cleanup_task):
            raise WorkerError("worker could not be reaped after termination")

    async def close(self) -> None:
        """Settle operations and reap idempotently; caller cancellation cannot stop cleanup."""
        self._check_loop()
        self._closed = True
        self._fail(WorkerError("worker client is closed"), failed=False)
        if not await asyncio.shield(self._cleanup_task):
            raise WorkerError("worker could not be reaped after termination")


async def spawn_multiplexed_worker(
    cmd: Sequence[str],
    env: Optional[dict] = None,
    cwd: Optional[str] = None,
    *,
    mailbox_capacity: int = 64,
    max_buffered_chars: int = 1024 * 1024,
    max_message_bytes: int = _MAX_MESSAGE_BYTES,
    max_request_bytes: int = _MAX_REQUEST_BYTES,
) -> MultiplexedWorkerClient:
    """Start a native worker on the caller's loop, requiring explicit multiplexing.

    Use from async application startup and await client.close() at shutdown.
    Sequential workers continue to use the synchronous spawn_worker factory.
    """
    _validate_limits(
        mailbox_capacity=mailbox_capacity,
        max_buffered_chars=max_buffered_chars,
        max_message_bytes=max_message_bytes,
        max_request_bytes=max_request_bytes,
    )
    proc = await asyncio.create_subprocess_exec(
        *cmd,
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        env=env,
        cwd=cwd,
        limit=max_message_bytes,
    )
    try:
        msg = await _read_message(proc.stdout, max_message_bytes)
        if msg.get("ready") is not True:
            raise WorkerError(f"worker did not report ready: {msg}")
        if msg.get("multiplexed") is not True:
            raise WorkerError(
                "worker does not support required multiplexing",
                code="unsupported_multiplexing",
            )
        return MultiplexedWorkerClient(
            proc,
            max_named_sessions=msg.get("max_named_sessions", 0),
            max_inflight_requests=msg.get("max_inflight_requests", 64),
            mailbox_capacity=mailbox_capacity,
            max_buffered_chars=max_buffered_chars,
            max_message_bytes=max_message_bytes,
            max_request_bytes=max_request_bytes,
        )
    except BaseException:
        cleanup = asyncio.create_task(_shutdown_async_process(proc))
        while True:
            try:
                await asyncio.shield(cleanup)
                break
            except asyncio.CancelledError:
                if cleanup.cancelled():
                    raise
                # No client was returned to own this child. Finish bounded
                # cleanup even if the startup caller cancels repeatedly.
                continue
        raise
