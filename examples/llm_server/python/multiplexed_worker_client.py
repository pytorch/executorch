# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Request-scoped JSONL transport for explicitly multiplexed workers.

Only the reader consumes stdout and only the writer touches stdin. Neither runs
user callbacks. A reservation is a handle that can be cancelled before submitting
``generate`` to an executor; ``generate`` drains its own bounded mailbox on the
calling thread. Terminal delivery has a separate slot, independent of token
capacity. A locally failed request remains registered until its wire terminal
arrives, so late messages cannot be mistaken for another request.
"""

import json
import math
import subprocess
import threading
from collections import deque
from dataclasses import dataclass, field
from typing import Optional

from .worker_client import (
    _decode_worker_json,
    _shutdown_process,
    _UINT64_MAX,
    WorkerClient,
    WorkerError,
)


# Match the native worker's fixed default, independently of inbound limits.
_MAX_REQUEST_BYTES = 1024 * 1024


@dataclass
class _Request:
    request_id: int
    op: str
    changed: threading.Condition
    ack: str = "done"
    submitted: bool = False
    sending: bool = False
    wire_done: bool = False
    consumed: bool = False
    cancel_requested: bool = False
    cancel_pending: bool = False
    tokens: deque = field(default_factory=deque)
    terminal: Optional[object] = None


@dataclass
class _Cancellation:
    target_request_id: int
    sending: bool = False


class MultiplexedWorkerClient:
    """Synchronous compatibility surface with concurrent request-scoped I/O.

    ``max_inflight_requests`` bounds reservations, queued work, and unconsumed
    completions. There is a separate, equally bounded budget for cancellation
    operations. ``cancel`` never waits for pipe I/O or a worker acknowledgement.
    ``stop`` is only a compatibility helper and refuses ambiguous cancellation.
    """

    supports_multiplexing = True
    supports_cancel = True

    def __init__(
        self,
        proc: subprocess.Popen,
        max_named_sessions: int = 0,
        max_inflight_requests: int = 64,
        mailbox_capacity: int = 64,
        max_message_chars: int = 1024 * 1024,
    ):
        for name, value in (
            ("max_named_sessions", max_named_sessions),
            ("max_inflight_requests", max_inflight_requests),
            ("mailbox_capacity", mailbox_capacity),
            ("max_message_chars", max_message_chars),
        ):
            if type(value) is not int or value < (
                0 if name == "max_named_sessions" else 1
            ):
                raise WorkerError(f"invalid {name}: {value!r}")
        self.max_named_sessions = max_named_sessions
        self.max_inflight_requests = max_inflight_requests
        self._mailbox_capacity = mailbox_capacity
        self._max_message_chars = max_message_chars
        self._proc = proc
        self._streams = (proc.stdin, proc.stdout)
        self._lock = threading.RLock()
        self._write_ready = threading.Condition(self._lock)
        self._requests = {}
        self._controls = {}
        self._writes = deque()
        self._next_request_id = 1
        self._terminal_error = None
        self._failed = False
        self._closed = False
        self._cleanup_lock = threading.Lock()
        self._reaped = False
        self._reader = threading.Thread(
            target=self._read_loop, name="worker-reader", daemon=True
        )
        self._writer = threading.Thread(
            target=self._write_loop, name="worker-writer", daemon=True
        )
        self._writer.start()
        self._reader.start()

    @property
    def healthy(self) -> bool:
        """Whether the transport can accept work without a known failure."""
        with self._lock:
            return self._terminal_error is None and self._proc.poll() is None

    @property
    def failed(self) -> bool:
        """Whether a permanent transport failure has occurred."""
        with self._lock:
            return self._failed

    @property
    def closed(self) -> bool:
        """Whether normal shutdown has been requested."""
        with self._lock:
            return self._closed

    def _ensure_usable_locked(self):
        if self._terminal_error is not None:
            raise self._terminal_error

    def _fail_locked(self, error, failed=True):
        if self._terminal_error is None:
            self._terminal_error = error
            self._failed = failed
        for state in self._requests.values():
            state.wire_done = True
            if state.terminal is None:
                state.tokens.clear()
                state.terminal = self._terminal_error
            state.changed.notify_all()
        self._requests.clear()
        self._controls.clear()
        self._writes.clear()
        self._write_ready.notify()

    def _allocate_id_locked(self):
        self._ensure_usable_locked()
        if self._next_request_id > _UINT64_MAX:
            self._fail_locked(WorkerError("worker request ids exhausted"))
            raise self._terminal_error
        request_id = self._next_request_id
        self._next_request_id += 1
        return request_id

    def _reserve_locked(self, op="generate", ack="done"):
        self._ensure_usable_locked()
        if len(self._requests) >= self.max_inflight_requests:
            raise WorkerError(
                "worker request capacity exhausted", code="capacity_exhausted"
            )
        request_id = self._allocate_id_locked()
        state = _Request(request_id, op, threading.Condition(self._lock), ack)
        self._requests[request_id] = state
        return state

    @staticmethod
    def _validate_id(request_id):
        if type(request_id) is not int or not 1 <= request_id <= _UINT64_MAX:
            raise WorkerError("request_id must be in [1, UINT64_MAX]")

    def reserve_request(self) -> int:
        """Allocate a cancellable handle before executor submission."""
        with self._lock:
            return self._reserve_locked().request_id

    def release_request(self, request_id: int) -> bool:
        """Release an unused reservation after failed executor submission."""
        self._validate_id(request_id)
        with self._lock:
            self._ensure_usable_locked()
            state = self._requests.get(request_id)
            if state is None or state.submitted:
                return False
            state.wire_done = True
            state.changed.notify_all()
            del self._requests[request_id]
            return True

    def wait_for_request(self, request_id: int) -> None:
        """Wait for wire settlement, even after the local consumer has failed.

        Retired IDs are already settled. Transport failure or shutdown settles
        every waiter without retaining a history of completed requests.
        """
        self._validate_id(request_id)
        with self._lock:
            state = self._requests.get(request_id)
            if state is not None:
                state.changed.wait_for(lambda: state.wire_done)

    def _retire_locked(self, state):
        if state.consumed and state.wire_done:
            self._requests.pop(state.request_id, None)

    def _cancel_locked(self, state):
        if state.cancel_requested:
            return True
        if state.wire_done:
            return False
        if not state.sending:
            self._writes = deque(
                item for item in self._writes if item[0] != state.request_id
            )
            state.cancel_requested = True
            state.wire_done = True
            if state.terminal is None:
                state.terminal = {
                    "done": True,
                    "cancelled": True,
                    "finish_reason": "stop",
                }
            state.changed.notify_all()
            return True
        if len(self._controls) >= self.max_inflight_requests:
            state.cancel_pending = True
            return True
        cancel_id = self._allocate_id_locked()
        payload = self._encode_request(
            {
                "op": "cancel",
                "request_id": cancel_id,
                "target_request_id": state.request_id,
            }
        )
        state.cancel_pending = False
        self._controls[cancel_id] = _Cancellation(state.request_id)
        state.cancel_requested = True
        self._writes.append((cancel_id, payload))
        self._write_ready.notify()
        return True

    def cancel(self, request_id: int) -> bool:
        """Latch cancellation, returning False if inactive or unavailable.

        True means accepted locally, not acknowledged by the worker. If the
        control budget is full, bounded per-request intent is retried when an
        ACK frees capacity. Completion still arrives through the owning generate
        call. Cancellation before the writer starts completes locally without
        sending generation to the worker.
        """
        self._validate_id(request_id)
        with self._lock:
            if self._terminal_error is not None:
                return False
            state = self._requests.get(request_id)
            if state is None or state.op != "generate":
                return False
            return self._cancel_locked(state)

    def stop(self) -> bool:
        """Cancel the sole active generation; never choose among concurrent ones."""
        with self._lock:
            active = [
                s
                for s in self._requests.values()
                if s.op == "generate" and not s.wire_done
            ]
            if self._terminal_error is not None or len(active) != 1:
                return False
            return self._cancel_locked(active[0])

    def reset(self) -> None:
        """Legacy no-op; persistent state is reset with reset_session."""
        with self._lock:
            self._ensure_usable_locked()

    @staticmethod
    def _encode_request(request):
        payload = json.dumps(request, allow_nan=False, ensure_ascii=True) + "\n"
        if len(payload.encode("utf-8")) > _MAX_REQUEST_BYTES:
            raise WorkerError(
                "worker request exceeds the 1 MiB frame limit", code="invalid_argument"
            )
        return payload

    def _submit(self, request, request_id=None, op="generate", ack="done"):
        with self._lock:
            self._ensure_usable_locked()
            if request_id is None:
                state = self._reserve_locked(op, ack)
            else:
                self._validate_id(request_id)
                state = self._requests.get(request_id)
                if state is None or state.op != op or state.submitted:
                    raise WorkerError(f"request id {request_id} is not reserved")
            state.submitted = True
        try:
            # Encoding and pipe I/O stay outside the dispatcher lock.
            payload = self._encode_request(dict(request, request_id=state.request_id))
            with self._lock:
                self._ensure_usable_locked()
                if not state.wire_done:
                    self._writes.append((state.request_id, payload))
                    self._write_ready.notify()
        except BaseException:
            with self._lock:
                state.wire_done = state.consumed = True
                state.changed.notify_all()
                self._retire_locked(state)
            raise
        return state

    def _write_loop(self):
        try:
            stdin = self._streams[0]
            while True:
                with self._write_ready:
                    self._write_ready.wait_for(
                        lambda: self._writes or self._terminal_error is not None
                    )
                    if self._terminal_error is not None:
                        return
                    request_id, payload = self._writes.popleft()
                    state = self._requests.get(request_id)
                    if state is not None:
                        if state.wire_done:
                            continue
                        state.sending = True
                    else:
                        control = self._controls.get(request_id)
                        if control is None:
                            continue
                        control.sending = True
                written = stdin.write(payload)
                if written is not None and written != len(payload):
                    raise WorkerError("short write to worker stdin")
                stdin.flush()
        except BaseException as error:  # noqa: B036 - settle thread failures
            # An interrupted partial frame is unrecoverable, never resume writing.
            with self._lock:
                self._fail_locked(WorkerError(f"worker write failed: {error}"))

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
        if "finish_reason" in msg and msg["finish_reason"] not in (
            "stop",
            "length",
        ):
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

    def _dispatch_cancel_locked(self, msg, kind):
        request_id = msg["request_id"]
        control = self._controls[request_id]
        if not control.sending or kind not in ("cancelled", "error"):
            raise WorkerError("unexpected cancellation response")
        del self._controls[request_id]
        target = self._requests.get(control.target_request_id)
        if kind == "error" and target is not None and target.terminal is None:
            target.tokens.clear()
            target.terminal = WorkerError(msg["error"], code=msg.get("code"))
            target.changed.notify_all()
        # Pending cancellation is stored on the bounded request
        # registry, not another queue. Retry as soon as ACK capacity frees.
        for pending in self._requests.values():
            if pending.cancel_pending and not pending.wire_done:
                self._cancel_locked(pending)
                if len(self._controls) >= self.max_inflight_requests:
                    break

    def _dispatch_locked(self, msg, kind):
        request_id = msg["request_id"]
        if request_id in self._controls:
            self._dispatch_cancel_locked(msg, kind)
            return
        state = self._requests.get(request_id)
        if state is None or not state.sending or state.wire_done:
            raise WorkerError(f"response for inactive request {request_id}")
        if kind == "token" and state.op == "generate":
            if state.terminal is not None:
                return
            if len(state.tokens) == self._mailbox_capacity:
                state.tokens.clear()
                state.terminal = WorkerError(
                    "worker token mailbox overflow", code="slow_consumer"
                )
                self._cancel_locked(state)
            else:
                state.tokens.append(msg["token"])
        elif kind in (state.ack, "error"):
            state.wire_done = True
            if state.terminal is None:
                state.terminal = (
                    WorkerError(msg["error"], code=msg.get("code"))
                    if kind == "error"
                    else msg
                )
            self._retire_locked(state)
        else:
            raise WorkerError(f"unexpected response for {state.op}: {kind}")
        state.changed.notify_all()

    def _read_loop(self):
        try:
            stdout = self._streams[1]
            while True:
                line = stdout.readline(self._max_message_chars + 1)
                if not line:
                    raise WorkerError("worker exited mid-request")
                if len(line) > self._max_message_chars or not line.endswith("\n"):
                    raise WorkerError("worker response is oversized or incomplete")
                msg = _decode_worker_json(line)
                kind = self._validate_message(msg)
                with self._lock:
                    if self._terminal_error is not None:
                        return
                    self._dispatch_locked(msg, kind)
        except BaseException as error:  # noqa: B036 - settle thread failures
            with self._lock:
                self._fail_locked(
                    error
                    if isinstance(error, WorkerError)
                    else WorkerError(f"worker read failed: {error}")
                )

    def _consume(self, state, token_callback=None, stats_callback=None):
        try:
            while True:
                with state.changed:
                    state.changed.wait_for(
                        lambda: state.tokens or state.terminal is not None
                    )
                    token = state.tokens.popleft() if state.tokens else None
                    terminal = state.terminal if token is None else None
                if token is not None:
                    if token_callback is not None:
                        token_callback(token)
                elif isinstance(terminal, WorkerError):
                    raise terminal
                else:
                    if state.op == "generate":
                        WorkerClient._on_done(terminal, stats_callback)
                    return
        finally:
            with self._lock:
                if (
                    state.op == "generate"
                    and not state.wire_done
                    and self._terminal_error is None
                ):
                    if state.terminal is None:
                        state.terminal = WorkerError("request consumer interrupted")
                    self._cancel_locked(state)
                state.tokens.clear()
                state.consumed = True
                self._retire_locked(state)

    def generate(
        self, prompt, config, token_callback=None, stats_callback=None, request_id=None
    ):
        """Drain one request's mailbox, invoking callbacks on the calling thread."""
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
        state = self._submit(request, request_id=request_id)
        self._consume(state, token_callback, stats_callback)

    def _op(self, op, session_id, ack):
        state = self._submit({"op": op, "session_id": session_id}, op=op, ack=ack)
        self._consume(state)

    def open_session(self, session_id: str) -> None:
        """Wait for a named session's admission acknowledgement."""
        self._op("open", session_id, "opened")

    def reset_session(self, session_id: str) -> None:
        """Wait for the named session's replacement acknowledgement."""
        self._op("reset", session_id, "reset")

    def close_session(self, session_id: str) -> None:
        """Wait for the named session's logical close acknowledgement."""
        self._op("close", session_id, "closed")

    def _cleanup(self, failed):
        with self._lock:
            if not failed:
                self._closed = True
            self._fail_locked(
                WorkerError(
                    "worker client aborted" if failed else "worker client is closed"
                ),
                failed,
            )
        with self._cleanup_lock:
            if not self._reaped:
                self._reaped = _shutdown_process(self._proc, self._streams)
            if self._reaped:
                self._reader.join(timeout=5)
                self._writer.join(timeout=5)

    def abort(self) -> None:
        """Fail outstanding operations and terminate/reap the worker."""
        self._cleanup(True)

    def close(self) -> None:
        """Settle outstanding operations and shut down the worker idempotently."""
        self._cleanup(False)
