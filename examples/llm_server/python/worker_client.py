# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Generic control-plane client for a model-execution worker process.

Model execution runs in a separate C++ worker process — the Python server is
HTTP/control plane only and never loads a model, links a backend, or imports a
pybind module. This client spawns a worker binary and drives generation over
JSONL on the worker's stdin/stdout. The protocol is model-agnostic: the same
client serves a TextLLM worker, a Qwen worker, or any future model worker; only
the binary and its launch args differ.

Protocol (one JSON object per line; full reference in cpp/worker_loop.h): a
per-request `generate` (a `prompt` or `prompt_segments` form, optional
`session_id`) streams `{"token"}` then a `{"done", ...}` carrying warm-resume
stats and optional `generated_token_ids`; `open`/`close`/`reset` ops manage named
sessions; failures return `{"error", "code"?}`. The shapes this client builds and
parses are in generate()/_on_done() below.

The worker's stdout carries ONLY protocol JSON; its logs go to stderr. One
request at a time per worker; the caller (SessionRuntime) serializes. A worker
hosts one engine and routes requests to per-session_id state (anonymous requests
share a scratch session); execution is synchronous.
"""

import errno
import json
import logging
import os
import subprocess
import threading
from dataclasses import dataclass, field
from typing import Callable, Optional, Sequence

logger = logging.getLogger(__name__)

_CONTROL_FD_ENV = "EXECUTORCH_LLM_WORKER_CONTROL_FD"
_CANCEL_FRAME_BYTES = 8
_UINT64_MAX = (1 << 64) - 1
_PROCESS_WAIT_TIMEOUT_SECONDS = 5


@dataclass
class WorkerStats:
    """Usage reported by a worker at the end of a request."""

    num_prompt_tokens: int = 0
    num_generated_tokens: int = 0
    # Why generation stopped, as the worker saw it: "stop" (EOS / cooperative
    # stop) or "length" (ran to max_new, possibly clamped to the context window).
    # None if the worker didn't report it (older worker / fake).
    finish_reason: Optional[str] = None
    # Warm-resume accounting: how many prompt tokens were served from the
    # session's resident KV state vs actually prefilled this request, and why
    # ("new"|"exact_prefix"|"dirty"|"mismatch"|"equal"). Not exposed as OpenAI
    # usage; logged for measuring warm-resume hit rate. None on older workers.
    reused_prompt_tokens: int = 0
    prefilled_prompt_tokens: int = 0
    session_reset_reason: Optional[str] = None
    prefill_ms: float = 0.0
    decode_ms: float = 0.0
    total_ms: float = 0.0
    prefill_tok_s: float = 0.0
    decode_tok_s: float = 0.0
    vision_encoder_ms: Optional[float] = None
    # The exact (non-terminal) token ids generated this turn. The control plane
    # stores these per session and splices them back as an `ids` prompt segment
    # next turn, so a prior assistant span is an exact token extension instead of
    # a lossy chat-template re-render. Empty on older workers and cancelled turns.
    generated_token_ids: list = field(default_factory=list)
    # True when an out-of-band cancellation ended this request. Older workers
    # omit the field and therefore report False.
    cancelled: bool = False


class WorkerError(RuntimeError):
    """A worker process failed, exited, or reported a generation error.

    `code` carries the worker's structured error code when present
    ("capacity_exhausted", "unsupported_session"), so the HTTP layer can map it
    to the right status; None for unstructured failures.
    """

    def __init__(self, message: str, code: Optional[str] = None):
        super().__init__(message)
        self.code = code


def _decode_worker_json(line: str) -> dict:
    try:
        msg = json.loads(line)
    except json.JSONDecodeError as error:
        raise WorkerError(f"invalid worker JSON: {line.rstrip()!r}") from error
    if not isinstance(msg, dict):
        raise WorkerError(f"invalid worker message: expected object, got {msg!r}")
    return msg


def _close_fd(fd: Optional[int]) -> None:
    if fd is None:
        return
    try:
        os.close(fd)
    except OSError:
        pass


def _try_process_call(call, *args, **kwargs) -> bool:
    try:
        call(*args, **kwargs)
        return True
    except Exception:  # noqa: BLE001 - process shutdown remains best-effort
        return False


def _close_process_streams(streams: tuple) -> None:
    # Closing buffered stdout while another thread is reading it can block, so
    # callers invoke this only after the process has been reaped.
    for stream in streams:
        close = getattr(stream, "close", None)
        if close is not None:
            _try_process_call(close)


def _shutdown_process(proc: subprocess.Popen, streams: Optional[tuple] = None) -> bool:
    """Invalidate process pipes and synchronously reap a child, best effort."""
    if streams is None:
        streams = (getattr(proc, "stdin", None), getattr(proc, "stdout", None))
    try:
        proc.stdin = None
        proc.stdout = None
    except Exception:  # noqa: BLE001 - fake/foreign Popen objects may be immutable
        pass

    try:
        running = proc.poll() is None
    except Exception:  # noqa: BLE001 - still attempt bounded termination
        running = True
    if running:
        _try_process_call(proc.terminate)

    reaped = _try_process_call(proc.wait, timeout=_PROCESS_WAIT_TIMEOUT_SECONDS)
    if not reaped:
        _try_process_call(proc.kill)
        reaped = _try_process_call(proc.wait, timeout=_PROCESS_WAIT_TIMEOUT_SECONDS)
    if reaped:
        _close_process_streams(streams)
    return reaped


class WorkerClient:
    """Drives one model-execution worker process over synchronous JSONL.

    SessionRuntime should call ``reserve_request()`` synchronously before
    submitting generation to its executor, then pass the returned id as
    ``generate(..., request_id=request_id)``. If executor submission itself
    fails, ``release_request(request_id)`` releases the unused reservation.
    Direct callers may omit ``request_id``; generate then reserves internally.

    Session operations and generation are serialized by the JSONL lock. Request
    reservation, cancellation, health, and shutdown use a separate state lock,
    so ``stop()`` never waits behind a blocking stdout read.
    """

    def __init__(
        self,
        proc: subprocess.Popen,
        max_named_sessions: int = 0,
        control_fd: Optional[int] = None,
    ):
        self._proc = proc
        self._lock = threading.Lock()
        self._state_lock = threading.Lock()
        self._control_fd = control_fd
        self._next_request_id = 1
        self._active_request_id: Optional[int] = None
        self._generating_request_id: Optional[int] = None
        self._cancel_delivery: Optional[bool] = None
        self._terminal_error: Optional[WorkerError] = None
        self._failed = False
        self._closed = False
        self._cleanup_in_progress = False
        self._cleanup_complete = False
        self._cleanup_done = threading.Event()
        self._streams_to_close = (
            getattr(proc, "stdin", None),
            getattr(proc, "stdout", None),
        )
        # Named sessions this worker can host (0 = scratch-only / single session).
        self.max_named_sessions = max_named_sessions
        # spawn_worker only passes the descriptor after positive negotiation.
        self.supports_cancel = control_fd is not None

    @property
    def healthy(self) -> bool:
        """Whether this client can accept work without a known terminal failure."""
        with self._state_lock:
            if self._terminal_error is not None:
                return False
        try:
            return self._proc.poll() is None
        except Exception:  # noqa: BLE001 - a broken process handle is unhealthy
            return False

    @property
    def failed(self) -> bool:
        with self._state_lock:
            return self._failed

    @property
    def closed(self) -> bool:
        with self._state_lock:
            return self._closed

    def _record_failure(self, error: WorkerError) -> WorkerError:
        """Record and return the first terminal WorkerError for stable re-raises."""
        with self._state_lock:
            if self._terminal_error is None:
                self._terminal_error = error
                self._failed = True
            return self._terminal_error

    def _ensure_usable(self) -> None:
        with self._state_lock:
            terminal_error = self._terminal_error
        if terminal_error is not None:
            raise terminal_error
        try:
            returncode = self._proc.poll()
        except Exception as error:  # noqa: BLE001 - process state is now ambiguous
            failure = self._record_failure(
                WorkerError("failed to query worker process state")
            )
            raise failure from error
        if returncode is not None:
            raise self._record_failure(
                WorkerError(f"worker exited (code {returncode}); restart the server")
            )

    def reset(self) -> None:
        # Legacy no-op; reset is explicit via reset_session, or handled by the
        # worker's prefill plan. It still observes the terminal lifecycle.
        self._ensure_usable()

    def reserve_request(self) -> int:
        """Reserve the next cancellation id before executor submission.

        IDs are process-lifetime monotonic uint64 values. Exhaustion is terminal:
        wrapping could make a stale cancellation target a different request.
        """
        self._ensure_usable()
        with self._state_lock:
            if self._terminal_error is not None:
                raise self._terminal_error
            if self._active_request_id is not None:
                raise WorkerError("a worker request is already reserved")
            if self._next_request_id > _UINT64_MAX:
                error = WorkerError("worker cancellation request ids exhausted")
                self._terminal_error = error
                self._failed = True
                raise error
            request_id = self._next_request_id
            self._next_request_id += 1
            self._active_request_id = request_id
            self._cancel_delivery = None
            return request_id

    def release_request(self, request_id: int) -> bool:
        """Release a reservation when generation was never submitted.

        Returns False for a stale/non-matching id or once generation has begun.
        """
        with self._state_lock:
            if self._terminal_error is not None:
                raise self._terminal_error
            if (
                self._active_request_id != request_id
                or self._generating_request_id is not None
            ):
                return False
            self._active_request_id = None
            self._cancel_delivery = None
            return True

    def _begin_generation(self, request_id: int) -> None:
        if not isinstance(request_id, int) or isinstance(request_id, bool):
            raise WorkerError("request_id must be a uint64")
        if request_id <= 0 or request_id > _UINT64_MAX:
            raise WorkerError("request_id must be in [1, UINT64_MAX]")
        self._ensure_usable()
        with self._state_lock:
            if self._terminal_error is not None:
                raise self._terminal_error
            if self._active_request_id != request_id:
                raise WorkerError(f"request id {request_id} is not reserved")
            if self._generating_request_id is not None:
                raise WorkerError("a worker generation is already active")
            self._generating_request_id = request_id

    def _finish_generation(self, request_id: int) -> None:
        with self._state_lock:
            if self._generating_request_id == request_id:
                self._generating_request_id = None
            # A stale generation must never clear a newer reservation.
            if self._active_request_id == request_id:
                self._active_request_id = None
                self._cancel_delivery = None

    def _poison_control_locked(self) -> None:
        control_fd = self._control_fd
        self._control_fd = None
        self.supports_cancel = False
        _close_fd(control_fd)

    def stop(self) -> bool:
        """Deliver one cancellation frame for the active request, at most once.

        The cached bool reports whether a complete 8-byte little-endian frame was
        written. Unsupported, inactive, closed, and failed clients return False.
        EAGAIN is a per-request delivery failure; partial writes and other OS
        errors also disable the channel because framing or descriptor health is
        no longer trustworthy.
        """
        with self._state_lock:
            if self._terminal_error is not None:
                return False
            control_fd = self._control_fd
            request_id = self._active_request_id
            if not self.supports_cancel or control_fd is None or request_id is None:
                return False
            if self._cancel_delivery is not None:
                return self._cancel_delivery
            frame = request_id.to_bytes(_CANCEL_FRAME_BYTES, "little")
            try:
                written = os.write(control_fd, frame)
            except BlockingIOError:
                self._cancel_delivery = False
                return False
            except OSError as error:
                self._cancel_delivery = False
                if error.errno not in (errno.EAGAIN, errno.EWOULDBLOCK):
                    self._poison_control_locked()
                return False
            if written != len(frame):
                self._cancel_delivery = False
                self._poison_control_locked()
                return False
            self._cancel_delivery = True
            return True

    def open_session(self, session_id: str) -> None:
        """Admit a named session (idempotent). Raises WorkerError with a `code`
        ("capacity_exhausted" / "unsupported_session") if the worker refuses."""
        self._op({"op": "open", "session_id": session_id}, ack_key="opened")

    def close_session(self, session_id: str) -> None:
        """Destroy a named session, freeing its state (idempotent)."""
        self._op({"op": "close", "session_id": session_id}, ack_key="closed")

    def reset_session(self, session_id: str) -> None:
        """Clear a named session's context (KV/recurrent + resident tokens) but
        keep its capacity slot allocated (idempotent)."""
        self._op({"op": "reset", "session_id": session_id}, ack_key="reset")

    def _write_request(self, request: dict) -> None:
        payload = json.dumps(request) + "\n"
        stdin = self._proc.stdin
        if stdin is None:
            raise self._record_failure(WorkerError("worker stdin is closed"))
        try:
            written = stdin.write(payload)
            if written is not None and written != len(payload):
                raise OSError("short write to worker stdin")
            stdin.flush()
        except (OSError, ValueError) as error:
            failure = self._record_failure(WorkerError("worker stdin is closed"))
            raise failure from error

    def _read_message(self) -> dict:
        stdout = self._proc.stdout
        if stdout is None:
            raise self._record_failure(WorkerError("worker stdout is closed"))
        try:
            line = stdout.readline()
        except (OSError, ValueError) as error:
            failure = self._record_failure(WorkerError("worker stdout is closed"))
            raise failure from error
        if not line:
            raise self._record_failure(WorkerError("worker exited mid-request"))
        try:
            return _decode_worker_json(line)
        except WorkerError as error:
            raise self._record_failure(error)

    def _op(self, request: dict, ack_key: str) -> None:
        with self._lock:
            self._ensure_usable()
            self._write_request(request)
            msg = self._read_message()
            if msg.get(ack_key):
                return
            if "error" in msg:
                raise WorkerError(str(msg["error"]), code=msg.get("code"))
            raise self._record_failure(
                WorkerError(f"unexpected worker response: {msg}")
            )

    @staticmethod
    def _on_done(msg: dict, stats_callback) -> None:
        reason = msg.get("session_reset_reason")
        if reason is not None and logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                "warm-resume: reason=%s reused=%d prefilled=%d",
                reason,
                msg.get("reused_prompt_tokens", 0),
                msg.get("prefilled_prompt_tokens", 0),
            )
        if stats_callback is not None:
            stats_callback(
                WorkerStats(
                    num_prompt_tokens=msg.get("prompt_tokens", 0),
                    num_generated_tokens=msg.get("completion_tokens", 0),
                    finish_reason=msg.get("finish_reason"),
                    reused_prompt_tokens=msg.get("reused_prompt_tokens", 0),
                    prefilled_prompt_tokens=msg.get("prefilled_prompt_tokens", 0),
                    session_reset_reason=reason,
                    prefill_ms=msg.get("prefill_ms", 0.0),
                    decode_ms=msg.get("decode_ms", 0.0),
                    total_ms=msg.get("total_ms", 0.0),
                    prefill_tok_s=msg.get("prefill_tok_s", 0.0),
                    decode_tok_s=msg.get("decode_tok_s", 0.0),
                    vision_encoder_ms=msg.get("vision_encoder_ms"),
                    cancelled=bool(msg.get("cancelled", False)),
                    generated_token_ids=msg.get("generated_token_ids", []),
                )
            )

    def _generate_locked(self, request: dict, token_callback, stats_callback) -> None:
        self._ensure_usable()
        self._write_request(request)
        while True:
            msg = self._read_message()
            if "token" in msg:
                if token_callback is not None:
                    try:
                        token_callback(msg["token"])
                    except Exception as error:
                        failure = self._record_failure(
                            WorkerError("token callback failed during worker response")
                        )
                        raise failure from error
            elif msg.get("done"):
                self._on_done(msg, stats_callback)
                return
            elif "error" in msg:
                raise WorkerError(str(msg["error"]), code=msg.get("code"))
            else:
                raise self._record_failure(
                    WorkerError(f"unexpected worker response: {msg}")
                )

    def generate(
        self,
        prompt,
        config,
        token_callback=None,
        stats_callback=None,
        request_id: Optional[int] = None,
    ):
        request = {
            "max_new_tokens": getattr(config, "max_new_tokens", -1),
            "temperature": getattr(config, "temperature", 0.0),
            "top_p": getattr(config, "top_p", 1.0),
            "top_k": getattr(config, "top_k", 0),
            "seed": getattr(config, "seed", 0),
            "stop": list(getattr(config, "stop", []) or []),
        }
        # Token-ID segments take precedence over the rendered string:
        # they let prior assistant spans be exact id runs, not lossy re-renders.
        # `is not None` (not truthiness): segments is a distinct prompt form, kept
        # whatever its content (the worker validates non-empty).
        segments = getattr(config, "prompt_segments", None)
        if segments is not None:
            request["prompt_segments"] = segments
        else:
            request["prompt"] = prompt
        session_id = getattr(config, "session_id", None)
        if session_id:
            request["session_id"] = session_id

        if request_id is None:
            # Preserve direct-caller serialization: allocate only after earlier
            # JSONL work releases the lock.
            with self._lock:
                request_id = self.reserve_request()
                self._begin_generation(request_id)
                try:
                    with self._state_lock:
                        if self.supports_cancel:
                            request["cancel_request_id"] = request_id
                    self._generate_locked(request, token_callback, stats_callback)
                finally:
                    self._finish_generation(request_id)
            return

        self._begin_generation(request_id)
        try:
            with self._state_lock:
                if self.supports_cancel:
                    request["cancel_request_id"] = request_id
            with self._lock:
                self._generate_locked(request, token_callback, stats_callback)
        finally:
            self._finish_generation(request_id)

    def _start_cleanup(
        self, terminal_error: WorkerError, failed: bool
    ) -> tuple[bool, threading.Event]:
        """Enter a terminal state and choose one process-cleanup owner."""
        control_fd = None
        with self._state_lock:
            if self._terminal_error is None:
                self._terminal_error = terminal_error
                self._failed = failed
            elif failed and not self._closed:
                self._failed = True
            if not failed:
                self._closed = True
            if self._cleanup_complete:
                return False, self._cleanup_done
            if self._cleanup_in_progress:
                return False, self._cleanup_done
            self._cleanup_in_progress = True
            self._cleanup_done = threading.Event()
            control_fd = self._control_fd
            self._control_fd = None
            self.supports_cancel = False
            self._active_request_id = None
            self._generating_request_id = None
            self._cancel_delivery = None
            cleanup_done = self._cleanup_done
        _close_fd(control_fd)
        return True, cleanup_done

    def _finish_cleanup(self, reaped: bool, cleanup_done: threading.Event) -> None:
        with self._state_lock:
            self._cleanup_in_progress = False
            self._cleanup_complete = reaped
            if reaped:
                self._streams_to_close = (None, None)
            cleanup_done.set()

    def _cleanup(self, terminal_error: WorkerError, failed: bool) -> None:
        owner, cleanup_done = self._start_cleanup(terminal_error, failed)
        if not owner:
            cleanup_done.wait()
            return
        reaped = False
        try:
            reaped = _shutdown_process(self._proc, self._streams_to_close)
        finally:
            self._finish_cleanup(reaped, cleanup_done)
        if not reaped:
            logger.error("Model worker could not be reaped after termination")

    def abort(self) -> None:
        """Idempotently terminate and reap an unusable worker process."""
        self._cleanup(WorkerError("worker client aborted"), failed=True)

    def close(self) -> None:
        """Idempotently terminate and reap the worker during normal shutdown."""
        self._cleanup(WorkerError("worker client is closed"), failed=False)


def spawn_worker(
    cmd: Sequence[str],
    env: Optional[dict] = None,
    cwd: Optional[str] = None,
    popen: Callable[..., subprocess.Popen] = subprocess.Popen,
) -> WorkerClient:
    """Start a worker and wait for its additive readiness negotiation.

    POSIX workers inherit the read end of a cancellation pipe through
    ``EXECUTORCH_LLM_WORKER_CONTROL_FD``. The parent retains a nonblocking writer
    only when readiness includes ``{"supports_cancel": true}``; old workers keep
    their original JSONL request shape and behavior.
    """
    logger.info("Starting model worker: %s", cmd[0])
    control_read_fd: Optional[int] = None
    control_write_fd: Optional[int] = None
    proc: Optional[subprocess.Popen] = None
    popen_kwargs = {
        "stdin": subprocess.PIPE,
        "stdout": subprocess.PIPE,
        "text": True,
        "bufsize": 1,
        "env": env,
        "cwd": cwd,
    }

    try:
        if os.name == "posix":
            control_read_fd, control_write_fd = os.pipe()
            os.set_blocking(control_write_fd, False)
            child_env = dict(os.environ if env is None else env)
            child_env[_CONTROL_FD_ENV] = str(control_read_fd)
            popen_kwargs["env"] = child_env
            popen_kwargs["pass_fds"] = (control_read_fd,)
        proc = popen(list(cmd), **popen_kwargs)
    except Exception:
        _close_fd(control_write_fd)
        if proc is not None:
            _shutdown_process(proc)
        raise
    finally:
        _close_fd(control_read_fd)

    try:
        if proc.stdout is None:
            raise WorkerError("worker failed to start (no stdout pipe).")
        line = proc.stdout.readline()
        if not line:
            raise WorkerError("worker failed to start (no output; check its stderr).")
        msg = _decode_worker_json(line)
        if not msg.get("ready"):
            raise WorkerError(f"worker did not report ready: {msg}")
        max_named = int(msg.get("max_named_sessions", 0))
        supports_cancel = msg.get("supports_cancel") is True
        if not supports_cancel:
            _close_fd(control_write_fd)
            control_write_fd = None
        logger.info(
            "Model worker ready (max_named_sessions=%d, supports_cancel=%s).",
            max_named,
            supports_cancel and control_write_fd is not None,
        )
        client = WorkerClient(
            proc,
            max_named_sessions=max_named,
            control_fd=control_write_fd,
        )
        control_write_fd = None  # ownership transferred to WorkerClient
        return client
    except Exception:
        _close_fd(control_write_fd)
        _shutdown_process(proc)
        raise
