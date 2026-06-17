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

import json
import logging
import subprocess
import threading
from dataclasses import dataclass, field
from typing import Callable, Optional, Sequence

logger = logging.getLogger(__name__)


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
    # The exact (non-terminal) token ids generated this turn. The control plane
    # stores these per session and splices them back as an `ids` prompt segment
    # next turn, so a prior assistant span is an exact token extension instead of
    # a lossy chat-template re-render. Empty on older workers.
    generated_token_ids: list = field(default_factory=list)


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
    except json.JSONDecodeError as e:
        raise WorkerError(f"invalid worker JSON: {line.rstrip()!r}") from e
    if not isinstance(msg, dict):
        raise WorkerError(f"invalid worker message: expected object, got {msg!r}")
    return msg


class WorkerClient:
    """Drives one model-execution worker process over JSONL (raw transport).

    Exposes ``generate(prompt, config, token_callback, stats_callback)`` /
    ``stop()`` plus the session ops ``open_session`` / ``close_session`` /
    ``reset_session`` that SessionRuntime drives. Calls are serialized by a lock
    and by SessionRuntime (one in-flight request). The control plane never
    executes model code.
    """

    def __init__(self, proc: subprocess.Popen, max_named_sessions: int = 0):
        self._proc = proc
        self._lock = threading.Lock()
        # Named sessions this worker can host (0 = scratch-only / single session).
        self.max_named_sessions = max_named_sessions

    def reset(self) -> None:
        # Legacy no-op; reset is explicit via reset_session, or handled by the
        # worker's prefill plan.
        pass

    def stop(self) -> None:
        # No-op: a worker request is synchronous over the JSONL pipe and is
        # NOT interruptible mid-generation. The in-flight request runs to
        # completion and head-of-line blocks every other session on this worker
        # until it finishes. Real cancellation needs a protocol change (a control
        # pipe, non-blocking stdin polling between decode steps, or request ids +
        # an out-of-band cancel op).
        pass

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

    def _op(self, request: dict, ack_key: str) -> None:
        with self._lock:
            if self._proc.poll() is not None:
                raise WorkerError(
                    f"worker exited (code {self._proc.returncode}); restart the server"
                )
            try:
                self._proc.stdin.write(json.dumps(request) + "\n")
                self._proc.stdin.flush()
            except (BrokenPipeError, ValueError) as e:
                raise WorkerError("worker stdin is closed") from e
            line = self._proc.stdout.readline()
            if not line:
                raise WorkerError("worker exited mid-request")
            msg = _decode_worker_json(line)
            if msg.get(ack_key):
                return
            if "error" in msg:
                raise WorkerError(msg["error"], code=msg.get("code"))
            raise WorkerError(f"unexpected worker response: {msg}")

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
                    generated_token_ids=msg.get("generated_token_ids", []),
                )
            )

    def generate(self, prompt, config, token_callback=None, stats_callback=None):
        request = {
            "max_new_tokens": getattr(config, "max_new_tokens", -1),
            "temperature": getattr(config, "temperature", 0.0),
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
        with self._lock:
            if self._proc.poll() is not None:
                raise WorkerError(
                    f"worker exited (code {self._proc.returncode}); restart the server"
                )
            try:
                self._proc.stdin.write(json.dumps(request) + "\n")
                self._proc.stdin.flush()
            except (BrokenPipeError, ValueError) as e:
                raise WorkerError("worker stdin is closed") from e

            while True:
                line = self._proc.stdout.readline()
                if not line:
                    raise WorkerError("worker exited mid-request")
                msg = _decode_worker_json(line)
                if "token" in msg:
                    if token_callback is not None:
                        token_callback(msg["token"])
                elif msg.get("done"):
                    self._on_done(msg, stats_callback)
                    return
                elif "error" in msg:
                    raise WorkerError(msg["error"], code=msg.get("code"))
                else:
                    raise WorkerError(f"unexpected worker response: {msg}")

    def close(self) -> None:
        """Terminate the worker process (called at server shutdown)."""
        if self._proc.poll() is not None:
            return
        try:
            if self._proc.stdin is not None:
                self._proc.stdin.close()
        except OSError:
            pass
        try:
            self._proc.terminate()
            self._proc.wait(timeout=5)
        except Exception:  # noqa: BLE001 - shutdown best-effort
            self._proc.kill()


def spawn_worker(
    cmd: Sequence[str],
    env: Optional[dict] = None,
    cwd: Optional[str] = None,
    popen: Callable[..., subprocess.Popen] = subprocess.Popen,
) -> WorkerClient:
    """Start a worker process and block until it reports ``{"ready": true}``.

    `cmd` is the worker binary and its launch args (model/tokenizer paths). The
    worker loads the model once before reporting ready, so a slow load surfaces
    here rather than on the first request.
    """
    logger.info("Starting model worker: %s", cmd[0])
    proc = popen(
        list(cmd),
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        text=True,
        bufsize=1,
        env=env,
        cwd=cwd,
    )
    line = proc.stdout.readline()
    if not line:
        raise WorkerError("worker failed to start (no output; check its stderr).")
    msg = _decode_worker_json(line)
    if not msg.get("ready"):
        raise WorkerError(f"worker did not report ready: {msg}")
    max_named = int(msg.get("max_named_sessions", 0))
    logger.info("Model worker ready (max_named_sessions=%d).", max_named)
    return WorkerClient(proc, max_named_sessions=max_named)
