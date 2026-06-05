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

Protocol (one JSON object per line):
  worker -> stdout, once at startup:  {"ready": true}
  client -> stdin,  per request:      {"prompt": str, "max_new_tokens": int,
                                       "temperature": float, "stop": [str, ...]}
  worker -> stdout, per request:      {"token": str} *   (streamed)
                                      {"done": true, "prompt_tokens": int,
                                       "completion_tokens": int,
                                       "finish_reason": "stop" | "length"}
                                  or  {"error": str}

The worker's stdout carries ONLY protocol JSON; its logs go to stderr. One
request at a time per worker (one worker == one session); the caller serializes.
"""

import json
import logging
import subprocess
import threading
from dataclasses import dataclass
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


class WorkerError(RuntimeError):
    """A worker process failed, exited, or reported a generation error."""


class WorkerClient:
    """Drives one model-execution worker process over JSONL.

    Exposes the same ``generate(prompt, config, token_callback, stats_callback)``
    / ``reset()`` / ``stop()`` surface the runner pool expects, so a pool of
    workers is a drop-in for the (retired) in-process session pool. One worker
    hosts one session; calls are serialized by a lock (and by the pool's single
    slot per worker). The control plane never executes model code.
    """

    def __init__(self, proc: subprocess.Popen):
        self._proc = proc
        self._lock = threading.Lock()

    def reset(self) -> None:
        # The worker resets its session at the start of each request; nothing to
        # do here.
        pass

    def stop(self) -> None:
        # Best-effort: a request is synchronous and not interruptible mid-
        # generation in V1.
        pass

    def generate(self, prompt, config, token_callback=None, stats_callback=None):
        request = {
            "prompt": prompt,
            "max_new_tokens": getattr(config, "max_new_tokens", -1),
            "temperature": getattr(config, "temperature", 0.0),
            "stop": list(getattr(config, "stop", []) or []),
        }
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
                msg = json.loads(line)
                if "token" in msg:
                    if token_callback is not None:
                        token_callback(msg["token"])
                elif msg.get("done"):
                    if stats_callback is not None:
                        stats_callback(
                            WorkerStats(
                                msg.get("prompt_tokens", 0),
                                msg.get("completion_tokens", 0),
                                msg.get("finish_reason"),
                            )
                        )
                    return
                elif "error" in msg:
                    raise WorkerError(msg["error"])

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
    msg = json.loads(line)
    if not msg.get("ready"):
        raise WorkerError(f"worker did not report ready: {msg}")
    logger.info("Model worker ready.")
    return WorkerClient(proc)
