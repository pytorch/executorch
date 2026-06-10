# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Python's stateful local-LLM runtime over one C++ worker process.

This is the internal boundary between protocol adapters (OpenAI chat, future
native/agent surfaces) and the worker. The adapter speaks sessions, prompts, and
generation parameters; the worker (driven over JSONL by a WorkerClient) owns all
model execution and session state (KV/recurrent, resident token ids, warm-resume
prefix logic). The Python server never loads a model, links a backend, or imports
a runtime pybind.

A SessionRuntime owns exactly one worker and serializes access to it (one
in-flight request at a time), bridging the worker's blocking generate() into an
async token stream. Multi-worker scheduling / named-session affinity is out of
scope: a single worker already hosts many isolated sessions on one weight load,
routed by session_id inside the worker.
"""

import asyncio
import logging
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import AsyncIterator, Optional

logger = logging.getLogger(__name__)

_SENTINEL = object()


@dataclass
class PromptInput:
    """A prompt as either a single rendered string or token-ID segments. Exactly
    one of `text` / `segments` is set. Segments ([{"text": str} | {"ids": [int]}])
    let an adapter splice exact prior-turn token ids in place of a lossy
    re-render (see openai_transcript)."""

    text: Optional[str] = None
    segments: Optional[list] = None

    def __post_init__(self):
        if (self.text is None) == (self.segments is None):
            raise ValueError("exactly one of PromptInput.text / .segments must be set")
        if self.segments is not None and not self.segments:
            raise ValueError("PromptInput.segments must be non-empty")


@dataclass
class GenerationOptions:
    """Sampling/length knobs forwarded to the worker (only what we honor today)."""

    max_new_tokens: int
    temperature: float = 0.0
    stop: list[str] = field(default_factory=list)


@dataclass
class GenStats:
    """Per-request metadata the worker reports at the end of generation."""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    # Worker-reported stop reason ("stop" | "length"), or None if not reported.
    finish_reason: Optional[str] = None
    # Warm-resume accounting: tokens served from the session's resident
    # state vs prefilled this request, and why.
    reused_prompt_tokens: int = 0
    prefilled_prompt_tokens: int = 0
    session_reset_reason: Optional[str] = None
    # Exact token ids generated this turn, for an adapter's transcript
    # store. Empty when the worker doesn't report them (e.g. a stop-trimmed turn).
    generated_token_ids: list = field(default_factory=list)


# Forwarded to WorkerClient.generate() as the per-request config it reads fields
# off; keeps that low-level contract unchanged while the runtime's public surface
# is PromptInput + GenerationOptions + session_id.
@dataclass
class _WorkerRequest:
    max_new_tokens: int
    temperature: float
    stop: list[str]
    session_id: Optional[str]
    prompt_segments: Optional[list]


class SessionRuntime:
    """Stateful runtime over a single worker. `worker` is a WorkerClient (a fake
    in tests) exposing generate()/stop()/close() and the session ops
    open_session/reset_session/close_session."""

    def __init__(self, worker):
        self._worker = worker
        # One executor thread; the lock guarantees the worker is never driven by
        # two requests at once (it is single-in-flight).
        self._executor = ThreadPoolExecutor(max_workers=1)
        self._lock = asyncio.Lock()

    async def open(self, session_id: str) -> None:
        """Admit a named session before generation so a capacity refusal surfaces
        up front (the adapter maps it to an HTTP status) rather than mid-stream.
        Idempotent."""
        await self._session_op("open_session", session_id)

    async def reset(self, session_id: str) -> None:
        """Clear a named session's context (KV/recurrent + resident ids) but keep
        its capacity slot. Idempotent."""
        await self._session_op("reset_session", session_id)

    async def close(self, session_id: str) -> None:
        """Destroy a named session, freeing its state and slot. Idempotent."""
        await self._session_op("close_session", session_id)

    async def _session_op(self, method: str, session_id: str) -> None:
        op = getattr(self._worker, method, None)
        if op is None:
            return  # worker doesn't support sessions (e.g. a minimal fake)
        loop = asyncio.get_running_loop()
        async with self._lock:
            await loop.run_in_executor(self._executor, op, session_id)

    def stop(self) -> None:
        """Request the in-flight generation stop at the next token boundary."""
        self._worker.stop()

    async def generate_stream(
        self,
        session_id: Optional[str],
        prompt: PromptInput,
        options: GenerationOptions,
        stats: Optional[GenStats] = None,
    ) -> AsyncIterator[str]:
        """Yield generated text pieces from the worker, holding the worker lock
        for the whole generation. `stats` (if given) is filled in place with the
        worker's terminal metadata (per-request, so concurrent callers don't
        race). session_id None routes to the worker's anonymous scratch session."""
        out_stats = stats if stats is not None else GenStats()
        req = _WorkerRequest(
            max_new_tokens=options.max_new_tokens,
            temperature=options.temperature,
            stop=list(options.stop),
            session_id=session_id,
            prompt_segments=prompt.segments,
        )
        prompt_text = prompt.text or ""
        loop = asyncio.get_running_loop()
        queue: asyncio.Queue = asyncio.Queue()

        def token_cb(token: str) -> None:
            loop.call_soon_threadsafe(queue.put_nowait, token)

        def stats_cb(s) -> None:
            out_stats.prompt_tokens = s.num_prompt_tokens
            out_stats.completion_tokens = s.num_generated_tokens
            out_stats.finish_reason = getattr(s, "finish_reason", None)
            out_stats.reused_prompt_tokens = getattr(s, "reused_prompt_tokens", 0)
            out_stats.prefilled_prompt_tokens = getattr(s, "prefilled_prompt_tokens", 0)
            out_stats.session_reset_reason = getattr(s, "session_reset_reason", None)
            out_stats.generated_token_ids = getattr(s, "generated_token_ids", [])

        def run() -> None:
            try:
                self._worker.generate(prompt_text, req, token_cb, stats_cb)
            except Exception as e:  # noqa: BLE001 - surface to the stream consumer
                loop.call_soon_threadsafe(queue.put_nowait, e)
            finally:
                loop.call_soon_threadsafe(queue.put_nowait, _SENTINEL)

        async with self._lock:
            fut = loop.run_in_executor(self._executor, run)
            try:
                while True:
                    item = await queue.get()
                    if item is _SENTINEL:
                        break
                    if isinstance(item, Exception):
                        raise item
                    yield item
            except asyncio.CancelledError:
                # stop() is a no-op and we still `await fut` below, so a
                # cancelled/disconnected client does NOT interrupt the worker --
                # the in-flight generation runs to completion and head-of-line
                # blocks other sessions until it does. Real interruption needs a
                # worker-protocol cancel (see WorkerClient.stop).
                self._worker.stop()
                raise
            finally:
                await fut

    def close_worker(self) -> None:
        """Shut the worker process down and the executor (called at shutdown)."""
        close = getattr(self._worker, "close", None)
        if close is not None:
            close()
        self._executor.shutdown(wait=False, cancel_futures=True)
