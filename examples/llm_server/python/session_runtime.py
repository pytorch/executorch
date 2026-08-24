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
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import AsyncIterator, Optional

from .worker_client import WorkerError

logger = logging.getLogger(__name__)

_SENTINEL = object()
_DEFAULT_CANCEL_GRACE_SECONDS = 2.0
_DEFAULT_ABORT_TIMEOUT_SECONDS = 12.0


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
    top_p: float = 1.0
    top_k: int = 0
    seed: int = 0
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
    prefill_ms: float = 0.0
    decode_ms: float = 0.0
    total_ms: float = 0.0
    prefill_tok_s: float = 0.0
    decode_tok_s: float = 0.0
    vision_encoder_ms: Optional[float] = None
    # Exact token ids generated this turn, for an adapter's transcript
    # store. Empty when the worker doesn't report them (e.g. a stop-trimmed turn).
    generated_token_ids: list = field(default_factory=list)
    cancelled: bool = False


# Forwarded to WorkerClient.generate() as the per-request config it reads fields
# off; keeps that low-level contract unchanged while the runtime's public surface
# is PromptInput + GenerationOptions + session_id.
@dataclass
class _WorkerRequest:
    max_new_tokens: int
    temperature: float
    top_p: float
    top_k: int
    seed: int
    stop: list[str]
    session_id: Optional[str]
    prompt_segments: Optional[list]


class _GenerationBridge:
    def __init__(
        self,
        worker,
        prompt_text: str,
        request: _WorkerRequest,
        stats: GenStats,
        request_id: Optional[int],
    ):
        self._worker = worker
        self._prompt_text = prompt_text
        self._request = request
        self._stats = stats
        self._request_id = request_id
        self._loop = asyncio.get_running_loop()
        self.queue: asyncio.Queue = asyncio.Queue()
        self.drop_tokens = threading.Event()
        self.worker_done = threading.Event()

    def _enqueue_if_live(self, item) -> None:
        if not self.drop_tokens.is_set():
            self.queue.put_nowait(item)

    def _enqueue_terminal(self, item) -> None:
        if not self._loop.is_closed():
            self.queue.put_nowait(item)

    def token_cb(self, token: str) -> None:
        if not self.drop_tokens.is_set() and not self._loop.is_closed():
            self._loop.call_soon_threadsafe(self._enqueue_if_live, token)

    def stats_cb(self, s) -> None:
        self._stats.prompt_tokens = s.num_prompt_tokens
        self._stats.completion_tokens = s.num_generated_tokens
        self._stats.finish_reason = getattr(s, "finish_reason", None)
        self._stats.reused_prompt_tokens = getattr(s, "reused_prompt_tokens", 0)
        self._stats.prefilled_prompt_tokens = getattr(s, "prefilled_prompt_tokens", 0)
        self._stats.session_reset_reason = getattr(s, "session_reset_reason", None)
        self._stats.prefill_ms = getattr(s, "prefill_ms", 0.0)
        self._stats.decode_ms = getattr(s, "decode_ms", 0.0)
        self._stats.total_ms = getattr(s, "total_ms", 0.0)
        self._stats.prefill_tok_s = getattr(s, "prefill_tok_s", 0.0)
        self._stats.decode_tok_s = getattr(s, "decode_tok_s", 0.0)
        self._stats.vision_encoder_ms = getattr(s, "vision_encoder_ms", None)
        self._stats.cancelled = getattr(s, "cancelled", False)
        self._stats.generated_token_ids = getattr(s, "generated_token_ids", [])

    def run(self) -> None:
        try:
            kwargs = {}
            if self._request_id is not None:
                kwargs["request_id"] = self._request_id
            self._worker.generate(
                self._prompt_text,
                self._request,
                self.token_cb,
                self.stats_cb,
                **kwargs,
            )
        except Exception as error:  # noqa: BLE001 - surface to the stream consumer
            if not self._loop.is_closed():
                self._loop.call_soon_threadsafe(self._enqueue_terminal, error)
        finally:
            self.worker_done.set()
            if not self._loop.is_closed():
                self._loop.call_soon_threadsafe(self._enqueue_terminal, _SENTINEL)

    async def items(self) -> AsyncIterator[str]:
        while True:
            item = await self.queue.get()
            if item is _SENTINEL:
                return
            if isinstance(item, Exception):
                raise item
            yield item

    def drain(self) -> None:
        while True:
            try:
                self.queue.get_nowait()
            except asyncio.QueueEmpty:
                return


class SessionRuntime:
    """Stateful runtime over one single-in-flight WorkerClient.

    Cancellation first requests a cooperative token-boundary stop. If the
    request does not finish within the grace period, the worker is aborted and
    this runtime remains failed until its owning server is restarted.
    """

    def __init__(
        self,
        worker,
        *,
        cancel_grace_seconds: float = _DEFAULT_CANCEL_GRACE_SECONDS,
        abort_timeout_seconds: float = _DEFAULT_ABORT_TIMEOUT_SECONDS,
    ):
        if cancel_grace_seconds < 0.0 or abort_timeout_seconds <= 0.0:
            raise ValueError("cancellation timeouts must be nonnegative and positive")
        self._worker = worker
        self._executor = ThreadPoolExecutor(max_workers=1)
        self._lock = asyncio.Lock()
        self._cancel_grace_seconds = cancel_grace_seconds
        self._abort_timeout_seconds = abort_timeout_seconds
        self._failure: Optional[WorkerError] = None

    @property
    def healthy(self) -> bool:
        if self._failure is not None:
            return False
        worker_health = getattr(self._worker, "healthy", True)
        return bool(worker_health() if callable(worker_health) else worker_health)

    def _ensure_healthy(self) -> None:
        if self._failure is not None:
            raise self._failure
        if not self.healthy:
            self._failure = WorkerError("model worker is unavailable; restart the server")
            raise self._failure

    def _mark_failed(self, message: str) -> WorkerError:
        if self._failure is None:
            self._failure = WorkerError(message)
        return self._failure

    async def open(self, session_id: str) -> None:
        """Admit a named session before generation so capacity errors are early."""
        await self._session_op("open_session", session_id)

    async def reset(self, session_id: str) -> None:
        """Clear a named session's context while keeping its capacity slot."""
        await self._session_op("reset_session", session_id)

    async def close(self, session_id: str) -> None:
        """Destroy a named session and free its state and capacity slot."""
        await self._session_op("close_session", session_id)

    async def _session_op(self, method: str, session_id: str) -> None:
        op = getattr(self._worker, method, None)
        if op is None:
            return
        self._ensure_healthy()
        loop = asyncio.get_running_loop()
        async with self._lock:
            self._ensure_healthy()
            await loop.run_in_executor(self._executor, op, session_id)

    def stop(self) -> bool:
        """Request an in-flight generation stop at the next token boundary."""
        return bool(self._worker.stop())

    async def _wait_for_worker(
        self, future: asyncio.Future, timeout: float
    ) -> bool:
        try:
            await asyncio.wait_for(asyncio.shield(future), timeout=timeout)
            return True
        except asyncio.TimeoutError:
            return False

    async def _cancel_generation(
        self,
        bridge: _GenerationBridge,
        future: asyncio.Future,
        uses_reservation: bool,
    ) -> None:
        bridge.drop_tokens.set()
        try:
            stop_result = self._worker.stop()
            # Legacy in-process test workers return None after synchronously
            # releasing their generation gate. Real WorkerClient returns bool.
            delivered = bool(stop_result) if uses_reservation else True
        except Exception:  # noqa: BLE001 - escalation handles failed stop delivery
            delivered = False

        # A false delivery result can race with normal worker completion after
        # it clears the active request. Always allow the same bounded grace
        # period before declaring the transport unusable.
        if await self._wait_for_worker(future, self._cancel_grace_seconds):
            bridge.drain()
            return
        if not delivered:
            logger.warning("Worker cancellation signal was not delivered")

        self._mark_failed("model worker cancellation timed out; restart the server")
        abort = getattr(self._worker, "abort", None)
        if abort is not None:
            try:
                await asyncio.wait_for(
                    asyncio.to_thread(abort), timeout=self._abort_timeout_seconds
                )
            except Exception as error:  # noqa: BLE001 - worker is already failed
                logger.error("Failed to abort model worker cleanly: %s", error)

        if not future.done():
            await self._wait_for_worker(future, self._cancel_grace_seconds)
        if not future.done():
            # The executor thread may be an uncooperative test double. Retrieve
            # any eventual exception without allowing this stale future to block
            # the runtime lock or a later shutdown.
            future.add_done_callback(
                lambda done: done.exception() if not done.cancelled() else None
            )
        bridge.drain()

    @staticmethod
    async def _finish_cleanup(cleanup: asyncio.Task) -> None:
        """Wait for cleanup even if the caller task is cancelled repeatedly."""
        while not cleanup.done():
            try:
                await asyncio.shield(cleanup)
            except asyncio.CancelledError:
                continue
        await cleanup

    async def generate_stream(
        self,
        session_id: Optional[str],
        prompt: PromptInput,
        options: GenerationOptions,
        stats: Optional[GenStats] = None,
    ) -> AsyncIterator[str]:
        """Yield generated text while holding the one-worker serialization lock."""
        out_stats = stats if stats is not None else GenStats()
        request = _WorkerRequest(
            max_new_tokens=options.max_new_tokens,
            temperature=options.temperature,
            top_p=options.top_p,
            top_k=options.top_k,
            seed=options.seed,
            stop=list(options.stop),
            session_id=session_id,
            prompt_segments=prompt.segments,
        )

        self._ensure_healthy()
        async with self._lock:
            self._ensure_healthy()
            reserve = getattr(self._worker, "reserve_request", None)
            release = getattr(self._worker, "release_request", None)
            uses_reservation = callable(reserve)
            request_id = reserve() if uses_reservation else None
            bridge = _GenerationBridge(
                self._worker, prompt.text or "", request, out_stats, request_id
            )
            loop = asyncio.get_running_loop()
            try:
                future = loop.run_in_executor(self._executor, bridge.run)
            except BaseException:
                if request_id is not None and callable(release):
                    release(request_id)
                raise

            completed = False
            cleanup: Optional[asyncio.Task] = None
            try:
                async for item in bridge.items():
                    yield item
                completed = True
            except BaseException:
                if not bridge.worker_done.is_set():
                    cleanup = asyncio.create_task(
                        self._cancel_generation(bridge, future, uses_reservation)
                    )
                    await self._finish_cleanup(cleanup)
                raise
            finally:
                if completed or bridge.worker_done.is_set():
                    await asyncio.shield(future)
                    bridge.drop_tokens.set()
                    bridge.drain()
                elif cleanup is None and not future.done():
                    cleanup = asyncio.create_task(
                        self._cancel_generation(bridge, future, uses_reservation)
                    )
                    await self._finish_cleanup(cleanup)

    def close_worker(self) -> None:
        """Shut down the worker process and executor during server shutdown."""
        close = getattr(self._worker, "close", None)
        if close is not None:
            close()
        self._executor.shutdown(wait=False, cancel_futures=True)
