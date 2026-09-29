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

A SessionRuntime owns exactly one worker, bridging blocking generate() into
request-scoped async token streams. Explicitly multiplexed workers permit bounded
concurrency across sessions; legacy workers retain single-in-flight execution.
Named-session operations remain ordered. Multi-worker scheduling is out of scope.
"""

import asyncio
import logging
import threading
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
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
    # store. None means unknown/unsafe (e.g. a stop-trimmed turn); [] means the
    # worker explicitly reported a known-empty, resumable token sequence.
    generated_token_ids: Optional[list[int]] = None
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
        mailbox_capacity: int = 256,
        max_buffered_chars: int = 1024 * 1024,
    ):
        self._worker = worker
        self._prompt_text = prompt_text
        self._request = request
        self._stats = stats
        self._request_id = request_id
        self._loop = asyncio.get_running_loop()
        self._mailbox_capacity = mailbox_capacity
        self._max_buffered_chars = max_buffered_chars
        self._tokens = deque()
        self._buffered_chars = 0
        self._terminal = None
        self._mailbox_lock = threading.Lock()
        self._ready = asyncio.Event()
        self._wakeup_pending = False
        self.drop_tokens = threading.Event()
        self.worker_done = threading.Event()

    def _wake(self) -> None:
        with self._mailbox_lock:
            self._wakeup_pending = False
            self._ready.set()

    def _schedule_locked(self) -> None:
        if self._wakeup_pending or self._ready.is_set() or self._loop.is_closed():
            return
        self._wakeup_pending = True
        try:
            self._loop.call_soon_threadsafe(self._wake)
        except RuntimeError:
            self._wakeup_pending = False

    def finish(self, terminal=_SENTINEL) -> None:
        with self._mailbox_lock:
            if self._terminal is None:
                self._terminal = terminal
            self._schedule_locked()

    def cancel(self) -> bool:
        self.drop_tokens.set()
        if getattr(self._worker, "supports_multiplexing", False):
            return bool(self._worker.cancel(self._request_id))
        return bool(self._worker.stop())

    def token_cb(self, token: str) -> None:
        overflow = False
        with self._mailbox_lock:
            if self.drop_tokens.is_set() or self._terminal is not None:
                return
            if (
                len(self._tokens) >= self._mailbox_capacity
                or self._buffered_chars + len(token) > self._max_buffered_chars
            ):
                self._tokens.clear()
                self._buffered_chars = 0
                self._terminal = WorkerError(
                    "generation mailbox overflow", code="slow_consumer"
                )
                self.drop_tokens.set()
                overflow = True
            else:
                self._tokens.append(token)
                self._buffered_chars += len(token)
            self._schedule_locked()
        if overflow:
            self.cancel()
            # Multiplexed transports isolate callback failures. Legacy workers
            # must drain their untagged response through the wire terminal.
            if getattr(self._worker, "supports_multiplexing", False) is True:
                raise self._terminal

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
        self._stats.generated_token_ids = getattr(s, "generated_token_ids", None)

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
        except BaseException as error:
            self.finish(
                error if isinstance(error, Exception) else WorkerError(str(error))
            )
        finally:
            settle = getattr(self._worker, "wait_for_request", None)
            if self._request_id is not None and callable(settle):
                try:
                    settle(self._request_id)
                except BaseException as error:
                    self.finish(
                        error
                        if isinstance(error, Exception)
                        else WorkerError(str(error))
                    )
            self.worker_done.set()
            self.finish()

    async def items(self) -> AsyncIterator[str]:
        while True:
            with self._mailbox_lock:
                if self._tokens:
                    item = self._tokens.popleft()
                    self._buffered_chars -= len(item)
                else:
                    item = self._terminal
                    if item is None:
                        self._ready.clear()
            if item is None:
                await self._ready.wait()
            elif item is _SENTINEL:
                return
            elif isinstance(item, Exception):
                raise item
            else:
                yield item

    def drain(self) -> None:
        with self._mailbox_lock:
            self._tokens.clear()
            self._buffered_chars = 0


@dataclass
class _SessionLock:
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    users: int = 0


class _SessionLocks:
    """Loop-local locks whose references include both owners and waiters."""

    def __init__(self, limit=None):
        self._entries = {}
        self._users = 0
        self._limit = limit

    @asynccontextmanager
    async def hold(self, session_id):
        if self._limit is not None and self._users >= self._limit:
            raise WorkerError("request capacity exhausted", code="capacity_exhausted")
        self._users += 1
        entry = None
        if session_id is not None:
            entry = self._entries.setdefault(session_id, _SessionLock())
            entry.users += 1
        try:
            if entry is None:
                yield
            else:
                async with entry.lock:
                    yield
        finally:
            self._users -= 1
            if entry is not None:
                entry.users -= 1
                if not entry.users:
                    del self._entries[session_id]


@dataclass
class _RuntimeOperation:
    future: Optional[asyncio.Future] = None


class Generation:
    """Lazy request-scoped iterator; close it when abandoning consumption."""

    def __init__(self, runtime, session_id, prompt, options, stats):
        self.stats = stats if stats is not None else GenStats()
        self._bridge = None
        self._future = None
        self._prefetch = None
        self._reader = None
        self._read_done = None
        self._close_task = None
        self._cancel_requested = False
        self._closed = False
        self._iterator = runtime._generate_stream(session_id, prompt, options, self)

    @property
    def request_id(self):
        """Return the wire ID after lazy admission, or None before it starts."""
        return self._bridge._request_id if self._bridge is not None else None

    def cancel(self) -> bool:
        """Cancel only this request, including before its worker submission."""
        if self._closed:
            return False
        self._cancel_requested = True
        return self._bridge.cancel() if self._bridge is not None else True

    def __aiter__(self):
        """Iterate this request's raw tokens."""
        return self

    async def __anext__(self):
        """Read the next token, consuming the retained preflight token first."""
        if self._closed:
            raise StopAsyncIteration
        if self._reader is not None:
            raise RuntimeError("generation already has an active reader")
        self._reader = asyncio.current_task()
        self._read_done = asyncio.get_running_loop().create_future()
        try:
            if self._prefetch is not None:
                prefetch, self._prefetch = self._prefetch, None
                return await prefetch
            return await self._iterator.__anext__()
        except BaseException:
            self._closed = True
            raise
        finally:
            self._reader = None
            self._read_done.set_result(None)

    async def wait_ready(self) -> None:
        """Start generation and retain its first raw token, terminal, or error.

        One prefetch task/token per admitted request lets HTTP map admission
        errors before sending headers without pre-opening native sessions.
        """
        if self._closed:
            return
        if self._prefetch is None:
            self._prefetch = asyncio.create_task(self._iterator.__anext__())
        try:
            await asyncio.shield(self._prefetch)
        except StopAsyncIteration:
            self._closed = True
        except BaseException:
            await self.aclose()
            raise

    async def __aenter__(self):
        """Return the handle without triggering admission."""
        return self

    async def __aexit__(self, *args):
        """Settle abandoned iteration when leaving the context."""
        await self.aclose()

    async def _close(self):
        if self._reader is not None:
            self._reader.cancel()
            await SessionRuntime._finish_cleanup(self._read_done)
        if self._prefetch is not None:
            prefetch, self._prefetch = self._prefetch, None
            if not prefetch.done():
                prefetch.cancel()
            try:
                await SessionRuntime._finish_cleanup(prefetch)
            except (asyncio.CancelledError, Exception):
                pass  # The preflight/consumer owns delivery of its failure.
        await self._iterator.aclose()

    async def aclose(self):
        """Cancel and join an active read before releasing this generation."""
        self._closed = True
        if self._close_task is None:
            self._close_task = asyncio.create_task(self._close())
        await SessionRuntime._finish_cleanup(self._close_task)

    async def result(self) -> GenStats:
        """Drain remaining text and return finalized generation statistics."""
        async for _ in self:
            pass
        if self._future is not None:
            await asyncio.shield(self._future)
        return self.stats


class SessionRuntime:
    """Stateful runtime over one legacy or explicitly multiplexed worker.

    Legacy cancellation timeout aborts the worker. Multiplexed timeout retains
    only that request's admission and session lease until settlement or shutdown;
    unrelated requests are never aborted to clean up a disconnected consumer.
    """

    def __init__(
        self,
        worker,
        *,
        cancel_grace_seconds: float = _DEFAULT_CANCEL_GRACE_SECONDS,
        abort_timeout_seconds: float = _DEFAULT_ABORT_TIMEOUT_SECONDS,
        max_concurrent_requests: Optional[int] = None,
        mailbox_capacity: int = 256,
        max_buffered_chars: int = 1024 * 1024,
    ):
        if cancel_grace_seconds < 0.0 or abort_timeout_seconds <= 0.0:
            raise ValueError("cancellation timeouts must be nonnegative and positive")
        self._worker = worker
        self.supports_multiplexing = (
            getattr(worker, "supports_multiplexing", False) is True
        )
        capacity = (
            getattr(worker, "max_inflight_requests", 64)
            if self.supports_multiplexing
            else 1
        )
        if max_concurrent_requests is not None:
            capacity = min(capacity, max_concurrent_requests)
        for value in (capacity, mailbox_capacity, max_buffered_chars):
            if type(value) is not int or value <= 0:
                raise ValueError("request and mailbox limits must be positive integers")
        self.max_concurrent_requests = capacity
        self._mailbox_capacity = mailbox_capacity
        self._max_buffered_chars = max_buffered_chars
        self._executor = ThreadPoolExecutor(max_workers=capacity)
        self._lock = asyncio.Lock()
        self._session_locks = _SessionLocks()
        self._admitted = 0
        self._settlements = set()
        self._bridges = set()
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
            self._failure = WorkerError(
                "model worker is unavailable; restart the server"
            )
            raise self._failure

    def _mark_failed(self, message: str) -> WorkerError:
        if self._failure is None:
            self._failure = WorkerError(message)
        return self._failure

    async def open(self, session_id: str) -> None:
        """Admit a named session before generation so capacity errors are early."""
        await self._session_op("open_session", session_id)

    async def reset(self, session_id: str) -> None:
        """Replace a named session's context under the same public ID."""
        await self._session_op("reset_session", session_id)

    async def close(self, session_id: str) -> None:
        """Destroy a named session and free its state and capacity slot."""
        await self._session_op("close_session", session_id)

    async def _release_operation(self, operation, lease) -> None:
        try:
            await asyncio.shield(operation.future)
        except Exception:
            pass  # A disconnected caller no longer consumes lifecycle errors.
        finally:
            await lease.__aexit__(None, None, None)
            self._admitted -= 1

    @asynccontextmanager
    async def _operation(self, session_id):
        self._ensure_healthy()
        operation = _RuntimeOperation()
        if not self.supports_multiplexing:
            async with self._lock:
                self._ensure_healthy()
                yield operation
            return
        if self._admitted >= self.max_concurrent_requests:
            raise WorkerError(
                "worker request capacity exhausted", code="capacity_exhausted"
            )
        self._admitted += 1
        lease = self._session_locks.hold(session_id)
        entered = False
        try:
            await lease.__aenter__()
            entered = True
            self._ensure_healthy()
            yield operation
        finally:
            if entered and operation.future is not None and not operation.future.done():
                settlement = asyncio.create_task(
                    self._release_operation(operation, lease)
                )
                self._settlements.add(settlement)
                settlement.add_done_callback(self._settlements.discard)
            else:
                if entered:
                    await lease.__aexit__(None, None, None)
                self._admitted -= 1

    async def _session_op(self, method: str, session_id: str) -> None:
        op = getattr(self._worker, method, None)
        if op is None:
            return
        self._ensure_healthy()
        loop = asyncio.get_running_loop()
        async with self._operation(session_id) as operation:
            operation.future = loop.run_in_executor(self._executor, op, session_id)
            await asyncio.shield(operation.future)

    def stop(self) -> bool:
        """Legacy stop helper; multiplexed callers must cancel their Generation."""
        if self.supports_multiplexing:
            return False
        return bool(self._worker.stop())

    async def _wait_for_worker(self, future: asyncio.Future, timeout: float) -> bool:
        try:
            await asyncio.wait_for(asyncio.shield(future), timeout=timeout)
            return True
        except asyncio.TimeoutError:
            return False
        except asyncio.CancelledError:
            if future.cancelled():
                return True
            raise

    async def _cancel_generation(
        self,
        bridge: _GenerationBridge,
        future: asyncio.Future,
        uses_reservation: bool,
    ) -> None:
        bridge.drop_tokens.set()
        try:
            stop_result = bridge.cancel()
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
        if self.supports_multiplexing:
            # _operation keeps the thread admission and named-session lease
            # until the worker future settles, without holding up this consumer.
            bridge.drain()
            return

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
    async def _finish_cleanup(cleanup: asyncio.Future) -> None:
        """Wait for cleanup even if the caller task is cancelled repeatedly."""
        while not cleanup.done():
            try:
                await asyncio.shield(cleanup)
            except asyncio.CancelledError:
                continue
        await cleanup

    def generate_stream(
        self,
        session_id: Optional[str],
        prompt: PromptInput,
        options: GenerationOptions,
        stats: Optional[GenStats] = None,
    ) -> Generation:
        """Return a lazy, cancellable stream compatible with async iteration."""
        return Generation(self, session_id, prompt, options, stats)

    async def _generate_stream(self, session_id, prompt, options, generation):
        out_stats = generation.stats
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
        async with self._operation(session_id) as operation:
            reserve = getattr(self._worker, "reserve_request", None)
            release = getattr(self._worker, "release_request", None)
            uses_reservation = callable(reserve)
            request_id = reserve() if uses_reservation else None
            bridge = _GenerationBridge(
                self._worker,
                prompt.text or "",
                request,
                out_stats,
                request_id,
                self._mailbox_capacity,
                self._max_buffered_chars,
            )
            generation._bridge = bridge
            if generation._cancel_requested:
                bridge.cancel()
            loop = asyncio.get_running_loop()
            try:
                future = loop.run_in_executor(self._executor, bridge.run)
                operation.future = future
                generation._future = future
                self._bridges.add(bridge)
                future.add_done_callback(lambda _: self._bridges.discard(bridge))
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
        error = self._mark_failed("model worker is closed")
        for bridge in tuple(self._bridges):
            bridge.drop_tokens.set()
            bridge.drain()
            bridge.finish(error)
        close = getattr(self._worker, "close", None)
        if close is not None:
            close()
        self._executor.shutdown(wait=False, cancel_futures=True)
