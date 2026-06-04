# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Pool of runner sessions + the streaming bridge.

Each pooled item exposes a uniform ``generate(prompt, config, token_cb,
stats_cb)`` / ``stop()`` surface. With a tokenizer it's a PrefixCachingSession
(turn-to-turn KV prefix reuse via seek/prefill_tokens); without one it's a
_StatelessRunner.

Session lifecycle: sessions are conversation-scoped, not request-scoped. The
pool keeps them warm for the process lifetime — acquire(prompt) routes a request
to the idle session whose KV already holds the longest matching prefix, and
release returns it to the pool with KV intact so the conversation's next turn
reuses it (concurrent conversations keep their caches instead of round-robin
eviction). A cache session is reset only on an unrecoverable error, and torn
down at shutdown — never reset per request, which would discard the cache reuse
exists to exploit. Stateless mode (no tokenizer) is the deliberate exception:
_StatelessRunner resets before each request because it does no prefix reuse.
There is no idle-eviction policy yet — N is fixed at construction.

A pool of N gives concurrency (the pybindings release the GIL during
prefill/decode). The cost is N x the per-session KV cache, which dominates
memory (e.g. ~7.5 GB at 32K context for a 0.6B model vs ~0.5 GB of weights), so
N is bounded by RAM, not compute.
"""

import asyncio
import logging
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import AsyncIterator, Callable, Optional

from executorch.extension.llm.runner import GenerationConfig, TextLLMRunner

from .prefix_cache import longest_common_prefix, PrefixCachingSession

logger = logging.getLogger(__name__)

_SENTINEL = object()
_MAX_CONTEXT_KEY = "get_max_context_len"
_MAX_SEQ_KEY = "get_max_seq_len"


def _register_llm_ops() -> None:
    """Register ExecuTorch LLM custom + quantized ops in this process.

    LLM .pte files exported with use_sdpa_with_kv_cache use llama::custom_sdpa,
    and quantized exports (e.g. embedding_quantize / 8da4w) use
    quantized_decomposed ops. The C++ runners link these; the Python runtime
    must import them or load_method('forward') fails with "Missing operator".
    Harmless if a build has them statically registered.
    """
    try:
        import executorch.extension.llm.custom_ops.custom_ops  # noqa: F401
    except Exception as e:  # noqa: BLE001
        logger.debug("custom_ops not imported (%s); assuming statically linked", e)
    try:
        import executorch.kernels.quantized  # noqa: F401
    except Exception as e:  # noqa: BLE001
        logger.debug(
            "quantized kernels not imported (%s); assuming statically linked", e
        )


@dataclass
class GenStats:
    prompt_tokens: int = 0
    completion_tokens: int = 0


def _admit_session_count(requested, capacity, production, allow_weight_duplication):
    """How many *physical* sessions to create.

    `capacity` is the authoritative serving-capacity dict — from the pool's own
    engine, or explicitly supplied for a factory-backed pool (a model launcher
    that builds sessions itself). When present, clamp to
    max_physical_sessions_without_weight_duplication unless the operator opts
    into duplication. This applies to BOTH the engine path and the factory path:
    a factory-backed launcher MUST pass capacity so it cannot bypass enforcement
    and silently duplicate weights / run unsafe concurrent backend calls (e.g. a
    single-slot CUDA model would otherwise load N copies of its weights).

    With no capacity: a standalone TextLLMRunner in production forces 1 (it can't
    share weights or serialize concurrent backend calls). A bare test factory
    (no capacity declared) keeps the requested N for convenience."""
    n = max(1, requested)
    if capacity is not None:
        # The escape hatch relaxes ONLY the weight-duplication clamp — the engine
        # still serializes backend execution internally, so N sessions are safe
        # (just memory-costly). It does not apply to the no-capacity path below.
        if allow_weight_duplication:
            return n
        cap = int(capacity.get("max_physical_sessions_without_weight_duplication", 1))
        limit = cap if cap > 0 else 1
        if n > limit:
            logger.warning(
                "Backend hosts %d physical session(s) without weight duplication; "
                "clamping num_runners %d->%d. Concurrent requests queue on the "
                "resident session(s).",
                limit,
                n,
                limit,
            )
            return limit
        return n
    # No capacity declared: standalone TextLLMRunner. N>1 in production is unsafe
    # REGARDLESS of weight-duplication willingness — concurrent backend calls into
    # separate Modules corrupt the heap (no engine-owned serialization), so the
    # escape hatch does NOT relax this. Fix thread-safety or use the engine path
    # first.
    if production and n > 1:
        logger.warning(
            "No shared-weight engine (no --hf-tokenizer, or data_path set); forcing "
            "num_runners=1 — %d standalone runners would duplicate weights and run "
            "unsafe concurrent backend calls.",
            n,
        )
        return 1
    return n


class _StatelessRunner:
    """Resets the KV cache before each request (no prefix reuse). Used when no
    tokenizer is available to do token-level prefix matching."""

    def __init__(self, runner):
        self._runner = runner

    def generate(self, prompt, config, token_callback=None, stats_callback=None):
        self._runner.reset()
        self._runner.generate(prompt, config, token_callback, stats_callback)

    def stop(self):
        self._runner.stop()


class RunnerPool:
    def __init__(
        self,
        model_path: Optional[str] = None,
        tokenizer_path: Optional[str] = None,
        data_path: Optional[str] = None,
        num_runners: int = 1,
        runner_factory: Optional[Callable[[], object]] = None,
        tokenizer: Optional[object] = None,
        allow_weight_duplication_for_parallel_runners: bool = False,
        serving_capacity: Optional[dict] = None,
    ):
        # serving_capacity: authoritative physical-session capacity for a
        # factory-backed pool (a model launcher building its own sessions). The
        # engine path derives this from the engine; a factory path MUST pass it
        # (e.g. {"max_physical_sessions_without_weight_duplication": 1}) or N>1
        # would silently duplicate weights — RunnerPool does not infer capacity
        # from an opaque factory.
        # runner_factory is the test/extensibility seam: tests inject a fake.
        # Production wiring:
        #  - With a tokenizer (prefix cache enabled, --hf-tokenizer) and engine
        #    support: one LLMEngine; each session reuses the engine resources and
        #    backend execution is serialized. The factory yields an LLMSession
        #    that PrefixCachingSession drives via prefill_tokens + decode_one
        #    (exact ids -> turn-to-turn reuse incl. completions).
        #  - Without a tokenizer (or with .ptd): the standalone TextLLMRunner
        #    generate() path (no shared weights / no engine serialization).
        production = runner_factory is None
        self._engine = None
        self._max_context_len = None
        self._max_seq_len = None
        if runner_factory is None:
            _register_llm_ops()
            if tokenizer is not None and data_path is None:
                from executorch.extension.llm.runner import LLMEngine

                self._engine = LLMEngine(
                    model_path=model_path, tokenizer_path=tokenizer_path
                )
                metadata = self._engine.metadata()
                self._max_context_len = metadata.get(_MAX_CONTEXT_KEY)
                self._max_seq_len = metadata.get(_MAX_SEQ_KEY)
                _engine = self._engine

                def runner_factory():
                    return _engine.create_session()

            else:
                if tokenizer is not None:
                    logger.warning(
                        "Prefix cache requested, but LLMEngine is unavailable for this "
                        "artifact path (for example data_path/.ptd). Falling back to "
                        "stateless TextLLMRunner; token-step APIs are exposed only "
                        "through LLMSession."
                    )

                def runner_factory():
                    return TextLLMRunner(
                        model_path=model_path,
                        tokenizer_path=tokenizer_path,
                        data_path=data_path,
                    )

        def make_session(index):
            runner = runner_factory()
            if tokenizer is not None and (self._engine is not None or not production):
                # Drive an LLMSession-shaped object via decode_one with prefix
                # reuse. Production raw TextLLMRunner fallback deliberately does
                # not use this path; token-step pybinds live only on LLMSession.
                return PrefixCachingSession(
                    runner,
                    tokenizer,
                    index=index,
                    max_context_len=self._max_context_len,
                    max_seq_len=self._max_seq_len,
                )
            return _StatelessRunner(runner)

        # Capacity is authoritative from the owned engine; for a factory-backed
        # pool it must be supplied explicitly (the factory is opaque to us).
        capacity = (
            self._engine.serving_capacity()
            if self._engine is not None
            else serving_capacity
        )
        n = _admit_session_count(
            num_runners,
            capacity,
            production,
            allow_weight_duplication_for_parallel_runners,
        )
        self._tokenizer = tokenizer
        self._sessions = [make_session(i) for i in range(n)]
        self._busy = [False] * n
        self._cond = asyncio.Condition()
        self._executor = ThreadPoolExecutor(max_workers=n)

    def _pick(self, prompt: str) -> int:
        """Index of an idle session, preferring the one whose KV already holds the
        longest token prefix of `prompt` (so a conversation's next turn lands on
        the runner that can reuse its cache). Tie -> emptiest cache, to avoid
        evicting a longer cache that likely belongs to another live conversation."""
        idle = [i for i, b in enumerate(self._busy) if not b]
        if self._tokenizer is None or not prompt:
            return idle[0]
        try:
            pids = self._tokenizer.encode(prompt, add_special_tokens=False)
        except (
            Exception
        ):  # noqa: BLE001 - routing is best-effort; fall back to any idle
            return idle[0]

        def key(i: int):
            cached = getattr(self._sessions[i], "cached_tokens", None) or []
            return (longest_common_prefix(cached, pids), -len(cached))

        return max(idle, key=key)

    @asynccontextmanager
    async def acquire(self, prompt: str = ""):
        async with self._cond:
            while all(self._busy):
                await self._cond.wait()
            idx = self._pick(prompt)
            self._busy[idx] = True
        try:
            yield self._sessions[idx]
        finally:
            async with self._cond:
                self._busy[idx] = False
                self._cond.notify()

    async def generate_stream(
        self,
        runner,
        prompt: str,
        config: GenerationConfig,
        stats: Optional[GenStats] = None,
    ) -> AsyncIterator[str]:
        """Yield generated text tokens. If `stats` is given it's filled in place
        with token counts (per-request, so concurrent streams don't race)."""
        out_stats = stats if stats is not None else GenStats()
        loop = asyncio.get_running_loop()
        queue: asyncio.Queue = asyncio.Queue()

        def token_cb(token: str) -> None:
            loop.call_soon_threadsafe(queue.put_nowait, token)

        def stats_cb(s) -> None:
            out_stats.prompt_tokens = s.num_prompt_tokens
            out_stats.completion_tokens = s.num_generated_tokens

        def run() -> None:
            try:
                # `runner` is a session wrapper: PrefixCachingSession reuses the
                # shared prefix; _StatelessRunner resets first. Cache policy lives
                # in the wrapper, not here.
                runner.generate(prompt, config, token_cb, stats_cb)
            except Exception as e:  # noqa: BLE001 - surface to the stream consumer
                loop.call_soon_threadsafe(queue.put_nowait, e)
            finally:
                loop.call_soon_threadsafe(queue.put_nowait, _SENTINEL)

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
            runner.stop()
            raise
        finally:
            await fut
