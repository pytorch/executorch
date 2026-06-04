# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Pool-level contract tests: abort-on-cancel and concurrency isolation.

Written with asyncio.run (sync test bodies) to avoid depending on an async
pytest plugin.
"""

import asyncio
import threading

from executorch.extension.llm.server.python.prefix_cache import PrefixCachingSession
from executorch.extension.llm.server.python.runner_pool import (
    _admit_session_count,
    _StatelessRunner,
    RunnerPool,
)


class _FakeEngine:
    """serving_capacity()-shaped fake."""

    def __init__(self, max_physical):
        self._cap = max_physical

    def serving_capacity(self):
        return {
            "max_physical_sessions_without_weight_duplication": self._cap,
            "estimated_bytes_per_session": 0,
        }


def _cap(max_physical):
    return _FakeEngine(max_physical).serving_capacity()


# Admission clamps physical sessions to the no-duplication capacity.
def test_admit_clamps_to_serving_capacity():
    # Single-slot (XNNPACK): 4 requested -> 1 physical session.
    assert (
        _admit_session_count(
            4, _cap(1), production=True, allow_weight_duplication=False
        )
        == 1
    )
    # Capacity hosts 4 without duplication: honor the request.
    assert (
        _admit_session_count(
            4, _cap(4), production=True, allow_weight_duplication=False
        )
        == 4
    )
    # Request below capacity is untouched.
    assert (
        _admit_session_count(
            2, _cap(4), production=True, allow_weight_duplication=False
        )
        == 2
    )
    # Unknown/zero capacity -> conservative 1.
    assert (
        _admit_session_count(
            4, _cap(0), production=True, allow_weight_duplication=False
        )
        == 1
    )
    # No capacity in production -> force 1 (standalone can't share weights).
    assert (
        _admit_session_count(4, None, production=True, allow_weight_duplication=False)
        == 1
    )
    # Explicit opt-in relaxes the weight-duplication clamp (engine still
    # serializes execution, so N sessions are safe).
    assert (
        _admit_session_count(4, _cap(1), production=True, allow_weight_duplication=True)
        == 4
    )
    # ...but opt-in does NOT relax the no-capacity standalone safety clamp:
    # concurrent backend calls into separate Modules corrupt the heap.
    assert (
        _admit_session_count(4, None, production=True, allow_weight_duplication=True)
        == 1
    )
    # Bare test factory (production=False, no capacity declared) is left alone.
    assert (
        _admit_session_count(4, None, production=False, allow_weight_duplication=False)
        == 4
    )


# The footgun fix: a factory-backed pool that declares capacity is clamped too,
# so a model launcher cannot silently duplicate weights via num_runners>1.
def test_admit_clamps_factory_path_with_capacity():
    # Factory path (production=False) + capacity=1 -> clamp to 1, NOT 4.
    assert (
        _admit_session_count(
            4, _cap(1), production=False, allow_weight_duplication=False
        )
        == 1
    )


def test_runner_pool_factory_capacity_clamps_sessions():
    # serving_capacity=1 with num_runners=4 must build exactly one session.
    created = []

    def factory():
        r = _EchoRunner()
        created.append(r)
        return r

    pool = RunnerPool(
        runner_factory=factory,
        num_runners=4,
        serving_capacity={"max_physical_sessions_without_weight_duplication": 1},
    )
    assert len(created) == 1
    assert len(pool._sessions) == 1


class _BlockingRunner:
    """Emits one token, then blocks until stop() is called."""

    def __init__(self):
        self._gate = threading.Event()
        self.stopped = False

    def reset(self):
        pass

    def stop(self):
        self.stopped = True
        self._gate.set()

    def generate(self, prompt, config, token_callback=None, stats_callback=None):
        if token_callback:
            token_callback("TOKEN")
        self._gate.wait(timeout=5)


class _EchoRunner:
    """Emits the prompt back as a single token; used to detect cross-talk."""

    def reset(self):
        pass

    def stop(self):
        pass

    def generate(self, prompt, config, token_callback=None, stats_callback=None):
        if token_callback:
            token_callback(prompt)


# (7) Client disconnect / cancellation must stop the runner.
def test_cancellation_calls_stop():
    async def scenario():
        runner = _BlockingRunner()
        pool = RunnerPool(runner_factory=lambda: runner, num_runners=1)
        async with pool.acquire() as r:
            agen = pool.generate_stream(r, "p", None).__aiter__()
            assert await agen.__anext__() == "TOKEN"  # runner now blocking
            nxt = asyncio.ensure_future(agen.__anext__())
            await asyncio.sleep(0.05)
            nxt.cancel()
            try:
                await nxt
            except asyncio.CancelledError:
                pass
            for _ in range(100):  # let the worker observe stop()
                if runner.stopped:
                    break
                await asyncio.sleep(0.02)
        assert runner.stopped

    asyncio.run(scenario())


# (8) Concurrent requests don't interleave / corrupt each other.
def test_concurrent_requests_isolated():
    async def scenario():
        pool = RunnerPool(runner_factory=_EchoRunner, num_runners=2)

        async def one(prompt):
            async with pool.acquire() as r:
                return "".join([t async for t in pool.generate_stream(r, prompt, None)])

        out = await asyncio.gather(one("AAA"), one("BBB"), one("CCC"))
        assert sorted(out) == ["AAA", "BBB", "CCC"]

    asyncio.run(scenario())


class _FakeTok:
    def encode(self, text, add_special_tokens=False):
        return list(text.encode("utf-8"))


class _CachingSession:
    """LLMSession-shaped fake: decode_one emits `gen_ids`, EOS on the last."""

    def __init__(self, gen_ids=(33,)):  # 33 -> byte '!'
        self.gen_ids = list(gen_ids)
        self.seeks = []
        self.prefilled = []
        self._pos = 0
        self._cursor = 0

    def prefill_tokens(self, ids):
        self.prefilled.append(list(ids))
        self._pos += len(ids)

    def decode_one(self, temperature=-1.0):
        tid = self.gen_ids[self._cursor] if self._cursor < len(self.gen_ids) else 0
        self._cursor += 1
        self._pos += 1
        return {
            "token_id": tid,
            "text": bytes([tid % 128]),
            "is_eos": self._cursor >= len(self.gen_ids),
        }

    def seek(self, p):
        self.seeks.append(p)
        self._pos = p
        self._cursor = 0

    def position(self):
        return self._pos

    def reset(self):
        self._pos = 0
        self._cursor = 0

    def stop(self):
        pass


# A tokenizer makes the pool wrap sessions in PrefixCachingSession and reuse the
# shared prefix across requests.
def test_pool_prefix_caching_reuses_across_requests():
    async def scenario():
        fake = _CachingSession(gen_ids=[33])  # cache b"abc" + [33]
        pool = RunnerPool(
            runner_factory=lambda: fake, num_runners=1, tokenizer=_FakeTok()
        )
        async with pool.acquire() as obj:
            assert isinstance(obj, PrefixCachingSession)
            _ = [t async for t in pool.generate_stream(obj, "abc", None)]
        # Turn 2 = b"abc!XY"; cache holds b"abc"+[33]=b"abc!" (prompt + the exact
        # generated id), so reuse is 4 (incl. the completion) and only b"XY"
        # prefills — completion reuse, not just the static prefix.
        async with pool.acquire() as obj:
            _ = [t async for t in pool.generate_stream(obj, "abc!XY", None)]
        assert fake.seeks[-1] == 4
        assert fake.prefilled[-1] == list(b"XY")

    asyncio.run(scenario())


# No tokenizer -> stateless wrapper that resets each request (no reuse).
def test_pool_stateless_without_tokenizer():
    async def scenario():
        resets = {"n": 0}

        class R:
            def reset(self):
                resets["n"] += 1

            def generate(
                self, prompt, config, token_callback=None, stats_callback=None
            ):
                if token_callback:
                    token_callback("x")

            def stop(self):
                pass

        pool = RunnerPool(runner_factory=R, num_runners=1)  # no tokenizer
        async with pool.acquire() as obj:
            assert isinstance(obj, _StatelessRunner)
            _ = [t async for t in pool.generate_stream(obj, "p", None)]
        assert resets["n"] == 1  # stateless path resets before generation

    asyncio.run(scenario())


# M4: acquire(prompt) routes to the idle session whose KV holds the longest
# matching prefix; a new conversation lands elsewhere instead of evicting it.
def test_pool_affinity_routing():
    async def scenario():
        pool = RunnerPool(
            runner_factory=_CachingSession, num_runners=2, tokenizer=_FakeTok()
        )

        async with pool.acquire("AAAA") as sA:  # conversation A caches on some session
            _ = [t async for t in pool.generate_stream(sA, "AAAA", None)]
        async with pool.acquire("AAAABB") as s2:  # continuation shares "AAAA"
            assert s2 is sA  # affinity hit: routed back to A's session
            _ = [t async for t in pool.generate_stream(s2, "AAAABB", None)]
        async with pool.acquire("ZZZZ") as s3:  # new conversation, no shared prefix
            assert s3 is not sA  # routed to the empty session, A's cache preserved

    asyncio.run(scenario())
