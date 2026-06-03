# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Tests for the prefix-cache session driver over the LLMSession API, using a
fake session (prefill_tokens/decode_one/seek/position/reset) + a byte-level fake
tokenizer so reuse decisions and exact-id cache tracking are verified without a
real engine."""

import pytest

from executorch.extension.llm.server.python.prefix_cache import (
    longest_common_prefix,
    PrefixCachingSession,
)


class FakeTokenizer:
    # Byte-level: prefix-preserving and deterministic — ideal for LCP testing.
    def encode(self, text, add_special_tokens=False):
        return list(text.encode("utf-8"))


class FakeSession:
    """LLMSession-shaped fake: decode_one() emits `gen_ids` in order, signalling
    is_eos on the last one. Tracks position like the real session."""

    def __init__(self, gen_ids=(10,), fail_seek=False):
        self.gen_ids = list(gen_ids)
        self.fail_seek = fail_seek
        self._cursor = 0
        self._pos = 0
        self.seeks = []
        self.prefilled = []
        self.reset_count = 0

    def prefill_tokens(self, ids):
        assert len(ids) >= 1, "prefill_tokens must get >=1 token"
        self.prefilled.append(list(ids))
        self._pos += len(ids)

    def decode_one(self, temperature=-1.0):
        tid = self.gen_ids[self._cursor] if self._cursor < len(self.gen_ids) else 0
        self._cursor += 1
        self._pos += 1
        return {
            "token_id": tid,
            "text": bytes([65 + (tid % 26)]),  # arbitrary 1-byte piece
            "is_eos": self._cursor >= len(self.gen_ids),
        }

    def seek(self, pos):
        if self.fail_seek:
            raise RuntimeError("seek unsupported (SWA)")
        self.seeks.append(pos)
        self._pos = pos
        self._cursor = 0

    def position(self):
        return self._pos

    def reset(self):
        self.reset_count += 1
        self._pos = 0
        self._cursor = 0

    def stop(self):
        pass


def _sess(gen_ids=(10,), fail_seek=False):
    return PrefixCachingSession(FakeSession(gen_ids, fail_seek), FakeTokenizer())


class FakeConfig:
    def __init__(self, max_new_tokens=-1, seq_len=-1, temperature=0.0):
        self.max_new_tokens = max_new_tokens
        self.seq_len = seq_len
        self.temperature = temperature

    def resolve_max_new_tokens(self, max_context_len, num_tokens_occupied):
        if self.seq_len == -1 and self.max_new_tokens == -1:
            result = max_context_len - num_tokens_occupied
        elif self.seq_len == -1:
            result = min(self.max_new_tokens, max_context_len - num_tokens_occupied)
        elif self.max_new_tokens == -1:
            result = min(self.seq_len, max_context_len) - num_tokens_occupied
        else:
            result = min(
                min(self.seq_len, max_context_len) - num_tokens_occupied,
                self.max_new_tokens,
            )
        return max(0, result)


def test_longest_common_prefix():
    assert longest_common_prefix([1, 2, 3], [1, 2, 9]) == 2
    assert longest_common_prefix([], [1]) == 0
    assert longest_common_prefix([1, 2], [1, 2, 3]) == 2


def test_first_turn_prefills_all_and_tracks_exact_ids():
    s = _sess(gen_ids=[10, 11])
    s.generate("abc", config=None)
    assert s._session.seeks == [0]  # seek(0) on a fresh session
    assert s._session.prefilled == [list(b"abc")]
    # cache = prompt ids + EXACT generated ids (decode_one token_ids).
    assert s.cached_tokens == list(b"abc") + [10, 11]


def test_completion_reuse_via_exact_ids():
    # Turn 2 includes turn-1's completion; exact-id tracking lets us reuse it.
    s = _sess(gen_ids=[120, 121])  # ids 120,121 -> bytes 'y','z' (120%26=16 -> 'Q'..)
    s.generate("abc", None)  # cache -> b"abc" + [120,121]
    cached_after_t1 = list(s.cached_tokens)
    # Build a turn-2 prompt whose ids are exactly the cached ids + a new suffix.
    s._session.gen_ids = [99]
    # Monkeypatch the tokenizer to return cached + new (simulating prompt=history).
    s._tok = type(
        "T",
        (),
        {
            "encode": lambda self, t, add_special_tokens=False: cached_after_t1
            + [55, 56]
        },
    )()
    s.generate("ignored", None)
    # Reused the whole cached prefix (capped at len-1), prefilled only the tail.
    assert s._session.seeks[-1] == len(cached_after_t1)
    assert s._session.prefilled[-1] == [55, 56]


def test_mid_divergence_partial_reuse():
    s = _sess(gen_ids=[10])
    s.generate("abcdef", None)  # cache -> b"abcdef" + [10]
    s.generate("abcXYZ", None)  # shares b"abc"
    assert s._session.seeks[-1] == 3
    assert s._session.prefilled[-1] == list(b"XYZ")


def test_reuse_capped_at_resident_position():
    # If _cached ever exceeds resident position, reuse is capped at position().
    s = _sess(gen_ids=[10])
    s._cached = list(b"abc!")  # seed 4 cached
    s._session._pos = 3  # but only 3 resident
    s.generate("abc!XY", None)
    assert s._session.seeks[-1] == 3  # capped, not 4
    assert s._session.prefilled[-1] == list(b"!XY")


def test_unset_max_tokens_resolves_against_context_capacity():
    session = FakeSession(gen_ids=[10, 11, 12, 13])
    s = PrefixCachingSession(
        session,
        FakeTokenizer(),
        max_context_len=5,
        max_seq_len=5,
    )
    out = []
    s.generate("abc", FakeConfig(max_new_tokens=-1), token_callback=out.append)
    assert out == ["K", "L"]  # prompt pos 3 leaves exactly 2 decode slots
    assert s.cached_tokens == list(b"abc") + [10, 11]
    assert session.position() == 5


def test_explicit_max_tokens_clamped_to_context_capacity():
    session = FakeSession(gen_ids=[10, 11, 12, 13])
    s = PrefixCachingSession(
        session,
        FakeTokenizer(),
        max_context_len=5,
        max_seq_len=5,
    )
    s.generate("abc", FakeConfig(max_new_tokens=10))
    assert s.cached_tokens == list(b"abc") + [10, 11]


def test_budget_exceeded_after_prefill_resets_and_clears_cache():
    # The post-prefill context-budget check can fail AFTER seek()/prefill_tokens()
    # already mutated KV to the new prompt. The session must reset + clear _cached
    # before raising, so a released warm session can't be affinity-matched on a
    # stale _cached and seek() into KV that holds different tokens.
    session = FakeSession(gen_ids=[10])
    s = PrefixCachingSession(session, FakeTokenizer(), max_context_len=3)
    s._cached = list(b"prev")  # pretend a prior turn cached something
    with pytest.raises(RuntimeError, match="context capacity"):
        s.generate("abcde", None)  # 5 prompt tokens > max_context_len 3 -> budget 0
    assert session.reset_count >= 1  # KV cleared, not left holding "abcde"
    assert s.cached_tokens == []  # _cached no longer describes a stale prompt


def test_warm_reuse_matches_cold_output():
    # A warm session (reuses a shared prefix) emits the same tokens as a cold
    # session that full-prefills the same prompt — reuse must not perturb output.
    warm = _sess(gen_ids=[65, 66, 67])
    warm.generate("system prefix ", None)  # warm the cache
    warm_out = []
    warm.generate("system prefix and more", None, token_callback=warm_out.append)

    cold = _sess(gen_ids=[65, 66, 67])  # fresh: seek(0) + full prefill
    cold_out = []
    cold.generate("system prefix and more", None, token_callback=cold_out.append)
    assert "".join(warm_out) == "".join(cold_out)


def test_fallback_on_seek_failure():
    # SWA model where seek() raises -> reset + full prefill, still correct.
    s = _sess(gen_ids=[10], fail_seek=True)
    out = []
    s.generate("abc", None, token_callback=lambda t: out.append(t))
    assert s._session.reset_count >= 1
    assert s._session.prefilled[-1] == list(b"abc")  # full prefill after fallback
    assert s.cached_tokens == list(b"abc") + [10]


def test_generation_error_propagates_without_retry():
    # A failure during decode_one must propagate (after reset), not be retried.
    class FailingSession(FakeSession):
        def decode_one(self, temperature=-1.0):
            raise RuntimeError("backend boom")

    s = PrefixCachingSession(FailingSession(), FakeTokenizer())
    with pytest.raises(RuntimeError, match="backend boom"):
        s.generate("abc", None)
    assert s._session.reset_count >= 1
    assert s.cached_tokens == []
