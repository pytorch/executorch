# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Tests for the model-agnostic SessionGenerateAdapter (no model/GPU)."""

from executorch.extension.llm.server.python.session_generate import (
    SessionGenerateAdapter,
    utf8_complete_prefix_len,
)


class FakeSession:
    def __init__(self, pieces):
        # pieces: list of (text_bytes, is_eos)
        self._pieces = list(pieces)
        self._i = 0
        self.reset_count = 0
        self.stopped = False
        self.prefilled = None

    def prefill_tokens(self, ids, temperature=-1.0):
        self.prefilled = list(ids)
        self.prefill_temperature = temperature
        self._i = 0

    def decode_one(self, temperature=-1.0):
        self.temperature = temperature
        if self._i >= len(self._pieces):
            return {"token_id": 0, "text": b"", "is_eos": True}
        text, is_eos = self._pieces[self._i]
        self._i += 1
        return {"token_id": 100 + self._i, "text": text, "is_eos": is_eos}

    def reset(self):
        self.reset_count += 1
        self._i = 0

    def stop(self):
        self.stopped = True


class FakeTokenizer:
    def encode(self, text, add_special_tokens=False):
        assert add_special_tokens is False
        return [1, 2, 3, 4]


class _Cfg:
    __slots__ = ("max_new_tokens", "temperature")

    def __init__(self, max_new_tokens=64, temperature=0.0):
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature


def _run(pieces, cfg=None):
    session = FakeSession(pieces)
    adapter = SessionGenerateAdapter(session, FakeTokenizer())
    out, stats = [], {}
    adapter.generate(
        "hello",
        cfg or _Cfg(),
        token_callback=out.append,
        stats_callback=lambda s: stats.update(
            prompt=s.num_prompt_tokens, gen=s.num_generated_tokens
        ),
    )
    return session, "".join(out), stats


def test_utf8_complete_prefix_len():
    assert utf8_complete_prefix_len(b"\xe2\x82") == 0  # incomplete 3-byte
    assert utf8_complete_prefix_len(b"\xe2\x82\xac") == 3  # complete €
    assert utf8_complete_prefix_len(b"hi\xe2\x82") == 2
    assert utf8_complete_prefix_len(b"ascii") == 5


def test_basic_generation():
    session, text, stats = _run([(b"Hello", False), (b" world", False), (b"", True)])
    assert text == "Hello world"
    assert session.prefilled == [1, 2, 3, 4]
    assert stats == {"prompt": 4, "gen": 2}  # EOS step not counted


def test_utf8_split_across_pieces():
    _, text, _ = _run(
        [(b"\xe2", False), (b"\x82", False), (b"\xac", False), (b"", True)]
    )
    assert text == "€"


def test_respects_max_new_tokens():
    _, text, stats = _run([(b"a", False)] * 100, cfg=_Cfg(max_new_tokens=3))
    assert text == "aaa"
    assert stats["gen"] == 3


def test_cooperative_stop():
    session = FakeSession([(b"a", False)] * 100)
    adapter = SessionGenerateAdapter(session, FakeTokenizer())
    out = []

    def cb(t):
        out.append(t)
        if len(out) == 2:
            adapter.stop()

    adapter.generate("hi", _Cfg(100), token_callback=cb)
    assert out == ["a", "a"]
    assert session.stopped is True


def test_temperature_passed_to_decode():
    session, _, _ = _run([(b"x", False), (b"", True)], cfg=_Cfg(temperature=0.0))
    assert session.temperature == 0.0
