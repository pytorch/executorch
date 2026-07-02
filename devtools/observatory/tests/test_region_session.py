# Copyright (c) Qualcomm Innovation Center, Inc.
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Unit tests for the Region/Session model added in RFC §4.5.

Covers:
- enter_context with explicit region_name (outermost == Session).
- enter_context without region_name at outermost (auto-default name).
- enter_context without region_name at inner level (config-only override).
- Records carry region_stack snapshot at collect() time.
- Sibling outermost regions produce multiple Sessions.
- Default-name auto-incrementing across siblings.
- Lens session hooks fire only at Session boundaries (outermost), not at
  inner Region boundaries.
"""

from __future__ import annotations

from typing import Any, List, Optional, Tuple

import pytest

from executorch.devtools.observatory import Observatory
from executorch.devtools.observatory.interfaces import (
    Lens,
    ObservationContext,
    Session,
)


class _HookProbe(Lens):
    """Minimal lens that records every hook invocation for inspection."""

    events: List[Tuple[str, Optional[str]]] = []

    @classmethod
    def get_name(cls) -> str:
        return "hook_probe"

    @classmethod
    def setup(cls) -> None:
        cls.events = []

    @classmethod
    def clear(cls) -> None:
        cls.events = []

    @classmethod
    def on_session_start(cls, ctx: ObservationContext) -> None:
        cls.events.append(("session_start", None))
        return None

    @classmethod
    def on_session_end(cls, ctx: ObservationContext) -> None:
        cls.events.append(("session_end", None))
        return None

    @classmethod
    def observe(cls, artifact: Any, ctx: ObservationContext) -> Any:
        cls.events.append(("observe", ctx.shared_state.get("record_name")))
        return artifact

    @classmethod
    def digest(cls, observation: Any, ctx: ObservationContext) -> Any:
        return None


@pytest.fixture(autouse=True)
def _reset_observatory():
    """Fresh Observatory + lens registry per test."""

    Observatory.clear()
    Observatory._lens_registry = []
    Observatory._lenses_initialized = True  # skip default-lens registration
    Observatory.register_lens(_HookProbe)
    yield
    Observatory.clear()
    Observatory._lens_registry = []
    Observatory._lenses_initialized = False


def test_outermost_with_name_opens_session_and_tags_records():
    """Outermost enter_context(name) opens a Session named after the region."""

    with Observatory.enter_context("aot"):
        Observatory.collect("foo", object())
        Observatory.collect("bar", object())

    assert "aot" in Observatory._sessions
    assert Observatory._sessions["aot"].name == "aot"
    assert Observatory._sessions["aot"].end_ts is not None  # closed cleanly

    records = list(Observatory._records.values())
    assert len(records) == 2
    for r in records:
        assert r.session_id == "aot"
        assert r.region_stack == ["aot"]


def test_outermost_without_name_uses_default():
    """Outermost enter_context() auto-generates 'default' session name."""

    with Observatory.enter_context():
        Observatory.collect("foo", object())

    assert "default" in Observatory._sessions
    rec = list(Observatory._records.values())[0]
    assert rec.session_id == "default"
    assert rec.region_stack == ["default"]


def test_inner_without_name_is_config_only_override():
    """Inner enter_context(config=...) without region_name does NOT push a Region."""

    with Observatory.enter_context("aot"):
        Observatory.collect("a", object())
        with Observatory.enter_context(config={"foo": "bar"}):
            Observatory.collect("b", object())
        Observatory.collect("c", object())

    records = list(Observatory._records.values())
    assert len(records) == 3
    # All records belong to "aot" with the same region_stack
    for r in records:
        assert r.session_id == "aot"
        assert r.region_stack == ["aot"]
    # Only one Session in the archive
    assert list(Observatory._sessions.keys()) == ["aot"]


def test_inner_with_name_pushes_region_but_not_session():
    """Inner enter_context(name) pushes a Region under the active Session."""

    with Observatory.enter_context("aot"):
        Observatory.collect("a", object())
        with Observatory.enter_context("quantize"):
            Observatory.collect("b", object())
            with Observatory.enter_context("prepare"):
                Observatory.collect("c", object())
        Observatory.collect("d", object())

    records = list(Observatory._records.values())
    by_name = {r.name: r for r in records}
    assert by_name["a"].region_stack == ["aot"]
    assert by_name["b"].region_stack == ["aot", "quantize"]
    assert by_name["c"].region_stack == ["aot", "quantize", "prepare"]
    assert by_name["d"].region_stack == ["aot"]
    # Still one Session
    assert list(Observatory._sessions.keys()) == ["aot"]


def test_sibling_outermost_produces_two_sessions():
    """Two sequential outermost enter_context blocks open two Sessions."""

    with Observatory.enter_context("aot"):
        Observatory.collect("a", object())
    with Observatory.enter_context("device"):
        Observatory.collect("b", object())

    assert list(Observatory._sessions.keys()) == ["aot", "device"]
    by_name = {r.name: r for r in Observatory._records.values()}
    assert by_name["a"].session_id == "aot"
    assert by_name["b"].session_id == "device"


def test_default_name_auto_increments():
    """Sibling outermost calls without name use 'default', 'default-2', ..."""

    with Observatory.enter_context():
        Observatory.collect("r0", object())
    with Observatory.enter_context():
        Observatory.collect("r1", object())
    with Observatory.enter_context():
        Observatory.collect("r2", object())

    assert list(Observatory._sessions.keys()) == ["default", "default-2", "default-3"]


def test_lens_hooks_fire_only_at_session_boundary():
    """on_session_start/end fire once per outermost block, not per inner Region."""

    with Observatory.enter_context("aot"):
        with Observatory.enter_context("quantize"):
            Observatory.collect("a", object())
        with Observatory.enter_context("lower"):
            Observatory.collect("b", object())

    starts = [e for e in _HookProbe.events if e[0] == "session_start"]
    ends = [e for e in _HookProbe.events if e[0] == "session_end"]
    observes = [e for e in _HookProbe.events if e[0] == "observe"]

    assert len(starts) == 1, _HookProbe.events
    assert len(ends) == 1, _HookProbe.events
    assert len(observes) == 2


def test_collect_outside_any_context_is_noop():
    """collect() outside any enter_context block is a no-op."""

    Observatory.collect("orphan", object())
    assert len(Observatory._records) == 0
    assert len(Observatory._sessions) == 0


def test_records_are_time_ordered():
    """Records preserve insertion order (time-ordered for the left panel)."""

    with Observatory.enter_context("run"):
        Observatory.collect("first", object())
        Observatory.collect("second", object())
        Observatory.collect("third", object())

    names = [r.name for r in Observatory._records.values()]
    assert names == ["first", "second", "third"]


def test_session_name_uniqueness_enforced():
    """Re-using a Session name in a sibling block raises RuntimeError."""

    with Observatory.enter_context("phase"):
        pass
    with pytest.raises(RuntimeError, match="phase"):
        with Observatory.enter_context("phase"):
            pass


def test_lens_hooks_fire_on_exception_inside_block():
    """on_session_end runs even when the block raises."""

    with pytest.raises(ValueError):
        with Observatory.enter_context("crashy"):
            raise ValueError("boom")

    starts = [e for e in _HookProbe.events if e[0] == "session_start"]
    ends = [e for e in _HookProbe.events if e[0] == "session_end"]
    assert len(starts) == 1
    assert len(ends) == 1
    assert Observatory._sessions["crashy"].end_ts is not None


def test_enable_context_alias_still_works():
    """enable_context(config=...) keeps working as a thin alias of enter_context."""

    with Observatory.enable_context(config={"x": 1}):
        Observatory.collect("foo", object())
        with Observatory.enable_context(config={"y": 2}):
            Observatory.collect("bar", object())

    assert list(Observatory._sessions.keys()) == ["default"]
    by_name = {r.name: r for r in Observatory._records.values()}
    assert by_name["foo"].session_id == "default"
    assert by_name["bar"].session_id == "default"
