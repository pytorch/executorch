# Copyright (c) Qualcomm Innovation Center, Inc.
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Unit tests for the session-as-a-first-class-unit RFC.

Covers the new payload contract introduced when the legacy flat
``SessionResult.start_data`` / ``end_data`` mirrors were dropped:

- ``Session.archive`` field defaults and propagation.
- ``SessionResult`` shape (no flat dicts).
- ``Frontend.dashboard`` signature is ``(session, session_records, analysis)``
  and is invoked once per (Session, lens) pair.
- Top-level payload keys ``archives``, ``sessions`` and the nested
  ``dashboard[lens][session_id]`` shape.
- ``export_json`` round-trip via ``_load_archive_sessions`` for both the new
  shape and the legacy nested ``session: {sessions, ...}`` shape.
- ``compare_archives`` produces N archives, prefixes session ids and record
  names, leaves region_stack alone, and rewrites graph_ref + per_layer_accuracy
  graph_ref.
"""

from __future__ import annotations

import json
import os
import tempfile
from typing import Any, Dict, List, Optional

import pytest

from executorch.devtools.observatory import Observatory
from executorch.devtools.observatory.interfaces import (
    AnalysisResult,
    Frontend,
    Lens,
    ObservationContext,
    Session,
    SessionResult,
    TableBlock,
    TableRecordSpec,
    ViewList,
)
from executorch.devtools.observatory.observatory import _NonFiniteFloatAsStringJSONEncoder


class _DashboardProbe(Lens):
    """Lens that records every dashboard call so tests can inspect arity/order."""

    calls: List[Dict[str, Any]] = []
    start_payload: Optional[Dict[str, Any]] = None
    end_payload: Optional[Dict[str, Any]] = None

    @classmethod
    def get_name(cls) -> str:
        return "probe"

    @classmethod
    def setup(cls) -> None:
        cls.calls = []
        cls.start_payload = {"hello": "start"}
        cls.end_payload = {"hello": "end"}

    @classmethod
    def clear(cls) -> None:
        cls.calls = []

    @classmethod
    def on_session_start(cls, ctx: ObservationContext) -> Optional[Dict[str, Any]]:
        return cls.start_payload

    @classmethod
    def on_session_end(cls, ctx: ObservationContext) -> Optional[Dict[str, Any]]:
        return cls.end_payload

    class _F(Frontend):
        def dashboard(self, session, session_records, analysis):
            _DashboardProbe.calls.append(
                {
                    "session_id": session.id,
                    "session_name": session.name,
                    "archive": session.archive,
                    "records": [r.name for r in session_records],
                    "start": dict(session.start_data),
                    "end": dict(session.end_data),
                }
            )
            return ViewList(
                blocks=[
                    TableBlock(
                        id="probe_block",
                        title="Probe",
                        record=TableRecordSpec(data={"sid": session.id}),
                    )
                ]
            )

    @staticmethod
    def get_frontend_spec() -> Frontend:
        return _DashboardProbe._F()


@pytest.fixture(autouse=True)
def _reset_observatory():
    Observatory.clear()
    Observatory._lens_registry = []
    Observatory._lenses_initialized = True
    Observatory.register_lens(_DashboardProbe)
    yield
    Observatory.clear()
    Observatory._lens_registry = []
    Observatory._lenses_initialized = False


# ---------------------------------------------------------------------------
# Session/SessionResult dataclass shape
# ---------------------------------------------------------------------------


def test_session_archive_field_defaults_to_default():
    s = Session(id="sid", name="n", start_ts=0.0)
    assert s.archive == "default"


def test_session_result_has_no_flat_dicts():
    sr = SessionResult()
    assert not hasattr(sr, "start_data"), "legacy flat start_data should be removed"
    assert not hasattr(sr, "end_data"), "legacy flat end_data should be removed"
    assert sr.sessions == []


def test_frontend_dashboard_signature_is_session_only():
    import inspect

    params = list(inspect.signature(Frontend().dashboard).parameters.keys())
    assert params == ["session", "session_records", "analysis"]


# ---------------------------------------------------------------------------
# Per-session lens hook payloads land on the right Session
# ---------------------------------------------------------------------------


def test_session_payloads_land_per_session():
    with Observatory.enter_context("alpha"):
        Observatory.collect("rec", object())
    with Observatory.enter_context("beta"):
        Observatory.collect("rec_b", object())

    sessions = Observatory._session_result.sessions
    assert [s.name for s in sessions] == ["alpha", "beta"]
    for s in sessions:
        assert s.archive == "default"
        assert s.start_data == {"probe": {"hello": "start"}}
        assert s.end_data == {"probe": {"hello": "end"}}


# ---------------------------------------------------------------------------
# Payload contract: archives + sessions + dashboard[lens][session_id]
# ---------------------------------------------------------------------------


def _build_payload() -> Dict[str, Any]:
    return Observatory._generate_report_payload(
        list(Observatory._records.values()),
        Observatory._session_result,
        {},
        Observatory._lens_registry,
    )


def test_payload_top_level_keys_archives_and_sessions():
    with Observatory.enter_context("only"):
        Observatory.collect("r", object())

    payload = _build_payload()

    assert "archives" in payload
    assert "sessions" in payload
    assert "session" not in payload, "legacy 'session' block must be gone"
    assert payload["archives"] == [{"label": "default", "session_ids": ["only"]}]
    assert len(payload["sessions"]) == 1
    s = payload["sessions"][0]
    assert s["id"] == "only"
    assert s["archive"] == "default"
    assert s["start_data"] == {"probe": {"hello": "start"}}
    assert s["end_data"] == {"probe": {"hello": "end"}}


def test_dashboard_is_nested_lens_then_session_id():
    with Observatory.enter_context("s1"):
        Observatory.collect("r1", object())
    with Observatory.enter_context("s2"):
        Observatory.collect("r2", object())

    payload = _build_payload()

    assert set(payload["dashboard"].keys()) == {"probe"}
    assert set(payload["dashboard"]["probe"].keys()) == {"s1", "s2"}
    s1_view = payload["dashboard"]["probe"]["s1"]
    assert s1_view["blocks"][0]["record"]["data"] == {"sid": "s1"}


def test_dashboard_invoked_once_per_session_lens_pair_with_filtered_records():
    with Observatory.enter_context("s1"):
        Observatory.collect("r_in_s1", object())
    with Observatory.enter_context("s2"):
        Observatory.collect("r_in_s2_a", object())
        Observatory.collect("r_in_s2_b", object())

    _build_payload()  # triggers dashboard calls

    calls = _DashboardProbe.calls
    assert len(calls) == 2
    by_session = {c["session_id"]: c for c in calls}
    assert by_session["s1"]["records"] == ["r_in_s1"]
    assert by_session["s2"]["records"] == ["r_in_s2_a", "r_in_s2_b"]
    # session_records is correctly filtered to that session only.


# ---------------------------------------------------------------------------
# JSON round-trip + forward-compat shim for legacy archives
# ---------------------------------------------------------------------------


def test_export_json_writes_new_shape(tmp_path):
    with Observatory.enter_context("only"):
        Observatory.collect("r", object())

    path = tmp_path / "out.json"
    Observatory.export_json(str(path))
    data = json.loads(path.read_text())

    assert set(data.keys()) == {"records", "sessions"}
    assert "session" not in data
    assert data["sessions"][0]["archive"] == "default"


def test_load_archive_sessions_reads_new_shape():
    payload = {
        "sessions": [
            {
                "id": "s1",
                "name": "s1",
                "archive": "Foo",
                "start_ts": 1.0,
                "end_ts": 2.0,
                "start_data": {"x": 1},
                "end_data": {},
            }
        ]
    }
    sr = Observatory._load_archive_sessions(payload)
    assert len(sr.sessions) == 1
    assert sr.sessions[0].archive == "Foo"
    assert sr.sessions[0].start_data == {"x": 1}


def test_load_archive_sessions_reads_legacy_nested_shape():
    """Old archives nested sessions under a top-level ``session`` key and
    omitted ``archive``. The shim must lift them and synthesize the field."""
    legacy = {
        "session": {
            "sessions": [
                {
                    "id": "old",
                    "name": "old",
                    "start_ts": 0.0,
                    "end_ts": 1.0,
                    "start_data": {},
                    "end_data": {},
                }
            ],
            # The legacy flat dicts still exist on disk in old archives.
            "start_data": {"foo": 1},
            "end_data": {},
        }
    }
    sr = Observatory._load_archive_sessions(legacy, default_archive="legacy_archive")
    assert len(sr.sessions) == 1
    assert sr.sessions[0].id == "old"
    assert sr.sessions[0].archive == "legacy_archive"


def test_load_archive_sessions_handles_empty_payload():
    sr = Observatory._load_archive_sessions({})
    assert sr.sessions == []


# ---------------------------------------------------------------------------
# compare_archives produces N archives, no region_stack prefix
# ---------------------------------------------------------------------------


def _write_archive(tmp_path, name: str, archive_label_in_archive: str = "default") -> str:
    """Run a self-contained collection and return the archive path."""
    Observatory.clear()
    Observatory._lens_registry = []
    Observatory._lenses_initialized = True
    Observatory.register_lens(_DashboardProbe)
    with Observatory.enter_context(name):
        Observatory.collect("Annotated Model", object())
    p = tmp_path / f"{name}.json"
    Observatory.export_json(str(p))
    return str(p)


def test_compare_archives_produces_n_archives_with_correct_session_grouping(tmp_path):
    a = _write_archive(tmp_path, "archA")
    b = _write_archive(tmp_path, "archB")

    out_html = tmp_path / "compare.html"
    Observatory.compare_archives(
        archive_paths=[a, b],
        labels=["XNN", "QNN"],
        html_path=str(out_html),
    )

    sessions = Observatory._session_result.sessions
    assert len(sessions) == 2
    by_archive = {s.archive: s for s in sessions}
    assert set(by_archive.keys()) == {"XNN", "QNN"}
    # Session ids carry the archive prefix.
    assert by_archive["XNN"].id == "XNN/archA"
    assert by_archive["QNN"].id == "QNN/archB"


def test_compare_archives_does_not_prepend_archive_to_region_stack(tmp_path):
    a = _write_archive(tmp_path, "alpha")
    b = _write_archive(tmp_path, "beta")

    out_html = tmp_path / "compare.html"
    Observatory.compare_archives(
        archive_paths=[a, b],
        labels=["A", "B"],
        html_path=str(out_html),
    )

    # Archive grouping is supplied by Session.archive, not by mutating the
    # region_stack — region trees stay unmodified.
    for rec in Observatory._records.values():
        assert rec.region_stack and rec.region_stack[0] != "A"
        assert rec.region_stack and rec.region_stack[0] != "B"


def test_compare_archives_rewrites_graph_ref_and_per_layer_graph_ref(tmp_path):
    """Both the graph and per_layer_accuracy digests carry a graph_ref that
    must match the (now-prefixed) record name in compare mode."""
    # Manually build two archives whose records carry graph + per_layer_accuracy
    # digests so we can verify both rewrites in one test.
    def _archive_with_graph_refs(label_seed: str) -> Dict[str, Any]:
        return {
            "records": [
                {
                    "name": "Annotated Model",
                    "timestamp": 0.0,
                    "session_id": label_seed,
                    "region_stack": [label_seed],
                    "data": {
                        "graph": {"graph_ref": "Annotated Model", "base": {}},
                        "per_layer_accuracy": {
                            "graph_ref": "Annotated Model",
                            "match_count": 0,
                        },
                    },
                }
            ],
            "sessions": [
                {
                    "id": label_seed,
                    "name": label_seed,
                    "archive": "default",
                    "start_ts": 0.0,
                    "end_ts": 1.0,
                    "start_data": {},
                    "end_data": {},
                }
            ],
        }

    a = tmp_path / "a.json"
    b = tmp_path / "b.json"
    a.write_text(json.dumps(_archive_with_graph_refs("seed_a")))
    b.write_text(json.dumps(_archive_with_graph_refs("seed_b")))

    Observatory.compare_archives(
        archive_paths=[str(a), str(b)],
        labels=["A", "B"],
        html_path=str(tmp_path / "out.html"),
    )

    recs = list(Observatory._records.values())
    names = {r.name for r in recs}
    assert "A/Annotated Model" in names
    assert "B/Annotated Model" in names

    for rec in recs:
        assert rec.data["graph"]["graph_ref"] == rec.name
        assert rec.data["per_layer_accuracy"]["graph_ref"] == rec.name


# ---------------------------------------------------------------------------
# HTML payload encoding survives the new dashboard nested shape
# ---------------------------------------------------------------------------


def test_payload_serializes_to_json_cleanly():
    """Regression: the new dashboard[lens][session_id] structure must JSON
    encode without errors (covers the _encode_html_blocks traversal update)."""
    with Observatory.enter_context("ses"):
        Observatory.collect("r", object())
    payload = _build_payload()
    # Round-trip through the actual encoder used by export_html_report.
    s = json.dumps(payload, cls=_NonFiniteFloatAsStringJSONEncoder)
    assert "session_ids" in s
    assert "ses" in s
