# Copyright (c) Qualcomm Innovation Center, Inc.
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Tests for the Report (JSON) feature: Frontend.json_report hook, the
framework assembler (_generate_json_report_payload / export_report_json),
and the two demo lens implementations (accuracy, per_layer_accuracy).

Report (JSON) contract (see RFC_SESSIONS.md):
  {
    "title": ...,
    "generated_at": ...,
    "archives": [{label, session_ids}, ...],
    "sessions": [{id, name, archive, start_ts, end_ts}, ...],
    "lenses": {
      <lens_name>: {
        <archive_label>: {
          <session_id>: <lens_dict>
        }
      }
    }
  }
"""

from __future__ import annotations

import json
import math
from typing import Any, Dict, List, Optional

import pytest

from executorch.devtools.observatory import Observatory
from executorch.devtools.observatory.interfaces import (
    AnalysisResult,
    Frontend,
    Lens,
    ObservationContext,
    RecordDigest,
    Session,
    SessionResult,
    ViewList,
)
from executorch.devtools.observatory.lenses.accuracy import AccuracyLens
from executorch.devtools.observatory.lenses.per_layer_accuracy import PerLayerAccuracyLens


def _session(
    sid: str = "s", archive: str = "default", start_ts: float = 0.0
) -> Session:
    return Session(id=sid, name=sid, archive=archive, start_ts=start_ts, end_ts=1.0)


def _rec(
    name: str,
    session_id: str = "s",
    data: Optional[Dict[str, Any]] = None,
) -> RecordDigest:
    return RecordDigest(name=name, timestamp=0.0, session_id=session_id, data=data or {})


@pytest.fixture(autouse=True)
def _reset_observatory():
    Observatory.clear()
    Observatory._lens_registry = []
    Observatory._lenses_initialized = True
    yield
    Observatory.clear()
    Observatory._lens_registry = []
    Observatory._lenses_initialized = False


# ---------------------------------------------------------------------------
# Base-class contract
# ---------------------------------------------------------------------------


def test_frontend_json_report_default_returns_none():
    assert Frontend().json_report(_session(), [], AnalysisResult()) is None


# ---------------------------------------------------------------------------
# Framework: payload shape
# ---------------------------------------------------------------------------


class _JsonProbe(Lens):
    calls: List[Dict[str, Any]] = []

    @classmethod
    def get_name(cls) -> str:
        return "probe"

    @classmethod
    def setup(cls) -> None:
        cls.calls = []

    @classmethod
    def clear(cls) -> None:
        cls.calls = []

    class _F(Frontend):
        def json_report(self, session, session_records, analysis):
            _JsonProbe.calls.append(
                {"session_id": session.id, "archive": session.archive}
            )
            return {"ok": True, "n": len(session_records)}

    @staticmethod
    def get_frontend_spec() -> Frontend:
        return _JsonProbe._F()


def test_json_report_invoked_once_per_session_lens():
    Observatory.register_lens(_JsonProbe)
    with Observatory.enter_context("s1", archive="arch_a"):
        Observatory.collect("r1", object())
    with Observatory.enter_context("s2", archive="arch_a"):
        Observatory.collect("r2", object())

    Observatory._generate_json_report_payload(
        list(Observatory._records.values()),
        Observatory._session_result,
        {},
        Observatory._lens_registry,
    )

    assert len(_JsonProbe.calls) == 2
    session_ids = {c["session_id"] for c in _JsonProbe.calls}
    assert session_ids == {"s1", "s2"}


def test_json_report_top_level_shape():
    Observatory.register_lens(_JsonProbe)
    with Observatory.enter_context("ses", archive="myarchive"):
        Observatory.collect("r", object())

    payload = Observatory._generate_json_report_payload(
        list(Observatory._records.values()),
        Observatory._session_result,
        {},
        Observatory._lens_registry,
    )

    assert set(payload.keys()) == {"title", "generated_at", "archives", "sessions", "lenses"}
    assert payload["archives"] == [{"label": "myarchive", "session_ids": ["ses"]}]
    assert len(payload["sessions"]) == 1
    s = payload["sessions"][0]
    assert s["id"] == "ses"
    assert s["archive"] == "myarchive"
    # Stripped: no start_data / end_data.
    assert "start_data" not in s
    assert "end_data" not in s
    assert payload["lenses"] == {"probe": {"myarchive": {"ses": {"ok": True, "n": 1}}}}


def test_lens_returning_none_omitted_no_ghost_keys():
    """Lenses that return None must not appear in lenses block at all."""

    class _Silent(Lens):
        @classmethod
        def get_name(cls) -> str:
            return "silent"

        class _F(Frontend):
            pass  # json_report not overridden => returns None

        @staticmethod
        def get_frontend_spec() -> Frontend:
            return _Silent._F()

    Observatory.register_lens(_Silent)
    with Observatory.enter_context("ses"):
        Observatory.collect("r", object())

    payload = Observatory._generate_json_report_payload(
        list(Observatory._records.values()),
        Observatory._session_result,
        {},
        Observatory._lens_registry,
    )

    assert "silent" not in payload["lenses"]


def test_compare_mode_groups_by_archive_in_lens_block(tmp_path):
    """In compare mode the lenses block must be keyed
    lenses[lens_name][archive_label][session_id]."""

    Observatory.register_lens(_JsonProbe)

    def _write_archive(name: str) -> str:
        Observatory.clear()
        Observatory._lens_registry = []
        Observatory._lenses_initialized = True
        Observatory.register_lens(_JsonProbe)
        with Observatory.enter_context(name):
            Observatory.collect("r", object())
        p = tmp_path / f"{name}.json"
        Observatory.export_json(str(p))
        return str(p)

    a = _write_archive("archA")
    b = _write_archive("archB")

    Observatory.compare_archives(
        archive_paths=[a, b],
        labels=["XNN", "QNN"],
        html_path=str(tmp_path / "out.html"),
    )

    payload = Observatory._generate_json_report_payload(
        list(Observatory._records.values()),
        Observatory._session_result,
        {},
        Observatory._lens_registry,
    )

    assert "XNN" in payload["lenses"]["probe"]
    assert "QNN" in payload["lenses"]["probe"]
    # session ids are prefixed with the archive label in compare mode.
    assert any(sid.startswith("XNN/") for sid in payload["lenses"]["probe"]["XNN"])


def test_export_report_json_writes_indented_file(tmp_path):
    Observatory.register_lens(_JsonProbe)
    with Observatory.enter_context("ses"):
        Observatory.collect("r", object())

    path = tmp_path / "report.json"
    Observatory.export_report_json(str(path), title="test")
    raw = path.read_text()
    data = json.loads(raw)
    assert data["title"] == "test"
    assert "probe" in data["lenses"]
    # Indented format: the file should contain newlines.
    assert "\n" in raw


def test_export_report_json_survives_nan_values(tmp_path):
    class _NaNLens(Lens):
        @classmethod
        def get_name(cls) -> str:
            return "nan_lens"

        class _F(Frontend):
            def json_report(self, session, session_records, analysis):
                return {"metric": float("nan")}

        @staticmethod
        def get_frontend_spec() -> Frontend:
            return _NaNLens._F()

    Observatory.register_lens(_NaNLens)
    with Observatory.enter_context("s"):
        Observatory.collect("r", object())

    path = tmp_path / "nan.json"
    Observatory.export_report_json(str(path))
    data = json.loads(path.read_text())
    # _NonFiniteFloatAsStringJSONEncoder encodes nan as a string.
    assert data["lenses"]["nan_lens"]["default"]["s"]["metric"] == "nan"


# ---------------------------------------------------------------------------
# AccuracyLens.json_report
# ---------------------------------------------------------------------------


class TestAccuracyJsonReport:
    def setup_method(self):
        self.f = AccuracyLens.get_frontend_spec()
        self.s = _session()

    def test_returns_none_when_no_accuracy_records(self):
        assert self.f.json_report(self.s, [_rec("r")], AnalysisResult()) is None

    def test_aggregates_mean_min_max_and_worst(self):
        records = [
            _rec("r1", data={"accuracy": {"psnr": 30.0, "mse": 0.10}}),
            _rec("r2", data={"accuracy": {"psnr": 50.0, "mse": 0.02}}),
        ]
        result = self.f.json_report(self.s, records, AnalysisResult())

        assert result is not None
        assert result["records_measured"] == 2
        psnr = result["metrics"]["psnr"]
        assert psnr["mean"] == pytest.approx(40.0)
        assert psnr["min"] == pytest.approx(30.0)
        assert psnr["max"] == pytest.approx(50.0)
        assert psnr["worst_record"] == "r1"  # lower psnr = worse
        mse = result["metrics"]["mse"]
        assert mse["mean"] == pytest.approx(0.06)
        assert mse["worst_record"] == "r1"   # mse=0.10 > 0.02: higher mse = worse quality

    def test_excludes_internal_and_suffix_keys(self):
        records = [
            _rec(
                "r1",
                data={
                    "accuracy": {
                        "psnr": 40.0,
                        "psnr_min": 35.0,
                        "psnr_max": 45.0,
                        "psnr_worst_idx": 7,
                        "_num_samples": 10,
                    }
                },
            )
        ]
        result = self.f.json_report(self.s, records, AnalysisResult())
        assert "psnr" in result["metrics"]
        for forbidden in ("psnr_min", "psnr_max", "psnr_worst_idx", "_num_samples"):
            assert forbidden not in result["metrics"], f"{forbidden} should not appear"

    def test_records_measured_counts_records_with_accuracy_only(self):
        records = [
            _rec("r1", data={"accuracy": {"psnr": 40.0}}),
            _rec("r2", data={}),  # no accuracy
        ]
        result = self.f.json_report(self.s, records, AnalysisResult())
        assert result["records_measured"] == 1


# ---------------------------------------------------------------------------
# PerLayerAccuracyLens.json_report
# ---------------------------------------------------------------------------


def _pla_digest(anchor: str = "Exported Float", rows: Optional[List] = None) -> dict:
    if rows is None:
        rows = [
            {
                "from_node_root": "layer_0", "target_node": "t0",
                "psnr": 30.0, "cosine_sim": 0.90, "mse": 0.05, "abs_err": 0.1,
            },
            {
                "from_node_root": "layer_1", "target_node": "t1",
                "psnr": 10.0, "cosine_sim": 0.60, "mse": 0.50, "abs_err": 0.9,
            },
            {
                "from_node_root": "layer_2", "target_node": "t2",
                "psnr": 45.0, "cosine_sim": 0.99, "mse": 0.01, "abs_err": 0.01,
            },
        ]
    return {
        "anchor_record": anchor,
        "sample_source": "accuracy.worst[mse]",
        "match_count": len(rows),
        "rows": rows,
    }


class TestPerLayerAccuracyJsonReport:
    def setup_method(self):
        self.f = PerLayerAccuracyLens.get_frontend_spec()
        self.s = _session()

    def _analysis_with_ranges(self) -> AnalysisResult:
        ar = AnalysisResult()
        ar.global_data["metric_ranges"] = {"psnr": [10.0, 45.0]}
        return ar

    def test_returns_none_when_no_per_layer_records(self):
        assert self.f.json_report(self.s, [_rec("r")], AnalysisResult()) is None

    def test_returns_none_when_rows_empty(self):
        records = [_rec("r", data={"per_layer_accuracy": _pla_digest(rows=[])})]
        assert self.f.json_report(self.s, records, AnalysisResult()) is None

    def test_top_level_fields_present(self):
        records = [_rec("target", data={"per_layer_accuracy": _pla_digest()})]
        result = self.f.json_report(self.s, records, self._analysis_with_ranges())
        assert result is not None
        assert result["anchor"] == "Exported Float"
        assert result["target"] == "target"
        assert result["n_layers"] == 3
        assert result["sample_source"] == "accuracy.worst[mse]"
        assert result["metric_ranges"] == {"psnr": [10.0, 45.0]}

    def test_worst_layers_psnr_sorted_ascending(self):
        """Lower psnr = worse quality; first entry in worst_layers should be
        the minimum psnr row."""
        records = [_rec("t", data={"per_layer_accuracy": _pla_digest()})]
        result = self.f.json_report(self.s, records, AnalysisResult())
        psnr_list = result["worst_layers"]["psnr"]
        assert psnr_list[0]["layer"] == "layer_1"  # psnr=10 is the worst
        # Check ascending sort for psnr
        psnr_vals = [e["psnr"] for e in psnr_list]
        assert psnr_vals == sorted(psnr_vals)

    def test_worst_layers_mse_sorted_descending(self):
        """Higher mse = worse quality; first entry should be max mse."""
        records = [_rec("t", data={"per_layer_accuracy": _pla_digest()})]
        result = self.f.json_report(self.s, records, AnalysisResult())
        mse_list = result["worst_layers"]["mse"]
        assert mse_list[0]["layer"] == "layer_1"  # mse=0.50 is the worst
        mse_vals = [e["mse"] for e in mse_list]
        assert mse_vals == sorted(mse_vals, reverse=True)

    def test_top_n_config_knob(self):
        """json_report_top_n=1 should return only 1 row per metric.
        The config value is threaded through analyze() → analysis.global_data
        so it survives outside a live enable_context block."""
        rows = [
            {
                "from_node_root": f"l{i}", "target_node": f"t{i}",
                "psnr": float(i), "cosine_sim": float(i) / 100,
                "mse": float(100 - i), "abs_err": float(100 - i) / 100,
            }
            for i in range(20)
        ]
        records = [_rec("t", data={"per_layer_accuracy": _pla_digest(rows=rows)})]
        # Run analyze with the custom config so global_data carries top_n=1.
        analysis = PerLayerAccuracyLens.analyze(
            records, {"per_layer_accuracy": {"json_report_top_n": 1}}
        )
        result = self.f.json_report(self.s, records, analysis)

        for metric in ("psnr", "cosine_sim", "mse", "abs_err"):
            assert len(result["worst_layers"][metric]) == 1, \
                f"Expected 1 entry for {metric}, got {len(result['worst_layers'][metric])}"
