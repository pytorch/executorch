# Copyright (c) Qualcomm Innovation Center, Inc.
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Unit tests for the new per-session dashboards on accuracy and
pipeline_graph_collector lenses."""

from __future__ import annotations

from typing import Any, Dict, List

import pytest

from executorch.devtools.observatory import Observatory
from executorch.devtools.observatory.interfaces import (
    AnalysisResult,
    RecordDigest,
    Session,
    SessionResult,
)
from executorch.devtools.observatory.lenses.accuracy import AccuracyLens
from executorch.devtools.observatory.lenses.pipeline_graph_collector import (
    PipelineGraphCollectorLens,
)


def _make_session(sid: str = "s1", archive: str = "default") -> Session:
    return Session(id=sid, name=sid, start_ts=0.0, end_ts=1.0, archive=archive)


def _rec(name: str, session_id: str, region_stack: List[str], data: Dict[str, Any]) -> RecordDigest:
    return RecordDigest(
        name=name,
        timestamp=0.0,
        session_id=session_id,
        region_stack=region_stack,
        data=data,
    )


# ---------------------------------------------------------------------------
# AccuracyLens dashboard
# ---------------------------------------------------------------------------


class TestAccuracyDashboard:
    def setup_method(self):
        self.frontend = AccuracyLens.get_frontend_spec()
        self.session = _make_session("ses_a")

    def test_returns_none_when_no_records_have_accuracy(self):
        records = [_rec("r1", "ses_a", ["edge"], {})]
        assert self.frontend.dashboard(self.session, records, AnalysisResult()) is None

    def test_aggregates_means_across_records(self):
        records = [
            _rec("r1", "ses_a", ["edge"], {"accuracy": {"mse": 0.10, "psnr": 30.0}}),
            _rec("r2", "ses_a", ["edge"], {"accuracy": {"mse": 0.30, "psnr": 50.0}}),
        ]
        view = self.frontend.dashboard(self.session, records, AnalysisResult())

        assert view is not None
        assert len(view.blocks) == 1
        block = view.blocks[0]
        assert block.id == "accuracy_session_summary"
        data = block.record.data
        assert data["records_measured"] == 2
        assert data["mse_mean"] == pytest.approx(0.20)
        assert data["psnr_mean"] == pytest.approx(40.0)

    def test_excludes_internal_and_suffix_keys_from_aggregate(self):
        records = [
            _rec(
                "r1",
                "ses_a",
                [],
                {
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
        view = self.frontend.dashboard(self.session, records, AnalysisResult())
        data = view.blocks[0].record.data
        assert "psnr_mean" in data
        assert "psnr_min_mean" not in data
        assert "psnr_max_mean" not in data
        assert "psnr_worst_idx_mean" not in data
        assert "_num_samples_mean" not in data

    def test_handles_records_with_only_error_message(self):
        records = [
            _rec("r1", "ses_a", [], {"accuracy": {"error_message": "boom"}}),
            _rec("r2", "ses_a", [], {"accuracy": {"mse": 0.1}}),
        ]
        view = self.frontend.dashboard(self.session, records, AnalysisResult())
        data = view.blocks[0].record.data
        assert data["records_measured"] == 2
        assert data["mse_mean"] == pytest.approx(0.1)
        # error_message is a string, not aggregated.
        assert "error_message_mean" not in data

    def test_record_view_unchanged_for_simple_digest(self):
        # Sanity check that the record view (not the dashboard) still works
        # after the dashboard addition; original record() path should return
        # at least one block for a non-empty digest.
        view = self.frontend.record(
            digest={"psnr": 40.0, "_num_samples": 1},
            analysis={"global": {}, "record": {}},
            context={"index": 0, "name": "r1"},
        )
        assert view is not None
        assert len(view.blocks) >= 1


# ---------------------------------------------------------------------------
# PipelineGraphCollectorLens dashboard
# ---------------------------------------------------------------------------


class TestPipelineGraphCollectorDashboard:
    def setup_method(self):
        self.frontend = PipelineGraphCollectorLens.get_frontend_spec()
        self.session = _make_session("ses_a")

    def test_returns_none_when_no_records(self):
        assert self.frontend.dashboard(self.session, [], AnalysisResult()) is None

    def test_counts_records_by_innermost_region(self):
        records = [
            _rec("r1", "ses_a", ["quantization"], {}),
            _rec("r2", "ses_a", ["quantization"], {}),
            _rec("r3", "ses_a", ["edge"], {}),
            _rec("r4", "ses_a", ["edge", "etrecord"], {}),
            _rec("r5", "ses_a", ["edge", "etrecord"], {}),
        ]
        view = self.frontend.dashboard(self.session, records, AnalysisResult())
        assert view is not None
        data = view.blocks[0].record.data
        assert data["total_records"] == 5
        assert data["quantization"] == 2
        assert data["edge"] == 1
        # etrecord is the *innermost* region for r4, r5 — they count under
        # etrecord, not under edge.
        assert data["etrecord"] == 2

    def test_unscoped_records_are_grouped_under_unscoped_label(self):
        records = [_rec("r1", "ses_a", [], {})]
        view = self.frontend.dashboard(self.session, records, AnalysisResult())
        data = view.blocks[0].record.data
        assert data["total_records"] == 1
        assert data["(unscoped)"] == 1

    def test_block_ordering_and_id(self):
        records = [_rec("r1", "ses_a", ["edge"], {})]
        view = self.frontend.dashboard(self.session, records, AnalysisResult())
        block = view.blocks[0]
        assert block.id == "pipeline_graph_summary"
        assert block.title == "Pipeline Graphs"
