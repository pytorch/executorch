# Copyright (c) Qualcomm Innovation Center, Inc.
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Unit tests for the two minor-issue fixes that landed alongside the
session-as-first-class-unit RFC:

1. ``PipelineGraphCollectorLens.close_aot_regions`` and the ``to_executorch``
   patch boundary -- AOT regions (quantization/edge/etrecord) must close
   before runtime work begins so device-side regions are siblings of
   ``edge``, not children.

2. ``compare_archives`` rewrites both ``graph.graph_ref`` AND
   ``per_layer_accuracy.graph_ref`` to the prefixed record name so the
   per-layer accuracy graph viewer finds its asset in compare reports.
"""

from __future__ import annotations

import contextlib
import json
from typing import List

import pytest

from executorch.devtools.observatory import Observatory
from executorch.devtools.observatory.lenses.pipeline_graph_collector import (
    PipelineGraphCollectorLens,
)


@pytest.fixture(autouse=True)
def _reset_observatory():
    Observatory.clear()
    Observatory._lens_registry = []
    Observatory._lenses_initialized = True
    yield
    Observatory.clear()
    Observatory._lens_registry = []
    Observatory._lenses_initialized = False
    # Make sure AOT-region state is fully reset between tests.
    PipelineGraphCollectorLens._etrecord_stack = None
    PipelineGraphCollectorLens._edge_stack = None
    PipelineGraphCollectorLens._quantization_stack = None
    PipelineGraphCollectorLens._enter_context_fn = None


# ---------------------------------------------------------------------------
# close_aot_regions: closes etrecord, edge, quantization (in that order)
# ---------------------------------------------------------------------------


class TestCloseAotRegions:
    def test_closes_all_three_open_stacks(self):
        # Wire up the lens with the live framework so opening/closing
        # actually mutates Observatory._region_stack.
        PipelineGraphCollectorLens._enter_context_fn = Observatory.enter_context

        with Observatory.enable_context(config={}):
            with Observatory.enter_context("session"):
                # Open all three AOT regions.
                PipelineGraphCollectorLens._transition_to_quantization()
                assert PipelineGraphCollectorLens._quantization_stack is not None
                # Transition to edge -> closes quantization, opens edge.
                PipelineGraphCollectorLens._transition_to_edge()
                # Open nested etrecord under edge.
                PipelineGraphCollectorLens._ensure_etrecord_region()
                assert PipelineGraphCollectorLens._edge_stack is not None
                assert PipelineGraphCollectorLens._etrecord_stack is not None

                # Capture the region-stack depth WHILE AOT regions are open.
                depth_with_aot = len(Observatory._region_stack)

                # Close all AOT regions and verify the framework's stack
                # popped back to just the session.
                PipelineGraphCollectorLens.close_aot_regions()

                assert PipelineGraphCollectorLens._etrecord_stack is None
                assert PipelineGraphCollectorLens._edge_stack is None
                assert PipelineGraphCollectorLens._quantization_stack is None
                assert len(Observatory._region_stack) == depth_with_aot - 2

    def test_idempotent_when_no_regions_open(self):
        # Should not raise even when nothing is open.
        PipelineGraphCollectorLens.close_aot_regions()
        PipelineGraphCollectorLens.close_aot_regions()  # second call also fine.

    def test_record_after_close_lands_at_session_root_not_under_edge(self):
        """A record collected after close_aot_regions should have an empty
        region_stack (or the session-only stack) -- proving that subsequent
        regions become session-root siblings."""
        PipelineGraphCollectorLens._enter_context_fn = Observatory.enter_context

        with Observatory.enable_context(config={}):
            with Observatory.enter_context("session"):
                PipelineGraphCollectorLens._transition_to_edge()
                # While edge is open, a record would have ["session", "edge"].
                Observatory.collect("under_edge", object())

                PipelineGraphCollectorLens.close_aot_regions()

                # Now we open a sibling region (mimicking AdbLens "device").
                with Observatory.enter_context("device"):
                    Observatory.collect("under_device", object())

        recs = {r.name: r for r in Observatory._records.values()}
        # The outer enable_context adds a synthetic "default" outermost
        # region; the explicit "session" sits inside it. AOT regions
        # appear after the explicit session, runtime regions appear
        # after AOT closes.
        assert recs["under_edge"].region_stack[-2:] == ["session", "edge"]
        # Critical: device sits at session root, NOT under edge.
        assert recs["under_device"].region_stack[-2:] == ["session", "device"]
        assert "edge" not in recs["under_device"].region_stack


# ---------------------------------------------------------------------------
# Per-layer accuracy graph_ref rewrite in compare mode
# ---------------------------------------------------------------------------


class TestPerLayerAccuracyGraphRefRewrite:
    def test_compare_archives_rewrites_per_layer_graph_ref(self, tmp_path):
        """Both graph and per_layer_accuracy digests carry graph_ref keyed
        by record name. compare_archives prefixes record names with the
        archive label and must rewrite BOTH digest fields so the per-layer
        graph viewer's asset lookup succeeds."""

        def _archive(seed: str):
            return {
                "records": [
                    {
                        "name": "Quantized Model",
                        "timestamp": 0.0,
                        "session_id": seed,
                        "region_stack": ["edge"],
                        "data": {
                            "graph": {
                                "graph_ref": "Quantized Model",
                                "base": {"nodes": [], "meta": {}},
                            },
                            "per_layer_accuracy": {
                                "graph_ref": "Quantized Model",
                                "match_count": 0,
                                "rows": [],
                            },
                        },
                    }
                ],
                "sessions": [
                    {
                        "id": seed,
                        "name": seed,
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
        a.write_text(json.dumps(_archive("seed_a")))
        b.write_text(json.dumps(_archive("seed_b")))

        Observatory.compare_archives(
            archive_paths=[str(a), str(b)],
            labels=["XNN", "QNN"],
            html_path=str(tmp_path / "compare.html"),
        )

        records = list(Observatory._records.values())
        assert len(records) == 2
        names = {r.name for r in records}
        assert names == {"XNN/Quantized Model", "QNN/Quantized Model"}

        for rec in records:
            assert rec.data["graph"]["graph_ref"] == rec.name, (
                "graph.graph_ref must match the prefixed record name"
            )
            assert rec.data["per_layer_accuracy"]["graph_ref"] == rec.name, (
                "per_layer_accuracy.graph_ref must match the prefixed record "
                "name -- otherwise the compare-mode per-layer viewer renders "
                "an empty canvas because graph_assets is keyed by prefixed name"
            )
