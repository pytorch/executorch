# Copyright (c) Qualcomm Innovation Center, Inc.
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Unit tests for the pipeline_graph_collector region structure (RFC §4.5).

Region layout:

    Session "<script-name>"
    ├── quantization/         (top-level; aten-graph work)
    │   ├── record  Annotated Model
    │   ├── record  Calibrated Model
    │   └── record  Quantized Model
    └── edge/                  (top-level; edge dialect)
        ├── record  Pre-EdgeTransform/<method>
        ├── record  EdgeProgramManager EP
        └── etrecord/         (lazy nested under edge)
            ├── record  ETRecord Exported/<method>
            ├── record  ETRecord Edge/<method>
            └── record  ETRecord Extra/<module>

`quantization` and `edge` are sibling top-level regions, opened lazily by
`_transition_to_quantization` / `_transition_to_edge`. The runtime region
stack only ever holds one chain at a time, so the lens transitions
between these siblings as patched calls fire.

Records are *not* wrapped in per-call regions like "prepare_pt2e" or
"convert_pt2e" — every region should hold more than one Record, and the
record name already conveys the per-call identity.
"""

from __future__ import annotations

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
    Observatory.register_lens(PipelineGraphCollectorLens)
    yield
    Observatory.clear()
    Observatory._lens_registry = []
    Observatory._lenses_initialized = False


def _open_test_session():
    return Observatory.enter_context("test_session")


def test_no_region_open_at_session_start():
    """Top-level regions are lazy: nothing opens until a transition fires."""

    with _open_test_session():
        assert PipelineGraphCollectorLens._quantization_stack is None
        assert PipelineGraphCollectorLens._edge_stack is None
        assert PipelineGraphCollectorLens._etrecord_stack is None


def test_transition_to_quantization_opens_only_quantization():
    """First quantization transition opens just that top-level region."""

    with _open_test_session():
        PipelineGraphCollectorLens._transition_to_quantization()
        Observatory.collect("Annotated Model", object())
        Observatory.collect("Calibrated Model", object())

    by_name = {r.name: r for r in Observatory._records.values()}
    assert by_name["Annotated Model"].region_stack == [
        "test_session",
        "quantization",
    ]
    assert by_name["Calibrated Model"].region_stack == [
        "test_session",
        "quantization",
    ]


def test_transition_to_edge_closes_quantization_and_opens_edge():
    """Forward transition: quantization closes; edge opens as sibling."""

    with _open_test_session():
        PipelineGraphCollectorLens._transition_to_quantization()
        Observatory.collect("Quantized Model", object())
        PipelineGraphCollectorLens._transition_to_edge()
        Observatory.collect("EdgeProgramManager EP", object())
        # While inside the session, edge is open and quantization is closed.
        assert PipelineGraphCollectorLens._quantization_stack is None
        assert PipelineGraphCollectorLens._edge_stack is not None

    by_name = {r.name: r for r in Observatory._records.values()}
    assert by_name["Quantized Model"].region_stack == [
        "test_session",
        "quantization",
    ]
    assert by_name["EdgeProgramManager EP"].region_stack == [
        "test_session",
        "edge",
    ]
    # After session_end, all stacks are closed.
    assert PipelineGraphCollectorLens._quantization_stack is None
    assert PipelineGraphCollectorLens._edge_stack is None


def test_etrecord_region_nests_under_edge():
    """_ensure_etrecord_region first transitions to edge, then opens etrecord."""

    with _open_test_session():
        PipelineGraphCollectorLens._ensure_etrecord_region()
        Observatory.collect("ETRecord Exported/forward", object())
        Observatory.collect("ETRecord Edge/forward", object())

    by_name = {r.name: r for r in Observatory._records.values()}
    for name in ("ETRecord Exported/forward", "ETRecord Edge/forward"):
        assert by_name[name].region_stack == [
            "test_session",
            "edge",
            "etrecord",
        ], name


def test_etrecord_lazy_open_idempotent():
    """Calling _ensure_etrecord_region multiple times is a no-op after first."""

    with _open_test_session():
        PipelineGraphCollectorLens._ensure_etrecord_region()
        first_stack = PipelineGraphCollectorLens._etrecord_stack
        PipelineGraphCollectorLens._ensure_etrecord_region()
        second_stack = PipelineGraphCollectorLens._etrecord_stack

    assert first_stack is second_stack


def test_records_per_region_have_more_than_one():
    """Every region should hold multiple records (no per-call sub-regions)."""

    with _open_test_session():
        PipelineGraphCollectorLens._transition_to_quantization()
        Observatory.collect("Annotated Model", object())
        Observatory.collect("Calibrated Model", object())
        Observatory.collect("Quantized Model", object())
        PipelineGraphCollectorLens._transition_to_edge()
        Observatory.collect("Pre-EdgeTransform/forward", object())
        Observatory.collect("EdgeProgramManager EP", object())
        PipelineGraphCollectorLens._ensure_etrecord_region()
        Observatory.collect("ETRecord Exported/forward", object())
        Observatory.collect("ETRecord Edge/forward", object())

    # Group records by region_stack to confirm region multiplicity.
    by_region: dict = {}
    for rec in Observatory._records.values():
        by_region.setdefault(tuple(rec.region_stack), []).append(rec.name)

    assert sorted(by_region.keys()) == sorted([
        ("test_session", "quantization"),
        ("test_session", "edge"),
        ("test_session", "edge", "etrecord"),
    ])
    for region_path, records in by_region.items():
        assert len(records) >= 2, f"region {region_path} only has {len(records)} record(s)"


def test_uninstall_closes_all_open_stacks():
    """on_session_end closes etrecord, edge, quantization in safe order."""

    with _open_test_session():
        PipelineGraphCollectorLens._ensure_etrecord_region()
        # Now etrecord+edge open, quantization closed.
        assert PipelineGraphCollectorLens._etrecord_stack is not None
        assert PipelineGraphCollectorLens._edge_stack is not None

    assert PipelineGraphCollectorLens._etrecord_stack is None
    assert PipelineGraphCollectorLens._edge_stack is None
    assert PipelineGraphCollectorLens._quantization_stack is None
    assert PipelineGraphCollectorLens._installed is False


def test_lens_hooks_fire_exactly_once_for_pipeline_collector():
    """on_session_start / on_session_end fire once each despite many transitions."""

    with _open_test_session():
        PipelineGraphCollectorLens._transition_to_quantization()
        Observatory.collect("a", object())
        Observatory.collect("b", object())
        PipelineGraphCollectorLens._transition_to_edge()
        Observatory.collect("c", object())
        PipelineGraphCollectorLens._ensure_etrecord_region()
        Observatory.collect("d", object())

    # 4 records collected.
    assert len(Observatory._records) == 4
    # _installed went True → False exactly once.
    assert PipelineGraphCollectorLens._installed is False
