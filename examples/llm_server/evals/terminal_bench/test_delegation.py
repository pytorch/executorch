# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import replace

import pytest

from .delegation import decide_delegation, DecodeRates, DelegationEstimate

SNAPSHOT = {
    "max_inflight_requests": 4,
    "submitted_generations": 0,
    "waiting_for_runtime_lease": 0,
}
ESTIMATE = DelegationEstimate(
    uncached_input_tokens=200,
    prefill_tokens_s=100,
    decode_tokens_s=20,
    child_tool_s=0,
    parent_work_s=30,
    inline_s=50,
    integration_s=1,
)


@pytest.mark.parametrize(
    "uncached,prefill,decode,parallel,allowed",
    [
        (200, 100, 20, 31, True),
        (20000, 100, 20, 211, False),
        (2000, 1000, 40, 31, True),
        (2000, 5, 2, 501, False),
    ],
)
def test_counterfactual_cost_cases(uncached, prefill, decode, parallel, allowed):
    estimate = replace(
        ESTIMATE,
        uncached_input_tokens=uncached,
        prefill_tokens_s=prefill,
        decode_tokens_s=decode,
    )
    decision = decide_delegation(
        SNAPSHOT,
        estimate,
        child_available=True,
        expected_output_tokens=200,
        expected_thinking_tokens=0,
        output_limit=1536,
    )
    assert decision["allowed"] is allowed
    assert decision["expected_parallel_s"] == parallel


@pytest.mark.parametrize(
    "estimate,reason",
    [
        (None, "unknown_cost"),
        (replace(ESTIMATE, prefill_tokens_s=None), "unknown_cost"),
        (replace(ESTIMATE, child_tool_s=None), "unknown_cost"),
        (replace(ESTIMATE, decode_tokens_s=0), "invalid_estimate"),
        (replace(ESTIMATE, decode_tokens_s=float("nan")), "invalid_estimate"),
        (replace(ESTIMATE, inline_s=float("inf")), "invalid_estimate"),
        (replace(ESTIMATE, uncached_input_tokens=-1), "invalid_estimate"),
        (replace(ESTIMATE, parent_work_s=0), "no_independent_work"),
        (replace(ESTIMATE, inline_s=31), "no_estimated_benefit"),
        (replace(ESTIMATE, integration_s=21), "no_estimated_benefit"),
        (replace(ESTIMATE, child_tool_s=40), "no_estimated_benefit"),
    ],
)
def test_uncertain_or_unhelpful_children_stay_inline(estimate, reason):
    decision = decide_delegation(
        SNAPSHOT,
        estimate,
        child_available=True,
        expected_output_tokens=200,
        expected_thinking_tokens=0,
        output_limit=1536,
    )
    assert not decision["allowed"]
    assert decision["reason"] == reason


def test_fully_cached_work_does_not_require_a_prefill_rate():
    estimate = replace(ESTIMATE, uncached_input_tokens=0, prefill_tokens_s=None)
    decision = decide_delegation(
        SNAPSHOT,
        estimate,
        child_available=True,
        expected_output_tokens=200,
        expected_thinking_tokens=0,
        output_limit=1536,
    )
    assert decision["allowed"]
    assert decision["expected_child_s"] == 10


@pytest.mark.parametrize(
    "updates,allowed,reason",
    [
        ({"submitted_generations": 2}, True, "estimated_benefit"),
        ({"submitted_generations": 3}, False, "request_capacity"),
        ({"max_inflight_requests": 1}, False, "request_capacity"),
        ({"waiting_for_runtime_lease": 1}, False, "request_capacity"),
        ({"submitted_generations": None}, False, "unknown_capacity"),
    ],
)
def test_reserve_request_capacity_for_parent(updates, allowed, reason):
    decision = decide_delegation(
        SNAPSHOT | updates,
        ESTIMATE,
        child_available=True,
        expected_output_tokens=200,
        expected_thinking_tokens=0,
        output_limit=1536,
    )
    assert decision["allowed"] is allowed
    assert decision["reason"] == reason


@pytest.mark.parametrize("forecast,allowed", [(200, True), (1000, False)])
def test_forecast_prices_work_independently_of_enforced_cap(forecast, allowed):
    decision = decide_delegation(
        SNAPSHOT,
        ESTIMATE,
        child_available=True,
        expected_output_tokens=forecast,
        expected_thinking_tokens=0,
        output_limit=1536,
    )
    assert decision["allowed"] is allowed
    assert decision["expected_child_s"] == 2 + forecast / 20
    assert decision["expected_output_tokens"] == forecast
    assert decision["output_limit"] == 1536


@pytest.mark.parametrize(
    "forecast,reason",
    [
        (None, "invalid_output_forecast"),
        (True, "invalid_output_forecast"),
        ("200", "invalid_output_forecast"),
        (1.5, "invalid_output_forecast"),
        (0, "invalid_output_forecast"),
        (-1, "invalid_output_forecast"),
        (1537, "forecast_exceeds_budget"),
    ],
)
def test_invalid_or_infeasible_forecast_stays_inline(forecast, reason):
    decision = decide_delegation(
        SNAPSHOT,
        ESTIMATE,
        child_available=True,
        expected_output_tokens=forecast,
        expected_thinking_tokens=0,
        output_limit=1536,
    )
    assert not decision["allowed"]
    assert decision["reason"] == reason


@pytest.mark.parametrize(
    "thinking,allowed,total,reason",
    [
        (None, False, None, "unknown_cost"),
        (0, True, 200, "estimated_benefit"),
        (800, False, 1000, "no_estimated_benefit"),
        (1336, False, 1536, "no_estimated_benefit"),
        (1337, False, 1537, "forecast_exceeds_budget"),
        (-1, False, None, "invalid_thinking_forecast"),
        (True, False, None, "invalid_thinking_forecast"),
        (1.5, False, None, "invalid_thinking_forecast"),
    ],
)
def test_thinking_contributes_to_decode_cost_and_total_budget(
    thinking, allowed, total, reason
):
    decision = decide_delegation(
        SNAPSHOT,
        ESTIMATE,
        child_available=True,
        expected_output_tokens=200,
        expected_thinking_tokens=thinking,
        output_limit=1536,
    )
    assert decision["allowed"] is allowed
    assert decision["reason"] == reason
    assert decision["expected_total_tokens"] == total
    if "expected_child_s" in decision:
        assert decision["expected_child_s"] == 2 + total / 20


@pytest.mark.parametrize(
    "shared,allowed,finish",
    [(22.5, True, 600 / 22.5), (15, False, 40), (10, False, 60)],
)
def test_batching_prices_both_parent_slowdown_and_aggregate_gain(
    shared, allowed, finish
):
    estimate = replace(
        ESTIMATE,
        uncached_input_tokens=0,
        solo_decode_tokens_s=30,
        decode_tokens_s=shared,
        parent_decode_tokens=600,
        parent_work_s=20,
        inline_s=40,
        integration_s=0,
    )
    decision = decide_delegation(
        SNAPSHOT,
        estimate,
        child_available=True,
        expected_output_tokens=600,
        expected_thinking_tokens=0,
        output_limit=1536,
    )
    assert decision["allowed"] is allowed
    assert decision["expected_child_s"] == pytest.approx(finish)
    assert decision["expected_parent_s"] == pytest.approx(finish)
    assert decision["parent_delay_s"] == pytest.approx(finish - 20)


def test_surviving_sequence_returns_to_solo_rate():
    estimate = replace(
        ESTIMATE,
        uncached_input_tokens=0,
        solo_decode_tokens_s=30,
        decode_tokens_s=22.5,
        parent_decode_tokens=600,
        parent_work_s=20,
        inline_s=30,
        integration_s=0,
    )
    decision = decide_delegation(
        SNAPSHOT,
        estimate,
        child_available=True,
        expected_output_tokens=300,
        expected_thinking_tokens=0,
        output_limit=1536,
    )
    assert decision["expected_child_s"] == pytest.approx(300 / 22.5)
    assert decision["expected_parent_s"] == pytest.approx(300 / 22.5 + 300 / 30)
    assert decision["allowed"]


def test_child_prefill_delays_parent_decoding():
    estimate = replace(
        ESTIMATE,
        uncached_input_tokens=1000,
        prefill_tokens_s=100,
        solo_decode_tokens_s=30,
        decode_tokens_s=22.5,
        parent_decode_tokens=600,
        parent_work_s=20,
        inline_s=35,
        integration_s=0,
    )
    decision = decide_delegation(
        SNAPSHOT,
        estimate,
        child_available=True,
        expected_output_tokens=600,
        expected_thinking_tokens=0,
        output_limit=1536,
    )
    assert decision["expected_parent_s"] == pytest.approx(10 + 600 / 22.5)
    assert not decision["allowed"]


def test_native_decode_rates_ignore_request_counts_and_mixed_prefill():
    rates = DecodeRates()
    idle = {
        "decode_tokens": 0,
        "decode_only_ms": 0,
        "mixed_ms": 0,
        "mean_decode_width": 0,
    }
    solo = {
        "decode_tokens": 300,
        "decode_only_ms": 10000,
        "mixed_ms": 0,
        "mean_decode_width": 1,
    }
    shared = {
        "decode_tokens": 900,
        "decode_only_ms": 10000 + 600 / 45 * 1000,
        "mixed_ms": 0,
        "mean_decode_width": 1.5,
    }
    rates.observe(idle)
    rates.observe(solo)
    assert rates.solo == 30
    assert rates.shared_fraction == 0.5
    assert rates.source == "no_batching_gain"
    rates.observe(shared)
    assert rates.shared_fraction == pytest.approx(0.75)
    assert rates.source == "observed_aggregate"
    rates.observe(
        {**shared, "decode_tokens": 1000, "decode_only_ms": 30000, "mixed_ms": 1000}
    )
    assert rates.shared_fraction == pytest.approx(0.75)
    rates.observe(idle)
    assert rates.solo is None
    assert rates.shared_fraction == 0.5
