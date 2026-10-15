# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import replace

import pytest

from .delegation import decide_delegation, DelegationEstimate

SNAPSHOT = {
    "max_inflight_requests": 4,
    "submitted_generations": 0,
    "waiting_for_runtime_lease": 0,
}
ESTIMATE = DelegationEstimate(30, 50, 1, 32, 20, 2)


def decide(estimate=ESTIMATE, snapshot=SNAPSHOT, **kwargs):
    return decide_delegation(
        snapshot,
        estimate,
        **(
            {
                "child_available": True,
                "expected_output_tokens": 200,
                "expected_thinking_tokens": 100,
                "output_limit": 1536,
            }
            | kwargs
        ),
    )


def test_saving_and_parent_slowdown_are_reported_separately():
    result = decide()
    assert result["allowed"]
    assert result["expected_parallel_s"] == 33
    assert result["parent_delay_s"] == 2
    assert result["expected_total_tokens"] == 300


@pytest.mark.parametrize(
    "estimate,reason",
    [
        (None, "unknown_cost"),
        (replace(ESTIMATE, child_s=55), "no_estimated_benefit"),
        (replace(ESTIMATE, integration_s=25), "no_estimated_benefit"),
        (replace(ESTIMATE, parent_work_s=0), "no_independent_work"),
        (replace(ESTIMATE, child_s=float("nan")), "invalid_estimate"),
        (replace(ESTIMATE, prefill_s=-1), "invalid_estimate"),
        (replace(ESTIMATE, relative_error=1), "invalid_estimate"),
        (replace(ESTIMATE, relative_error=0.3), "no_estimated_benefit"),
    ],
)
def test_unknown_invalid_or_uncertain_savings_stay_inline(estimate, reason):
    result = decide(estimate)
    assert not result["allowed"]
    assert result["reason"] == reason


@pytest.mark.parametrize(
    "kwargs,reason",
    [
        ({"child_available": False}, "no_child_budget"),
        ({"expected_output_tokens": 0}, "invalid_output_forecast"),
        ({"expected_output_tokens": True}, "invalid_output_forecast"),
        ({"expected_output_tokens": 2000}, "forecast_exceeds_budget"),
        ({"expected_thinking_tokens": None}, "unknown_cost"),
        ({"expected_thinking_tokens": -1}, "invalid_thinking_forecast"),
        ({"expected_thinking_tokens": 1400}, "forecast_exceeds_budget"),
    ],
)
def test_forecasts_and_total_budget(kwargs, reason):
    result = decide(**kwargs)
    assert not result["allowed"]
    assert result["reason"] == reason


@pytest.mark.parametrize(
    "changes,reason",
    [
        ({"submitted_generations": 3}, "request_capacity"),
        ({"waiting_for_runtime_lease": 1}, "request_capacity"),
        ({"submitted_generations": None}, "unknown_capacity"),
    ],
)
def test_request_capacity(changes, reason):
    assert decide(snapshot=SNAPSHOT | changes)["reason"] == reason
