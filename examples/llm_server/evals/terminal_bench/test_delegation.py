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
ESTIMATE = DelegationEstimate(
    uncached_input_tokens=200,
    output_tokens=200,
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
        SNAPSHOT, estimate, child_available=True, output_limit=200
    )
    assert decision["allowed"] is allowed
    assert decision["parallel_s_at_cap"] == parallel


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
        (replace(ESTIMATE, output_tokens=True), "invalid_estimate"),
        (replace(ESTIMATE, output_tokens=201), "invalid_estimate"),
        (replace(ESTIMATE, parent_work_s=0), "no_independent_work"),
        (replace(ESTIMATE, inline_s=31), "no_estimated_benefit"),
        (replace(ESTIMATE, integration_s=21), "no_estimated_benefit"),
        (replace(ESTIMATE, child_tool_s=40), "no_estimated_benefit"),
    ],
)
def test_uncertain_or_unhelpful_children_stay_inline(estimate, reason):
    decision = decide_delegation(
        SNAPSHOT, estimate, child_available=True, output_limit=200
    )
    assert not decision["allowed"]
    assert decision["reason"] == reason


def test_fully_cached_work_does_not_require_a_prefill_rate():
    estimate = replace(ESTIMATE, uncached_input_tokens=0, prefill_tokens_s=None)
    decision = decide_delegation(
        SNAPSHOT, estimate, child_available=True, output_limit=200
    )
    assert decision["allowed"]
    assert decision["child_s_at_cap"] == 10


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
        SNAPSHOT | updates, ESTIMATE, child_available=True, output_limit=200
    )
    assert decision["allowed"] is allowed
    assert decision["reason"] == reason
