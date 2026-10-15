# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import pytest

from .delegation import decide_delegation

SNAPSHOT = {
    "max_inflight_requests": 4,
    "submitted_generations": 0,
    "waiting_for_runtime_lease": 0,
}


def decide(snapshot=SNAPSHOT, **kwargs):
    return decide_delegation(
        snapshot,
        **(
            {
                "child_available": True,
                "expected_output_tokens": 200,
                "output_limit": 1536,
            }
            | kwargs
        ),
    )


def test_available_limits_admit_without_inference_estimates():
    result = decide()
    assert result["allowed"]
    assert result["reason"] == "within_limits"


@pytest.mark.parametrize(
    "kwargs,reason",
    [
        ({"child_available": False}, "no_child_budget"),
        ({"expected_output_tokens": 0}, "invalid_output_forecast"),
        ({"expected_output_tokens": True}, "invalid_output_forecast"),
        ({"expected_output_tokens": 2000}, "forecast_exceeds_budget"),
    ],
)
def test_proposal_and_output_limit(kwargs, reason):
    result = decide(**kwargs)
    assert not result["allowed"]
    assert result["reason"] == reason


@pytest.mark.parametrize(
    "changes,reason",
    [
        ({"submitted_generations": 3}, "inflight_request_limit"),
        ({"waiting_for_runtime_lease": 1}, "inflight_request_limit"),
        ({"submitted_generations": None}, "unknown_request_limit"),
    ],
)
def test_inflight_request_limit(changes, reason):
    assert decide(snapshot=SNAPSHOT | changes)["reason"] == reason
