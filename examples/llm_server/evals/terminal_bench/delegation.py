# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Check explicit limits before launching one background child."""


def decide_delegation(
    snapshot,
    *,
    child_available,
    expected_output_tokens,
    output_limit,
):
    decision = {
        "allowed": False,
        "reason": "no_child_budget",
        "expected_output_tokens": expected_output_tokens,
        "output_limit": output_limit,
    }
    if not child_available:
        return decision
    if type(expected_output_tokens) is not int or expected_output_tokens <= 0:
        return decision | {"reason": "invalid_output_forecast"}
    if expected_output_tokens > output_limit:
        return decision | {"reason": "forecast_exceeds_budget"}
    counts = [
        snapshot.get(k)
        for k in (
            "max_inflight_requests",
            "submitted_generations",
            "waiting_for_runtime_lease",
        )
    ]
    if any(type(v) is not int or v < 0 for v in counts):
        return decision | {"reason": "unknown_request_limit"}
    request_limit, submitted, waiting = counts
    if waiting or request_limit - submitted < 2:
        return decision | {"reason": "inflight_request_limit"}
    return decision | {"allowed": True, "reason": "within_limits"}
