# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Admission compares calibrated critical-path costs for the same remaining work."""

import math
from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class DelegationEstimate:
    parent_work_s: float
    inline_s: float
    integration_s: float
    parent_s: float
    child_s: float
    prefill_s: float
    relative_error: float = 0.0


def decide_delegation(
    snapshot,
    estimate,
    *,
    child_available,
    expected_output_tokens,
    expected_thinking_tokens,
    output_limit,
):
    decision = {
        "allowed": False,
        "reason": "no_child_budget",
        "expected_output_tokens": expected_output_tokens,
        "expected_thinking_tokens": expected_thinking_tokens,
        "expected_total_tokens": None,
        "output_limit": output_limit,
    }
    if not child_available:
        return decision
    if type(expected_output_tokens) is not int or expected_output_tokens <= 0:
        return decision | {"reason": "invalid_output_forecast"}
    if expected_output_tokens > output_limit:
        return decision | {"reason": "forecast_exceeds_budget"}
    if expected_thinking_tokens is None:
        return decision | {"reason": "unknown_cost"}
    if type(expected_thinking_tokens) is not int or expected_thinking_tokens < 0:
        return decision | {"reason": "invalid_thinking_forecast"}
    total = expected_output_tokens + expected_thinking_tokens
    decision["expected_total_tokens"] = total
    if total > output_limit:
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
    if estimate is None:
        return decision | {"reason": "unknown_cost"}
    values = asdict(estimate)
    if (
        any(
            type(v) not in (int, float) or not math.isfinite(v) or v < 0
            for v in values.values()
        )
        or estimate.relative_error >= 1
    ):
        return decision | {"reason": "invalid_estimate"}
    if estimate.parent_work_s == 0:
        return decision | {"reason": "no_independent_work"}
    parallel = max(estimate.parent_s, estimate.child_s) + estimate.integration_s
    upper = parallel * (1 + estimate.relative_error)
    lower = estimate.inline_s * (1 - estimate.relative_error)
    allowed = upper < lower
    return decision | {
        "allowed": allowed,
        "reason": "estimated_benefit" if allowed else "no_estimated_benefit",
        "estimate": values,
        "expected_parallel_s": parallel,
        "expected_child_s": estimate.child_s,
        "expected_parent_s": estimate.parent_s,
        "parent_delay_s": max(0, estimate.parent_s - estimate.parent_work_s),
        "inline_s": estimate.inline_s,
        "parallel_upper_s": upper,
        "inline_lower_s": lower,
    }
