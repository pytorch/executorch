# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Delegation admission from explicit workload estimates; unknown costs stay inline."""

import math
from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class DelegationEstimate:
    # Totals cover the whole child, including subsequent requests and tool work.
    uncached_input_tokens: int
    output_tokens: int
    prefill_tokens_s: float | None
    decode_tokens_s: float
    child_tool_s: float
    parent_work_s: float
    inline_s: float
    integration_s: float


def decide_delegation(snapshot, estimate, *, child_available, output_limit):
    decision = {"allowed": False, "reason": "no_child_budget"}
    if not child_available:
        return decision

    counts = [
        snapshot.get(key)
        for key in (
            "max_inflight_requests",
            "submitted_generations",
            "waiting_for_runtime_lease",
        )
    ]
    if any(type(value) is not int or value < 0 for value in counts):
        return {**decision, "reason": "unknown_capacity"}
    capacity, submitted, waiting = counts
    # Reserve request slots only, not decode or memory capacity.
    if waiting or capacity - submitted < 2:
        return {**decision, "reason": "request_capacity"}
    if estimate is None:
        return {**decision, "reason": "unknown_cost"}

    if (
        type(estimate.uncached_input_tokens) is not int
        or estimate.uncached_input_tokens < 0
        or type(estimate.output_tokens) is not int
        or not 0 < estimate.output_tokens <= output_limit
    ):
        return {**decision, "reason": "invalid_estimate"}

    nonnegative = [
        estimate.child_tool_s,
        estimate.parent_work_s,
        estimate.inline_s,
        estimate.integration_s,
    ]
    positive = [estimate.decode_tokens_s]
    if estimate.uncached_input_tokens:
        positive.append(estimate.prefill_tokens_s)
    if any(value is None for value in nonnegative + positive):
        return {**decision, "reason": "unknown_cost"}
    if any(
        type(value) not in (int, float) or not math.isfinite(value) or value < 0
        for value in nonnegative + positive
    ) or any(value <= 0 for value in positive):
        return {**decision, "reason": "invalid_estimate"}
    if estimate.parent_work_s == 0:
        return {**decision, "reason": "no_independent_work"}

    decision["estimate"] = asdict(estimate)
    prefill_s = (
        estimate.uncached_input_tokens / estimate.prefill_tokens_s
        if estimate.uncached_input_tokens
        else 0
    )
    child_s = (
        prefill_s
        + estimate.output_tokens / estimate.decode_tokens_s
        + estimate.child_tool_s
    )
    parallel_s = max(estimate.parent_work_s, child_s) + estimate.integration_s
    allowed = math.isfinite(parallel_s) and parallel_s < estimate.inline_s
    return {
        **decision,
        "allowed": allowed,
        "reason": "estimated_benefit" if allowed else "no_estimated_benefit",
        "child_s_at_cap": child_s,
        "parallel_s_at_cap": parallel_s,
        "inline_s": estimate.inline_s,
    }
