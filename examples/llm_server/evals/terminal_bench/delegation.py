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
    prefill_tokens_s: float | None
    decode_tokens_s: float
    child_tool_s: float
    parent_work_s: float
    inline_s: float
    integration_s: float
    solo_decode_tokens_s: float | None = None
    parent_decode_tokens: int = 0
    decode_rate_source: str = "provided"


class DecodeRates:
    """Recent native decode windows; no batching gain assumed before observation."""

    def __init__(self):
        self.previous = None
        self.solo = None
        self.shared_fraction = 0.5
        self.source = "no_batching_gain"

    def observe(self, engine):
        previous, self.previous = self.previous, engine
        if not engine or not previous:
            return
        if engine["decode_tokens"] < previous["decode_tokens"]:
            self.solo = None
            self.shared_fraction = 0.5
            self.source = "no_batching_gain"
            return
        tokens = engine["decode_tokens"] - previous["decode_tokens"]
        elapsed = engine["decode_only_ms"] - previous["decode_only_ms"]
        mixed = engine["mixed_ms"] - previous["mixed_ms"]
        if tokens <= 0 or elapsed <= 0 or mixed != 0 or not engine["mean_decode_width"]:
            return
        steps = round(engine["decode_tokens"] / engine["mean_decode_width"])
        previous_steps = (
            round(previous["decode_tokens"] / previous["mean_decode_width"])
            if previous["mean_decode_width"]
            else 0
        )
        if steps <= previous_steps:
            return
        width = tokens / (steps - previous_steps)
        aggregate = tokens * 1000 / elapsed
        if width <= 1:
            self.solo = aggregate
        elif width <= 2 and self.solo is not None:
            self.shared_fraction = min(1.0, aggregate / (2 * self.solo))
            self.source = "observed_aggregate"


def _decode_finish_times(estimate, child_tokens, prefill_s):
    solo = estimate.solo_decode_tokens_s or estimate.decode_tokens_s
    parent_tokens = estimate.parent_decode_tokens
    parent_tools = estimate.parent_work_s - parent_tokens / solo
    child_ready = prefill_s + estimate.child_tool_s
    if not parent_tokens:
        return parent_tools, child_ready + child_tokens / solo
    # Charge child prefill as a shared stall; tools can run during it.
    ready = [max(parent_tools, prefill_s), child_ready]
    tokens = [parent_tokens, child_tokens]
    first = 0 if ready[0] <= ready[1] else 1
    second = 1 - first
    remaining = tokens[first] - (ready[second] - ready[first]) * solo
    finish = [0.0, 0.0]
    if remaining <= 0:
        finish[first] = ready[first] + tokens[first] / solo
        finish[second] = ready[second] + tokens[second] / solo
    else:
        overlap = min(remaining, tokens[second])
        shared_end = ready[second] + overlap / estimate.decode_tokens_s
        finish[first] = shared_end + (remaining - overlap) / solo
        finish[second] = shared_end + (tokens[second] - overlap) / solo
    return finish


def _cost_error(estimate):
    if estimate is None:
        return "unknown_cost"
    if (
        type(estimate.uncached_input_tokens) is not int
        or estimate.uncached_input_tokens < 0
    ):
        return "invalid_estimate"
    nonnegative = [
        estimate.child_tool_s,
        estimate.parent_work_s,
        estimate.inline_s,
        estimate.integration_s,
    ]
    positive = [estimate.decode_tokens_s]
    if estimate.solo_decode_tokens_s is not None:
        positive.append(estimate.solo_decode_tokens_s)
    if estimate.uncached_input_tokens:
        positive.append(estimate.prefill_tokens_s)
    if any(value is None for value in nonnegative + positive):
        return "unknown_cost"
    if any(
        type(value) not in (int, float) or not math.isfinite(value) or value < 0
        for value in nonnegative + positive
    ) or any(value <= 0 for value in positive):
        return "invalid_estimate"
    solo = estimate.solo_decode_tokens_s or estimate.decode_tokens_s
    if (
        type(estimate.parent_decode_tokens) is not int
        or estimate.parent_decode_tokens < 0
        or estimate.parent_decode_tokens / solo > estimate.parent_work_s
    ):
        return "invalid_estimate"
    return None


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
        return {**decision, "reason": "invalid_output_forecast"}
    if expected_output_tokens > output_limit:
        return {**decision, "reason": "forecast_exceeds_budget"}
    if expected_thinking_tokens is None:
        return {**decision, "reason": "unknown_cost"}
    if type(expected_thinking_tokens) is not int or expected_thinking_tokens < 0:
        return {**decision, "reason": "invalid_thinking_forecast"}
    total_tokens = expected_output_tokens + expected_thinking_tokens
    decision["expected_total_tokens"] = total_tokens
    if total_tokens > output_limit:
        return {**decision, "reason": "forecast_exceeds_budget"}

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
    error = _cost_error(estimate)
    if error:
        return {**decision, "reason": error}
    if estimate.parent_work_s == 0:
        return {**decision, "reason": "no_independent_work"}

    decision["estimate"] = asdict(estimate)
    prefill_s = (
        estimate.uncached_input_tokens / estimate.prefill_tokens_s
        if estimate.uncached_input_tokens
        else 0
    )
    parent_s, child_s = _decode_finish_times(estimate, total_tokens, prefill_s)
    parallel_s = max(parent_s, child_s) + estimate.integration_s
    allowed = math.isfinite(parallel_s) and parallel_s < estimate.inline_s
    return {
        **decision,
        "allowed": allowed,
        "reason": "estimated_benefit" if allowed else "no_estimated_benefit",
        "expected_child_s": child_s,
        "expected_parent_s": parent_s,
        "parent_delay_s": max(0, parent_s - estimate.parent_work_s),
        "expected_parallel_s": parallel_s,
        "inline_s": estimate.inline_s,
    }
