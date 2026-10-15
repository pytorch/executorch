# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""A calibrated roofline approximation for dense, non-speculative inference."""

import math
import statistics
from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class AttentionGroup:
    layers: int
    query_heads: int
    kv_heads: int
    head_dim: int
    kv_bytes: int
    window: int = 0


@dataclass(frozen=True)
class ModelCost:
    weight_bytes: int
    linear_parameters: int
    output_parameters: int
    attention: tuple[AttentionGroup, ...]

    @classmethod
    def from_dict(cls, data):
        model = cls(
            **(
                data
                | {"attention": tuple(AttentionGroup(**g) for g in data["attention"])}
            )
        )
        values = [model.weight_bytes, model.linear_parameters, model.output_parameters]
        for group in model.attention:
            values.extend(
                [
                    group.layers,
                    group.query_heads,
                    group.kv_heads,
                    group.head_dim,
                    group.kv_bytes,
                ]
            )
            if type(group.window) is not int or group.window < 0:
                raise ValueError("Attention windows must be nonnegative integers")
        if not model.attention or any(type(v) is not int or v <= 0 for v in values):
            raise ValueError("Model costs must be positive integers")
        return model

    def attention_cost(self, position, tokens):
        flops = traffic = 0
        for group in self.attention:
            end = position + tokens
            if group.window:
                below = max(0, min(tokens, group.window - position))
                pairs = (
                    below * position
                    + below * (below + 1) / 2
                    + (tokens - below) * group.window
                )
                # A prefill chunk can touch the window at its start and new KV.
                attended = min(position, group.window - 1) + tokens
            else:
                pairs = tokens * position + tokens * (tokens + 1) / 2
                attended = end
            flops += 4 * group.layers * group.query_heads * group.head_dim * pairs
            traffic += (
                2
                * group.layers
                * group.kv_heads
                * group.head_dim
                * group.kv_bytes
                * attended
            )
        return flops, traffic


@dataclass(frozen=True)
class InferenceCost:
    model: ModelCost
    effective_bandwidth_bytes_s: float
    effective_compute_flops_s: float
    # Empirical timing corrections, not physical byte counts.
    pair_weight_factor: float
    mixed_factor: float
    chunk_size: int
    relative_error: float
    prefill_overhead_s: float = 0.0

    @classmethod
    def from_dict(cls, data):
        model = cls(**(data | {"model": ModelCost.from_dict(data["model"])}))
        values = (
            model.effective_bandwidth_bytes_s,
            model.effective_compute_flops_s,
            model.pair_weight_factor,
            model.mixed_factor,
        )
        if any(not math.isfinite(v) or v <= 0 for v in values):
            raise ValueError("Invalid inference calibration")
        if type(model.chunk_size) is not int or model.chunk_size < 1:
            raise ValueError("Invalid calibration chunk size")
        if not math.isfinite(model.relative_error) or not 0 <= model.relative_error < 1:
            raise ValueError("Invalid calibration uncertainty")
        if not math.isfinite(model.prefill_overhead_s) or model.prefill_overhead_s < 0:
            raise ValueError("Invalid prefill overhead")
        return model

    def step(self, inputs):
        token_batch_size = sum(i["tokens"] for i in inputs)
        generation_batch_size = sum(i["decode"] for i in inputs)
        prefill = any(not i["decode"] for i in inputs)
        if generation_batch_size > 2:
            raise ValueError("Calibration supports generation batch sizes up to two")
        attention = [
            self.model.attention_cost(i["position"], i["tokens"]) for i in inputs
        ]
        linear_flops = 2 * (
            self.model.linear_parameters * token_batch_size
            + self.model.output_parameters * len(inputs)
        )
        weight_time_factor = (
            self.pair_weight_factor if generation_batch_size == 2 and not prefill else 1
        )
        parameter_load_s = (
            weight_time_factor
            * self.model.weight_bytes
            / self.effective_bandwidth_bytes_s
        )
        linear_compute_s = linear_flops / self.effective_compute_flops_s
        linear_s = max(parameter_load_s, linear_compute_s)
        attention_s = max(
            sum(f for f, _ in attention) / self.effective_compute_flops_s,
            sum(b for _, b in attention) / self.effective_bandwidth_bytes_s,
        )
        return (
            linear_s + attention_s + (self.prefill_overhead_s if prefill else 0)
        ) * (self.mixed_factor if generation_batch_size and prefill else 1)

    def prefill(self, position, tokens):
        seconds = 0.0
        while tokens:
            chunk = min(tokens, self.chunk_size)
            seconds += self.step(
                [{"position": position, "tokens": chunk, "decode": False}]
            )
            tokens -= chunk
            position += chunk
        return seconds

    def to_dict(self):
        return asdict(self)


def fit_cost(model, samples):
    samples = [s for s in samples if s["succeeded"] and s["elapsed_ms"] > 0]
    solo = [s for s in samples if len(s["inputs"]) == 1 and s["inputs"][0]["decode"]]
    pairs = [
        s
        for s in samples
        if len(s["inputs"]) == 2 and all(i["decode"] for i in s["inputs"])
    ]
    prefill = [s for s in samples if all(not i["decode"] for i in s["inputs"])]
    mixed = [
        s
        for s in samples
        if any(i["decode"] for i in s["inputs"])
        and any(not i["decode"] for i in s["inputs"])
    ]
    if len(solo) < 8 or len(pairs) < 8 or len(prefill) < 2 or not mixed:
        raise ValueError(
            "Calibration needs solo, overlapping decode, prefill and mixed steps"
        )
    effective_bandwidth_bytes_s = statistics.median(
        (model.weight_bytes + model.attention_cost(s["inputs"][0]["position"], 1)[1])
        / (s["elapsed_ms"] / 1000)
        for s in solo
    )
    points = [
        (
            (
                2
                * (
                    model.linear_parameters * sum(i["tokens"] for i in s["inputs"])
                    + model.output_parameters * len(s["inputs"])
                )
                + sum(
                    model.attention_cost(i["position"], i["tokens"])[0]
                    for i in s["inputs"]
                )
            )
            / 1e12,
            s["elapsed_ms"] / 1000,
        )
        for s in prefill
    ]
    large = [
        (work, seconds)
        for work, seconds in points
        if work >= max(work for work, _ in points) / 2
    ]
    effective_compute_flops_s = statistics.median(
        work * 1e12 / seconds for work, seconds in large
    )
    weight_seconds = model.weight_bytes / effective_bandwidth_bytes_s
    small = [
        seconds - weight_seconds
        for work, seconds in points
        if work * 1e12 / effective_compute_flops_s <= weight_seconds
    ]
    overhead = max(0.0, statistics.median(small)) if small else 0.0
    effective_compute_flops_s = statistics.median(
        work * 1e12 / (seconds - overhead)
        for work, seconds in large
        if seconds > overhead
    )
    factors = []
    for sample in pairs:
        attention = [model.attention_cost(i["position"], 1) for i in sample["inputs"]]
        attention_s = max(
            sum(f for f, _ in attention) / effective_compute_flops_s,
            sum(b for _, b in attention) / effective_bandwidth_bytes_s,
        )
        factors.append(
            max(
                0.01,
                (sample["elapsed_ms"] / 1000 - attention_s)
                * effective_bandwidth_bytes_s
                / model.weight_bytes,
            )
        )
    model_cost = InferenceCost(
        model,
        effective_bandwidth_bytes_s,
        effective_compute_flops_s,
        statistics.median(factors),
        1.0,
        max(i["tokens"] for s in samples for i in s["inputs"] if not i["decode"]),
        0.0,
        overhead,
    )
    mixed_factor = statistics.median(
        s["elapsed_ms"] / 1000 / model_cost.step(s["inputs"]) for s in mixed
    )
    return InferenceCost(
        model,
        effective_bandwidth_bytes_s,
        effective_compute_flops_s,
        model_cost.pair_weight_factor,
        mixed_factor,
        model_cost.chunk_size,
        0.0,
        overhead,
    )


def validation_error(cost, samples):
    groups = {}
    for sample in samples:
        if not sample["succeeded"] or sample["elapsed_ms"] <= 0:
            continue
        kind = (
            sum(i["decode"] for i in sample["inputs"]),
            any(not i["decode"] for i in sample["inputs"]),
        )
        groups.setdefault(kind, []).append(
            abs(cost.step(sample["inputs"]) / (sample["elapsed_ms"] / 1000) - 1)
        )
    if sum(map(len, groups.values())) < 16 or not all(
        kind in groups for kind in ((0, True), (1, False), (2, False), (1, True))
    ):
        raise ValueError("Too few validation steps")
    return max(
        sorted(errors)[math.ceil(0.9 * len(errors)) - 1] for errors in groups.values()
    )
