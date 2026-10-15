# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import replace

import pytest

from .inference_cost import (
    AttentionGroup,
    fit_cost,
    InferenceCost,
    ModelCost,
    validation_error,
)

MODEL = ModelCost(
    16_762_600_196,
    25_163_726_848,
    1_344_831_488,
    (AttentionGroup(13, 32, 2, 128, 2), AttentionGroup(39, 32, 2, 128, 2, 2048)),
)
COST = InferenceCost(MODEL, 438e9, 11.64e12, 1.0, 1.0, 256, 0.05)


def input_at(position, tokens=1, decode=True):
    return {"position": position, "tokens": tokens, "decode": decode}


@pytest.mark.parametrize("batch_size", [1, 2])
@pytest.mark.parametrize("flops_s", [1.5e11, 1e13])
def test_generation_matches_scaling_book_roofline(batch_size, flops_s):
    model = ModelCost(
        2_000_000_000, 900_000_000, 100_000_000, (AttentionGroup(4, 4, 4, 64, 2),)
    )
    cost = InferenceCost(model, 1e11, flops_s, 1.0, 1.0, 256, 0.0)
    positions = [1023, 8191][:batch_size]
    kv_bytes = 2 * 4 * 4 * 64 * 2 * sum(p + 1 for p in positions)
    # Attention is bandwidth-bound here; linear operations cross the roofline
    # at batch size 1.5 when FLOPs/s is 1.5e11.
    expected_s = kv_bytes / 1e11 + max(2 * batch_size * 1e9 / flops_s, 2e9 / 1e11)
    assert cost.step([input_at(p) for p in positions]) == pytest.approx(expected_s)


def test_weight_sharing_increases_aggregate_but_slows_each_sequence():
    solo = COST.step([input_at(8192)])
    pair = COST.step([input_at(8192), input_at(8192)])
    assert solo < pair < 1.1 * solo
    assert 2 / pair > 1.8 / solo
    assert replace(COST, pair_weight_factor=2).step(
        [input_at(8192), input_at(8192)]
    ) == pytest.approx(2 * solo)


def test_compute_saturation_limits_batching_gain():
    cost = replace(COST, effective_compute_flops_s=1e12)
    solo = cost.step([input_at(8192)])
    paired = cost.step([input_at(8192), input_at(8192)])
    assert paired == pytest.approx(2 * solo)


def test_cached_prefix_skips_linear_work_but_still_costs_attention():
    assert COST.prefill(7936, 256) < COST.prefill(0, 8192) / 20
    assert COST.prefill(32768, 256) > COST.prefill(0, 256)
    assert COST.step([input_at(32768)]) > COST.step([input_at(8192)])


def test_sliding_window_caps_attention_work():
    local = replace(MODEL, attention=(AttentionGroup(52, 32, 2, 128, 2, 2048),))
    assert local.attention_cost(8192, 256) == local.attention_cost(32768, 256)
    global_only = replace(MODEL, attention=(AttentionGroup(52, 32, 2, 128, 2),))
    assert (
        global_only.attention_cost(32768, 256)[0]
        > global_only.attention_cost(8192, 256)[0]
    )


def test_small_prefill_has_a_weight_read_floor():
    assert COST.prefill(0, 1) > COST.prefill(0, 256) / 256


def test_mixed_prefill_increases_decode_step_latency():
    decode = COST.step([input_at(8192)])
    mixed = COST.step([input_at(8192), input_at(0, 255, False)])
    assert mixed > decode


def measurements(cost, offset=0):
    shapes = (
        [input_at(1024 + offset)],
        [input_at(1024 + offset), input_at(2048 + offset)],
        [input_at(1024 + offset, 256, False)],
        [input_at(1024 + offset), input_at(2048 + offset, 255, False)],
    )
    return [
        {"succeeded": True, "inputs": shape, "elapsed_ms": 1000 * cost.step(shape)}
        for shape in shapes
        for _ in range(12)
    ]


def test_calibration_requires_actual_paired_measurements():
    samples = [s for s in measurements(COST) if len(s["inputs"]) == 1]
    with pytest.raises(ValueError, match="overlapping"):
        fit_cost(MODEL, samples)


def test_fit_predicts_unseen_context_and_roundtrips():
    observed = replace(COST, pair_weight_factor=1.4, mixed_factor=1.1)
    fitted = fit_cost(MODEL, measurements(observed))
    assert fitted.pair_weight_factor == pytest.approx(1.4, rel=0.02)
    assert validation_error(fitted, measurements(observed, 4096)) < 0.1
    assert InferenceCost.from_dict(fitted.to_dict()) == fitted


def test_tiny_prefills_expose_fixed_cost_instead_of_distorting_compute():
    observed = replace(COST, prefill_overhead_s=0.17)
    samples = measurements(observed)
    for size in (4, 8, 128, 256):
        shape = [input_at(2048, size, False)]
        samples.append(
            {
                "succeeded": True,
                "inputs": shape,
                "elapsed_ms": 1000 * observed.step(shape),
            }
        )
    fitted = fit_cost(MODEL, samples)
    assert fitted.prefill_overhead_s == pytest.approx(0.17, abs=0.002)
    assert fitted.effective_compute_flops_s == pytest.approx(
        COST.effective_compute_flops_s, rel=0.02
    )
    for size in (1, 16, 64, 192):
        shape = [input_at(8192, size, False)]
        assert fitted.step(shape) == pytest.approx(observed.step(shape), rel=0.02)


def test_validation_does_not_hide_bad_prefill_among_many_decodes():
    samples = measurements(COST)
    for sample in samples:
        if any(not i["decode"] for i in sample["inputs"]):
            sample["elapsed_ms"] *= 2
    samples += [measurements(COST)[0]] * 1000
    assert validation_error(COST, samples) == pytest.approx(0.5)
