# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

"""CPU authority for the Gemma 4 MTP top-k route.

`topk_reference` transcribes `runtime/ops/topk/topk.wgsl` bit-for-bit: the same
32-entry heap, the same bit-pattern comparator, and the same emission order. It
is the oracle the WG64-staged shader must reproduce exactly, so it deliberately
does NOT adopt `torch.topk` tie behaviour.

`AUTHORITY_SHA256` pins the exported corpus: any change to the transcription or
to `topk_cases()` moves the digest and has to be re-committed on purpose.
"""

from __future__ import annotations

import hashlib
import json
import os
import struct
import unittest

from pathlib import Path
from typing import Mapping

INPUT_WIDTH = 2048
OUTPUT_WIDTH = 32

# Committed digest of the canonical authority payload; see the module docstring.
AUTHORITY_SHA256 = "c7289f314f7c364251ae228f0c28cc300677db333c649c5a99c39994a259d50d"

_U32 = 0xFFFFFFFF


def f32_bits(value: float) -> int:
    return struct.unpack("<I", struct.pack("<f", value))[0]


def bits_f32(bits: int) -> float:
    return struct.unpack("<f", struct.pack("<I", bits & _U32))[0]


def _is_nan_bits(bits: int) -> bool:
    return (bits & 0x7F800000) == 0x7F800000 and (bits & 0x007FFFFF) != 0


def float_less_than_bits(lhs: int, rhs: int) -> bool:
    lhs_nan = _is_nan_bits(lhs)
    rhs_nan = _is_nan_bits(rhs)
    if lhs_nan or rhs_nan:
        return (not lhs_nan) and rhs_nan

    lhs_magnitude = lhs & 0x7FFFFFFF
    rhs_magnitude = rhs & 0x7FFFFFFF
    if lhs_magnitude == 0 and rhs_magnitude == 0:
        return False

    lhs_negative = (lhs & 0x80000000) != 0
    rhs_negative = (rhs & 0x80000000) != 0
    if lhs_negative != rhs_negative:
        return lhs_negative
    if lhs_negative:
        return lhs > rhs
    return lhs < rhs


def _greater(lhs: int, rhs: int) -> bool:
    return float_less_than_bits(rhs, lhs)


def _push_heap(
    heap_values: list[int],
    heap_indices: list[int],
    initial_hole: int,
    top: int,
    value_bits: int,
    value_index: int,
) -> None:
    hole = initial_hole
    while hole > top:
        parent = (hole - 1) // 2
        if not _greater(heap_values[parent], value_bits):
            break
        heap_values[hole] = heap_values[parent]
        heap_indices[hole] = heap_indices[parent]
        hole = parent
    heap_values[hole] = value_bits
    heap_indices[hole] = value_index


def _adjust_heap(
    heap_values: list[int],
    heap_indices: list[int],
    initial_hole: int,
    length: int,
    value_bits: int,
    value_index: int,
) -> None:
    top = initial_hole
    hole = initial_hole
    second_child = initial_hole
    while second_child < (length - 1) // 2:
        second_child = 2 * (second_child + 1)
        if _greater(heap_values[second_child], heap_values[second_child - 1]):
            second_child -= 1
        heap_values[hole] = heap_values[second_child]
        heap_indices[hole] = heap_indices[second_child]
        hole = second_child
    if (length & 1) == 0 and second_child == (length - 2) // 2:
        second_child = 2 * (second_child + 1)
        heap_values[hole] = heap_values[second_child - 1]
        heap_indices[hole] = heap_indices[second_child - 1]
        hole = second_child - 1
    _push_heap(heap_values, heap_indices, hole, top, value_bits, value_index)


def topk_reference(scores_bits: list[int]) -> tuple[list[int], list[int]]:
    """Return (values_bits, indices) exactly as `topk.wgsl` emits them."""
    if len(scores_bits) != INPUT_WIDTH:
        raise ValueError(f"topk authority requires {INPUT_WIDTH} scores")

    heap_values = list(scores_bits[:OUTPUT_WIDTH])
    heap_indices = list(range(OUTPUT_WIDTH))

    for parent in range(OUTPUT_WIDTH // 2 - 1, -1, -1):
        _adjust_heap(
            heap_values,
            heap_indices,
            parent,
            OUTPUT_WIDTH,
            heap_values[parent],
            heap_indices[parent],
        )

    heap_root = heap_values[0]
    for index in range(OUTPUT_WIDTH, INPUT_WIDTH):
        value_bits = scores_bits[index]
        if _greater(value_bits, heap_root):
            _adjust_heap(heap_values, heap_indices, 0, OUTPUT_WIDTH, value_bits, index)
            heap_root = heap_values[0]

    last = OUTPUT_WIDTH
    while last > 1:
        last -= 1
        value_bits = heap_values[last]
        value_index = heap_indices[last]
        heap_values[last] = heap_values[0]
        heap_indices[last] = heap_indices[0]
        _adjust_heap(heap_values, heap_indices, 0, last, value_bits, value_index)

    return heap_values, heap_indices


def _lcg(seed: int, count: int) -> list[int]:
    state = seed & _U32
    drawn: list[int] = []
    for _ in range(count):
        state = (1664525 * state + 1013904223) & _U32
        drawn.append(state)
    return drawn


def _ramp_row() -> list[int]:
    return [f32_bits(-1024.0 + index * 1.0) for index in range(INPUT_WIDTH)]


def _random_row(seed: int) -> list[int]:
    return [
        f32_bits((draw / 2**32) * 200.0 - 100.0) for draw in _lcg(seed, INPUT_WIDTH)
    ]


def _interior_ties_row() -> list[int]:
    row = _random_row(7)
    for index in range(200, 264):
        row[index] = f32_bits(50.0)
    return row


def _boundary_ties_row() -> list[int]:
    row = [f32_bits(-1.0)] * INPUT_WIDTH
    for index in range(0, 40):
        row[index * 7] = f32_bits(3.5)
    return row


def _all_equal_row() -> list[int]:
    return [f32_bits(2.25)] * INPUT_WIDTH


def _nan_row() -> list[int]:
    row = _random_row(11)
    row[5] = 0x7FC00001
    row[600] = 0x7F800001
    row[1900] = 0xFFC00007
    return row


def _signed_zero_row() -> list[int]:
    row = [f32_bits(-3.0)] * INPUT_WIDTH
    for index in range(0, 64, 2):
        row[index] = 0x00000000
        row[index + 1] = 0x80000000
    return row


def _infinity_row() -> list[int]:
    row = _random_row(13)
    row[0] = 0xFF800000
    row[1] = 0x7F800000
    row[1000] = 0x7F800000
    row[2047] = 0xFF800000
    return row


def _descending_row() -> list[int]:
    return [f32_bits(float(INPUT_WIDTH - index)) for index in range(INPUT_WIDTH)]


# The only corpus rows whose emitted top-32 exercises the `lhs > rhs` branch.
def _all_negative_row() -> list[int]:
    return [f32_bits(-1.0 - (draw / 2**32) * 100.0) for draw in _lcg(23, INPUT_WIDTH)]


def _straddling_zero_row() -> list[int]:
    row = _all_negative_row()
    for slot, index in enumerate(range(31, INPUT_WIDTH, 173)):
        row[index] = f32_bits(0.25 + slot * 0.5)
    return row


# `export_topk_artifacts.py` writes these names sorted; that sorted list is the
# `kCases[]` contract in `test/native/test_topk.cpp`.
def topk_cases() -> dict[str, list[int]]:
    return {
        "ordinary": _ramp_row(),
        "random_seeded": _random_row(3),
        "interior_ties": _interior_ties_row(),
        "boundary_ties": _boundary_ties_row(),
        "all_equal": _all_equal_row(),
        "nan_payloads": _nan_row(),
        "signed_zeros": _signed_zero_row(),
        "infinities": _infinity_row(),
        "descending": _descending_row(),
        "all_negative": _all_negative_row(),
        "straddling_zero": _straddling_zero_row(),
    }


def authority_digest(body: Mapping[str, object]) -> str:
    """SHA-256 of the canonical authority payload, excluding the seal itself."""
    payload = {key: value for key, value in body.items() if key != "sha256"}
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def build_authority() -> dict[str, object]:
    cases: dict[str, object] = {}
    for name, scores in topk_cases().items():
        values, indices = topk_reference(scores)
        cases[name] = {
            "indices": indices,
            "scores_bits": scores,
            "values_bits": values,
        }
    body: dict[str, object] = {
        "cases": cases,
        "input_width": INPUT_WIDTH,
        "output_width": OUTPUT_WIDTH,
        "schema_version": 1,
    }
    body["sha256"] = authority_digest(body)
    return body


class TestEagleTopKCpu(unittest.TestCase):
    def test_eager_reference_is_repeatable(self) -> None:
        first = build_authority()
        second = build_authority()
        self.assertEqual(first, second)

        for name, case in first["cases"].items():
            self.assertEqual(len(case["values_bits"]), OUTPUT_WIDTH, name)
            self.assertEqual(len(case["indices"]), OUTPUT_WIDTH, name)
            self.assertEqual(len(set(case["indices"])), OUTPUT_WIDTH, name)
            for index in case["indices"]:
                self.assertTrue(0 <= index < INPUT_WIDTH, name)
            for slot, index in enumerate(case["indices"]):
                self.assertEqual(
                    case["values_bits"][slot], case["scores_bits"][index], name
                )

        receipt = os.environ.get("EAGLE_TOPK_EAGER_RECEIPT")
        if receipt:
            path = Path(receipt)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(
                json.dumps(first, indent=2, sort_keys=True) + "\n", encoding="utf-8"
            )

    def test_authority_matches_the_committed_digest(self) -> None:
        authority = build_authority()
        self.assertEqual(authority["sha256"], AUTHORITY_SHA256)
        self.assertEqual(authority_digest(authority), AUTHORITY_SHA256)

    def test_emission_order_is_descending(self) -> None:
        for name, scores in topk_cases().items():
            values, _ = topk_reference(scores)
            for slot in range(1, OUTPUT_WIDTH):
                self.assertFalse(
                    float_less_than_bits(values[slot - 1], values[slot]),
                    f"{name} slot {slot} is not in descending order",
                )

    def test_matches_sorted_selection_on_distinct_values(self) -> None:
        # With all-distinct finite values the heap order is the unique answer, so
        # a plain sort is an independent cross-check of the transcription.
        shipped = topk_cases()
        names = (
            "ordinary",
            "random_seeded",
            "descending",
            "all_negative",
            "straddling_zero",
        )
        for name in names:
            scores = shipped[name]
            self.assertEqual(len(set(scores)), INPUT_WIDTH, name)
            expected = sorted(
                range(INPUT_WIDTH), key=lambda i: bits_f32(scores[i]), reverse=True
            )[:OUTPUT_WIDTH]
            _, indices = topk_reference(scores)
            self.assertEqual(indices, expected, name)

    def test_negative_rows_keep_negatives_inside_the_selected_top_k(self) -> None:
        # Without this the exported corpus never reaches `topk.wgsl:62`.
        shipped = topk_cases()
        negatives = shipped["all_negative"]
        _, indices = topk_reference(negatives)
        self.assertTrue(all(bits_f32(negatives[i]) < 0.0 for i in indices))

        straddling = shipped["straddling_zero"]
        _, indices = topk_reference(straddling)
        selected = [bits_f32(straddling[i]) for i in indices]
        self.assertEqual(len([value for value in selected if value > 0.0]), 12)
        self.assertEqual(len([value for value in selected if value < 0.0]), 20)

    def test_comparator_ranks_nan_above_every_non_nan(self) -> None:
        for nan in (0x7FC00001, 0x7F800001, 0xFFC00007):
            for other in (f32_bits(3.5), f32_bits(-3.5), 0x7F800000, 0xFF800000):
                self.assertTrue(float_less_than_bits(other, nan))
                self.assertFalse(float_less_than_bits(nan, other))

    def test_comparator_treats_any_two_nans_as_equal(self) -> None:
        nans = (0x7FC00001, 0x7F800001, 0xFFC00007)
        for lhs in nans:
            for rhs in nans:
                self.assertFalse(float_less_than_bits(lhs, rhs))

    def test_comparator_treats_signed_zeros_as_equal(self) -> None:
        self.assertFalse(float_less_than_bits(0x80000000, 0x00000000))
        self.assertFalse(float_less_than_bits(0x00000000, 0x80000000))

    def test_comparator_ranks_every_negative_below_every_positive(self) -> None:
        for negative in (f32_bits(-1e-30), f32_bits(-1.0), 0xFF800000):
            for positive in (f32_bits(1e-30), f32_bits(1.0), 0x7F800000):
                self.assertTrue(float_less_than_bits(negative, positive))
                self.assertFalse(float_less_than_bits(positive, negative))

    def test_comparator_orders_two_negatives_by_magnitude(self) -> None:
        self.assertTrue(float_less_than_bits(f32_bits(-2.0), f32_bits(-1.0)))
        self.assertFalse(float_less_than_bits(f32_bits(-1.0), f32_bits(-2.0)))
        self.assertTrue(float_less_than_bits(0xFF800000, f32_bits(-3.4e38)))

    def test_comparator_orders_two_positives_by_magnitude(self) -> None:
        self.assertTrue(float_less_than_bits(f32_bits(1.0), f32_bits(2.0)))
        self.assertFalse(float_less_than_bits(f32_bits(2.0), f32_bits(1.0)))
        self.assertTrue(float_less_than_bits(f32_bits(3.4e38), 0x7F800000))

    def test_nan_sorts_above_every_finite_value(self) -> None:
        values, indices = topk_reference(_nan_row())
        self.assertEqual(indices[:3], [5, 600, 1900])
        for bits in values[:3]:
            self.assertTrue(_is_nan_bits(bits))
        self.assertFalse(_is_nan_bits(values[3]))

    def test_positive_infinity_beats_finite_and_negative_infinity_loses(self) -> None:
        _, indices = topk_reference(_infinity_row())
        self.assertIn(1, indices)
        self.assertIn(1000, indices)
        self.assertNotIn(0, indices)
        self.assertNotIn(2047, indices)

    def test_signed_zeros_compare_equal_and_beat_negatives(self) -> None:
        values, indices = topk_reference(_signed_zero_row())
        self.assertEqual(sorted(indices), list(range(OUTPUT_WIDTH)))
        for bits in values:
            self.assertIn(bits, (0x00000000, 0x80000000))

    def test_tie_rule_is_not_lowest_index_wins(self) -> None:
        # `tie_by_low_index` is an explicitly killed mutation of this authority.
        _, indices = topk_reference(_all_equal_row())
        self.assertNotEqual(indices, list(range(OUTPUT_WIDTH)))

    def test_rejects_wrong_input_width(self) -> None:
        with self.assertRaisesRegex(ValueError, str(INPUT_WIDTH)):
            topk_reference([f32_bits(0.0)] * (INPUT_WIDTH - 1))
