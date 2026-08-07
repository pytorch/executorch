# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

"""CPU authority for the Gemma 4 MTP scatter routes.

`scatter_serial` transcribes `runtime/ops/scatter/scatter.wgsl` (ascending
last-write-wins, mirroring portable `op_scatter.cpp`).
`scatter_unique_indices.wgsl` is arbitrary-write-wins and is therefore only
equivalent when the destinations are pairwise distinct — the provenance the
official top-32 token ordering guarantees and this module enforces.
"""

from __future__ import annotations

import unittest

VOCAB_SIZE = 262144
SELECTED_COUNT = 4096
CENTROIDS = 2048
TOKENS_PER_CENTROID = 128
SELECTED_CENTROIDS = 32

_U32 = 0xFFFFFFFF


def scatter_serial(
    base: list[float], indices: list[int], source: list[float]
) -> list[float]:
    """Ascending last-write-wins with the shader's verbatim range guard."""
    if len(base) != VOCAB_SIZE:
        raise ValueError(f"scatter authority requires a {VOCAB_SIZE}-wide base")
    if len(indices) != SELECTED_COUNT or len(source) != SELECTED_COUNT:
        raise ValueError(f"scatter authority requires {SELECTED_COUNT} writes")
    out = list(base)
    for i in range(SELECTED_COUNT):
        destination = indices[i]
        if 0 <= destination < VOCAB_SIZE:
            out[destination] = source[i]
    return out


def destinations_are_pairwise_distinct(indices: list[int]) -> bool:
    """Whether the parallel route is well defined: distinct *surviving* writes."""
    live = [d for d in indices if 0 <= d < VOCAB_SIZE]
    return len(set(live)) == len(live)


def has_official_provenance(indices: list[int]) -> bool:
    """The exporter-certified contract: every destination in range and distinct."""
    return all(0 <= d < VOCAB_SIZE for d in indices) and len(set(indices)) == len(
        indices
    )


def scatter_parallel(
    base: list[float], indices: list[int], source: list[float]
) -> list[float]:
    """Order-independent transcription of `scatter_unique_indices.wgsl`."""
    if len(base) != VOCAB_SIZE:
        raise ValueError(f"scatter authority requires a {VOCAB_SIZE}-wide base")
    if len(indices) != SELECTED_COUNT or len(source) != SELECTED_COUNT:
        raise ValueError(f"scatter authority requires {SELECTED_COUNT} writes")
    written: dict[int, float] = {}
    # Descending: a duplicate destination keeps the lowest index, not serial's last.
    for i in range(SELECTED_COUNT - 1, -1, -1):
        destination = indices[i]
        if 0 <= destination < VOCAB_SIZE:
            written[destination] = source[i]
    out = list(base)
    for destination, value in written.items():
        out[destination] = value
    return out


def scatter_unique(
    base: list[float], indices: list[int], source: list[float]
) -> list[float]:
    """The parallel route; refuses the inputs on which it is not well defined."""
    if not destinations_are_pairwise_distinct(indices):
        raise ValueError("scatter_src_unique requires pairwise-distinct destinations")
    return scatter_parallel(base, indices, source)


def _lcg(seed: int, count: int) -> list[int]:
    state = seed & _U32
    drawn: list[int] = []
    for _ in range(count):
        state = (1664525 * state + 1013904223) & _U32
        drawn.append(state)
    return drawn


def token_ordering(seed: int = 17) -> list[list[int]]:
    """A deterministic [2048, 128] permutation of range(262144)."""
    tokens = list(range(VOCAB_SIZE))
    draws = _lcg(seed, VOCAB_SIZE)
    for i in range(VOCAB_SIZE - 1, 0, -1):
        j = draws[i] % (i + 1)
        tokens[i], tokens[j] = tokens[j], tokens[i]
    return [
        tokens[row * TOKENS_PER_CENTROID : (row + 1) * TOKENS_PER_CENTROID]
        for row in range(CENTROIDS)
    ]


def official_selected_indices(selected_centroids: list[int]) -> list[int]:
    """`token_ordering[topk_indices]` flattened, i.e. the shipped provenance."""
    if len(selected_centroids) != SELECTED_CENTROIDS:
        raise ValueError(f"expected {SELECTED_CENTROIDS} selected centroids")
    if len(set(selected_centroids)) != SELECTED_CENTROIDS:
        raise ValueError("selected centroids must be distinct")
    ordering = token_ordering()
    flattened: list[int] = []
    for row in selected_centroids:
        if not 0 <= row < CENTROIDS:
            raise ValueError(f"centroid row {row} out of range")
        flattened.extend(ordering[row])
    return flattened


def base_row() -> list[float]:
    return [float((index % 1021) - 510) * 0.5 for index in range(VOCAB_SIZE)]


def source_row(seed: int) -> list[float]:
    return [(draw / 2**32) * 20.0 - 10.0 for draw in _lcg(seed, SELECTED_COUNT)]


def _selected_centroids() -> list[int]:
    return [row * 61 % CENTROIDS for row in range(SELECTED_CENTROIDS)]


def _differing_positions(left: list[float], right: list[float]) -> list[int]:
    return [index for index in range(VOCAB_SIZE) if left[index] != right[index]]


def _duplicate_indices() -> list[int]:
    indices = official_selected_indices(_selected_centroids())
    indices[10] = indices[4000]
    indices[11] = indices[4000]
    return indices


def _negative_indices() -> list[int]:
    indices = official_selected_indices(_selected_centroids())
    indices[0] = -1
    indices[1] = -VOCAB_SIZE
    return indices


def _out_of_range_indices() -> list[int]:
    indices = official_selected_indices(_selected_centroids())
    indices[2] = VOCAB_SIZE
    indices[3] = VOCAB_SIZE + 7
    return indices


def _boundary_indices() -> list[int]:
    indices = official_selected_indices(_selected_centroids())
    indices[0] = 0
    indices[1] = VOCAB_SIZE - 1
    seen: set[int] = set()
    for position, destination in enumerate(indices):
        while destination in seen:
            destination = (destination + 1) % VOCAB_SIZE
        indices[position] = destination
        seen.add(destination)
    return indices


# Fixture contract consumed by `webgpu_scatter_test`. `equivalent` marks the cases
# on which the parallel route must match serial bit-for-bit; `provenance` marks the
# strictly smaller set an exporter may certify for `et_vk.scatter_src_unique`.
def scatter_cases() -> dict[str, dict[str, object]]:
    official = official_selected_indices(_selected_centroids())
    return {
        "official_unique": {
            "indices": official,
            "source": source_row(5),
        },
        "boundary_unique": {
            "indices": _boundary_indices(),
            "source": source_row(19),
        },
        "reversed_unique": {
            "indices": list(reversed(official)),
            "source": source_row(23),
        },
        "duplicate_destinations": {
            "indices": _duplicate_indices(),
            "source": source_row(29),
        },
        "negative_destinations": {
            "indices": _negative_indices(),
            "source": source_row(31),
        },
        "out_of_range_destinations": {
            "indices": _out_of_range_indices(),
            "source": source_row(37),
        },
    }


EQUIVALENT_CASES = (
    "official_unique",
    "boundary_unique",
    "reversed_unique",
    "negative_destinations",
    "out_of_range_destinations",
)
PROVENANCE_CASES = ("official_unique", "boundary_unique", "reversed_unique")


class TestScatterCpu(unittest.TestCase):
    def test_official_provenance_is_a_full_permutation(self) -> None:
        ordering = token_ordering()
        self.assertEqual(len(ordering), CENTROIDS)
        flattened = [token for row in ordering for token in row]
        self.assertEqual(len(flattened), VOCAB_SIZE)
        self.assertEqual(len(set(flattened)), VOCAB_SIZE)

    def test_official_selection_is_pairwise_distinct(self) -> None:
        indices = official_selected_indices(_selected_centroids())
        self.assertEqual(len(indices), SELECTED_COUNT)
        self.assertTrue(destinations_are_pairwise_distinct(indices))

    def test_case_labels_match_the_computed_predicates(self) -> None:
        cases = scatter_cases()
        for name, case in cases.items():
            indices = case["indices"]
            assert isinstance(indices, list)
            self.assertEqual(
                destinations_are_pairwise_distinct(indices),
                name in EQUIVALENT_CASES,
                f"{name} mislabels parallel-route equivalence",
            )
            self.assertEqual(
                has_official_provenance(indices),
                name in PROVENANCE_CASES,
                f"{name} mislabels official provenance",
            )
        # Provenance is strictly stronger than equivalence: the out-of-range and
        # negative rows are bit-equivalent yet must never be certified.
        self.assertLess(set(PROVENANCE_CASES), set(EQUIVALENT_CASES))

    def test_unique_route_matches_serial_on_official_indices(self) -> None:
        base = base_row()
        cases = scatter_cases()
        for name in EQUIVALENT_CASES:
            case = cases[name]
            indices = case["indices"]
            source = case["source"]
            assert isinstance(indices, list) and isinstance(source, list)
            self.assertEqual(
                _differing_positions(
                    scatter_unique(base, indices, source),
                    scatter_serial(base, indices, source),
                ),
                [],
                name,
            )

    def test_parallel_route_disagrees_with_serial_on_duplicates(self) -> None:
        # Why the distinctness guard exists: the two routes are only equal there.
        base = base_row()
        case = scatter_cases()["duplicate_destinations"]
        indices = case["indices"]
        source = case["source"]
        assert isinstance(indices, list) and isinstance(source, list)
        parallel = scatter_parallel(base, indices, source)
        serial = scatter_serial(base, indices, source)
        duplicated = indices[4000]
        self.assertEqual([indices[10], indices[11]], [duplicated, duplicated])
        self.assertNotEqual(source[10], source[4000])
        self.assertEqual(_differing_positions(parallel, serial), [duplicated])
        self.assertEqual(parallel[duplicated], source[10])
        self.assertEqual(serial[duplicated], source[4000])

    def test_unique_route_rejects_duplicate_destinations(self) -> None:
        base = base_row()
        case = scatter_cases()["duplicate_destinations"]
        indices = case["indices"]
        source = case["source"]
        assert isinstance(indices, list) and isinstance(source, list)
        self.assertFalse(destinations_are_pairwise_distinct(indices))
        with self.assertRaisesRegex(ValueError, "pairwise-distinct"):
            scatter_unique(base, indices, source)
        # The duplicate-safe route stays defined: the later write wins.
        result = scatter_serial(base, indices, source)
        self.assertEqual(result[indices[4000]], source[4000])

    def test_out_of_range_and_negative_destinations_are_dropped(self) -> None:
        base = base_row()
        for name in ("negative_destinations", "out_of_range_destinations"):
            case = scatter_cases()[name]
            indices = case["indices"]
            source = case["source"]
            assert isinstance(indices, list) and isinstance(source, list)
            result = scatter_serial(base, indices, source)
            dropped = [
                position
                for position, destination in enumerate(indices)
                if not 0 <= destination < VOCAB_SIZE
            ]
            self.assertTrue(dropped, name)
            written = {
                indices[position]
                for position in range(SELECTED_COUNT)
                if position not in set(dropped)
            }
            for index in range(VOCAB_SIZE):
                if index not in written:
                    self.assertEqual(result[index], base[index], f"{name}@{index}")

    def test_boundary_destinations_are_written(self) -> None:
        base = base_row()
        case = scatter_cases()["boundary_unique"]
        indices = case["indices"]
        source = case["source"]
        assert isinstance(indices, list) and isinstance(source, list)
        result = scatter_unique(base, indices, source)
        self.assertEqual(result[0], source[indices.index(0)])
        self.assertEqual(result[VOCAB_SIZE - 1], source[indices.index(VOCAB_SIZE - 1)])

    def test_rejects_wrong_widths(self) -> None:
        base = base_row()
        indices = official_selected_indices(_selected_centroids())
        source = source_row(5)
        with self.assertRaisesRegex(ValueError, str(VOCAB_SIZE)):
            scatter_serial(base[:-1], indices, source)
        with self.assertRaisesRegex(ValueError, str(SELECTED_COUNT)):
            scatter_serial(base, indices[:-1], source)
        with self.assertRaisesRegex(ValueError, str(SELECTED_COUNT)):
            scatter_serial(base, indices, source[:-1])

    def test_rejects_non_distinct_centroid_selection(self) -> None:
        rows = _selected_centroids()
        rows[1] = rows[0]
        with self.assertRaisesRegex(ValueError, "distinct"):
            official_selected_indices(rows)
