# Copyright 2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import random
from dataclasses import dataclass
from typing import Any, Tuple

import torch

from executorch.backends.arm.test import common
from executorch.backends.arm.test.tester.test_pipeline import TosaPipelineINT


InputT = Tuple[Any, ...]


class HighRankPermuteViewModel(torch.nn.Module):
    def __init__(self, ops: list[tuple[str, Any]]):
        super().__init__()
        self.ops = ops
        self.block = torch.nn.Sequential(
            torch.nn.Conv2d(
                in_channels=3,
                out_channels=64,
                kernel_size=3,
                stride=2,
                padding=1,
            ),
            torch.nn.ReLU(),
        )

    def forward(self, x):
        x = self.block(x).permute(0, 2, 3, 1)
        for kind, payload in self.ops:
            if kind == "reshape":
                x = x.reshape(payload)
            elif kind == "permute":
                x = x.permute(payload)
            else:
                raise AssertionError(f"Unknown op kind: {kind}")
        return x


@dataclass(frozen=True)
class TransposeInvariantCase:
    module: torch.nn.Module
    inputs: InputT
    expected_transposes: int


def _random_non_identity_permutation(
    rng: random.Random, rank: int
) -> tuple[int, ...] | None:
    if rank < 2:
        return None
    base = list(range(rank))
    for _ in range(12):
        candidate = base[:]
        rng.shuffle(candidate)
        if candidate != base:
            return tuple(candidate)
    return None


def _reshape_add_singleton(rng: random.Random, shape: list[int]) -> list[int] | None:
    if len(shape) >= 6:
        return None
    pos = rng.randrange(0, len(shape) + 1)
    return shape[:pos] + [1] + shape[pos:]


def _reshape_remove_singleton(rng: random.Random, shape: list[int]) -> list[int] | None:
    singleton_indices = [idx for idx, dim in enumerate(shape) if dim == 1]
    if not singleton_indices:
        return None
    idx = rng.choice(singleton_indices)
    return shape[:idx] + shape[idx + 1 :]


def _reshape_split_dim(rng: random.Random, shape: list[int]) -> list[int] | None:
    if len(shape) >= 6:
        return None
    candidates = [idx for idx, dim in enumerate(shape) if dim > 1]
    if not candidates:
        return None
    idx = rng.choice(candidates)
    dim = shape[idx]
    factors = [f for f in range(2, min(dim, 64) + 1) if dim % f == 0]
    if not factors:
        return None
    first = rng.choice(factors)
    second = dim // first
    return shape[:idx] + [first, second] + shape[idx + 1 :]


def _reshape_merge_adjacent(rng: random.Random, shape: list[int]) -> list[int] | None:
    if len(shape) < 2:
        return None
    idx = rng.randrange(0, len(shape) - 1)
    merged = shape[idx] * shape[idx + 1]
    return shape[:idx] + [merged] + shape[idx + 2 :]


def _generate_chain(
    rng: random.Random,
    start_shape: list[int],
    steps: int,
) -> list[tuple[str, Any]]:
    shape = list(start_shape)
    ops: list[tuple[str, Any]] = []
    saw_high_rank_permute = False

    reshape_transforms = [
        _reshape_add_singleton,
        _reshape_remove_singleton,
        _reshape_split_dim,
        _reshape_merge_adjacent,
    ]

    for _ in range(steps):
        do_permute = rng.random() < 0.55
        if do_permute:
            permutation = _random_non_identity_permutation(rng, len(shape))
            if permutation is not None:
                ops.append(("permute", permutation))
                shape = [shape[idx] for idx in permutation]
                if len(permutation) > 4:
                    saw_high_rank_permute = True
                continue

        rng.shuffle(reshape_transforms)
        for transform in reshape_transforms:
            new_shape = transform(rng, shape)
            if new_shape is None or new_shape == shape:
                continue
            ops.append(("reshape", tuple(new_shape)))
            shape = new_shape
            break

    while len(shape) <= 4:
        new_shape = _reshape_add_singleton(rng, shape)
        if new_shape is None:
            break
        ops.append(("reshape", tuple(new_shape)))
        shape = new_shape

    if not saw_high_rank_permute:
        permutation = _random_non_identity_permutation(rng, len(shape))
        if permutation is not None and len(permutation) > 4:
            ops.append(("permute", permutation))

    return ops


def _build_high_rank_permute_cases() -> dict[str, TransposeInvariantCase]:
    rng = random.Random(
        20260225
    )  # nosec B311: deterministic RNG for test case generation
    start_shape = [1, 16, 16, 64]
    expected_transpose_counts = [4, 3, 3, 3, 2, 3, 3, 3, 3, 2]
    cases: dict[str, TransposeInvariantCase] = {}
    for idx in range(10):
        ops = _generate_chain(rng, start_shape, steps=8)
        cases[f"high_rank_permute_fuzz_case_{idx}"] = TransposeInvariantCase(
            module=HighRankPermuteViewModel(ops).eval(),
            inputs=(torch.randn(1, 3, 32, 32),),
            expected_transposes=expected_transpose_counts[idx],
        )
    return cases


@common.parametrize("case", _build_high_rank_permute_cases())
def test_transpose_invariants_tosa_INT_high_rank_permute_view(
    case: TransposeInvariantCase,
) -> None:
    pipeline = TosaPipelineINT[InputT](
        case.module,
        case.inputs,
        aten_op=[],
        exir_op=[],
        run_on_tosa_ref_model=False,
    )
    pipeline.count_tosa_ops({"TRANSPOSE": case.expected_transposes})
    pipeline.run()
