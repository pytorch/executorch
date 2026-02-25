# Copyright 2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import random
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from executorch.backends.arm.test import common
from executorch.backends.arm.test.tester.test_pipeline import TosaPipelineINT
from executorch.backends.arm.tosa.compile_spec import TosaCompileSpec


class HighRankPermuteViewModel(torch.nn.Module):
    def __init__(self, ops: list[tuple[str, Any]]):
        super().__init__()
        self.ops = ops
        self.block = nn.Sequential(
            nn.Conv2d(
                in_channels=3,
                out_channels=64,
                kernel_size=3,
                stride=2,
                padding=1,
            ),
            nn.ReLU(),
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

    # Ensure each case has at least one rank>4 permute.
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


def _build_cases() -> dict[str, HighRankPermuteViewModel]:
    rng = random.Random(
        20260225
    )  # nosec B311: deterministic RNG for test case generation
    start_shape = [1, 16, 16, 64]  # conv output from input 1x3x32x32 after NHWC permute
    cases: dict[str, HighRankPermuteViewModel] = {}
    for idx in range(10):
        ops = _generate_chain(rng, start_shape, steps=8)
        cases[f"fuzz_case_{idx}"] = HighRankPermuteViewModel(ops)
    return cases


def _run_model(model: torch.nn.Module, out_dir: str) -> Path:
    sample = torch.randn(1, 3, 32, 32)
    pipeline = TosaPipelineINT[tuple[torch.Tensor]](
        model.eval(),
        (sample,),
        aten_op=[],
        exir_op=[],
        run_on_tosa_ref_model=False,
        custom_path=out_dir,
        tosa_debug_mode=TosaCompileSpec.DebugMode.JSON,
        tosa_extensions=["int16", "int4", "cf"],
    )
    pipeline.run()

    tosa_files = sorted(Path(out_dir).glob("*.tosa"))
    assert tosa_files, f"No TOSA artifacts found in {out_dir}"
    return tosa_files[0]


def _assert_transpose_invariants(tosa_path: Path) -> int:
    import tosa.Op as Op  # type: ignore[import-not-found,import-untyped]
    from tosa.TosaGraph import (  # type: ignore[import-not-found,import-untyped]
        TosaGraph,
    )
    from tosa.TransposeAttribute import (  # type: ignore[import-not-found,import-untyped]
        TransposeAttribute,
    )

    graph = TosaGraph.GetRootAs(tosa_path.read_bytes(), 0)
    block = graph.Regions(0).Blocks(0)

    shape_by_name = {
        block.Tensors(i).Name().decode(): list(block.Tensors(i).ShapeAsNumpy())
        for i in range(block.TensorsLength())
    }

    op_enum = Op.Op()
    op_value_to_name = {
        getattr(op_enum, name): name for name in dir(op_enum) if name.isupper()
    }

    high_rank_transpose_count = 0
    for i in range(block.OperatorsLength()):
        op = block.Operators(i)
        if op_value_to_name.get(op.Op()) != "TRANSPOSE":
            continue

        inputs = [op.Inputs(j).decode() for j in range(op.InputsLength())]
        outputs = [op.Outputs(j).decode() for j in range(op.OutputsLength())]
        assert len(inputs) == 1 and len(outputs) == 1, (
            f"Unexpected TRANSPOSE arity at op #{i}: "
            f"{len(inputs)} inputs, {len(outputs)} outputs"
        )

        attr_tbl = op.Attribute()
        transpose_attr = TransposeAttribute()
        transpose_attr.Init(attr_tbl.Bytes, attr_tbl.Pos)
        perms = [int(perm) for perm in transpose_attr.PermsAsNumpy()]

        in_shape = [int(v) for v in shape_by_name[inputs[0]]]
        out_shape = [int(v) for v in shape_by_name[outputs[0]]]

        rank = len(in_shape)
        assert (
            len(perms) == rank
        ), f"Invalid TRANSPOSE rank at op #{i}: len(perms)={len(perms)} rank={rank}"
        assert sorted(perms) == list(
            range(rank)
        ), f"Invalid TRANSPOSE permutation at op #{i}: perms={perms}, rank={rank}"
        expected_out_shape = [in_shape[perm] for perm in perms]
        assert expected_out_shape == out_shape, (
            f"Invalid TRANSPOSE shape mapping at op #{i}: "
            f"perms={perms}, in_shape={in_shape}, out_shape={out_shape}, "
            f"expected_out_shape={expected_out_shape}"
        )
        if rank > 4:
            high_rank_transpose_count += 1

    return high_rank_transpose_count


@common.parametrize("model", _build_cases())
def test_high_rank_permute_view_tosa_INT_transpose_invariants(
    model: torch.nn.Module, tmp_path
):
    out_dir = tmp_path / "high_rank_permute_view_fuzz"
    out_dir.mkdir(parents=True, exist_ok=True)
    tosa_path = _run_model(model, str(out_dir))
    assert tosa_path.exists(), f"Missing TOSA dump: {tosa_path}"
    high_rank_count = _assert_transpose_invariants(tosa_path)
    assert (
        high_rank_count > 0
    ), "Expected at least one rank>4 TRANSPOSE in generated case."
