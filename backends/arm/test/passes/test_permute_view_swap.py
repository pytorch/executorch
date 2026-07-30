# Copyright 2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from itertools import permutations
from typing import cast, Sequence

import pytest
import sympy  # type: ignore[import-untyped]
import torch
from executorch.backends.arm._passes import CanonicalizeViewCopyPermutePass
from executorch.exir.dialects._ops import ops as exir_ops
from torch._subclasses.fake_tensor import FakeTensorMode
from torch.fx.experimental.symbolic_shapes import ShapeEnv

_Dim = int | torch.SymInt


def _numel(shape: list[int]) -> int:
    numel = 1
    for dim in shape:
        numel *= dim
    return numel


def _factorizations(numel: int, max_rank: int) -> list[list[int]]:
    shapes: list[list[int]] = []

    def recurse(remaining: int, rank: int, shape: list[int]) -> None:
        if rank == 0:
            if remaining == 1:
                shapes.append(list(shape))
            return

        for dim in range(1, remaining + 1):
            if remaining % dim == 0:
                shape.append(dim)
                recurse(remaining // dim, rank - 1, shape)
                shape.pop()

    for rank in range(1, max_rank + 1):
        recurse(numel, rank, [])

    return shapes


def _meta_tensor(shape: list[int]) -> torch.Tensor:
    return torch.empty(tuple(shape), device="meta")


def _permute_view_swap(
    x: torch.Tensor, permute_dims: list[int], output_shape: Sequence[_Dim]
) -> tuple[list[_Dim], list[int]] | None:
    graph = torch.fx.Graph()
    input_node = graph.placeholder("x")
    input_node.meta["val"] = x
    normalized_dims = [dim if dim >= 0 else dim + len(x.shape) for dim in permute_dims]
    permute = graph.call_function(
        exir_ops.edge.aten.permute_copy.default,
        args=(input_node, permute_dims),
    )
    permute.meta["val"] = x.new_empty(tuple(x.shape[dim] for dim in normalized_dims))
    view = graph.call_function(
        exir_ops.edge.aten.view_copy.default,
        args=(permute, output_shape),
    )
    view.meta["val"] = x.new_empty(tuple(output_shape))
    return CanonicalizeViewCopyPermutePass()._permute_view_swap(x, permute, view)


def _assert_swap_matches_tensor_behavior(
    input_shape: list[int],
    permute_dims: list[int],
    output_shape: list[int],
    swapped_args: tuple[Sequence[_Dim], Sequence[int]],
) -> None:
    view_shape, swapped_permute_dims = swapped_args
    normalized_dims = [
        dim if dim >= 0 else dim + len(input_shape) for dim in permute_dims
    ]

    data = torch.arange(_numel(input_shape)).reshape(input_shape)
    original = data.permute(normalized_dims).contiguous().reshape(output_shape)
    swapped = (
        data.reshape(cast(list[int], list(view_shape)))
        .permute(list(swapped_permute_dims))
        .contiguous()
    )

    assert list(swapped.shape) == output_shape
    torch.testing.assert_close(swapped, original)


@pytest.mark.parametrize(
    "input_shape, permute_dims, output_shape, expected",
    [
        ([2, 3, 4], [2, 0, 1], [4, 6], ([6, 4], [1, 0])),
        ([2, 3, 4], [1, 2, 0], [12, 2], ([2, 12], [1, 0])),
        ([2, 3, 4], [0, 2, 1], [2, 2, 2, 3], ([2, 3, 2, 2], [0, 2, 3, 1])),
        ([2, 3, 4], [-1, 0, 1], [4, 6], ([6, 4], [1, 0])),
        ([2, 3, 4], [2, 0, 1], [1, 4, 6], ([1, 6, 4], [0, 2, 1])),
        ([2, 3, 4], [2, 0, 1], [4, 1, 6], ([1, 6, 4], [2, 0, 1])),
        ([2, 3, 4], [2, 0, 1], [4, 6, 1], ([6, 4, 1], [1, 0, 2])),
        (
            [2, 3, 4],
            [2, 0, 1],
            [2, 2, 2, 3],
            ([2, 3, 2, 2], [2, 3, 0, 1]),
        ),
        ([2, 3, 4], [1, 2, 0], [3, 2, 2, 2], ([2, 3, 2, 2], [1, 2, 3, 0])),
        ([2, 3, 4, 5], [0, 2, 3, 1], [2, 20, 3], ([2, 3, 20], [0, 2, 1])),
        ([2, 3, 4, 5], [2, 3, 0, 1], [20, 2, 3], ([2, 3, 20], [2, 0, 1])),
        ([2, 3, 4, 5], [2, 3, 0, 1], [4, 5, 6], ([6, 4, 5], [1, 2, 0])),
        ([2, 3, 4, 5], [3, 0, 1, 2], [5, 24], ([24, 5], [1, 0])),
        ([1, 2, 3, 4], [2, 3, 0, 1], [12, 2], ([2, 12], [1, 0])),
        ([2, 1, 3, 4], [1, 3, 0, 2], [4, 6], ([6, 4], [1, 0])),
        ([2, 3, 1, 4], [2, 0, 3, 1], [2, 2, 2, 3], ([2, 3, 2, 2], [0, 2, 3, 1])),
        ([2, 2, 3], [2, 0, 1], [3, 2, 2], ([2, 2, 3], [2, 0, 1])),
        ([2, 2, 3], [1, 2, 0], [2, 3, 2], ([2, 2, 3], [1, 2, 0])),
        ([2, 2, 2, 3], [3, 0, 1, 2], [3, 8], ([8, 3], [1, 0])),
        ([2, 2, 2, 3], [3, 0, 1, 2], [3, 2, 2, 2], ([2, 2, 2, 3], [3, 0, 1, 2])),
    ],
)
def test_permute_view_swap_expected_rewrites(
    input_shape: list[int],
    permute_dims: list[int],
    output_shape: list[int],
    expected: tuple[list[int], list[int]],
) -> None:
    swapped_args = _permute_view_swap(
        _meta_tensor(input_shape), permute_dims, output_shape
    )

    assert swapped_args == expected
    _assert_swap_matches_tensor_behavior(
        input_shape, permute_dims, output_shape, swapped_args
    )


@pytest.mark.parametrize(
    "input_shape, permute_dims, output_shape",
    [
        ([2, 3, 4], [1, 0, 2], [3, 8]),
        ([2, 3, 4], [1, 0, 2], [6, 4]),
        ([2, 3, 4], [2, 1, 0], [4, 6]),
        ([2, 3, 4], [0, 2, 1], [8, 3]),
        ([2, 3, 4, 5], [3, 0, 1, 2], [10, 12]),
        ([2, 3, 4, 5], [1, 3, 0, 2], [15, 8]),
        ([2, 3, 4], [2, 0, 1], [5, 5]),
    ],
)
def test_permute_view_swap_rejects_unsupported_rewrites(
    input_shape: list[int], permute_dims: list[int], output_shape: list[int]
) -> None:
    assert (
        _permute_view_swap(_meta_tensor(input_shape), permute_dims, output_shape)
        is None
    )


def test_permute_view_swap_generated_cases_are_semantically_valid() -> None:
    input_shapes = [
        [2, 3],
        [2, 4],
        [2, 3, 4],
        [2, 2, 3],
        [1, 2, 3],
        [2, 1, 3],
        [2, 3, 1],
        [2, 2, 2, 3],
    ]

    total_cases = 0
    accepted_cases = 0
    rejected_cases = 0

    for input_shape in input_shapes:
        output_shapes = _factorizations(_numel(input_shape), max_rank=4)
        for permute_dims in permutations(range(len(input_shape))):
            for output_shape in output_shapes:
                total_cases += 1
                swapped_args = _permute_view_swap(
                    _meta_tensor(input_shape), list(permute_dims), output_shape
                )
                if swapped_args is None:
                    rejected_cases += 1
                    continue

                accepted_cases += 1
                _assert_swap_matches_tensor_behavior(
                    input_shape, list(permute_dims), output_shape, swapped_args
                )

    assert total_cases > 1000
    assert accepted_cases > 200
    assert rejected_cases > 200


def _make_symint(
    shape_env: ShapeEnv, symbol: str, hint: int, min: int = 1, max: int = 64
) -> torch.SymInt:
    symint = shape_env.create_symintnode(sympy.Symbol(symbol), hint=hint)
    assert isinstance(symint, torch.SymInt)
    shape_env.constrain_symbol_range(
        symint.node.expr, compiler_min=min, compiler_max=max
    )
    return symint


def test_permute_view_swap_preserves_symbolic_dimensions() -> None:
    shape_env = ShapeEnv()
    batch = _make_symint(shape_env, "batch", hint=2)

    with FakeTensorMode(shape_env=shape_env, allow_non_fake_inputs=True):
        x = torch.empty((batch, 2, 3, 4), device="cpu")
        assert _permute_view_swap(x, [2, 3, 0, 1], [12, batch, 2]) == (
            [batch, 2, 12],
            [2, 0, 1],
        )
        assert _permute_view_swap(x, [0, 2, 3, 1], [batch, 12, 2]) == (
            [batch, 2, 12],
            [0, 2, 1],
        )

        x = torch.empty((batch, 2, 10, 10), device="cpu")
        assert _permute_view_swap(x, [0, 1, 2, 3], [batch, 2, 5, 2, 5, 2]) == (
            [batch, 2, 5, 2, 5, 2],
            [0, 1, 2, 3, 4, 5],
        )
