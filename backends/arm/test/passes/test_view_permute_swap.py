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


def _view_permute_swap(
    x: torch.Tensor, view_shape: Sequence[_Dim], permute_dims: list[int]
) -> tuple[list[int], list[_Dim]] | None:
    graph = torch.fx.Graph()
    input_node = graph.placeholder("x")
    input_node.meta["val"] = x
    view = graph.call_function(
        exir_ops.edge.aten.view_copy.default,
        args=(input_node, view_shape),
    )
    view.meta["val"] = x.new_empty(tuple(view_shape))
    normalized_dims = [
        dim if dim >= 0 else dim + len(view_shape) for dim in permute_dims
    ]
    permute = graph.call_function(
        exir_ops.edge.aten.permute_copy.default,
        args=(view, permute_dims),
    )
    permute.meta["val"] = x.new_empty(tuple(view_shape[dim] for dim in normalized_dims))
    return CanonicalizeViewCopyPermutePass()._view_permute_swap(view, permute)


def _assert_swap_matches_tensor_behavior(
    input_shape: list[int],
    view_shape: list[int],
    permute_dims: list[int],
    swapped_args: tuple[Sequence[int], Sequence[_Dim]],
) -> None:
    swapped_permute_dims, output_shape = swapped_args
    normalized_dims = [
        dim if dim >= 0 else dim + len(view_shape) for dim in permute_dims
    ]

    data = torch.arange(_numel(input_shape)).reshape(input_shape)
    original = data.reshape(view_shape).permute(normalized_dims).contiguous()
    swapped = (
        data.permute(list(swapped_permute_dims))
        .contiguous()
        .reshape(cast(list[int], list(output_shape)))
    )

    assert list(swapped.shape) == list(original.shape)
    torch.testing.assert_close(swapped, original)


@pytest.mark.parametrize(
    "input_shape, view_shape, permute_dims, expected",
    [
        ([2, 3, 4], [2, 3, 2, 2], [0, 2, 3, 1], ([0, 2, 1], [2, 2, 2, 3])),
        ([2, 3, 4], [2, 3, 2, 2], [2, 3, 0, 1], ([2, 0, 1], [2, 2, 2, 3])),
        ([2, 3, 4], [2, 12], [1, 0], ([1, 2, 0], [12, 2])),
        ([2, 3, 4], [6, 4], [1, 0], ([2, 0, 1], [4, 6])),
        ([2, 3, 4], [6, 4], [-1, 0], ([2, 0, 1], [4, 6])),
        ([2, 3, 4], [1, 6, 4], [0, 2, 1], ([2, 0, 1], [1, 4, 6])),
        ([2, 3, 4], [1, 6, 4], [2, 0, 1], ([2, 0, 1], [4, 1, 6])),
        ([2, 3, 4], [1, 6, 4], [2, 1, 0], ([2, 0, 1], [4, 6, 1])),
        ([2, 3, 4], [2, 3, 2, 2], [1, 2, 3, 0], ([1, 2, 0], [3, 2, 2, 2])),
        ([2, 3, 4, 5], [2, 3, 20], [0, 2, 1], ([0, 2, 3, 1], [2, 20, 3])),
        ([2, 3, 4, 5], [2, 3, 20], [2, 0, 1], ([2, 3, 0, 1], [20, 2, 3])),
        ([2, 3, 4, 5], [6, 4, 5], [1, 2, 0], ([2, 3, 0, 1], [4, 5, 6])),
        ([2, 3, 4, 5], [24, 5], [1, 0], ([3, 0, 1, 2], [5, 24])),
        ([1, 2, 3, 4], [2, 12], [1, 0], ([0, 2, 3, 1], [12, 2])),
        ([2, 1, 3, 4], [6, 4], [1, 0], ([1, 3, 0, 2], [4, 6])),
        ([2, 3, 1, 4], [2, 3, 2, 2], [0, 2, 3, 1], ([0, 2, 3, 1], [2, 2, 2, 3])),
        ([2, 2, 3], [2, 2, 3], [2, 0, 1], ([2, 0, 1], [3, 2, 2])),
        ([2, 2, 3], [2, 2, 3], [1, 2, 0], ([1, 2, 0], [2, 3, 2])),
        ([2, 2, 2, 3], [8, 3], [1, 0], ([3, 0, 1, 2], [3, 8])),
        ([2, 2, 2, 3], [2, 2, 2, 3], [3, 0, 1, 2], ([3, 0, 1, 2], [3, 2, 2, 2])),
        (
            [2, 3, 4, 5, 6, 7, 8, 9],
            [1, 2, 3, 4, 5, 6, 7, 8, 9],
            [0, 8, 7, 6, 5, 4, 3, 2, 1],
            ([7, 6, 5, 4, 3, 2, 1, 0], [1, 9, 8, 7, 6, 5, 4, 3, 2]),
        ),
    ],
)
def test_view_permute_swap_expected_rewrites(
    input_shape: list[int],
    view_shape: list[int],
    permute_dims: list[int],
    expected: tuple[list[int], list[int]],
) -> None:
    swapped_args = _view_permute_swap(
        _meta_tensor(input_shape), view_shape, permute_dims
    )

    assert swapped_args == expected
    _assert_swap_matches_tensor_behavior(
        input_shape, view_shape, permute_dims, swapped_args
    )


@pytest.mark.parametrize(
    "input_shape, view_shape, permute_dims",
    [
        ([2, 3, 4], [2, 4, 3], [0, 2, 1]),
        ([4], [2, 2], [1, 0]),
    ],
)
def test_view_permute_swap_rejects_unsupported_rewrites(
    input_shape: list[int], view_shape: list[int], permute_dims: list[int]
) -> None:
    assert (
        _view_permute_swap(_meta_tensor(input_shape), view_shape, permute_dims) is None
    )


def test_view_permute_swap_generated_cases_are_semantically_valid() -> None:
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
        view_shapes = _factorizations(_numel(input_shape), max_rank=4)
        for view_shape in view_shapes:
            for permute_dims in permutations(range(len(view_shape))):
                total_cases += 1
                swapped_args = _view_permute_swap(
                    _meta_tensor(input_shape), view_shape, list(permute_dims)
                )
                if swapped_args is None:
                    rejected_cases += 1
                    continue

                accepted_cases += 1
                _assert_swap_matches_tensor_behavior(
                    input_shape, view_shape, list(permute_dims), swapped_args
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


def test_view_permute_swap_preserves_symbolic_dimensions() -> None:
    shape_env = ShapeEnv()
    batch = _make_symint(shape_env, "batch", hint=2)

    with FakeTensorMode(shape_env=shape_env, allow_non_fake_inputs=True):
        x = torch.empty((batch, 2, 3, 4), device="cpu")
        assert _view_permute_swap(x, [batch, 2, 12], [0, 2, 1]) == (
            [0, 2, 3, 1],
            [batch, 12, 2],
        )
        assert _view_permute_swap(x, [batch, 2, 3, 4], [2, 3, 0, 1]) == (
            [2, 3, 0, 1],
            [3, 4, batch, 2],
        )

        x = torch.empty((batch, 2, 10, 10), device="cpu")
        assert _view_permute_swap(x, [batch, 2, 5, 2, 5, 2], [0, 1, 2, 3, 4, 5]) == (
            [0, 1, 2, 3],
            [batch, 2, 5, 2, 5, 2],
        )
