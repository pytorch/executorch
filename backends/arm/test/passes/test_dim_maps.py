# Copyright 2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from itertools import combinations, permutations
from typing import cast, Sequence, TypeVar

import sympy  # type: ignore[import-untyped]
import torch

from executorch.backends.arm._passes.dim_maps import (
    normalize_view_shape,
    PermuteMap,
    ViewMap,
)
from torch.fx.experimental.symbolic_shapes import ShapeEnv


_RNG = torch.Generator().manual_seed(0)
_T = TypeVar("_T")
_Dim = int | torch.SymInt
_DimT = TypeVar("_DimT", bound=_Dim)


def _make_symint(
    shape_env: ShapeEnv, symbol: str, hint: int, min: int = 1, max: int = 64
) -> torch.SymInt:
    symint = shape_env.create_symintnode(sympy.Symbol(symbol), hint=hint)
    assert isinstance(symint, torch.SymInt)
    shape_env.constrain_symbol_range(
        symint.node.expr, compiler_min=min, compiler_max=max
    )
    return symint


def _numel(shape: list[int]) -> int:
    numel = 1
    for dim in shape:
        numel *= dim
    return numel


def _factorizations(numel: int, rank: int) -> list[list[int]]:
    shapes: list[list[int]] = []

    def recurse(remaining: int, remaining_rank: int, shape: list[int]) -> None:
        if remaining_rank == 0:
            if remaining == 1:
                shapes.append(list(shape))
            return

        for dim in range(1, remaining + 1):
            if remaining % dim == 0:
                shape.append(dim)
                recurse(remaining // dim, remaining_rank - 1, shape)
                shape.pop()

    recurse(numel, rank, [])
    return shapes


def _randint(low: int, high: int) -> int:
    return int(torch.randint(low, high + 1, (), generator=_RNG).item())


def _choice(choices: list[_T]) -> _T:
    return choices[_randint(0, len(choices) - 1)]


def _shuffle(values: list[int]) -> None:
    indices = torch.randperm(len(values), generator=_RNG).tolist()
    values[:] = [values[index] for index in indices]


def _random_shape(rank: int, max_dim: int = 4) -> list[int]:
    return [_randint(1, max_dim) for _ in range(rank)]


def _random_view_shape(numel: int, max_rank: int = 4) -> list[int]:
    rank = _randint(1, max_rank)
    return _choice(_factorizations(numel, rank))


def _tensor(shape: list[int]) -> torch.Tensor:
    return torch.arange(_numel(shape), dtype=torch.float32).reshape(shape)


def _inverse_permutation(permutation: list[int]) -> list[int]:
    inverse = [0] * len(permutation)
    for index, dim in enumerate(permutation):
        inverse[dim] = index
    return inverse


def _permute_map(permutation: list[int]) -> PermuteMap:
    graph = torch.fx.Graph()
    x = graph.placeholder("x")
    permute = graph.call_function(torch.ops.aten.permute.default, args=(x, permutation))
    return PermuteMap(permute)


def _all_dim_subsets(rank: int) -> list[list[int]]:
    return [
        list(dims)
        for subset_size in range(1, rank + 1)
        for dims in combinations(range(rank), subset_size)
    ]


def _reduce_shape(shape: Sequence[_DimT], dims: list[int]) -> list[_DimT]:
    reduced_shape = list(shape)
    for dim in dims:
        reduced_shape[dim] = cast(_DimT, 1)
    return reduced_shape


def _reduce(tensor: torch.Tensor, dims: list[int]) -> torch.Tensor:
    return tensor.sum(dim=tuple(dims), keepdim=True)


def _same(lhs: torch.Tensor, rhs: torch.Tensor) -> bool:
    return lhs.shape == rhs.shape and torch.equal(lhs, rhs)


def _propose_permute_view_swap(
    input_shape: Sequence[_DimT],
    permutation: list[int],
    output_shape: Sequence[_DimT],
) -> tuple[list[_DimT], list[int]] | None:
    permuted_shape = [input_shape[dim] for dim in permutation]
    view_map = ViewMap.from_shapes(permuted_shape, output_shape)
    if not view_map.is_valid_map:
        return None

    permuted_axis = _inverse_permutation(permutation)
    target_axis_order = view_map.map_permutation(permuted_axis)
    if target_axis_order is None:
        return None

    return (
        [output_shape[target_axis] for target_axis in target_axis_order],
        _inverse_permutation(target_axis_order),
    )


def _propose_view_permute_swap(
    input_shape: Sequence[_DimT],
    view_shape: Sequence[_DimT],
    permutation: list[int],
) -> tuple[list[int], list[_DimT]] | None:
    view_map = ViewMap.from_shapes(input_shape, view_shape)
    if not view_map.is_valid_map:
        return None

    mapped_dims = view_map.map_permutation_inverse(permutation)
    if mapped_dims is None:
        return None

    output_shape = [view_shape[dim] for dim in permutation]
    return mapped_dims, output_shape


def _propose_reduction_view_swap(
    input_shape: Sequence[_DimT],
    source_dims: list[int],
    view_shape: Sequence[_DimT],
) -> tuple[list[_DimT], list[int]] | None:
    view_map = ViewMap.from_shapes(input_shape, view_shape)
    if not view_map.is_valid_map:
        return None

    target_dims = view_map.map_dim(source_dims)
    if target_dims is None:
        return None
    return list(view_shape), target_dims


def _propose_view_reduction_swap(
    input_shape: Sequence[_DimT],
    view_shape: Sequence[_DimT],
    target_dims: list[int],
) -> tuple[list[int], list[_DimT]] | None:
    view_map = ViewMap.from_shapes(input_shape, view_shape)
    if not view_map.is_valid_map:
        return None

    source_dims = view_map.map_dim_inverse(target_dims)
    if source_dims is None:
        return None
    return source_dims, _reduce_shape(view_shape, target_dims)


def _bruteforce_permute_view_swaps(
    x: torch.Tensor,
    permutation: list[int],
    output_shape: list[int],
) -> list[tuple[list[int], list[int]]]:
    original = x.permute(permutation).reshape(output_shape)
    candidates: list[tuple[list[int], list[int]]] = []
    for candidate_permutation in permutations(range(len(output_shape))):
        candidate_permutation_list = list(candidate_permutation)
        for candidate_shape in _factorizations(_numel(output_shape), len(output_shape)):
            candidate = x.reshape(candidate_shape).permute(candidate_permutation_list)
            if _same(original, candidate):
                candidates.append((candidate_shape, candidate_permutation_list))
    return candidates


def _bruteforce_view_permute_swaps(
    x: torch.Tensor,
    view_shape: list[int],
    permutation: list[int],
) -> list[tuple[list[int], list[int]]]:
    output_shape = [view_shape[dim] for dim in permutation]
    original = x.reshape(view_shape).permute(permutation)
    candidates: list[tuple[list[int], list[int]]] = []
    for candidate_permutation in permutations(range(len(x.shape))):
        candidate_permutation_list = list(candidate_permutation)
        candidate = x.permute(candidate_permutation_list).reshape(output_shape)
        if _same(original, candidate):
            candidates.append((candidate_permutation_list, output_shape))
    return candidates


def _bruteforce_reduction_view_swaps(
    x: torch.Tensor,
    source_dims: list[int],
    view_shape: list[int],
) -> list[tuple[list[int], list[int]]]:
    candidates: list[tuple[list[int], list[int]]] = []
    for target_dims in _all_dim_subsets(len(view_shape)):
        output_shape = _reduce_shape(view_shape, target_dims)
        reduced = _reduce(x, source_dims)
        if reduced.numel() != _numel(output_shape):
            continue
        original = reduced.reshape(output_shape)
        candidate = _reduce(x.reshape(view_shape), target_dims)
        if _same(original, candidate):
            candidates.append((view_shape, target_dims))
    return candidates


def _bruteforce_view_reduction_swaps(
    x: torch.Tensor,
    view_shape: list[int],
    target_dims: list[int],
) -> list[tuple[list[int], list[int]]]:
    original = _reduce(x.reshape(view_shape), target_dims)
    candidates: list[tuple[list[int], list[int]]] = []
    for source_dims in _all_dim_subsets(len(x.shape)):
        reduced = _reduce(x, source_dims)
        if reduced.numel() != original.numel():
            continue
        candidate = reduced.reshape(original.shape)
        if _same(original, candidate):
            candidates.append((source_dims, list(original.shape)))
    return candidates


def test_dim_map_maps_split_and_merged_prime_factor_groups() -> None:
    view_map = ViewMap.from_shapes([1, 2, 3, 4], [1, 6, 2, 2])

    assert view_map.is_valid_map
    assert view_map.map_dim(0) == [0]
    assert view_map.map_dim(1) is None
    assert view_map.map_dim(2) is None
    assert view_map.map_dim(3) == [2, 3]
    assert view_map.map_dim([1, 2]) == [1]
    assert view_map.map_dim([3, 1]) is None
    assert view_map.map_dim([3, 1, 2]) == [1, 2, 3]

    assert view_map.map_dim_inverse(0) is None
    assert view_map.map_dim_inverse(1) == [1, 2]
    assert view_map.map_dim_inverse(2) is None
    assert view_map.map_dim_inverse([3, 1, 2]) == [3, 1, 2]
    assert view_map.map_dim_inverse([2, 3, 1]) == [3, 1, 2]


def test_dim_map_groups_reordered_crossing_prime_factors() -> None:
    view_map = ViewMap.from_shapes([2, 3], [3, 2])

    assert view_map.is_valid_map
    assert view_map.map_dim(0) is None
    assert view_map.map_dim(1) is None
    assert view_map.map_dim([0, 1]) == [0, 1]
    assert view_map.map_dim_inverse(0) is None
    assert view_map.map_dim_inverse(1) is None
    assert view_map.map_dim_inverse([0, 1]) == [0, 1]


def test_dim_map_matches_view_map_docstring_example_reduction_dims() -> None:
    view_map = ViewMap.from_shapes([4, 3, 10], [2, 2, 1, 5, 3, 2])

    assert view_map.is_valid_map
    assert view_map.map_dim(0) == [0, 1]
    assert view_map.map_dim([1, 2]) == [3, 4, 5]
    assert view_map.map_dim_inverse([0, 1]) == [0]
    assert view_map.map_dim_inverse([3, 4, 5]) == [1, 2]

    assert view_map.map_dim(1) is None
    assert view_map.map_dim(2) is None
    assert view_map.map_dim_inverse(2) is None


def test_dim_map_matches_view_map_docstring_example_permutation_dims() -> None:
    view_map = ViewMap.from_shapes([4, 3, 10], [2, 2, 1, 5, 3, 2])

    assert view_map.map_permutation([1, 2, 0]) == [2, 3, 4, 5, 0, 1]
    assert view_map.map_permutation_inverse([2, 3, 4, 5, 0, 1]) == [1, 2, 0]

    assert view_map.map_permutation([0, 2, 1]) is None
    assert view_map.map_permutation([2, 0, 1]) is None


def test_dim_map_validates_reductions_by_whole_groups() -> None:
    view_map = ViewMap.from_shapes([2, 3], [3, 2])

    assert view_map.map_dim([0]) is None
    assert view_map.map_dim_inverse([1]) is None
    assert view_map.map_dim([0, 1]) == [0, 1]
    assert view_map.map_dim_inverse([0, 1]) == [0, 1]


def test_dim_map_validates_permuted_group_blocks() -> None:
    view_map = ViewMap.from_shapes([2, 3, 5], [3, 2, 5])

    assert view_map.map_permutation([0, 1, 2]) == [0, 1, 2]
    assert view_map.map_permutation([2, 0, 1]) == [2, 0, 1]
    assert view_map.map_permutation_inverse([0, 2, 1]) is None

    merged_view_map = ViewMap.from_shapes([2, 3], [6])
    assert merged_view_map.map_permutation([0, 1]) == [0]
    assert merged_view_map.map_permutation([1, 0]) is None


def test_extends_mapped_permutation_with_singletons() -> None:
    view_map = ViewMap.from_shapes([2, 2], [2, 1, 2])
    assert view_map.map_permutation([0, 1]) == [0, 1, 2]
    assert view_map.map_permutation([1, 0]) == [1, 2, 0]

    singleton_view_map = ViewMap.from_shapes([2], [1, 2])
    assert singleton_view_map.map_permutation([0]) == [0, 1]
    assert singleton_view_map.map_permutation_inverse([1, 0]) == [0]

    assert view_map.map_permutation([0, 0]) is None


def test_dim_map_uses_strict_no_mapping_for_singletons() -> None:
    view_map = ViewMap.from_shapes([1, 4], [4])

    assert view_map.is_valid_map
    assert view_map.map_dim(0) is None
    assert view_map.map_dim(1) == [0]
    assert view_map.map_dim_inverse(0) == [1]

    split_view_map = ViewMap.from_shapes([4], [2, 1, 2])
    assert split_view_map.map_dim(0) == [0, 2]
    assert split_view_map.map_dim_inverse(1) is None
    assert split_view_map.map_dim_inverse([0, 2]) == [0]


def test_dim_map_maps_reduced_singletons_only_when_unambiguous() -> None:
    split_singleton_view_map = ViewMap.from_shapes([1, 4], [1, 1, 4])
    assert split_singleton_view_map.map_dim(0) == [0, 1]

    squeezed_singleton_view_map = ViewMap.from_shapes([1, 50, 10, 1], [1, 50, 10])
    assert squeezed_singleton_view_map.map_dim(-1) is None
    assert squeezed_singleton_view_map.map_dim([0, -1]) == [0]


def test_dim_map_remaps_unit_slice_through_view() -> None:
    view_map = ViewMap.from_shapes([5, 2, 1, 4, 6], [5, 2, 4, 6])

    assert view_map.remap_unit_slice([5, 2, 3, 4, 6], 2, 0, 1) == (
        [5, 2, 12, 6],
        2,
        0,
        4,
    )
    assert view_map.remap_unit_slice([5, 2, 3, 4, 6], 2, 1, 2) == (
        [5, 2, 12, 6],
        2,
        4,
        8,
    )


def test_dim_map_remaps_unit_slice_through_flattening_view() -> None:
    view_map = ViewMap.from_shapes([5, 2, 1, 4, 6], [5, 2, 24])

    assert view_map.remap_unit_slice([5, 2, 3, 4, 6], 2, 1, 2) == (
        [5, 2, 72],
        2,
        24,
        48,
    )


def test_dim_map_does_not_remap_unit_slice_into_previous_axis() -> None:
    view_map = ViewMap.from_shapes([3, 3, 1], [3, 3])

    assert view_map.remap_unit_slice([3, 3, 3], 2, 0, 1) is None


def test_dim_map_preserves_symbolic_dimensions_as_prime_factors() -> None:
    shape_env = ShapeEnv()
    batch = _make_symint(shape_env, "batch", hint=4)

    view_map = ViewMap.from_shapes([batch, 6], [batch, 2, 3])

    assert view_map.is_valid_map
    assert view_map.map_dim(0) == [0]
    assert view_map.map_dim(1) == [1, 2]
    assert view_map.map_dim_inverse(0) == [0]


def test_normalize_view_shape_infers_concrete_dim_from_symints() -> None:
    shape_env = ShapeEnv()
    batch = _make_symint(shape_env, "batch", hint=4)

    normalized_shape = normalize_view_shape([batch, 6], [batch, -1])

    assert normalized_shape[0] is batch
    assert normalized_shape[1] == 6
    assert isinstance(normalized_shape[1], int)


def test_normalize_view_shape_preserves_symbolic_inferred_dim() -> None:
    shape_env = ShapeEnv()
    batch = _make_symint(shape_env, "batch", hint=4)

    normalized_shape = normalize_view_shape([2, batch], [-1])

    assert isinstance(normalized_shape[0], torch.SymInt)
    assert normalized_shape[0] is not batch
    assert sympy.simplify(normalized_shape[0].node.expr - 2 * batch.node.expr) == 0


def test_view_map_simplifies_constant_symint_dims() -> None:
    shape_env = ShapeEnv()
    batch = _make_symint(shape_env, "batch", hint=4)
    constant = batch * 6 // batch

    view_map = ViewMap.from_shapes([batch, constant], [batch, 2, 3])

    assert view_map.source_shape[1] == 6
    assert isinstance(view_map.source_shape[1], int)
    assert view_map.is_valid_map
    assert view_map.map_dim(1) == [1, 2]


def test_view_map_from_shapes_normalizes_symbolic_view_shape() -> None:
    shape_env = ShapeEnv()
    batch = _make_symint(shape_env, "batch", hint=4)

    view_map = ViewMap.from_shapes([batch, 6], [batch, -1])

    assert view_map.target_shape[0] is batch
    assert view_map.target_shape[1] == 6
    assert isinstance(view_map.target_shape[1], int)
    assert view_map.is_valid_map


def test_dim_map_permute_view_swap_preserves_symbolic_view_shape_dims() -> None:
    shape_env = ShapeEnv()
    batch = _make_symint(shape_env, "batch", hint=4)
    input_shape: list[_Dim] = [batch, 6]
    output_shape: list[_Dim] = [2, 3, batch]

    proposal = _propose_permute_view_swap(input_shape, [1, 0], output_shape)

    assert proposal is not None
    view_shape, permutation = proposal
    assert isinstance(view_shape[0], torch.SymInt)
    assert view_shape[0] is batch
    assert view_shape[1:] == [2, 3]
    assert permutation == [1, 2, 0]


def test_dim_map_view_permute_swap_preserves_symbolic_output_shape_dims() -> None:
    shape_env = ShapeEnv()
    batch = _make_symint(shape_env, "batch", hint=4)
    input_shape: list[_Dim] = [batch, 6]
    view_shape: list[_Dim] = [batch, 2, 3]

    proposal = _propose_view_permute_swap(input_shape, view_shape, [1, 2, 0])

    assert proposal is not None
    permutation, output_shape = proposal
    assert permutation == [1, 0]
    assert output_shape[:2] == [2, 3]
    assert isinstance(output_shape[2], torch.SymInt)
    assert output_shape[2] is batch


def test_dim_map_reduction_view_swap_preserves_symbolic_view_shape_dims() -> None:
    shape_env = ShapeEnv()
    batch = _make_symint(shape_env, "batch", hint=4)
    input_shape: list[_Dim] = [batch, 6]
    view_shape: list[_Dim] = [batch, 2, 3]

    proposal = _propose_reduction_view_swap(input_shape, [1], view_shape)

    assert proposal is not None
    view_shape, target_dims = proposal
    assert isinstance(view_shape[0], torch.SymInt)
    assert view_shape[0] is batch
    assert view_shape[1:] == [2, 3]
    assert target_dims == [1, 2]


def test_dim_map_view_reduction_swap_preserves_symbolic_output_shape_dims() -> None:
    shape_env = ShapeEnv()
    batch = _make_symint(shape_env, "batch", hint=4)
    input_shape: list[_Dim] = [batch, 6]
    view_shape: list[_Dim] = [batch, 2, 3]

    proposal = _propose_view_reduction_swap(input_shape, view_shape, [1, 2])

    assert proposal is not None
    source_dims, output_shape = proposal
    assert source_dims == [1]
    assert isinstance(output_shape[0], torch.SymInt)
    assert output_shape[0] is batch
    assert output_shape[1:] == [1, 1]


def test_permute_map_matches_docstring_reduction_identities() -> None:
    input_shape = [2, 3, 5]
    permutation = [2, 0, 1]
    permute_map = _permute_map(permutation)
    x = _tensor(input_shape)

    source_dims = [0, 2]
    target_dims = permute_map.map_dims(source_dims)
    assert target_dims == [1, 0]
    assert _same(
        _reduce(x, source_dims).permute(permutation),
        _reduce(x.permute(permutation), target_dims),
    )

    target_dims = [0, 2]
    source_dims = permute_map.map_dims_inverse(target_dims)
    assert source_dims == [2, 1]
    assert _same(
        _reduce(x.permute(permutation), target_dims),
        _reduce(x, source_dims).permute(permutation),
    )


def test_dim_map_randomized_permute_view_swaps_match_bruteforce() -> None:
    accepted = 0
    rejected = 0

    for _ in range(80):
        input_shape = _random_shape(_randint(1, 4), max_dim=3)
        permutation = list(range(len(input_shape)))
        _shuffle(permutation)
        output_shape = _random_view_shape(_numel(input_shape), max_rank=4)
        x = _tensor(input_shape)

        proposal = _propose_permute_view_swap(input_shape, permutation, output_shape)
        brute_force_swaps = _bruteforce_permute_view_swaps(x, permutation, output_shape)
        if proposal is None and brute_force_swaps:
            proposal = brute_force_swaps[0]

        if proposal is None:
            rejected += 1
            assert brute_force_swaps == []
            continue

        accepted += 1
        assert proposal in brute_force_swaps
        view_shape, new_permutation = proposal
        original = x.permute(permutation).reshape(output_shape)
        candidate = x.reshape(view_shape).permute(new_permutation)
        assert _same(original, candidate)


def test_dim_map_randomized_view_permute_swaps_match_bruteforce() -> None:
    accepted = 0
    rejected = 0

    for _ in range(80):
        input_shape = _random_shape(_randint(1, 4), max_dim=3)
        view_shape = _random_view_shape(_numel(input_shape), max_rank=4)
        permutation = list(range(len(view_shape)))
        _shuffle(permutation)
        x = _tensor(input_shape)

        proposal = _propose_view_permute_swap(input_shape, view_shape, permutation)
        brute_force_swaps = _bruteforce_view_permute_swaps(x, view_shape, permutation)
        if proposal is None and brute_force_swaps:
            proposal = brute_force_swaps[0]

        if proposal is None:
            rejected += 1
            assert brute_force_swaps == []
            continue

        accepted += 1
        assert proposal in brute_force_swaps
        new_permutation, output_shape = proposal
        original = x.reshape(view_shape).permute(permutation)
        candidate = x.permute(new_permutation).reshape(output_shape)
        assert _same(original, candidate)


def test_dim_map_randomized_reduction_view_swaps_match_bruteforce() -> None:
    accepted = 0
    rejected = 0

    for _ in range(80):
        input_shape = _random_shape(_randint(1, 4), max_dim=3)
        source_dims = _choice(_all_dim_subsets(len(input_shape)))
        view_shape = _random_view_shape(_numel(input_shape), max_rank=4)
        x = _tensor(input_shape)

        proposal = _propose_reduction_view_swap(input_shape, source_dims, view_shape)
        brute_force_swaps = _bruteforce_reduction_view_swaps(x, source_dims, view_shape)
        if proposal is None and brute_force_swaps:
            proposal = brute_force_swaps[0]

        if proposal is None:
            rejected += 1
            assert brute_force_swaps == []
            continue

        accepted += 1
        assert proposal in brute_force_swaps
        new_shape, target_dims = proposal
        output_shape = _reduce_shape(new_shape, target_dims)
        original = _reduce(x, source_dims).reshape(output_shape)
        candidate = _reduce(x.reshape(new_shape), target_dims)
        assert _same(original, candidate)


def test_dim_map_randomized_view_reduction_swaps_match_bruteforce() -> None:
    accepted = 0
    rejected = 0

    for _ in range(80):
        input_shape = _random_shape(_randint(1, 4), max_dim=3)
        view_shape = _random_view_shape(_numel(input_shape), max_rank=4)
        target_dims = _choice(_all_dim_subsets(len(view_shape)))
        x = _tensor(input_shape)

        proposal = _propose_view_reduction_swap(input_shape, view_shape, target_dims)
        brute_force_swaps = _bruteforce_view_reduction_swaps(x, view_shape, target_dims)
        if proposal is None and brute_force_swaps:
            proposal = brute_force_swaps[0]

        if proposal is None:
            rejected += 1
            assert brute_force_swaps == []
            continue

        accepted += 1
        assert proposal in brute_force_swaps
        source_dims, output_shape = proposal
        original = _reduce(x.reshape(view_shape), target_dims)
        candidate = _reduce(x, source_dims).reshape(output_shape)
        assert _same(original, candidate)


def test_permute_map_randomized_reduction_permute_swaps_match_bruteforce() -> None:
    for _ in range(80):
        input_shape = _random_shape(_randint(1, 4), max_dim=3)
        source_dims = _choice(_all_dim_subsets(len(input_shape)))
        permutation = list(range(len(input_shape)))
        _shuffle(permutation)
        permute_map = _permute_map(permutation)
        target_dims = permute_map.map_dims(source_dims)
        x = _tensor(input_shape)

        original = _reduce(x, source_dims).permute(permutation)
        candidate = _reduce(x.permute(permutation), target_dims)
        assert _same(original, candidate)

        brute_force_dims = [
            dims
            for dims in _all_dim_subsets(len(input_shape))
            if _same(original, _reduce(x.permute(permutation), dims))
        ]
        assert sorted(target_dims) in brute_force_dims


def test_permute_map_randomized_permute_reduction_swaps_match_bruteforce() -> None:
    for _ in range(80):
        input_shape = _random_shape(_randint(1, 4), max_dim=3)
        permutation = list(range(len(input_shape)))
        _shuffle(permutation)
        target_dims = _choice(_all_dim_subsets(len(input_shape)))
        permute_map = _permute_map(permutation)
        source_dims = permute_map.map_dims_inverse(target_dims)
        x = _tensor(input_shape)

        original = _reduce(x.permute(permutation), target_dims)
        candidate = _reduce(x, source_dims).permute(permutation)
        assert _same(original, candidate)

        brute_force_dims = [
            dims
            for dims in _all_dim_subsets(len(input_shape))
            if _same(original, _reduce(x, dims).permute(permutation))
        ]
        assert sorted(source_dims) in brute_force_dims
