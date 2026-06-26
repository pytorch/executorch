# Copyright 2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from collections import Counter, defaultdict, deque
from dataclasses import dataclass
from typing import cast, Iterable, Sequence

import sympy  # type: ignore[import-untyped]
import torch

from executorch.exir.dialects._ops import ops as exir_ops
from torch.fx import Node

_Dim = int | torch.SymInt
_FactorKey = tuple[str, int | str]


@dataclass(frozen=True)
class _Factor:
    key: _FactorKey
    axis: int


@dataclass
class _ViewGroups:
    source_axis_to_groups: list[list[int]]
    target_axis_to_groups: list[list[int]]
    group_to_source_axes: dict[int, list[int]]
    group_to_target_axes: dict[int, list[int]]


def _is_permutation(dims: Sequence[int], rank: int) -> bool:
    return sorted(dims) == list(range(rank))


def _normalize_dim(dim: int, rank: int) -> int:
    normalized = dim if dim >= 0 else dim + rank
    assert 0 <= normalized < rank, f"Invalid dim {dim} for rank {rank}"
    return normalized


def _normalize_dims(dims: int | Sequence[int], rank: int) -> list[int]:
    if isinstance(dims, int):
        return [_normalize_dim(dims, rank)]
    return [_normalize_dim(dim, rank) for dim in dims]


def _normalize_permutation(dims: Sequence[int], rank: int) -> list[int] | None:
    if len(dims) != rank:
        return None
    try:
        normalized = [_normalize_dim(dim, rank) for dim in dims]
    except AssertionError:
        return None
    return normalized if _is_permutation(normalized, rank) else None


def _extend_permutation_with_singletons(
    dims: Sequence[int], shape: Sequence[_Dim]
) -> list[int] | None:
    """Extend a partial permutation with missing singleton axes."""
    try:
        extended_dims = _normalize_dims(dims, len(shape))
    except AssertionError:
        return None
    if len(set(extended_dims)) != len(extended_dims):
        return None

    missing_dims = [dim for dim in range(len(shape)) if dim not in set(extended_dims)]
    if any(not _dim_equals(shape[dim], 1) for dim in missing_dims):
        return None

    for dim in reversed(missing_dims):
        insert_at = next(
            (
                index
                for index, existing_dim in enumerate(extended_dims)
                if existing_dim > dim
            ),
            len(extended_dims),
        )
        extended_dims.insert(insert_at, dim)
    return extended_dims if _is_permutation(extended_dims, len(shape)) else None


def _dim_expr(dim: _Dim) -> sympy.Basic:
    return sympy.Integer(dim) if isinstance(dim, int) else dim.node.expr


def _simplify_dim(dim: _Dim) -> _Dim:
    if isinstance(dim, int):
        return dim

    maybe_int = dim.node.maybe_as_int()
    return maybe_int if maybe_int is not None else dim


def _simplify_shape(shape: Iterable[_Dim]) -> list[_Dim]:
    return [_simplify_dim(dim) for dim in shape]


def _dim_equals(lhs: _Dim, rhs: _Dim) -> bool:
    lhs = _simplify_dim(lhs)
    rhs = _simplify_dim(rhs)
    if isinstance(lhs, int) and isinstance(rhs, int):
        return lhs == rhs
    return sympy.simplify(_dim_expr(lhs) - _dim_expr(rhs)) == 0


def _factor_int(dim: int) -> list[_FactorKey] | None:
    if dim < 1:
        return None
    factors: list[_FactorKey] = []
    divisor = 2
    while divisor * divisor <= dim:
        while dim % divisor == 0:
            factors.append(("int", divisor))
            dim //= divisor
        divisor += 1 if divisor == 2 else 2
    if dim > 1:
        factors.append(("int", dim))
    return factors


def _factor_dim(dim: _Dim) -> list[_FactorKey] | None:
    dim = _simplify_dim(dim)
    if _dim_equals(dim, 1):
        return []
    if isinstance(dim, int):
        return _factor_int(dim)
    return [("sym", sympy.srepr(_dim_expr(dim)))]


def _factor_shape(shape: Sequence[_Dim]) -> list[_Factor] | None:
    factors: list[_Factor] = []
    for axis, dim in enumerate(shape):
        dim_factors = _factor_dim(dim)
        if dim_factors is None:
            return None
        factors.extend(_Factor(factor, axis) for factor in dim_factors)
    return factors


def _dedupe(items: Iterable[int]) -> list[int]:
    deduped: list[int] = []
    seen: set[int] = set()
    for item in items:
        if item not in seen:
            deduped.append(item)
            seen.add(item)
    return deduped


def numel(shape: Iterable[_Dim]) -> _Dim:
    numel: _Dim = 1
    for dim in shape:
        numel = _simplify_dim(numel * _simplify_dim(dim))
    return numel


def same_numel(first_shape: Iterable[_Dim], second_shape: Iterable[_Dim]) -> bool:
    return _dim_equals(numel(first_shape), numel(second_shape))


def normalize_view_shape(
    source_shape: Sequence[_Dim], target_shape: Sequence[_Dim]
) -> list[_Dim]:
    """Normalize a view shape with <=1 unknown dim, indicated by -1."""
    source_shape = _simplify_shape(source_shape)
    normalized_shape = _simplify_shape(target_shape)
    inferred_dims = [
        index for index, dim in enumerate(normalized_shape) if _dim_equals(dim, -1)
    ]
    if not inferred_dims:
        return normalized_shape

    assert len(inferred_dims) == 1, f"Invalid view shape {target_shape}"
    inferred_dim = inferred_dims[0]
    known_shape = [
        dim for index, dim in enumerate(normalized_shape) if index != inferred_dim
    ]
    source_numel = numel(source_shape)
    known_numel = numel(known_shape)
    normalized_shape[inferred_dim] = _simplify_dim(
        source_numel if _dim_equals(known_numel, 1) else source_numel // known_numel
    )
    return normalized_shape


class _UnionFind:
    def __init__(self, size: int) -> None:
        self.parents = list(range(size))

    def find(self, item: int) -> int:
        parent = self.parents[item]
        if parent != item:
            self.parents[item] = self.find(parent)
        return self.parents[item]

    def union(self, first: int, second: int) -> None:
        first_root = self.find(first)
        second_root = self.find(second)
        if first_root != second_root:
            self.parents[second_root] = first_root


class ViewMap:
    """Maps dims before and after a view operator.

    The map models a view by expanding both shapes into ordered prime-factor
    streams and finding the permutation between them. Singleton dims are not counted.
    For example, the view from [4, 3, 10] to [2, 2, 1, 5, 3, 2] is represented as:

        Source:        [4, 3, 10]
        Source primes: [2, 2, 3, 2, 5]
        Permutation:   [0, 1, 4, 2, 3]
        Target primes: [2, 2, 5, 3, 2]
        Target:        [2, 2, 1, 5, 3, 2]

    Dim mappings are derived by unioning factors into groups where
    - factors from the same source axis belong to the same group;
       e.g. [2,2] [3] [2,5]
    - factors from the same target axis belong to the same group;
       e.g. [2] [2] [5] [3] [2]
    - factors whose source-to-target permutation order crosses belong to the same group.
       e.g. [2] [2] [3,5,2]

    The final groups are formed by the union of all conditions, in this case
    [2, 2] and [3, 5, 2]. A source dim maps to all target dims that share its
    group with any of its factors, and vice versa.

    Additional conditions apply for the map being valid depending on if the mapped dim
    is a reduction operator or a permutation operator, as described in the respective methods.

    SymInts are partially supported by factorizing them as single primes as the true
    value is not known, causing potentially fewer valid mappings.

    """

    def __init__(self, view_node: Node) -> None:
        """Build a view map from an FX view_copy node."""
        input_node = view_node.args[0]
        assert isinstance(input_node, Node) and (
            view_node.target == exir_ops.edge.aten.view_copy.default
        )
        input_val = input_node.meta["val"]
        assert isinstance(input_val, torch.Tensor)

        self.source_shape = _simplify_shape(cast(Sequence[_Dim], input_val.shape))
        self.target_shape = normalize_view_shape(
            self.source_shape, cast(Sequence[_Dim], view_node.args[1])
        )
        self._groups = self._build_groups(self.source_shape, self.target_shape)

    @classmethod
    def from_shapes(
        cls, source_shape: Sequence[_Dim], target_shape: Sequence[_Dim]
    ) -> ViewMap:
        """Build a view map directly from source and target shapes."""
        view_map = cls.__new__(cls)
        view_map.source_shape = _simplify_shape(source_shape)
        view_map.target_shape = normalize_view_shape(
            view_map.source_shape, target_shape
        )
        view_map._groups = cls._build_groups(
            view_map.source_shape, view_map.target_shape
        )
        return view_map

    @property
    def is_valid_map(self) -> bool:
        """Return whether the shapes can be represented by grouped factors."""
        return self._groups is not None

    @property
    def source_rank(self) -> int:
        """Return the source shape rank."""
        return len(self.source_shape)

    @property
    def target_rank(self) -> int:
        """Return the target shape rank."""
        return len(self.target_shape)

    def map_dim(
        self,
        source_dims: int | Sequence[int],
    ) -> list[int] | None:
        """Map source reduction dims (e.g. `x.sum(dim)`, `x.max(dim)`) to valid target
        reduction dims:

            x.op(dims).view(S) == x.view(S').op(mapped_dims)

        Reduction dims are valid only when the selected dims and mapped dims both cover
        complete groups. E.g. in the example view [4, 3, 10] -> [2, 2, 1, 5, 3, 2] the
        valid maps are.
            [0] <=> [0, 1] and [1, 2] <=> [3, 4, 5]
        """
        try:
            normalized_dims = _normalize_dims(source_dims, self.source_rank)
        except AssertionError:
            return None

        groups = self._valid_groups()
        if not self._is_valid_reduction(normalized_dims, groups.source_axis_to_groups):
            return None

        target_dims = self._map_dims(
            normalized_dims,
            groups.source_axis_to_groups,
            groups.group_to_target_axes,
        )
        if not target_dims or not self._is_valid_reduction(
            target_dims, groups.target_axis_to_groups
        ):
            return None
        return target_dims

    def map_dim_inverse(
        self,
        target_dims: int | Sequence[int],
    ) -> list[int] | None:
        """Map target reduction dims to valid source reduction dims, inverse map
        of map_dim.

        x.view(S).op(dims) == x.op(mapped_dims).view(S')

        """
        try:
            normalized_dims = _normalize_dims(target_dims, self.target_rank)
        except AssertionError:
            return None

        groups = self._valid_groups()
        if not self._is_valid_reduction(normalized_dims, groups.target_axis_to_groups):
            return None

        source_dims = self._map_dims(
            normalized_dims,
            groups.target_axis_to_groups,
            groups.group_to_source_axes,
        )
        if not source_dims or not self._is_valid_reduction(
            source_dims, groups.source_axis_to_groups
        ):
            return None
        return source_dims

    def map_permutation(
        self,
        source_permutation: Sequence[int],
    ) -> list[int] | None:
        """Map a source permutation to a valid target permutation.

        Permutation dims have an additional constraint on the order of dims:
        Dims are valid only when dims mapped through one group appear as contiguous
        increasing blocks dims in both source and target.

        In the example view [4, 3, 10] -> [2, 2, 1, 5, 3, 2], [1, 2, 0] is a valid
        permutation, but [2, 0, 1] and [0, 2, 1] are not since [1, 2] belong to the same
        group but are not a) contiguous, or b) in increasing order.

        """
        source_permutation = _normalize_permutation(
            source_permutation, self.source_rank
        )
        if source_permutation is None:
            return None

        groups = self._valid_groups()
        target_permutation = _extend_permutation_with_singletons(
            self._map_dims(
                source_permutation,
                groups.source_axis_to_groups,
                groups.group_to_target_axes,
            ),
            self.target_shape,
        )
        if target_permutation is None:
            return None

        return (
            target_permutation
            if self._matching_permuted_group_blocks(
                source_permutation,
                target_permutation,
                groups.source_axis_to_groups,
                groups.target_axis_to_groups,
            )
            else None
        )

    def map_permutation_inverse(
        self,
        target_permutation: Sequence[int],
    ) -> list[int] | None:
        """Map a target permutation to a valid source permutation.

        Inverse of map_permutation.

        """
        target_permutation = _normalize_permutation(
            target_permutation, self.target_rank
        )
        if target_permutation is None:
            return None

        groups = self._valid_groups()
        source_permutation = _extend_permutation_with_singletons(
            self._map_dims(
                target_permutation,
                groups.target_axis_to_groups,
                groups.group_to_source_axes,
            ),
            self.source_shape,
        )
        if source_permutation is None:
            return None

        return (
            source_permutation
            if self._matching_permuted_group_blocks(
                source_permutation,
                target_permutation,
                groups.source_axis_to_groups,
                groups.target_axis_to_groups,
            )
            else None
        )

    @staticmethod
    def _map_dims(
        source_dims: Iterable[int],
        source_axis_to_groups: Sequence[Sequence[int]],
        group_to_target_axes: dict[int, list[int]],
    ) -> list[int]:
        return _dedupe(
            target_axis
            for source_axis in source_dims
            for group in source_axis_to_groups[source_axis]
            for target_axis in group_to_target_axes[group]
        )

    @staticmethod
    def _matching_permuted_group_blocks(
        source_permutation: Sequence[int],
        target_permutation: Sequence[int],
        source_axis_to_groups: Sequence[Sequence[int]],
        target_axis_to_groups: Sequence[Sequence[int]],
    ) -> bool:
        """Return whether source and target permutations consume groups
        equally.
        """
        closed_groups: set[int] = set()
        source_index = 0
        target_index = 0

        while True:
            source_index, source_group = ViewMap._next_group(
                source_permutation, source_axis_to_groups, source_index
            )
            target_index, target_group = ViewMap._next_group(
                target_permutation, target_axis_to_groups, target_index
            )

            if source_group is None or target_group is None:
                return source_group is None and target_group is None
            if source_group != target_group or source_group in closed_groups:
                return False

            source_index, source_axes = ViewMap._consume_group(
                source_permutation,
                source_axis_to_groups,
                source_index,
                source_group,
            )
            target_index, target_axes = ViewMap._consume_group(
                target_permutation,
                target_axis_to_groups,
                target_index,
                target_group,
            )
            if source_axes != sorted(source_axes) or target_axes != sorted(target_axes):
                return False

            closed_groups.add(source_group)

    @staticmethod
    def _next_group(
        permutation: Sequence[int],
        axis_to_groups: Sequence[Sequence[int]],
        index: int,
    ) -> tuple[int, int | None]:
        """Return the next grouped axis index and group, skipping singletons."""
        while index < len(permutation):
            axis = permutation[index]
            axis_groups = axis_to_groups[axis]
            if not axis_groups:
                index += 1
                continue
            assert len(axis_groups) == 1
            return index, axis_groups[0]
        return index, None

    @staticmethod
    def _consume_group(
        permutation: Sequence[int],
        axis_to_groups: Sequence[Sequence[int]],
        index: int,
        group: int,
    ) -> tuple[int, list[int]]:
        """Consume one group block, ignoring singleton axes."""
        axes: list[int] = []
        while index < len(permutation):
            axis = permutation[index]
            axis_groups = axis_to_groups[axis]
            if not axis_groups:
                index += 1
                continue
            assert len(axis_groups) == 1
            if axis_groups[0] != group:
                break
            axes.append(axis)
            index += 1
        return index, axes

    @staticmethod
    def _is_valid_reduction(
        normalized_dims: Iterable[int],
        axis_to_groups: Sequence[Sequence[int]],
    ) -> bool:
        """Return whether dims cover every selected group in one shape."""
        normalized_dims = set(normalized_dims)
        if not normalized_dims:
            return False

        group_to_axes: dict[int, set[int]] = defaultdict(set)
        selected_groups: set[int] = set()
        for axis, groups in enumerate(axis_to_groups):
            for group in groups:
                group_to_axes[group].add(axis)
                if axis in normalized_dims:
                    selected_groups.add(group)

        if any(not axis_to_groups[axis] for axis in normalized_dims):
            return False

        return all(
            group_to_axes[group].issubset(normalized_dims) for group in selected_groups
        )

    @classmethod
    def _build_groups(
        cls, source_shape: Sequence[_Dim], target_shape: Sequence[_Dim]
    ) -> _ViewGroups | None:
        """Build source/target axis groups from ordered prime factors."""

        # Compute ordered prime factorizations of input and output shapes
        source_factors = _factor_shape(source_shape)
        target_factors = _factor_shape(target_shape)
        if (
            source_factors is None
            or target_factors is None
            or Counter(factor.key for factor in source_factors)
            != Counter(factor.key for factor in target_factors)
        ):
            return None
        source_factors = source_factors
        target_factors = target_factors

        # Compute prime factor permutation between input and output shapes
        factor_count = len(source_factors)
        permutation = cls._find_permutation(source_factors, target_factors)
        if permutation is None:
            return None
        # Find groups of factors that must be mapped together to preserve view equivalence
        union_find = _UnionFind(factor_count)
        cls._union_factors_sharing_axes(
            union_find, (factor.axis for factor in source_factors)
        )

        cls._union_factors_sharing_axes(
            union_find,
            (
                target_factors[permutation[source_position]].axis
                for source_position in range(factor_count)
            ),
        )

        cls._union_crossing_factors(union_find, permutation)

        # Create group data structure
        source_axis_groups: list[set[int]] = [set() for _ in source_shape]
        target_axis_groups: list[set[int]] = [set() for _ in target_shape]
        group_to_source_axes: dict[int, set[int]] = defaultdict(set)
        group_to_target_axes: dict[int, set[int]] = defaultdict(set)

        for source_position, source_factor in enumerate(source_factors):
            group = union_find.find(source_position)
            target_factor = target_factors[permutation[source_position]]

            source_axis_groups[source_factor.axis].add(group)
            target_axis_groups[target_factor.axis].add(group)
            group_to_source_axes[group].add(source_factor.axis)
            group_to_target_axes[group].add(target_factor.axis)

        return _ViewGroups(
            source_axis_to_groups=[sorted(groups) for groups in source_axis_groups],
            target_axis_to_groups=[sorted(groups) for groups in target_axis_groups],
            group_to_source_axes={
                group: sorted(axes) for group, axes in group_to_source_axes.items()
            },
            group_to_target_axes={
                group: sorted(axes) for group, axes in group_to_target_axes.items()
            },
        )

    @staticmethod
    def _find_permutation(
        X: Sequence[_Factor], Y: Sequence[_Factor]
    ) -> list[int] | None:
        """Computes the permutation from X -> Y, handling duplicates."""
        duplicates: dict[_FactorKey, deque[int]] = defaultdict(deque)
        for i, y in enumerate(Y):
            duplicates[y.key].append(i)

        permutation: list[int] = []
        for x in X:
            positions = duplicates[x.key]
            if not positions:
                return None
            permutation.append(positions.popleft())

        return permutation

    @staticmethod
    def _union_factors_sharing_axes(
        union_find: _UnionFind, axes: Iterable[int]
    ) -> None:
        """Union factor positions that belong to the same axis."""
        first_position_by_axis: dict[int, int] = {}
        for position, axis in enumerate(axes):
            if axis in first_position_by_axis:
                union_find.union(first_position_by_axis[axis], position)
            else:
                first_position_by_axis[axis] = position

    @staticmethod
    def _union_crossing_factors(
        union_find: _UnionFind, permutation: Sequence[int]
    ) -> None:
        """Union factor positions whose target ordering crosses."""
        for first in range(len(permutation)):
            for second in range(first + 1, len(permutation)):
                if permutation[first] > permutation[second]:
                    union_find.union(first, second)

    def _valid_groups(self) -> _ViewGroups:
        """Return built groups for a valid map."""
        assert self._groups is not None
        return self._groups


class PermuteMap:
    """Maps dims to equivalent dims before and after a permute."""

    def __init__(self, permute_node: Node) -> None:
        permute_dims = permute_node.args[1]
        assert isinstance(permute_dims, Sequence) and not isinstance(
            permute_dims, (str, bytes)
        )
        normalized = _normalize_permutation(
            cast(Sequence[int], permute_dims), len(cast(Sequence[int], permute_dims))
        )
        if normalized is None:
            raise ValueError(f"Invalid permute dims: {permute_dims}")
        self.permute_dims = normalized

    def map_dims(self, dims: int | Sequence[int]) -> list[int]:
        """Computes mapped dims s.t.

        x.op(dims).permute(P) == x.permute(P).op(mapped_dims)

        """
        normalized_dims = _normalize_dims(dims, len(self.permute_dims))
        inverse_permute = [0] * len(self.permute_dims)
        for target_dim, source_dim in enumerate(self.permute_dims):
            inverse_permute[source_dim] = target_dim
        return [inverse_permute[dim] for dim in normalized_dims]

    def map_dims_inverse(self, dims: int | Sequence[int]) -> list[int]:
        """Computes mapped dims s.t.

        x.permute(P).op(dims) == x.op(mapped_dims).permute(P)

        """
        normalized_dims = _normalize_dims(dims, len(self.permute_dims))
        return [self.permute_dims[dim] for dim in normalized_dims]
