# Copyright 2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from typing import cast, Iterable, Sequence

import sympy  # type: ignore[import-untyped]
import torch

from executorch.exir.dialects._ops import ops as exir_ops
from torch.fx import Node

_Dim = int | torch.SymInt
_Piece = tuple[int, int]


class ViewMap:
    """Map dimensions across a view."""

    def __init__(self, view_node: Node) -> None:
        input_node = view_node.args[0]
        assert isinstance(input_node, Node) and (
            view_node.target == exir_ops.edge.aten.view_copy.default
        )
        input_val = input_node.meta["val"]
        assert isinstance(input_val, torch.Tensor)

        self.source_shape = cast(list[_Dim], list(input_val.shape))
        self.target_shape = list(cast(Sequence[_Dim], view_node.args[1]))
        self.source_to_target_pieces = self._build_piece_map(
            self.source_shape, self.target_shape
        )

    @classmethod
    def from_shapes(
        cls, source_shape: Sequence[_Dim], target_shape: Sequence[_Dim]
    ) -> ViewMap:
        view_map = cls.__new__(cls)
        view_map.source_shape = list(source_shape)
        view_map.target_shape = list(target_shape)
        view_map.source_to_target_pieces = cls._build_piece_map(
            view_map.source_shape, view_map.target_shape
        )
        return view_map

    @property
    def is_valid_map(self) -> bool:
        return self.source_to_target_pieces is not None

    @property
    def source_rank(self) -> int:
        return len(self.source_shape)

    @property
    def target_rank(self) -> int:
        return len(self.target_shape)

    def map_source_to_target(
        self,
        source_dims: int | Sequence[int],
        *,
        include_unbacked_singletons: bool = False,
    ) -> list[int]:
        """Map source dims to target dims."""
        normalized_dims = self._normalize_dims(source_dims, self.source_rank)
        source_to_target_pieces = self._valid_source_to_target_pieces()

        mapped_dims = [
            target_axis
            for dim in normalized_dims
            for target_axis, _ in source_to_target_pieces[dim]
        ]

        mapped_dims_unique: list[int] = []
        for dim in mapped_dims:
            if dim not in mapped_dims_unique:
                mapped_dims_unique.append(dim)
        if include_unbacked_singletons:
            mapped_dims_unique = [
                target_axis
                for target_axis, target_dim in enumerate(self.target_shape)
                if target_axis not in mapped_dims_unique
                and self._dim_equals(target_dim, 1)
            ] + mapped_dims_unique
        return mapped_dims_unique

    def map_target_to_source(
        self,
        target_dims: int | Sequence[int],
        *,
        include_unbacked_singletons: bool = False,
    ) -> list[int]:
        """Map target dims to source dims."""
        normalized_dims = self._normalize_dims(target_dims, self.target_rank)
        source_to_target_pieces = self._valid_source_to_target_pieces()

        mapped_dims = [
            source_axis
            for dim in normalized_dims
            for source_axis, target_pieces in enumerate(source_to_target_pieces)
            if any(target_axis == dim for target_axis, _ in target_pieces)
        ]

        if include_unbacked_singletons:
            mapped_dims_unique: list[int] = []
            for dim in mapped_dims:
                if dim not in mapped_dims_unique:
                    mapped_dims_unique.append(dim)

            mapped_dims = mapped_dims_unique
            seen_dims = set(mapped_dims)
            for source_dim, source_dim_size in reversed(
                list(enumerate(self.source_shape))
            ):
                if source_dim not in seen_dims and self._dim_equals(source_dim_size, 1):
                    insert_at = next(
                        (
                            index
                            for index, dim in enumerate(mapped_dims)
                            if dim > source_dim
                        ),
                        len(mapped_dims),
                    )
                    mapped_dims.insert(insert_at, source_dim)
                    seen_dims.add(source_dim)

        return mapped_dims

    def validate_mapped_permute(
        self,
        input_permute_dims: Sequence[int],
        permute_dims: Sequence[int],
    ) -> bool:
        source_permute = self._normalize_dims(input_permute_dims, self.source_rank)
        target_permute = self._normalize_dims(permute_dims, self.target_rank)
        if not self._is_permutation(source_permute, self.source_rank):
            return False
        if len(set(target_permute)) != len(target_permute) or any(
            dim < 0 or dim >= self.target_rank for dim in target_permute
        ):
            return False

        target_to_source_pieces = self._target_to_source_pieces()
        required_target_axes = {
            target_axis
            for target_axis, pieces in enumerate(target_to_source_pieces)
            if any(
                not self._dim_equals(self.source_shape[source_axis], 1)
                for source_axis, _ in pieces
            )
        }
        if not required_target_axes.issubset(target_permute):
            return False

        permuted_pieces = [
            piece
            for target_axis in target_permute
            for piece in target_to_source_pieces[target_axis]
            if not self._dim_equals(self.source_shape[piece[0]], 1)
        ]

        source_axes: list[int] = []
        seen_axes: set[int] = set()
        index = 0

        while index < len(permuted_pieces):
            source_axis = permuted_pieces[index][0]
            parts: list[int] = []
            while (
                index < len(permuted_pieces)
                and permuted_pieces[index][0] == source_axis
            ):
                parts.append(permuted_pieces[index][1])
                index += 1

            if source_axis in seen_axes or parts != list(range(len(parts))):
                return False
            seen_axes.add(source_axis)
            source_axes.append(source_axis)

        return source_axes == [
            dim
            for dim in source_permute
            if not self._dim_equals(self.source_shape[dim], 1)
        ]

    def validate_mapped_reduction(self, dims: Iterable[int]) -> bool:
        target_dims = sorted(set(self._normalize_dims(list(dims), self.target_rank)))
        if not target_dims:
            return False
        return target_dims == list(range(target_dims[0], target_dims[-1] + 1))

    def validate_mapped_source_reduction(self, dims: Iterable[int]) -> bool:
        source_dims = sorted(set(self._normalize_dims(list(dims), self.source_rank)))
        if not source_dims:
            return False
        return source_dims == list(range(source_dims[0], source_dims[-1] + 1))

    def source_to_single_target_permutation(self) -> list[int] | None:
        mapped_dims = []
        for source_axis in range(self.source_rank):
            target_axes = self.map_source_to_target(source_axis)
            if len(target_axes) != 1:
                return None
            mapped_dims.append(target_axes[0])
        return (
            mapped_dims if self._is_permutation(mapped_dims, len(mapped_dims)) else None
        )

    def target_to_single_source_permutation(self) -> list[int] | None:
        mapped_dims = []
        for target_axis in range(self.target_rank):
            source_axes = self.map_target_to_source(target_axis)
            if len(source_axes) != 1:
                return None
            mapped_dims.append(source_axes[0])
        return (
            mapped_dims if self._is_permutation(mapped_dims, len(mapped_dims)) else None
        )

    def target_shape_for_source_shape(
        self, source_shape: Sequence[_Dim]
    ) -> list[_Dim] | None:
        if len(source_shape) != self.source_rank:
            return None

        target_shape: list[_Dim] = [1] * self.target_rank
        handled_target_axes = self._fill_coalesced_target_axes(
            source_shape, target_shape
        )

        for source_axis in range(self.source_rank):
            if self._fill_target_axes_for_source_axis(
                source_axis, source_shape, target_shape, handled_target_axes
            ):
                continue
            return None

        return target_shape if self.same_numel(source_shape, target_shape) else None

    def _fill_coalesced_target_axes(
        self,
        source_shape: Sequence[_Dim],
        target_shape: list[_Dim],
    ) -> set[int]:
        handled_target_axes: set[int] = set()
        for target_axis in range(self.target_rank):
            source_axes = self.map_target_to_source(target_axis)
            if len(source_axes) > 1:
                target_shape[target_axis] = self.numel(
                    source_shape[source_axis] for source_axis in source_axes
                )
                handled_target_axes.add(target_axis)
        return handled_target_axes

    def _fill_target_axes_for_source_axis(
        self,
        source_axis: int,
        source_shape: Sequence[_Dim],
        target_shape: list[_Dim],
        handled_target_axes: set[int],
    ) -> bool:
        target_axes = self.map_source_to_target(source_axis)
        if not target_axes:
            return True
        if any(target_axis in handled_target_axes for target_axis in target_axes):
            return True
        if len(target_axes) == 1:
            target_shape[target_axes[0]] = source_shape[source_axis]
            return True

        target_dims = [self.target_shape[target_axis] for target_axis in target_axes]
        if self._dim_equals(source_shape[source_axis], self.source_shape[source_axis]):
            self._set_target_dims(target_shape, target_axes, target_dims)
            return True
        if self._dim_equals(self.numel(target_dims), 1):
            target_shape[target_axes[0]] = source_shape[source_axis]
            return True
        if self._dim_equals(self.numel(target_dims), self.source_shape[source_axis]):
            self._set_target_dims(target_shape, target_axes, target_dims)
            return True
        return False

    @staticmethod
    def _set_target_dims(
        target_shape: list[_Dim],
        target_axes: Sequence[int],
        target_dims: Sequence[_Dim],
    ) -> None:
        for target_axis, target_dim in zip(target_axes, target_dims):
            target_shape[target_axis] = target_dim

    @classmethod
    def _build_piece_map(  # noqa: C901
        cls, source_shape: Sequence[_Dim], target_shape: Sequence[_Dim]
    ) -> list[list[_Piece]] | None:
        """Map each source axis to the target-axis pieces it becomes.

        Each piece is represented as (target_axis, source_axis_part). The part
        index lets the caller reject permutations that reorder pieces split out
        of the same source axis.

        E.g. a view [4, 3, 2] -> [2, 2, 6] is mapped as [[(0, 0), (1, 1)], [(2,
        0)], [(2, 0)]].

        """
        source_axes = [
            axis for axis, dim in enumerate(source_shape) if not cls._dim_equals(dim, 1)
        ]
        target_axes = [
            axis for axis, dim in enumerate(target_shape) if not cls._dim_equals(dim, 1)
        ]

        piece_map: list[list[_Piece]] = [[] for _ in source_shape]
        if not source_axes or not target_axes:
            if source_axes or target_axes:
                return None
            cls._add_singleton_pieces(piece_map, source_shape, target_shape)
            return piece_map

        source_index = 0
        target_index = 0
        source_part = 0
        source_axis = source_axes[source_index]
        target_axis = target_axes[target_index]
        source_remaining = source_shape[source_axis]
        target_remaining = target_shape[target_axis]

        while source_index < len(source_axes) and target_index < len(target_axes):
            piece_map[source_axis].append((target_axis, source_part))

            if cls._dim_equals(source_remaining, target_remaining):
                source_index += 1
                target_index += 1
                source_part = 0
                if source_index < len(source_axes):
                    source_axis = source_axes[source_index]
                    source_remaining = source_shape[source_axis]
                if target_index < len(target_axes):
                    target_axis = target_axes[target_index]
                    target_remaining = target_shape[target_axis]
                continue

            if isinstance(source_remaining, int) and isinstance(target_remaining, int):
                if (0 < target_remaining < source_remaining) and (
                    source_remaining % target_remaining == 0
                ):
                    source_remaining //= target_remaining
                    source_part += 1
                    target_index += 1
                    if target_index < len(target_axes):
                        target_axis = target_axes[target_index]
                        target_remaining = target_shape[target_axis]
                    continue

                if (0 < source_remaining < target_remaining) and (
                    target_remaining % source_remaining == 0
                ):
                    target_remaining //= source_remaining
                    source_index += 1
                    source_part = 0
                    if source_index < len(source_axes):
                        source_axis = source_axes[source_index]
                        source_remaining = source_shape[source_axis]
                    continue

                return None

            return None

        if source_index != len(source_axes) or target_index != len(target_axes):
            return None
        cls._add_singleton_pieces(piece_map, source_shape, target_shape)
        return piece_map

    @classmethod
    def _add_singleton_pieces(
        cls,
        piece_map: list[list[_Piece]],
        source_shape: Sequence[_Dim],
        target_shape: Sequence[_Dim],
    ) -> None:
        mapped_source_axes = {
            source_axis for source_axis, pieces in enumerate(piece_map) if pieces
        }
        mapped_target_axes = {
            target_axis for pieces in piece_map for target_axis, _ in pieces
        }
        source_singletons = [
            axis
            for axis, dim in enumerate(source_shape)
            if axis not in mapped_source_axes and cls._dim_equals(dim, 1)
        ]
        target_singletons = [
            axis
            for axis, dim in enumerate(target_shape)
            if axis not in mapped_target_axes and cls._dim_equals(dim, 1)
        ]

        if len(source_singletons) == len(target_singletons):
            for source_axis, target_axis in zip(source_singletons, target_singletons):
                piece_map[source_axis].append((target_axis, 0))
        elif len(source_singletons) == 1:
            for target_axis in target_singletons:
                piece_map[source_singletons[0]].append((target_axis, 0))
        elif len(target_singletons) == 1:
            for source_axis in source_singletons:
                piece_map[source_axis].append((target_singletons[0], 0))
        else:
            for source_axis, target_axis in zip(source_singletons, target_singletons):
                piece_map[source_axis].append((target_axis, 0))

    def _target_to_source_pieces(self) -> list[list[_Piece]]:
        target_to_source: list[list[_Piece]] = [[] for _ in self.target_shape]
        for source_axis, pieces in enumerate(self._valid_source_to_target_pieces()):
            for target_axis, source_part in pieces:
                target_to_source[target_axis].append((source_axis, source_part))
        return target_to_source

    def _valid_source_to_target_pieces(self) -> list[list[_Piece]]:
        assert self.source_to_target_pieces is not None
        return self.source_to_target_pieces

    @staticmethod
    def _normalize_dim(dim: int, rank: int) -> int:
        return dim if dim >= 0 else dim + rank

    @classmethod
    def _normalize_dims(cls, dims: int | Sequence[int], rank: int) -> list[int]:
        if isinstance(dims, int):
            return [cls._normalize_dim(dims, rank)]
        return [cls._normalize_dim(dim, rank) for dim in dims]

    @staticmethod
    def _is_permutation(dims: Sequence[int], rank: int) -> bool:
        return sorted(dims) == list(range(rank))

    @staticmethod
    def _dim_expr(dim: _Dim) -> sympy.Basic:
        return sympy.Integer(dim) if isinstance(dim, int) else dim.node.expr

    @classmethod
    def _dim_equals(cls, lhs: _Dim, rhs: _Dim) -> bool:
        if isinstance(lhs, int) and isinstance(rhs, int):
            return lhs == rhs
        return sympy.simplify(cls._dim_expr(lhs) - cls._dim_expr(rhs)) == 0

    @classmethod
    def shapes_equal(cls, lhs: Sequence[_Dim], rhs: Sequence[_Dim]) -> bool:
        return len(lhs) == len(rhs) and all(
            cls._dim_equals(lhs_dim, rhs_dim) for lhs_dim, rhs_dim in zip(lhs, rhs)
        )

    @staticmethod
    def numel(shape: Iterable[_Dim]) -> _Dim:
        numel: _Dim = 1
        for dim in shape:
            numel *= dim
        return numel

    @classmethod
    def same_numel(
        cls, first_shape: Iterable[_Dim], second_shape: Iterable[_Dim]
    ) -> bool:
        return cls.numel(first_shape) == cls.numel(second_shape)


class PermuteMap:
    """Map dimensions across a permute."""

    def __init__(self, permute_node: Node) -> None:
        permute_dims = permute_node.args[1]
        assert isinstance(permute_dims, Sequence) and not isinstance(
            permute_dims, (str, bytes)
        )
        self.permute_dims = list(cast(Sequence[int], permute_dims))

    def map_source_to_target(self, source_dims: int | Sequence[int]) -> list[int]:
        normalized_dims = ViewMap._normalize_dims(source_dims, len(self.permute_dims))
        inverse_permute = [0] * len(self.permute_dims)
        for target_dim, source_dim in enumerate(self.permute_dims):
            inverse_permute[source_dim] = target_dim
        return [inverse_permute[dim] for dim in normalized_dims]

    def map_target_to_source(self, target_dims: int | Sequence[int]) -> list[int]:
        normalized_dims = ViewMap._normalize_dims(target_dims, len(self.permute_dims))
        return [self.permute_dims[dim] for dim in normalized_dims]
