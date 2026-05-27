# Copyright 2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import io
import itertools
from contextlib import redirect_stderr, redirect_stdout
from math import prod
from typing import Any, Sequence, Set, Type

import torch
import tosa_serializer as ts
from executorch.backends.arm._passes.arm_pass import ArmPass
from executorch.backends.arm._passes.rewrite_slice import RewriteSlicePass
from executorch.backends.arm.arm_vela import vela_compile
from executorch.backends.arm.tosa.mapping import map_dtype
from executorch.backends.arm.tosa.specification import get_context_spec
from executorch.exir.dialects._ops import ops as exir_ops
from executorch.exir.pass_base import ExportPass


class DecomposePermuteForU55Pass(ArmPass):
    """Decompose U55 permutes into shape-safe permutes for large tensor shapes.

    Ethos-U55 has transpose shape constraints based on rank-dependent
    dimension-product limits. For shapes that violate this limit, rewrite
    a single permute into ``slice_copy -> permute -> cat`` so each emitted
    permute operates on a shape that satisfies the same product constraint.

    """

    _passes_required_after: Set[Type[ExportPass]] = {RewriteSlicePass}

    _PERMUTE_OPS = (
        exir_ops.edge.aten.permute.default,
        exir_ops.edge.aten.permute_copy.default,
    )
    _SLICE_OP = exir_ops.edge.aten.slice_copy.Tensor
    _CAT_OP = exir_ops.edge.aten.cat.default
    _MAX_PRODUCT = 2**16
    _VELA_TARGET = "ethos-u55-128"

    @classmethod
    def _violates_u55_worst_case_constraint(cls, shape: Sequence[int]) -> bool:
        """Checks the worst case scenario for a permute operation, any permute
        which does not violate this constraint is guaranteed to be supported.

        The check ensures that the product of any combination of rank(shape) - 2
        dims is smaller than 2**16, e.g. for a rank 4 tensor (N, C, H, W)
        N*C, N*H, N*W, C*H, C*W and H*W must all be smaller than 2**16.

        """

        rank = len(shape)
        if rank == 0:
            return False
        combination_len = max(1, rank - 2)
        for dims in itertools.combinations(shape, combination_len):
            if prod(dims) > cls._MAX_PRODUCT:
                return True
        return False

    @classmethod
    def _build_transpose_probe_tosa(
        cls,
        shape: tuple[int, ...],
        permutation: tuple[int, ...],
        dtype: Any,
    ) -> bytes:
        """Creates a tosa flatbuffer only containing a single transpose operator
        with given shape, permutation and dtype.
        """
        version = get_context_spec().version
        tosa_graph = ts.TosaSerializer(
            "",
            targetMajor=version.major,
            targetMinor=version.minor,
            targetPatch=version.micro,
            targetDraft=True if version.minor > 0 else False,
        )
        tosa_dtype = map_dtype(dtype)
        input_name = "probe_ifm"
        output_shape = [shape[idx] for idx in permutation]

        input_tensor = ts.TosaSerializerTensor(
            input_name,
            list(shape),
            tosa_dtype,
            data=None,
        )
        tosa_graph.addInputTensor(input_tensor)
        output_tensor = tosa_graph.addIntermediate(output_shape, tosa_dtype)

        attr = ts.TosaSerializerAttribute()
        attr.TransposeAttribute(list(permutation))
        tosa_graph.addOperator(
            ts.Op.TRANSPOSE,
            inputs=[input_name],
            outputs=[output_tensor.name],
            attributes=attr,
        )
        tosa_graph.addOutputTensor(output_tensor)
        return tosa_graph.serialize()

    @classmethod
    def _violates_exact_constraint(
        cls, shape: tuple[int, ...], permutation: tuple[int, ...], dtype: torch.dtype
    ) -> bool:
        """Performs a Vela compilation of a permute with given shape,
        permutation and dtype to check wheter it is supported.
        """

        # Lazy import to avoid circular dependency
        from executorch.backends.arm.ethosu.compile_spec import EthosUCompileSpec

        if dtype not in (torch.int8, torch.bool, torch.int16):
            return True

        try:
            tosa_flatbuffer = cls._build_transpose_probe_tosa(shape, permutation, dtype)
            compile_flags = EthosUCompileSpec(cls._VELA_TARGET).compiler_flags

            # Vela prints summaries to stdout/stderr even with verbose disabled.
            # Suppress this during pass-time probing.
            with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                vela_compile(tosa_flatbuffer, list(compile_flags), verbose=False)
            return False

        except Exception:
            return True

    @staticmethod
    def _chunk_ranges(size: int, max_chunk: int) -> list[tuple[int, int]]:
        ranges: list[tuple[int, int]] = []
        start = 0
        while start < size:
            end = min(start + max_chunk, size)
            ranges.append((start, end))
            start = end
        return ranges

    @classmethod
    def _find_safe_split_chunk_limits(
        cls, shape: Sequence[int]
    ) -> dict[int, int] | None:
        """Compute per-axis chunk limits that make the worst-case check pass.

        This is a greedy planner over an ``effective_shape``:
        - Identify violating dim-combinations for the coarse U55 rule.
        - Pick a splittable axis from those combinations.
        - Compute the largest safe chunk size for that axis given the other dims.
        - Apply that limit to ``effective_shape`` and continue.

        Returns:
            ``dict[axis, chunk_limit]`` when a valid split plan is found.
            ``None`` when no positive chunk limit can be found for any
            splittable axis in a violating state. This is treated as an
            exceptional condition by the caller.

        """
        rank = len(shape)
        if rank == 0:
            return {}

        combination_len = max(1, rank - 2)
        effective_shape = list(shape)
        split_chunk_limits: dict[int, int] = {}

        while cls._violates_u55_worst_case_constraint(effective_shape):
            violating_combos = cls._get_violating_combinations(
                effective_shape, rank, combination_len
            )
            if not violating_combos:
                break

            candidate_axes = cls._get_candidate_axes(effective_shape, violating_combos)
            if not candidate_axes:
                return None

            if cls._apply_first_valid_axis_reduction(
                effective_shape,
                violating_combos,
                candidate_axes,
                split_chunk_limits,
            ):
                continue

            # Violations remain, but no axis can be reduced with a positive
            # chunk limit in this state.
            return None

        return split_chunk_limits

    @classmethod
    def _get_violating_combinations(
        cls, effective_shape: Sequence[int], rank: int, combination_len: int
    ) -> list[tuple[int, ...]]:
        return [
            combo
            for combo in itertools.combinations(range(rank), combination_len)
            if prod(effective_shape[idx] for idx in combo) > cls._MAX_PRODUCT
        ]

    @staticmethod
    def _get_candidate_axes(
        effective_shape: Sequence[int], violating_combos: Sequence[tuple[int, ...]]
    ) -> list[int]:
        return sorted(
            {
                axis
                for combo in violating_combos
                for axis in combo
                if effective_shape[axis] > 1
            },
            key=lambda axis: effective_shape[axis],
            reverse=True,
        )

    @classmethod
    def _get_safe_chunk_limit_for_axis(
        cls,
        effective_shape: Sequence[int],
        violating_combos: Sequence[tuple[int, ...]],
        axis: int,
    ) -> int:
        # Worst product of all other dims in violating combos that include
        # this axis. The chunk limit for this axis is derived from this
        # worst-case term.
        max_other_product = 1
        for combo in violating_combos:
            if axis not in combo:
                continue
            other_product = 1
            for idx in combo:
                if idx != axis:
                    other_product *= effective_shape[idx]
            max_other_product = max(max_other_product, other_product)
        return cls._MAX_PRODUCT // max_other_product

    @classmethod
    def _apply_first_valid_axis_reduction(
        cls,
        effective_shape: list[int],
        violating_combos: Sequence[tuple[int, ...]],
        candidate_axes: Sequence[int],
        split_chunk_limits: dict[int, int],
    ) -> bool:
        for axis in candidate_axes:
            safe_chunk_limit = cls._get_safe_chunk_limit_for_axis(
                effective_shape, violating_combos, axis
            )
            if safe_chunk_limit < 1 or effective_shape[axis] <= safe_chunk_limit:
                continue

            split_chunk_limits[axis] = min(
                split_chunk_limits.get(axis, effective_shape[axis]),
                safe_chunk_limit,
            )
            effective_shape[axis] = safe_chunk_limit
            return True

        return False

    def _apply_split_permute_cat(
        self,
        op,
        input_node,
        normalized_permutation: tuple[int, ...],
        split_chunk_limits: dict[int, int],
        kwargs,
        meta,
    ):
        """Apply split -> permute -> concat decomposition.

        For each planned split axis, recursively emit ``slice_copy`` chunks.
        At the recursion leaf, emit one permute per chunk. Then concatenate
        chunk outputs along the corresponding output axis.

        """
        split_axes = sorted(split_chunk_limits.keys())
        input_shape = tuple(input_node.data.shape)

        def recurse(current, depth: int):
            if depth >= len(split_axes):
                return super(DecomposePermuteForU55Pass, self).call_operator(
                    op,
                    (current, normalized_permutation),
                    kwargs,
                    meta,
                    updated=True,
                )

            split_axis = split_axes[depth]
            split_ranges = self._chunk_ranges(
                input_shape[split_axis],
                split_chunk_limits[split_axis],
            )
            chunk_outputs = []
            for start, end in split_ranges:
                sliced = super(DecomposePermuteForU55Pass, self).call_operator(
                    self._SLICE_OP,
                    (current, split_axis, start, end),
                    {},
                    meta,
                    updated=True,
                )
                chunk_outputs.append(recurse(sliced, depth + 1))

            if len(chunk_outputs) == 1:
                return chunk_outputs[0]

            # After permute, the original split axis lands at this output index.
            cat_dim = normalized_permutation.index(split_axis)
            return super(DecomposePermuteForU55Pass, self).call_operator(
                self._CAT_OP,
                (chunk_outputs, cat_dim),
                {},
                meta,
                updated=True,
            )

        return recurse(input_node, 0)

    def call_operator(self, op, args, kwargs, meta):
        if op not in self._PERMUTE_OPS:
            return super().call_operator(op, args, kwargs, meta)

        spec = get_context_spec()
        if not spec.is_U55_subset:
            return super().call_operator(op, args, kwargs, meta)

        input_shape = args[0].data.shape
        rank = args[0].data.ndim
        dtype = args[0].data.dtype
        permutation = tuple(dim % rank for dim in args[1])

        # This is a quick check to avoid the overhead of the Vela compilation in 99% of cases.
        if not self._violates_u55_worst_case_constraint(input_shape):
            return super().call_operator(op, args, kwargs, meta)

        if not self._violates_exact_constraint(input_shape, permutation, dtype):
            return super().call_operator(op, args, kwargs, meta)

        split_chunk_limits = self._find_safe_split_chunk_limits(input_shape)
        if split_chunk_limits is None:
            raise RuntimeError(
                "DecomposePermuteForU55Pass could not find a valid split plan "
                f"for shape={tuple(input_shape)} perm={permutation}. "
                "This is expected to be extremely rare for real model graphs."
            )

        if not split_chunk_limits:
            return super().call_operator(op, args, kwargs, meta)

        return self._apply_split_permute_cat(
            op,
            args[0],
            permutation,
            split_chunk_limits,
            kwargs,
            meta,
        )
