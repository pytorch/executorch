# Copyright 2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
import operator
from typing import Set, Type

from executorch.backends.arm._passes import ArmOpTargetedPass

from executorch.exir.dialects._ops import ops as exir_ops
from executorch.exir.pass_base import ExportPass, ProxyValue

from torch import SymInt


class RewriteSlicePass(ArmOpTargetedPass):
    """Rewrite slice operations with step of 1 to TOSA slice operators."""

    _passes_required_after: Set[Type[ExportPass]] = set()
    target_ops = (exir_ops.edge.aten.slice_copy.Tensor,)

    def _normalize_slice_bound(
        self,
        bound: int | None,
        dim_size: int | SymInt | ProxyValue,
        *,
        is_start: bool,
        meta,
    ) -> int | SymInt | ProxyValue:
        if isinstance(dim_size, int):
            return (
                slice(bound, None).indices(dim_size)[0]
                if is_start
                else slice(None, bound).indices(dim_size)[1]
            )
        if bound is None:
            return 0 if is_start else dim_size
        if bound < 0:
            return super().call_sym(
                max,
                (super().call_sym(operator.add, (dim_size, bound), meta), 0),
                meta,
            )
        return super().call_sym(min, (bound, dim_size), meta)

    def call_operator(self, op, args, kwargs, meta, updated=False) -> ProxyValue:
        if op not in self.target_ops:
            return super().call_operator(op, args, kwargs, meta, updated)

        if len(args) == 5 and args[4] != 1:
            raise ValueError(
                f"Only slice with 4 arguments and step of 1 is supported, got {len(args)} arguments and step {args[4]}"
            )
        input, dim, start, end = args[:4]
        if isinstance(start, (SymInt, ProxyValue)) or isinstance(
            end, (SymInt, ProxyValue)
        ):
            return super().call_operator(op, args, kwargs, meta, updated)

        input_shape = input.data.shape
        size_list = list(meta.data["val"].shape)
        # TOSA SLICE rejects statically known non-positive sizes, including
        # dimensions that are not being sliced.
        if any(isinstance(size, int) and size <= 0 for size in size_list):
            return super().call_operator(op, args, kwargs, meta, updated)

        dim_size = input_shape[dim]
        if isinstance(dim_size, SymInt):
            dim_size = super().call_size_operator(input, dim, meta)
        start_index = self._normalize_slice_bound(
            start, dim_size, is_start=True, meta=meta
        )
        end_index = self._normalize_slice_bound(
            end, dim_size, is_start=False, meta=meta
        )

        start_list: list[int | SymInt | ProxyValue] = [0] * len(input_shape)
        start_list[dim] = start_index

        if any(isinstance(dim, (SymInt, ProxyValue)) for dim in start_list):
            starts = start_list
        else:
            starts = super().call_shape_operator(  # type: ignore[assignment]
                exir_ops.backend.tosa.CONST_SHAPE.default,
                (start_list,),
                {},
                meta,
                True,
            )
        if any(isinstance(d, SymInt) for d in size_list):
            # Express sizes as ProxyValues to allow for symbolic shapes
            input_shape_proxy = super().call_size_operator_all(input, meta)
            sizes = input_shape_proxy
            if isinstance(size_list[dim], SymInt):
                size = super().call_sym(operator.sub, (end_index, start_index), meta)
                sizes[dim] = super().call_sym(max, (size, 0), meta)
            else:
                sizes[dim] = size_list[dim]
        else:
            sizes = super().call_shape_operator(  # type: ignore[assignment]
                exir_ops.backend.tosa.CONST_SHAPE.default,
                (size_list,),
                {},
                meta,
                True,
            )

        return super().call_operator(
            exir_ops.backend.tosa.SLICE.default,
            (input, starts, sizes),
            kwargs,
            meta,
            updated=True,
        )
