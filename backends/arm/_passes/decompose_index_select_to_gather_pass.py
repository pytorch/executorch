# Copyright 2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import math
from typing import Set, Type

import torch

from executorch.backends.arm._passes import ArmOpTargetedPass
from executorch.backends.arm._passes.arm_pass_utils import (
    get_param_tensor,
    is_param_node,
)
from executorch.backends.arm._passes.convert_expand_copy_to_repeat import (
    ConvertExpandCopyToRepeatPass,
)
from executorch.backends.arm._passes.convert_squeezes_to_view import (
    ConvertSqueezesToViewPass,
)
from executorch.exir import ExportedProgram
from executorch.exir.dialects._ops import ops as exir_ops
from executorch.exir.pass_base import ExportPass


def _get_index_select_decomposition(op):
    """Return the operator overloads used to lower index_select via TOSA gather.

    Raises:
        RuntimeError: If the provided operator is not supported by this pass.

    """
    if op is exir_ops.edge.aten.index_select.default:
        return (
            exir_ops.edge.aten.view_copy.default,
            exir_ops.edge.aten.unsqueeze_copy.default,
            exir_ops.edge.aten.expand_copy.default,
            exir_ops.edge.dim_order_ops._to_dim_order_copy.default,
            exir_ops.backend.tosa.GATHER.default,
        )

    raise RuntimeError(f"Can't get index_select decomposition for op {op}")


class DecomposeIndexSelectToGatherPass(ArmOpTargetedPass):
    """Decompose edge index_select into a single backend TOSA gather.

    index_select(x, dim, index) semantics:
      - index is rank-1
      - output shape is x.shape with dim replaced by len(index)

    Lowering strategy:
      - Normalize dim to [0, rank)
      - Flatten x into values [N, K, C] where:
            N = prod(x.shape[:dim]) (or 1)
            K = x.shape[dim]
            C = prod(x.shape[dim+1:]) (or 1)
      - Expand indices to [N, W] where W = len(index):
            index [W] -> [1, W] -> expand [N, W]
      - tosa.GATHER(values=[N,K,C], indices=[N,W]) -> [N,W,C]
      - Reshape back to original rank with dim replaced by W

    Notes:
      - indices must be int32 (TOSA gather requirement).
      - bool input is cast to int8 for gather and cast back afterwards.

    """

    _passes_required_after: Set[Type[ExportPass]] = {
        ConvertExpandCopyToRepeatPass,
        ConvertSqueezesToViewPass,
    }

    target_ops = {
        exir_ops.edge.aten.index_select.default,
    }

    def __init__(
        self, exported_program: ExportedProgram | None = None, *args, **kwargs
    ) -> None:
        super().__init__(*args, **kwargs)
        self.exported_program = exported_program

    def call_operator(self, op, args, kwargs, meta):
        if op not in self.target_ops:
            return super().call_operator(op, args, kwargs, meta)

        x, dim, index = args

        x_t, idx_t = x.data, index.data  # FakeTensor
        x_shape, idx_shape = tuple(x_t.shape), tuple(idx_t.shape)
        x_rank, idx_rank = len(x_shape), len(idx_shape)

        if (
            x_rank >= 1
            and idx_rank == 1
            and self.exported_program is not None
            and is_param_node(self.exported_program, index.node)
            and all(isinstance(size, int) for size in x_shape)
        ):
            constant_index = get_param_tensor(self.exported_program, index.node)
            if constant_index is not None and constant_index.numel() > 0:
                indices = constant_index.tolist()
                dim_norm = dim % x_rank
                dim_size = x_shape[dim_norm]
                if any(index < 0 or index >= dim_size for index in indices):
                    raise RuntimeError(
                        f"index_select index out of range for dimension of size {dim_size}"
                    )
                if indices == list(range(indices[0], indices[0] + len(indices))):
                    return super().call_operator(
                        exir_ops.edge.aten.slice_copy.Tensor,
                        (x, dim_norm, indices[0], indices[-1] + 1),
                        {},
                        meta,
                        updated=True,
                    )

                slices = []
                for index_value in indices:
                    slices.append(
                        super().call_operator(
                            exir_ops.edge.aten.slice_copy.Tensor,
                            (x, dim_norm, index_value, index_value + 1),
                            {},
                            meta,
                            updated=True,
                        )
                    )
                return super().call_operator(
                    exir_ops.edge.aten.cat.default,
                    (slices, dim_norm),
                    {},
                    meta,
                    updated=True,
                )

        assert x_rank >= 1 and idx_rank == 1 and idx_t.dtype == torch.int32, (
            f"[{self.__class__.__name__}] unsupported index_select signature: "
            f"x_rank={x_rank}, index_rank={idx_rank}, index_dtype={idx_t.dtype} "
            "(expected x_rank>=1, index_rank==1, index_dtype=int32)."
        )

        (
            view_op,
            unsqueeze_op,
            expand_op,
            to_copy_op,
            tosa_gather_op,
        ) = _get_index_select_decomposition(op)

        # Compute flattening factors
        dim_norm = dim % x_rank  # Normalize dim
        before = list(x_shape[:dim_norm])
        after = list(x_shape[dim_norm + 1 :])

        N = math.prod(before) if before else 1
        K = x_shape[dim_norm]
        C = math.prod(after) if after else 1
        W = idx_shape[0]

        # Note:
        #   bool index_select (and the bool<->int8 casts) are INT-profile only;
        #   FP profile should reject these via the IndexSelectSupported check.
        # Optional: cast bool -> int8
        x_for_gather = x
        x_dtype = x_t.dtype
        if x_dtype == torch.bool:
            x_for_gather = super().call_operator(
                to_copy_op,
                (x,),
                {"dtype": torch.int8},
                meta,
                updated=True,
            )

        # x_for_gather: [d0, d1, ..., d{r-1}] -> values_nkc: [N, K, C]
        values_nkc = super().call_operator(
            view_op,
            (x_for_gather, [N, K, C]),
            {},
            meta,
            updated=True,
        )

        # index: [W] -> idx_1w: [1, W]
        idx_1w = super().call_operator(
            unsqueeze_op,
            (index, 0),
            {},
            meta,
            updated=True,
        )
        # idx_1w: [1, W] -> indices_nw: [N, W]
        indices_nw = super().call_operator(
            expand_op,
            (idx_1w, [N, W]),
            {},
            meta,
            updated=True,
        )

        # TOSA gather:
        # values_nkc: [N, K, C]
        # indices_nw: [N, W]
        # gathered_nwc: [N, W, C]
        gathered_nwc = super().call_operator(
            tosa_gather_op,
            (values_nkc, indices_nw),
            {},
            meta,
            updated=True,
        )

        # reshape back to original rank with dim replaced by W
        # gathered_nwc: [N, W, C] -> out: before + [W] + after
        out_shape = before + [W] + after
        out = super().call_operator(
            view_op,
            (gathered_nwc, out_shape),
            {},
            meta,
            updated=True,
        )

        # Optional: cast int8 -> bool
        if x_dtype == torch.bool:
            out = super().call_operator(
                to_copy_op,
                (out,),
                {"dtype": torch.bool},
                meta,
                updated=True,
            )

        return out
