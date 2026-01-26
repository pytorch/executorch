# Copyright 2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


import math
from typing import Set, Type

import torch

from executorch.backends.arm._passes import ArmPass
from executorch.backends.arm._passes.replace_scalar_with_tensor_pass import (
    ReplaceScalarWithTensorByProfilePass,
)
from executorch.exir.dialects._ops import ops as exir_ops
from executorch.exir.pass_base import ExportPass


def _get_unfold_copy_decomposition(op) -> tuple:
    """
    Returns the decomposition of the given aten.unfold_copy operation into
    its equivalent TOSA-supported operations

    Returns:
        A tuple of operator overloads used by the decomposition.

    Raises:
        RuntimeError: If the provided operator is not supported by this pass.
    """

    if op in DecomposeUnfoldToGatherPass._TARGET_OPS:
        return (
            exir_ops.edge.dim_order_ops._to_dim_order_copy.default,
            exir_ops.edge.aten.view_copy.default,
            exir_ops.edge.aten.arange.start_step,
            exir_ops.edge.aten.mul.Scalar,
            exir_ops.edge.aten.unsqueeze_copy.default,
            exir_ops.edge.aten.add.Tensor,
            exir_ops.edge.aten.expand_copy.default,
            exir_ops.backend.tosa.GATHER.default,
            exir_ops.edge.aten.permute_copy.default,
        )

    raise RuntimeError(f"Can't get unfold_copy decomposition for op {op}")


class DecomposeUnfoldToGatherPass(ArmPass):
    """
    Decompose unfold_copy with backend tosa.GATHER as the core op, plus other
    TOSA-supported ops to build indices and materialize the output layout.

    Supported op:
      y = x.unfold_copy(dimension=dim, size=C, step=S)

    Requirements:
      - size > 0, step > 0
      - input rank >= 1
      - dim may be negative (normalized against rank)
      - the selected dimension length K must satisfy K >= size
      - shape must be static enough to compute:
          U = floor((K - C) / S) + 1
        (i.e. K, C, S known at export time)

    Input / Output (PyTorch semantics):
      - Input:  x has shape [d0, d1, ..., d{r-1}]  (rank r)
      - Output: y has shape:
          [d0, ..., d{dim-1}, U, d{dim+1}, ..., d{r-1}, C]
        where U = floor((K - C) / S) + 1 and K = x.size(dim)

    Strategy (single gather):
      - If x is bool: cast to int8 for TOSA gather and cast back at the end.
      - Normalize dimension and reshape x to the 3D "values" tensor expected by
        TOSA gather:
            pre  = x.shape[:dim]
            post = x.shape[dim+1:]
            P = prod(pre)  (or 1 if empty)
            Q = prod(post) (or 1 if empty)
            K = x.shape[dim]
            values = reshape(x, [P, K, Q])
      - Build a 1D index vector of length U*C:
            idx2d[u,c] = u*S + c   for u in [0..U-1], c in [0..C-1]
            idx1d = reshape(idx2d, [U*C])
        Then expand across P:
            indices = expand(idx1d[None, :], [P, U*C])
      - Call backend gather:
            tosa.GATHER(values=[P,K,Q], indices=[P,U*C]) -> [P,U*C,Q]
      - Reshape and permute to match PyTorch unfold layout:
            [P,U*C,Q] -> [*pre, U, *post, C]
    """

    _passes_required_after: Set[Type[ExportPass]] = {
        ReplaceScalarWithTensorByProfilePass,
    }

    _TARGET_OPS = {
        exir_ops.edge.aten.unfold_copy.default,
    }

    _UnfoldCopyInfo = tuple[
        torch.Tensor,  # x_val (FakeTensor)
        int,  # C
        int,  # S
        int,  # K
        int,  # U
        int,  # UC
        list[int],  # pre
        list[int],  # post
        int,  # P
        int,  # Q
        bool,  # needs_bool_cast
    ]

    def _compute_unfold_copy_params(
        self,
        x: torch.Tensor,
        dim: int,
        size: int,
        step: int,
    ) -> _UnfoldCopyInfo:
        x_val = x.data  # FakeTensor
        x_shape = tuple(x_val.shape)
        rank = len(x_shape)

        # Checked by UnfoldCopySupported (programmer error if hit here).
        assert rank >= 1, "unfold_copy requires rank >= 1"
        assert size > 0 and step > 0, "unfold_copy requires size/step > 0"

        dim_norm = dim % rank
        C = size
        S = step

        K = x_shape[dim_norm]
        assert K >= C, f"unfold_copy requires K>=C (K={K}, C={C})"

        U = (K - C) // S + 1
        UC = U * C

        pre = list(x_shape[:dim_norm])
        post = list(x_shape[dim_norm + 1 :])

        P = math.prod(pre) if pre else 1
        Q = math.prod(post) if post else 1

        needs_bool_cast = x_val.dtype == torch.bool

        return (x_val, C, S, K, U, UC, pre, post, P, Q, needs_bool_cast)

    def call_operator(self, op, args, kwargs, meta):
        if op not in self._TARGET_OPS:
            return super().call_operator(op, args, kwargs, meta)

        x, dim, size, step = args

        (
            x_val,
            C,
            S,
            K,
            U,
            UC,
            pre,
            post,
            P,
            Q,
            needs_bool_cast,
        ) = self._compute_unfold_copy_params(x, dim, size, step)

        (
            to_copy_op,
            view_op,
            arange_op,
            mul_scalar_op,
            unsqueeze_op,
            add_tensor_op,
            expand_op,
            tosa_gather_op,
            permute_op,
        ) = _get_unfold_copy_decomposition(op)

        # Note:
        #   bool unfolding (and the bool<->int8 casts) are INT-profile only;
        #   FP profile should reject these via the UnfoldCopySupported check.
        # ---- optional bool -> int8 ----
        x_for_gather = x
        if needs_bool_cast:
            x_for_gather = super().call_operator(
                to_copy_op,
                (x,),
                {"dtype": torch.int8},
                meta,
                updated=True,
            )

        # ---- values: [P, K, Q] ----
        values_pkq = super().call_operator(
            view_op,
            (x_for_gather, [P, K, Q]),
            {},
            meta,
            updated=True,
        )

        # ---- build idx1d of length U*C ----
        # idx_u: [U]
        idx_u = super().call_operator(
            arange_op,
            (0, U),
            {
                "dtype": torch.int32,
                "device": x_val.device,
            },
            meta,
            updated=True,
        )
        # idx_u_scaled: [U] = idx_u * S
        idx_u_scaled = super().call_operator(
            mul_scalar_op,
            (idx_u, S),
            {},
            meta,
            updated=True,
        )
        # idx_u2d: [U,1]
        idx_u2d = super().call_operator(
            unsqueeze_op,
            (idx_u_scaled, 1),
            {},
            meta,
            updated=True,
        )

        # idx_c: [C]
        idx_c = super().call_operator(
            arange_op,
            (0, C),
            {
                "dtype": torch.int32,
                "device": x_val.device,
            },
            meta,
            updated=True,
        )
        # idx_c2d: [1,C]
        idx_c2d = super().call_operator(
            unsqueeze_op,
            (idx_c, 0),
            {},
            meta,
            updated=True,
        )

        # idx2d: [U,C] = idx_u2d + idx_c2d
        idx2d = super().call_operator(
            add_tensor_op,
            (idx_u2d, idx_c2d),
            {},
            meta,
            updated=True,
        )

        # idx1d: [U*C]
        idx1d = super().call_operator(
            view_op,
            (idx2d, [UC]),
            {},
            meta,
            updated=True,
        )

        # indices: [P, U*C] by expanding from [1, U*C]
        idx_1xUC = super().call_operator(
            unsqueeze_op,
            (idx1d, 0),
            {},
            meta,
            updated=True,
        )
        indices_puc = super().call_operator(
            expand_op,
            (idx_1xUC, [P, UC]),
            {},
            meta,
            updated=True,
        )

        # ---- backend tosa gather ----
        # tosa.GATHER(values=[P,K,Q], indices=[P,UC]) -> [P,UC,Q]
        gathered_pucq = super().call_operator(
            tosa_gather_op,
            (values_pkq, indices_puc),
            {},
            meta,
            updated=True,
        )

        # [P,UC,Q] -> [P,U,C,Q]
        pucq = super().call_operator(
            view_op,
            (gathered_pucq, [P, U, C, Q]),
            {},
            meta,
            updated=True,
        )

        # Unflatten Q back to post, and P back to pre:
        # current logical layout: [*pre_flat=P, U, C, *post_flat=Q]
        if post:
            # [P,U,C,Q] -> [P,U,C,*post]
            shaped = super().call_operator(
                view_op,
                (pucq, [P, U, C, *post]),
                {},
                meta,
                updated=True,
            )
        else:
            # [P,U,C,Q(=1)] -> [P,U,C]
            shaped = super().call_operator(
                view_op,
                (pucq, [P, U, C]),
                {},
                meta,
                updated=True,
            )

        if pre:
            # [P,U,C,*post] -> [*pre, U, C, *post]
            shaped = super().call_operator(
                view_op,
                (shaped, [*pre, U, C, *post]),
                {},
                meta,
                updated=True,
            )
        else:
            # drop P(=1): [P,U,C,*post] -> [U, C, *post]
            shaped = super().call_operator(
                view_op,
                (shaped, [U, C, *post]),
                {},
                meta,
                updated=True,
            )

        # Move C to the last dimension:
        # from [*pre, U, C, *post] -> [*pre, U, *post, C]
        pre_len = len(pre)
        post_len = len(post)
        rank_cur = pre_len + 1 + 1 + post_len  # pre + U + C + post
        c_idx = pre_len + 1

        perm = list(range(0, pre_len + 1)) + list(range(c_idx + 1, rank_cur)) + [c_idx]
        out = super().call_operator(
            permute_op,
            (shaped, perm),
            {},
            meta,
            updated=True,
        )

        # ---- optional int8 -> bool ----
        if needs_bool_cast:
            out = super().call_operator(
                to_copy_op,
                (out,),
                {"dtype": torch.bool},
                meta,
                updated=True,
            )

        return out
