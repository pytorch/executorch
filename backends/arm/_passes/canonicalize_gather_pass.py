# Copyright 2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


import logging
from typing import Set, Type

import torch
from executorch.backends.arm._passes import ArmPass
from executorch.exir.dialects._ops import ops as exir_ops
from executorch.exir.pass_base import ExportPass

logger = logging.getLogger(__name__)


class CanonicalizeGatherPass(ArmPass):
    """
    Canonicalize gather so it can be lowered to TOSA.GATHER via the backend dialect.

    This pass is intended to run only for nodes already gated by GatherSupported.

    Behavior:
      - Reshape x from [N,K] to [N,K,1] so values matches TOSA gather's [N,K,C].
      - Keep indices as [N,W]
      - Lower using tosa.GATHER.default, producing [N,W,1].
      - Reshape output to [N,W].
      - Only insert bool<->int8 casts when x is bool:
        * If x is bool: gather runs on int8 and output is cast back to bool.
        * If x is not bool: gather runs on original dtype and output keeps dtype.
    """

    _passes_required_after: Set[Type[ExportPass]] = set()

    _TARGET_OPS = {exir_ops.edge.aten.gather.default}

    def call_operator(self, op, args, kwargs, meta):
        if op not in self._TARGET_OPS:
            return super().call_operator(op, args, kwargs, meta)

        # edge.aten.gather.default: (x, dim, index) with kw-only sparse_grad
        x, dim, index = args

        # GatherSupported should have gated this already; treat violations as errors.
        x_shape = x.data.shape
        index_shape = index.data.shape
        if not (
            dim in (1, -1)
            and len(x_shape) == 2
            and len(index_shape) == 2
            and index_shape[0] == x_shape[0]
        ):
            raise RuntimeError(
                f"[{op}] Unexpected gather pattern; expected "
                f"x:[N,K], index:[N,W], dim in {{1,-1}}, matching N. "
                f"Got dim={dim}, x.shape={x_shape}, index.shape={index_shape}."
            )

        N, K = x_shape[0], x_shape[1]
        W = index_shape[1]

        view_op = exir_ops.edge.aten.view_copy.default
        to_copy_op = exir_ops.edge.dim_order_ops._to_dim_order_copy.default

        # Use backend dialect gather:
        # values:  [N,K,C]
        # indices: [N,W]
        # output:  [N,W,C]
        tosa_gather_op = exir_ops.backend.tosa.GATHER.default

        needs_bool_cast = x.data.dtype == torch.bool

        # bool -> int8 (only if needed)
        values_in = x
        if needs_bool_cast:
            values_in = super().call_operator(
                to_copy_op,
                (x,),
                {"dtype": torch.int8},
                meta,
                updated=True,
            )

        # [N,K] -> [N,K,1]
        values_3d = super().call_operator(
            view_op,
            (values_in, [N, K, 1]),
            {},
            meta,
            updated=True,
        )

        # indices stays [N,W]
        gathered_3d = super().call_operator(
            tosa_gather_op,
            (values_3d, index),
            {},
            meta,
            updated=True,
        )

        # [N,W,1] -> [N,W]
        gathered_2d = super().call_operator(
            view_op,
            (gathered_3d, [N, W]),
            {},
            meta,
            updated=True,
        )

        # int8 -> bool (only if needed)
        if needs_bool_cast:
            return super().call_operator(
                to_copy_op,
                (gathered_2d,),
                {"dtype": torch.bool},
                meta,
                updated=True,
            )

        return gathered_2d
