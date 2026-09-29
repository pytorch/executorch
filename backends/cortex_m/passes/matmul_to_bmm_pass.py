# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import cast, Dict

import torch
from executorch.exir.pass_base import ExportPass, NodeMetadata, ProxyValue

from torch._ops import OpOverload
from torch.fx.node import Argument


class MatmulToBmmPass(ExportPass):
    """
    Rewrites ``aten.matmul.default`` into ``aten.bmm.default`` so the cortex_m
    annotator and ``quantized_batch_matmul`` lowering pick it up.

    ``a @ b`` / ``torch.matmul`` is captured as ``aten.matmul.default`` at
    annotation time and only decomposes to ``bmm`` at ``to_edge`` -- after
    quantization -- so it would otherwise never receive qparams. This pass runs
    before annotation and normalizes matmul to bmm:

    - rank-3 @ rank-3 with matching batch dims: replaced directly with
      ``aten.bmm.default``.
    - rank>3 (e.g. attention [B,H,N,d]@[B,H,d,N]) with matching leading batch
      dims: the leading batch dims are folded into a single batch dim (reshape to
      3D), ``aten.bmm.default`` is applied, and the result is reshaped back to the
      original leading dims. This is required because the cortex_m bmm checker
      rejects non-rank-3.
    - everything else is left unchanged, including rank-2 (mm / linear territory)
      and broadcasting matmuls whose batch dims differ (``aten.bmm`` requires
      equal batch dims, so rewriting those would crash).
    """

    def call_operator(
        self,
        op: OpOverload,
        args: tuple[Argument, ...],
        kwargs: Dict[str, Argument],
        meta: NodeMetadata,
    ) -> ProxyValue:
        if op != torch.ops.aten.matmul.default:
            return super().call_operator(op, args, kwargs, meta)

        lhs = cast(ProxyValue, args[0])
        rhs = cast(ProxyValue, args[1])
        lhs_shape = lhs.to_tensor().shape
        rhs_shape = rhs.to_tensor().shape
        lhs_rank = len(lhs_shape)
        rhs_rank = len(rhs_shape)

        if lhs_rank == 3 and rhs_rank == 3 and lhs_shape[:-2] == rhs_shape[:-2]:
            return super().call_operator(
                torch.ops.aten.bmm.default, (lhs, rhs), {}, meta
            )

        if lhs_rank == rhs_rank and lhs_rank > 3 and lhs_shape[:-2] == rhs_shape[:-2]:
            batch_dims = list(lhs_shape[:-2])
            m, k = lhs_shape[-2], lhs_shape[-1]
            n = rhs_shape[-1]

            lhs_3d = super().call_operator(
                torch.ops.aten.reshape.default, (lhs, [-1, m, k]), {}, meta
            )
            rhs_3d = super().call_operator(
                torch.ops.aten.reshape.default, (rhs, [-1, k, n]), {}, meta
            )
            bmm_out = super().call_operator(
                torch.ops.aten.bmm.default, (lhs_3d, rhs_3d), {}, meta
            )
            return super().call_operator(
                torch.ops.aten.reshape.default,
                (bmm_out, batch_dims + [m, n]),
                {},
                meta,
            )

        return super().call_operator(op, args, kwargs, meta)
