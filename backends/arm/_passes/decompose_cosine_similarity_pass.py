# Copyright 2025 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Set, Type

import torch
from executorch.backends.arm._passes import ArmPass
from executorch.backends.arm._passes.convert_full_like_to_full_pass import (
    ConvertFullLikeToFullPass,
)

from executorch.backends.arm._passes.decompose_div_pass import DecomposeDivPass
from executorch.backends.arm._passes.decompose_sum_pass import DecomposeSumPass
from executorch.backends.arm._passes.insert_table_ops import InsertTableOpsPass
from executorch.exir.pass_base import ExportPass

torch_cosine_similarity = (torch.ops.aten.cosine_similarity.default,)


class DecomposeCosineSimilarityPass(ArmPass):
    """
    Decomposition of aten.cosine_similarity:

      dot    = sum(mul(x1, x2), dims, keepdim=False)
      norm   = pow( sum(mul(x, x), dims, keepdim=False), 0.5 )
      eps    = full( (), eps_scalar )
      n1c    = max(norm1, eps)
      n2c    = max(norm2, eps)
      denom  = mul(n1c, n2c)
      out    = div(dot, denom)
    """

    _passes_required_after: Set[Type[ExportPass]] = {
        DecomposeDivPass,
        DecomposeSumPass,
        ConvertFullLikeToFullPass,
        InsertTableOpsPass,
    }

    def call_operator(self, op, args, kwargs, meta):
        if op not in torch_cosine_similarity or not self.allowed_to_transform(meta):
            return super().call_operator(op, args, kwargs, meta)

        x1, x2 = args[0], args[1]
        dim = kwargs.get("dim", 1)
        eps = kwargs.get("eps", 1e-8)
        dims = [dim] if isinstance(dim, int) else list(dim)

        # 1) dot
        prod = super().call_operator(torch.ops.aten.mul.Tensor, (x1, x2), {}, meta)
        dot = super().call_operator(
            torch.ops.aten.sum.dim_IntList, (prod, dims, False), {}, meta
        )

        # 2a) norm1 = pow(sum(x1*x1), 0.5)
        x1_sq = super().call_operator(torch.ops.aten.mul.Tensor, (x1, x1), {}, meta)
        s1 = super().call_operator(
            torch.ops.aten.sum.dim_IntList, (x1_sq, dims, False), {}, meta
        )
        norm1 = super().call_operator(
            torch.ops.aten.pow.Tensor_Scalar, (s1, 0.5), {}, meta
        )

        # 2b) norm2 = pow(sum(x2*x2), 0.5)
        x2_sq = super().call_operator(torch.ops.aten.mul.Tensor, (x2, x2), {}, meta)
        s2 = super().call_operator(
            torch.ops.aten.sum.dim_IntList, (x2_sq, dims, False), {}, meta
        )
        norm2 = super().call_operator(
            torch.ops.aten.pow.Tensor_Scalar, (s2, 0.5), {}, meta
        )

        # 3) eps scalar - we need to broadcast ourselves as TOSA dont do this for scalar
        eps_t = super().call_operator(
            torch.ops.aten.full_like.default, (norm1, eps), {}, meta
        )

        # 4) clamp to avoid zero division
        n1c = super().call_operator(
            torch.ops.aten.maximum.default, (norm1, eps_t), {}, meta
        )
        n2c = super().call_operator(
            torch.ops.aten.maximum.default, (norm2, eps_t), {}, meta
        )

        # 5) denom and divide
        denom = super().call_operator(torch.ops.aten.mul.Tensor, (n1c, n2c), {}, meta)
        out = super().call_operator(torch.ops.aten.div.Tensor, (dot, denom), {}, meta)

        return out
