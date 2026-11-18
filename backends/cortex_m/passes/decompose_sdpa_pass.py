# Copyright 2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import math
from typing import Dict

import torch
from executorch.exir.pass_base import ExportPass, NodeMetadata, ProxyValue
from torch._ops import OpOverload
from torch.fx.node import Argument


class DecomposeSDPAPass(ExportPass):
    """
    Decomposes float32 scaled_dot_product_attention without mask, causal
    masking, dropout or GQA into matmul -> mul -> softmax -> matmul. Only
    static, non-empty shapes where query, key and value have the same leading
    dimensions are decomposed, since MatmulToBmmPass turns exactly those
    matmuls into bmm.

    The default SDPA decomposition splits the scale over query and key before
    the score matmul. Applying it once to the scores instead means the
    quantizer observes the scores before and after the scale, so their qparams
    differ by the scale and the mul can be folded during lowering.

    Without a mask a score row can only be all -inf for non-finite inputs, so
    regular softmax is used rather than _safe_softmax.
    """

    _known_kwargs = {"attn_mask", "dropout_p", "is_causal", "scale", "enable_gqa"}

    def call_operator(
        self,
        op: OpOverload,
        args: tuple[Argument, ...],
        kwargs: Dict[str, Argument],
        meta: NodeMetadata,
    ) -> ProxyValue:
        if op != torch.ops.aten.scaled_dot_product_attention.default:
            return super().call_operator(op, args, kwargs, meta)

        query, key, value = args[:3]
        attn_mask = args[3] if len(args) > 3 else kwargs.get("attn_mask")
        dropout_p = args[4] if len(args) > 4 else kwargs.get("dropout_p", 0.0)
        is_causal = args[5] if len(args) > 5 else kwargs.get("is_causal", False)
        scale = kwargs.get("scale")

        if (
            not isinstance(query, ProxyValue)
            or not isinstance(key, ProxyValue)
            or not isinstance(value, ProxyValue)
            or not kwargs.keys() <= self._known_kwargs
            or attn_mask is not None
            or dropout_p != 0.0
            or is_causal
            or kwargs.get("enable_gqa", False)
        ):
            return super().call_operator(op, args, kwargs, meta)

        shapes = [t.to_tensor().shape for t in (query, key, value)]
        if (
            # Scaling after the score matmul could overflow in half precision,
            # which is why the default decomposition splits the scale.
            query.to_tensor().dtype != torch.float32
            # MatmulToBmmPass reshapes with -1, which is ambiguous for empty
            # tensors.
            or not all(isinstance(dim, int) and dim > 0 for s in shapes for dim in s)
            or len(shapes[0]) < 3
            or not shapes[0][:-2] == shapes[1][:-2] == shapes[2][:-2]
        ):
            return super().call_operator(op, args, kwargs, meta)

        if scale is None:
            scale = 1.0 / math.sqrt(shapes[0][-1])

        key_t = super().call_operator(
            torch.ops.aten.transpose.int, (key, -2, -1), {}, meta
        )
        scores = super().call_operator(
            torch.ops.aten.matmul.default, (query, key_t), {}, meta
        )
        # A Python scalar, unlike mul.Scalar, becomes a constant buffer in
        # ScalarsToAttributePass rather than a full op evaluated at runtime.
        scores = super().call_operator(
            torch.ops.aten.mul.Tensor, (scores, scale), {}, meta
        )
        attn = super().call_operator(torch.ops.aten.softmax.int, (scores, -1), {}, meta)
        return super().call_operator(
            torch.ops.aten.matmul.default, (attn, value), {}, meta
        )
