# Copyright 2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import cast, Dict

import torch
from executorch.exir.pass_base import ExportPass, NodeMetadata, ProxyValue

from torch._ops import OpOverload
from torch.fx.node import Argument


class DecomposeMeanPass(ExportPass):
    """
    Rewrites AdaptiveAvgPool2d and spatial mean.dim into AvgPool2d, which
    CMSIS-NN has a kernel for. Both are spellings of the same classifier head.
    """

    def call_operator(
        self,
        op: OpOverload,
        args: tuple[Argument, ...],
        kwargs: Dict[str, Argument],
        meta: NodeMetadata,
    ) -> ProxyValue:
        if op == torch.ops.aten.adaptive_avg_pool2d.default:
            input_tensor = cast(ProxyValue, args[0]).to_tensor()
            shape = input_tensor.shape
            stride = [1, 1]
            kernel_size = [shape[-2], shape[-1]]

            new_args = (args[0], kernel_size, stride, [0, 0], 0, 0)

            adaptive_output = torch.ops.aten.adaptive_avg_pool2d.default(
                input_tensor, *args[1:]
            )
            avg_pool_output = torch.ops.aten.avg_pool2d.default(
                input_tensor, *new_args[1:]
            )

            if adaptive_output.shape == avg_pool_output.shape:
                new_op = torch.ops.aten.avg_pool2d.default
                return super().call_operator(new_op, new_args, kwargs, meta)

        if op == torch.ops.aten.mean.dim:
            decomposed = self._mean_dim_to_avg_pool2d(args, kwargs, meta)
            if decomposed is not None:
                return decomposed

        return super().call_operator(op, args, kwargs, meta)

    def _mean_dim_to_avg_pool2d(
        self,
        args: tuple[Argument, ...],
        kwargs: Dict[str, Argument],
        meta: NodeMetadata,
    ) -> ProxyValue | None:
        """A mean over both spatial dimensions of NCHW is an average pool
        covering the whole plane. Any other reduction is left alone."""
        # A mean over a constant arrives as a bare tensor rather than a proxy.
        if not isinstance(args[0], ProxyValue):
            return None

        input_tensor = args[0].to_tensor()
        # The rank matters as well as the dims: a 3-D mean([-2, -1]) normalizes
        # to the same pair and is not a spatial reduction.
        if input_tensor.dim() != 4:
            return None

        # The kernel size and the view both take the shape as literals, which a
        # symbolic dimension cannot supply.
        if any(not isinstance(d, int) for d in input_tensor.shape):
            return None

        dims = args[1]
        if not isinstance(dims, (list, tuple)) or not all(
            isinstance(d, int) for d in dims
        ):
            return None
        if sorted(cast(int, d) % 4 for d in dims) != [2, 3]:
            return None

        # dtype= would change the accumulation type, which avg_pool2d cannot do.
        if kwargs.get("dtype") is not None:
            return None

        n, c, h, w = input_tensor.shape
        pooled = super().call_operator(
            torch.ops.aten.avg_pool2d.default,
            (args[0], [h, w], [1, 1], [0, 0], False, False),
            {},
            meta,
        )

        keepdim = args[2] if len(args) > 2 else kwargs.get("keepdim", False)
        if keepdim:
            return pooled
        # avg_pool2d keeps the spatial dimensions; a mean without keepdim drops
        # them.
        return super().call_operator(
            torch.ops.aten.view.default, (pooled, [n, c]), {}, meta
        )
