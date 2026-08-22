# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from executorch.exir.dialects._ops import ops as exir_ops

from executorch.exir.pass_base import ExportPass


class MeanToSumDiv(ExportPass):
    def call_operator(self, op, args, kwargs, meta):
        if op != exir_ops.edge.aten.mean.dim:
            return super().call_operator(op, args, kwargs, meta)
        sum_res = super().call_operator(
            exir_ops.edge.aten.sum.dim_IntList, args, kwargs, meta
        )
        # args[0] is the input tensor
        shape = args[0].node.meta["val"].shape
        dtype = args[0].node.meta["val"].dtype
        dims_to_reduce = args[1]
        size = 1.0
        for dim in dims_to_reduce:
            size = size * shape[dim]

        size_tensor = super().call_operator(
            exir_ops.edge.aten.full.default,
            (
                [
                    1,
                ],
                size,
            ),
            {"dtype": dtype},
            meta,
        )

        return super().call_operator(
            exir_ops.edge.aten.div.Tensor, (sum_res, size_tensor), {}, meta
        )
