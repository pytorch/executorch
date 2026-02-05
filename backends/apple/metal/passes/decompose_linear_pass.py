# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch
from executorch.exir.dialects._ops import ops as exir_ops
from executorch.exir.pass_base import ExportPass


class DecomposeLinearPass(ExportPass):
    """
    Decompose aten.linear into matmul + add to avoid addmm.

    For 2D inputs, we unsqueeze to 3D before decomposition to force the matmul
    code path instead of addmm. The C++ implementation of aten.linear directly
    calls addmm for 2D inputs with bias, which would require implementing
    aoti_torch_mps_addmm_out. By unsqueezing to 3D, we force the matmul path,
    then squeeze back to 2D.
    """

    def call_operator(self, op, args, kwargs, meta):
        # Only intercept linear operations
        if op not in (exir_ops.edge.aten.linear.default, torch.ops.aten.linear.default):
            return super().call_operator(op, args, kwargs, meta)

        # Get input, weight, and bias arguments
        input_arg = args[0]
        weight_arg = args[1]
        bias_arg = args[2] if len(args) > 2 else None

        # Determine which ops to use based on the input operator
        if op == exir_ops.edge.aten.linear.default:
            t_op = exir_ops.edge.aten.t.default
            matmul_op = exir_ops.edge.aten.matmul.default
            add_op = exir_ops.edge.aten.add.Tensor
            unsqueeze_op = exir_ops.edge.aten.unsqueeze.default
            squeeze_op = exir_ops.edge.aten.squeeze.dims
        else:
            t_op = torch.ops.aten.t.default
            matmul_op = torch.ops.aten.matmul.default
            add_op = torch.ops.aten.add.Tensor
            unsqueeze_op = torch.ops.aten.unsqueeze.default
            squeeze_op = torch.ops.aten.squeeze.dims

        # Check if input is 2D from metadata
        needs_unsqueeze = len(meta["val"].shape) == 2

        # Unsqueeze 2D input to 3D: (M, K) -> (1, M, K)
        if needs_unsqueeze:
            input_arg = super().call_operator(unsqueeze_op, (input_arg, 0), {}, meta)

        # Transpose weight
        weight_t = super().call_operator(t_op, (weight_arg,), {}, meta)

        # Matmul
        result = super().call_operator(matmul_op, (input_arg, weight_t), {}, meta)

        # Add bias if present
        if bias_arg is not None:
            result = super().call_operator(add_op, (result, bias_arg), {}, meta)

        # Squeeze 3D output back to 2D: (1, M, N) -> (M, N)
        if needs_unsqueeze:
            result = super().call_operator(squeeze_op, (result, [0]), {}, meta)

        return result
