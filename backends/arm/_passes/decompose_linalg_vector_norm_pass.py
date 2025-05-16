# Copyright 2025 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch
from executorch.exir.pass_base import ExportPass


class DecomposeLinearVectorNormPass(ExportPass):
    """
    This pass decomposes aten.linalg_vector_norm.default into more primitive ops.
    We need to add this pass before quantization for graph annotation.
    By default, aten.linalg_vector_norm op is decomposed during legalization to Edge IR.

    The decomposition is as follows:

      For p == 1:
          out = REDUCE_SUM(ABS(x), dims, keepdim)

      For p == 2:
          out = SQRT(REDUCE_SUM(MUL(x, x), dims, keepdim))

      For arbitrary p:
          We dont support arbitrary p, because our decomposition looks like
          out = POW(REDUCE_SUM(POW(ABS(x), p), dims, keepdim), 1/p)
          In this case we need to wrap p into Tensor and we need to know
          dtype prior, but we dont know this from FX graph.
    """

    torch_linalg_vector_norm = (torch.ops.aten.linalg_vector_norm.default,)

    def call_operator(self, op, args, kwargs, meta):
        if op not in self.torch_linalg_vector_norm:
            return super().call_operator(op, args, kwargs, meta)

        # Extract inputs and optional arguments.
        # Expected args:
        #   args[0]: input tensor
        #   args[1]: norm order 'p' (optional, default: 2.0)
        #   args[2]: dimensions to reduce (should be provided)
        #   args[3]: keepdim flag (optional, default: False)
        input_tensor = args[0]
        norm_order = args[1] if len(args) > 1 else 2.0
        norm_dim = args[2] if len(args) > 2 else None
        keepdim = args[3] if len(args) > 3 else False

        if norm_order not in (1, 2):
            raise ValueError(
                f"The order of {norm_order}\n"
                f"is not supported for linalg_vector_norm operator"
            )

        if norm_dim is None:
            raise ValueError("The norm_dim for linalg_vector_norm is None.")

        dims = [norm_dim] if isinstance(norm_dim, int) else list(norm_dim)

        # Decomposition based on norm order.
        if norm_order == 1:
            op1 = super().call_operator(
                torch.ops.aten.abs.default, (input_tensor,), {}, meta
            )
            op2 = super().call_operator(
                torch.ops.aten.sum.dim_IntList, (op1, dims, keepdim), {}, meta
            )
            return op2

        elif norm_order == 2:
            # For p == 2, decomposition is sqrt(sum(x * x, dims, keepdim))
            op1 = super().call_operator(
                torch.ops.aten.mul.Tensor, (input_tensor, input_tensor), {}, meta
            )
            op2 = super().call_operator(
                torch.ops.aten.sum.dim_IntList, (op1, dims, keepdim), {}, meta
            )
            op3 = super().call_operator(torch.ops.aten.sqrt.default, (op2,), {}, meta)
            return op3
