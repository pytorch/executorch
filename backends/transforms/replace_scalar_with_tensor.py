# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# Copyright 2025 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Dict, Optional, Union

import torch
from executorch.exir.dialects._ops import ops as exir_ops
from executorch.exir.dialects.edge._ops import EdgeOpOverload
from executorch.exir.pass_base import ExportPass


class ReplaceScalarWithTensorArgPass(ExportPass):
    """
    For binary ops like add.Scalar, sub.Scalar mul.Scalar, and div.Scalar,
    replace the scalar arg with Tensor arg.
    """

    default_ops: Dict[EdgeOpOverload, EdgeOpOverload] = {
        exir_ops.edge.aten.add.Scalar: exir_ops.edge.aten.add.Tensor,
        exir_ops.edge.aten.sub.Scalar: exir_ops.edge.aten.sub.Tensor,
        exir_ops.edge.aten.mul.Scalar: exir_ops.edge.aten.mul.Tensor,
        exir_ops.edge.aten.div.Scalar: exir_ops.edge.aten.div.Tensor,
        torch.ops.aten.add.Scalar: torch.ops.aten.add.Tensor,
        torch.ops.aten.sub.Scalar: torch.ops.aten.sub.Tensor,
        torch.ops.aten.mul.Scalar: torch.ops.aten.mul.Tensor,
        torch.ops.aten.div.Scalar: torch.ops.aten.div.Tensor,
    }

    def __init__(
        self,
        scalar_to_tensor_ops: Optional[
            Dict[
                Union[EdgeOpOverload, torch._ops.OpOverload],
                Union[EdgeOpOverload, torch._ops.OpOverload],
            ]
        ] = None,
    ):
        if scalar_to_tensor_ops is not None:
            self.scalar_to_tensor_ops = scalar_to_tensor_ops
        else:
            self.scalar_to_tensor_ops = self.default_ops
        super().__init__()

    def get_replacement(self, op, args, kwargs, meta):
        return super().call_operator(
            # Replace with .Tensor variant.
            op=self.scalar_to_tensor_ops[op],
            args=(
                # Tensor arg.
                args[0],
                # Scalar arg - replace with aten.full tensor.
                super().call_operator(
                    exir_ops.edge.aten.full.default,
                    args=(
                        (1,),
                        args[1],
                    ),
                    kwargs={"dtype": args[0].to_tensor().dtype},
                    meta=meta,
                ),
                # Other args.
                *args[2:],
            ),
            kwargs=kwargs,
            meta=meta,
        )

    def call_operator(self, op, args, kwargs, meta):
        if op not in self.scalar_to_tensor_ops:
            return super().call_operator(op, args, kwargs, meta)

        # There must be exactly 2 args (3 for add and sub containing alpha)
        assert len(args) == 2 or len(args) == 3

        # If there are two args, just replace the op.
        if len(args) == 2:
            return self.get_replacement(op, args, kwargs, meta)

        # In case the op has three args, it must be scalar add/sub op.
        if (
            op not in {exir_ops.edge.aten.add.Scalar, exir_ops.edge.aten.sub.Scalar}
            or "alpha" in kwargs
        ):
            return super().call_operator(op, args, kwargs, meta)

        return self.get_replacement(op, args, kwargs, meta)
