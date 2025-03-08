# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# Copyright 2025 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Dict, Tuple

import torch
from executorch.exir.dialects._ops import ops as exir_ops
from executorch.exir.pass_base import ExportPass, NodeMetadata, ProxyValue
from torch.fx.node import Argument


class ReplaceScalarTensorWithFullPass(ExportPass):
    """
    aten.scalar_tensor can be replaced by aten.full with a shape of [1].
    """

    def call_operator(
        self,
        op,
        args: Tuple[Argument, ...],
        kwargs: Dict[str, Argument],
        meta: NodeMetadata,
    ) -> ProxyValue:
        if op not in {
            exir_ops.edge.aten.scalar_tensor.default,
            torch.ops.aten.scalar_tensor.default,
        }:
            return super().call_operator(op, args, kwargs, meta)

        return super().call_operator(
            exir_ops.edge.aten.full.default,
            (
                [1],
                args[0],
            ),
            {"dtype": kwargs["dtype"]},
            meta,
        )
