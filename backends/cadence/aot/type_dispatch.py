# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import torch
from executorch.backends.cadence.aot.pass_utils import (
    CadencePassAttribute,
    register_cadence_pass,
)
from executorch.exir.dialects._ops import ops as exir_ops
from executorch.exir.pass_base import ExportPass, NodeMetadata, ProxyValue
from torch._ops import OpOverload
from torch.fx.node import Argument


@register_cadence_pass(CadencePassAttribute(opt_level=4))
class CompileTimeTypeDispatchPass(ExportPass):
    """
    Replaces generic ops with ops that have explicit types.
    """

    def call_operator(
        self,
        op: OpOverload,
        args: tuple[Argument, ...],
        kwargs: dict[str, Argument],
        meta: NodeMetadata,
    ) -> ProxyValue:
        if op not in {
            exir_ops.edge.cadence.quantized_fully_connected.per_tensor,
        }:
            return super().call_operator(op, args, kwargs, meta)

        if (
            # pyre-ignore[16]: None has no attribute `to_tensor`.
            args[0].to_tensor().dtype == torch.int8
            and args[1].to_tensor().dtype == torch.int8
        ):
            return super().call_operator(
                exir_ops.edge.cadence.quantized_fully_connected_asym8sxasym8s_asym8s.per_tensor,
                args,
                kwargs,
                meta,
            )
        elif (
            args[0].to_tensor().dtype == torch.uint8
            and args[1].to_tensor().dtype == torch.uint8
        ):
            return super().call_operator(
                exir_ops.edge.cadence.quantized_fully_connected_asym8uxasym8u_asym8u.per_tensor,
                args,
                kwargs,
                meta,
            )
        else:
            raise RuntimeError(
                f"Unsupported input types for {op}: {args[0].to_tensor().dtype} and {args[1].to_tensor().dtype}"
            )
