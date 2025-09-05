# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# Copyright 2025 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


# This file contains all the functions that decompose one op into simpler ops in the
# graph.

# pyre-strict

from typing import Dict

from executorch.backends.cadence.aot.pass_utils import (
    CadencePassAttribute,
    register_cadence_pass,
)
from executorch.exir.dialects._ops import ops as exir_ops
from executorch.exir.dialects.edge._ops import EdgeOpOverload
from executorch.exir.pass_base import ExportPass, NodeMetadata, ProxyValue
from torch.fx.node import Argument


@register_cadence_pass(CadencePassAttribute(opt_level=0))
class DecomposeAtenApproxGeluPass(ExportPass):
    """
    Decompose the aten gelu op with an approximate arg to a series of simpler ops
    """

    def call_operator(
        self,
        op: EdgeOpOverload,
        args: tuple[Argument, ...],
        kwargs: Dict[str, Argument],
        meta: NodeMetadata,
    ) -> ProxyValue:
        # compute the approximate gelu (0.7978845608028654 is sqrt(2 / pi))
        # as 0.5 * x * (1 + torch.tanh(0.7978845608028654 * ( x + 0.044715 * x^3)))

        # Get 0.5 * x
        half = super().call_operator(
            exir_ops.edge.aten.mul.Tensor,
            (args[0], 0.5),
            {},
            meta,
        )

        scaled = super().call_operator(
            exir_ops.edge.aten.mul.Tensor,
            (args[0], 0.044715),
            {},
            meta,
        )

        # Get x^2 (note that we use mul.Tensor twice instead of pow.Tensor because
        # it is much more efficient on DSP backends)
        scaled_square = super().call_operator(
            exir_ops.edge.aten.mul.Tensor,
            (scaled, args[0]),
            {},
            meta,
        )

        # Get x^3
        scaled_cubed = super().call_operator(
            exir_ops.edge.aten.mul.Tensor,
            (scaled_square, args[0]),
            {},
            meta,
        )

        # Get x + 0.044715 * x^3
        inner_sum = super().call_operator(
            exir_ops.edge.aten.add.Tensor,
            (scaled_cubed, args[0]),
            {},
            meta,
        )

        # Get 0.7978845608028654 * ( x + 0.044715 * x^3)
        scaled_sum = super().call_operator(
            exir_ops.edge.aten.mul.Tensor,
            (inner_sum, 0.7978845608028654),
            {},
            meta,
        )

        # Get torch.tanh(0.7978845608028654 * ( x + 0.044715 * x^3))
        tanh = super().call_operator(
            exir_ops.edge.aten.tanh.default,
            (scaled_sum,),
            {},
            meta,
        )

        # Get 1 + torch.tanh(0.79788456 * ( x + 0.044715 * x^3))
        # TODO(): Check why this is not working properly with integer values (e.g. 1 instead of 1.)
        outer_sum = super().call_operator(
            exir_ops.edge.aten.add.Tensor,
            (tanh, 1.0),
            {},
            meta,
        )

        # Return the final result
        return super().call_operator(
            exir_ops.edge.aten.mul.Tensor,
            (half, outer_sum),
            {},
            meta,
        )


# This class encapsulates all the functions that decompose one op in the graph.
class CadenceDecomposeOpsInGraph:
    passes = [
        DecomposeAtenApproxGeluPass,
    ]
