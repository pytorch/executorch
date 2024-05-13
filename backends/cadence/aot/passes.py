# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from executorch.exir.dialects._ops import ops as exir_ops
from executorch.exir.pass_base import ExportPass


class ReplacePT2QuantWithCadenceQuant(ExportPass):
    """
    Replace the pt2 quantization ops with custom cadence quantization ops.
    """

    def call_operator(self, op, args, kwargs, meta):
        if op not in {exir_ops.edge.quantized_decomposed.quantize_per_tensor.default}:
            return super().call_operator(op, args, kwargs, meta)

        return super().call_operator(
            exir_ops.edge.cadence.quantize_per_tensor.default,
            args,
            kwargs,
            meta,
        )


class ReplacePT2DequantWithCadenceDequant(ExportPass):
    """
    Replace the pt2 dequantization ops with custom cadence dequantization ops.
    """

    def call_operator(self, op, args, kwargs, meta):
        if op not in {exir_ops.edge.quantized_decomposed.dequantize_per_tensor.default}:
            return super().call_operator(op, args, kwargs, meta)

        return super().call_operator(
            exir_ops.edge.cadence.dequantize_per_tensor.default,
            args,
            kwargs,
            meta,
        )
