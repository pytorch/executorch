# Copyright 2023 NXP
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
    batch_mat_mul_options

Representation of the TFLite operator 'BatchMatMul'.
"""
import executorch.backends.nxp.backend.ir.tflite_generator.meta.meta as meta
import flatbuffers as fb
from executorch.backends.nxp.backend.ir.lib.tflite import (
    BatchMatMulOptions,
    BuiltinOperator,
    BuiltinOptions,
)


class BatchMatMul(meta.BuiltinOptions):
    adj_x: bool
    adj_y: bool
    asymmetric_quantize_inputs: bool

    def __init__(
        self, adj_x: bool, adj_y: bool, asymmetric_quantize_inputs: bool
    ) -> None:
        super().__init__(
            BuiltinOptions.BuiltinOptions.BatchMatMulOptions,
            BuiltinOperator.BuiltinOperator.BATCH_MATMUL,
        )
        self.adj_x = adj_x
        self.adj_y = adj_y
        self.asymmetric_quantize_inputs = asymmetric_quantize_inputs

    def gen_tflite(self, builder: fb.Builder):
        BatchMatMulOptions.Start(builder)

        BatchMatMulOptions.AddAdjX(builder, self.adj_x)
        BatchMatMulOptions.AddAdjY(builder, self.adj_y)
        BatchMatMulOptions.AddAsymmetricQuantizeInputs(
            builder, self.asymmetric_quantize_inputs
        )

        return BatchMatMulOptions.End(builder)
