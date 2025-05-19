# Copyright 2023 NXP
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import executorch.backends.nxp.backend.ir.lib.tflite.BuiltinOperator as libBuiltinOperator
import executorch.backends.nxp.backend.ir.lib.tflite.BuiltinOptions as libBuiltinOptions
import executorch.backends.nxp.backend.ir.lib.tflite.QuantizeOptions as libQuantizeOptions
import executorch.backends.nxp.backend.ir.tflite_generator.meta.meta as meta
import flatbuffers as fb


class Dequantize(meta.BuiltinOptions):

    def __init__(self) -> None:
        super().__init__(
            libBuiltinOptions.BuiltinOptions.DequantizeOptions,
            libBuiltinOperator.BuiltinOperator.DEQUANTIZE,
        )

    def gen_tflite(self, builder: fb.Builder):
        libQuantizeOptions.Start(builder)

        return libQuantizeOptions.End(builder)
