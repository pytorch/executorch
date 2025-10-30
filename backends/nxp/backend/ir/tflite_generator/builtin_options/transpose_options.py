#
# Copyright 2023 Martin Pavella
# Copyright 2024 NXP
#
# License: MIT
# See the LICENSE_MIT for more details.
#
"""
    Transpose

Representation of the TFLite operator 'Transpose'.
"""

import executorch.backends.nxp.backend.ir.lib.tflite.BuiltinOperator as libBuiltinOperator
import executorch.backends.nxp.backend.ir.lib.tflite.BuiltinOptions as libBuiltinOptions
import executorch.backends.nxp.backend.ir.lib.tflite.TransposeOptions as libTransposeOptions
import executorch.backends.nxp.backend.ir.tflite_generator.meta.meta as meta
import flatbuffers as fb


class Transpose(meta.BuiltinOptions):
    def __init__(self) -> None:
        super().__init__(
            libBuiltinOptions.BuiltinOptions.TransposeOptions,
            libBuiltinOperator.BuiltinOperator.TRANSPOSE,
        )

    def gen_tflite(self, builder: fb.Builder):
        libTransposeOptions.Start(builder)
        return libTransposeOptions.End(builder)
