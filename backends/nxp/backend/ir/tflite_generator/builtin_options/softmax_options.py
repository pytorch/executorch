#
# Copyright 2023 Martin Pavella
# Copyright 2024 NXP
#
# License: MIT
# See the LICENSE_MIT for more details.
#
"""
    Softmax

Representation of the TFLite operator 'Softmax'.
"""

import executorch.backends.nxp.backend.ir.lib.tflite.BuiltinOperator as libBuiltinOperator
import executorch.backends.nxp.backend.ir.lib.tflite.BuiltinOptions as libBuiltinOptions
import executorch.backends.nxp.backend.ir.lib.tflite.SoftmaxOptions as libSoftmaxOptions
import executorch.backends.nxp.backend.ir.tflite_generator.meta.meta as meta
import flatbuffers as fb


class Softmax(meta.BuiltinOptions):
    beta: float

    def __init__(self, beta: float) -> None:
        super().__init__(
            libBuiltinOptions.BuiltinOptions.SoftmaxOptions,
            libBuiltinOperator.BuiltinOperator.SOFTMAX,
        )
        self.beta = beta

    def gen_tflite(self, builder: fb.Builder):
        libSoftmaxOptions.Start(builder)

        libSoftmaxOptions.AddBeta(builder, self.beta)

        return libSoftmaxOptions.End(builder)
