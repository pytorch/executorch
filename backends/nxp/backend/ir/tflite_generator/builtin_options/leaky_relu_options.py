#
# Copyright 2023 Martin Pavella
# Copyright 2024 NXP
#
# License: MIT
# See the LICENSE_MIT for more details.
#


import executorch.backends.nxp.backend.ir.lib.tflite.BuiltinOperator as libBuiltinOperator
import executorch.backends.nxp.backend.ir.lib.tflite.BuiltinOptions as libBuiltinOptions
import executorch.backends.nxp.backend.ir.lib.tflite.LeakyReluOptions as libLeakyReluOptions
import executorch.backends.nxp.backend.ir.tflite_generator.meta.meta as meta
import flatbuffers as fb


class LeakyRelu(meta.BuiltinOptions):
    alpha: float

    def __init__(self, alpha: float) -> None:
        super().__init__(
            libBuiltinOptions.BuiltinOptions.LeakyReluOptions,
            libBuiltinOperator.BuiltinOperator.LEAKY_RELU,
        )
        self.alpha = alpha

    def gen_tflite(self, builder: fb.Builder):
        libLeakyReluOptions.Start(builder)

        libLeakyReluOptions.AddAlpha(builder, self.alpha)

        return libLeakyReluOptions.End(builder)
