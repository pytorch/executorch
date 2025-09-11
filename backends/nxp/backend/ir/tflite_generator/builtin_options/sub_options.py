#
# Copyright 2023 Martin Pavella
# Copyright 2024 NXP
#
# License: MIT
# See the LICENSE_MIT for more details.
#
"""
    Sub

Representation of the TFLite operator 'Sub'.
"""

import executorch.backends.nxp.backend.ir.tflite_generator.meta.meta as meta
import flatbuffers as fb
from executorch.backends.nxp.backend.ir.lib.tflite import (
    ActivationFunctionType as libActivationFunctionType,
    BuiltinOperator as libBuiltinOperator,
    BuiltinOptions as libBuiltinOptions,
    SubOptions as libSubOptions,
)


class Sub(meta.BuiltinOptions):
    fused_activation_function: libActivationFunctionType.ActivationFunctionType

    # TODO potScaleInt16

    def __init__(
        self,
        fused_activation_function: libActivationFunctionType.ActivationFunctionType = libActivationFunctionType.ActivationFunctionType.NONE,
    ) -> None:
        super().__init__(
            libBuiltinOptions.BuiltinOptions.SubOptions,
            libBuiltinOperator.BuiltinOperator.SUB,
        )
        self.fused_activation_function = fused_activation_function

    def gen_tflite(self, builder: fb.Builder):
        libSubOptions.Start(builder)

        libSubOptions.AddFusedActivationFunction(
            builder, self.fused_activation_function
        )

        return libSubOptions.End(builder)
