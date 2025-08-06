#
# Copyright 2023 Martin Pavella
# Copyright 2024 NXP
#
# License: MIT
# See the LICENSE_MIT for more details.
#
"""
    Add

Representation of the TFLite operator 'Add'.
"""

import executorch.backends.nxp.backend.ir.tflite_generator.meta.meta as meta
import flatbuffers as fb
from executorch.backends.nxp.backend.ir.lib.tflite import (
    ActivationFunctionType as libActivationFunctionType,
    AddOptions as libAddOptions,
    BuiltinOperator as libBuiltinOperator,
    BuiltinOptions as libBuiltinOptions,
)


class Add(meta.BuiltinOptions):
    fused_activation_function: libActivationFunctionType.ActivationFunctionType

    # TODO potScaleInt16

    def __init__(
        self,
        fused_activation_function: libActivationFunctionType.ActivationFunctionType = libActivationFunctionType.ActivationFunctionType.NONE,
    ) -> None:
        super().__init__(
            libBuiltinOptions.BuiltinOptions.AddOptions,
            libBuiltinOperator.BuiltinOperator.ADD,
        )
        self.fused_activation_function = fused_activation_function

    def gen_tflite(self, builder: fb.Builder):
        libAddOptions.Start(builder)

        libAddOptions.AddFusedActivationFunction(
            builder, self.fused_activation_function
        )

        return libAddOptions.End(builder)
