#
# Copyright 2023 Martin Pavella
#
# License: MIT
# See the LICENSE_MIT for more details.
#
"""
    Sub

Representation of the TFLite operator 'Sub'.
"""

import flatbuffers as fb

import executorch.backends.nxp.backend.ir.tflite_generator.meta.meta as meta
from executorch.backends.nxp.backend.ir.lib.tflite import (
    BuiltinOptions as libBuiltinOptions,
    BuiltinOperator as libBuiltinOperator,
    ActivationFunctionType as libActivationFunctionType,
    SubOptions as libSubOptions
)


class Sub(meta.BuiltinOptions):
    fused_activation_function: libActivationFunctionType.ActivationFunctionType

    # TODO potScaleInt16

    def __init__(self,
                 fused_activation_function: libActivationFunctionType.ActivationFunctionType = libActivationFunctionType.ActivationFunctionType.NONE
                 ) -> None:
        super().__init__(libBuiltinOptions.BuiltinOptions.SubOptions,
                         libBuiltinOperator.BuiltinOperator.SUB)
        self.fused_activation_function = fused_activation_function

    def gen_tflite(self, builder: fb.Builder):
        libSubOptions.Start(builder)

        libSubOptions.AddFusedActivationFunction(builder, self.fused_activation_function)

        return libSubOptions.End(builder)
