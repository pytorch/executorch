#
# Copyright 2023 Martin Pavella
# Copyright 2023 NXP
#
# License: MIT
# See the LICENSE_MIT for more details.
#
"""
    Div

Representation of the TFLite operator 'Div'.
"""

import executorch.backends.nxp.backend.ir.tflite_generator.meta.meta as meta
import flatbuffers as fb
from executorch.backends.nxp.backend.ir.lib.tflite import DivOptions as libDivOptions
from executorch.backends.nxp.backend.ir.lib.tflite.ActivationFunctionType import (
    ActivationFunctionType,
)
from executorch.backends.nxp.backend.ir.lib.tflite.BuiltinOperator import (
    BuiltinOperator,
)
from executorch.backends.nxp.backend.ir.lib.tflite.BuiltinOptions import BuiltinOptions


class Div(meta.BuiltinOptions):
    fused_activation_function: ActivationFunctionType

    def __init__(
        self,
        fused_activation_function: ActivationFunctionType = ActivationFunctionType.NONE,
    ) -> None:
        super().__init__(BuiltinOptions.DivOptions, BuiltinOperator.DIV)
        self.fused_activation_function = fused_activation_function

    def gen_tflite(self, builder: fb.Builder):
        libDivOptions.Start(builder)

        libDivOptions.AddFusedActivationFunction(
            builder, self.fused_activation_function
        )

        return libDivOptions.End(builder)
