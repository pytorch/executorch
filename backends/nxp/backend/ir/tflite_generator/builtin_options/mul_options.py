#
# Copyright 2023 Martin Pavella
# Copyright 2023 NXP
#
# License: MIT
# See the LICENSE_MIT for more details.
#

import executorch.backends.nxp.backend.ir.tflite_generator.meta.meta as meta
import flatbuffers as fb
from executorch.backends.nxp.backend.ir.lib.tflite import MulOptions
from executorch.backends.nxp.backend.ir.lib.tflite.ActivationFunctionType import (
    ActivationFunctionType,
)
from executorch.backends.nxp.backend.ir.lib.tflite.BuiltinOperator import (
    BuiltinOperator,
)
from executorch.backends.nxp.backend.ir.lib.tflite.BuiltinOptions import BuiltinOptions


class Mul(meta.BuiltinOptions):
    fused_activation_function: ActivationFunctionType

    def __init__(
        self,
        fused_activation_function: ActivationFunctionType = ActivationFunctionType.NONE,
    ) -> None:
        super().__init__(BuiltinOptions.MulOptions, BuiltinOperator.MUL)
        self.fused_activation_function = fused_activation_function

    def gen_tflite(self, builder: fb.Builder):
        MulOptions.Start(builder)

        MulOptions.AddFusedActivationFunction(builder, self.fused_activation_function)

        return MulOptions.End(builder)
