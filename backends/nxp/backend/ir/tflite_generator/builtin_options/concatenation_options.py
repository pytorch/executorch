# Copyright 2023 NXP
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import executorch.backends.nxp.backend.ir.lib.tflite.ActivationFunctionType as libActivationFunctionType
import executorch.backends.nxp.backend.ir.lib.tflite.BuiltinOperator as libBuiltinOperator
import executorch.backends.nxp.backend.ir.lib.tflite.BuiltinOptions as libBuiltinOptions
import executorch.backends.nxp.backend.ir.lib.tflite.ConcatenationOptions as libConcatenationOptions
import executorch.backends.nxp.backend.ir.tflite_generator.meta.meta as meta
import flatbuffers as fb


class Concatenation(meta.BuiltinOptions):
    axis: int
    fused_activation_function: libActivationFunctionType.ActivationFunctionType

    def __init__(
        self,
        axis: int,
        fused_activation_function: libActivationFunctionType.ActivationFunctionType = libActivationFunctionType.ActivationFunctionType.NONE,
    ) -> None:
        super().__init__(
            libBuiltinOptions.BuiltinOptions.ConcatenationOptions,
            libBuiltinOperator.BuiltinOperator.CONCATENATION,
        )
        self.axis = axis
        self.fused_activation_function = fused_activation_function

    def gen_tflite(self, builder: fb.Builder):
        libConcatenationOptions.Start(builder)

        libConcatenationOptions.AddAxis(builder, self.axis)
        libConcatenationOptions.AddFusedActivationFunction(
            builder, self.fused_activation_function
        )

        return libConcatenationOptions.End(builder)
