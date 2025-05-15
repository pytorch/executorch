# Copyright 2024 NXP
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import executorch.backends.nxp.backend.ir.lib.tflite.TransposeConvOptions as libTransposeConvOptions
import executorch.backends.nxp.backend.ir.tflite_generator.meta.meta as meta
import flatbuffers as fb
from executorch.backends.nxp.backend.ir.lib.tflite.ActivationFunctionType import (
    ActivationFunctionType,
)
from executorch.backends.nxp.backend.ir.lib.tflite.BuiltinOperator import (
    BuiltinOperator,
)
from executorch.backends.nxp.backend.ir.lib.tflite.BuiltinOptions import BuiltinOptions
from executorch.backends.nxp.backend.ir.lib.tflite.Padding import Padding


class TransposeConv(meta.BuiltinOptions):
    padding: Padding
    stride_w: int
    stride_h: int
    fused_activation_function: ActivationFunctionType

    def __init__(
        self,
        padding: Padding = Padding.SAME,
        stride_w: int = 1,
        stride_h: int = 1,
        fused_activation_function: ActivationFunctionType = ActivationFunctionType.NONE,
    ) -> None:
        super().__init__(
            BuiltinOptions.TransposeConvOptions, BuiltinOperator.TRANSPOSE_CONV
        )
        self.padding = padding
        self.stride_w = stride_w
        self.stride_h = stride_h
        self.fused_activation_function = fused_activation_function

    def gen_tflite(self, builder: fb.Builder):
        libTransposeConvOptions.Start(builder)

        libTransposeConvOptions.AddPadding(builder, self.padding)
        libTransposeConvOptions.AddStrideW(builder, self.stride_w)
        libTransposeConvOptions.AddStrideH(builder, self.stride_h)
        libTransposeConvOptions.AddFusedActivationFunction(
            builder, self.fused_activation_function
        )

        return libTransposeConvOptions.End(builder)
