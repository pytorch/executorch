#
# Copyright 2023 Martin Pavella
# Copyright 2023 NXP
#
# License: MIT
# See the LICENSE_MIT for more details.
#

import executorch.backends.nxp.backend.ir.lib.tflite.Conv2DOptions as libConv2DOptions
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


class Conv2D(meta.BuiltinOptions):
    padding: Padding
    stride_w: int
    stride_h: int
    dilation_w_factor: int
    dilation_h_factor: int
    fused_activation_function: ActivationFunctionType

    def __init__(
        self,
        padding: Padding = Padding.SAME,
        stride_w: int = 1,
        stride_h: int = 1,
        dilation_w_factor: int = 1,
        dilation_h_factor: int = 1,
        fused_activation_function: ActivationFunctionType = ActivationFunctionType.NONE,
    ) -> None:
        super().__init__(BuiltinOptions.Conv2DOptions, BuiltinOperator.CONV_2D)
        self.padding = padding
        self.stride_w = stride_w
        self.stride_h = stride_h
        self.dilation_w_factor = dilation_w_factor
        self.dilation_h_factor = dilation_h_factor
        self.fused_activation_function = fused_activation_function

    def gen_tflite(self, builder: fb.Builder):
        libConv2DOptions.Start(builder)

        libConv2DOptions.AddPadding(builder, self.padding)
        libConv2DOptions.AddStrideW(builder, self.stride_w)
        libConv2DOptions.AddStrideH(builder, self.stride_h)
        libConv2DOptions.AddFusedActivationFunction(
            builder, self.fused_activation_function
        )
        libConv2DOptions.AddDilationWFactor(builder, self.dilation_w_factor)
        libConv2DOptions.AddDilationHFactor(builder, self.dilation_h_factor)

        return libConv2DOptions.End(builder)
