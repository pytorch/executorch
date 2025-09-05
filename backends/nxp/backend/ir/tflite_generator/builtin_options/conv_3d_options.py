# Copyright 2023 NXP
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import executorch.backends.nxp.backend.ir.lib.tflite.Conv3DOptions as libConv3DOptions
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


class Conv3D(meta.BuiltinOptions):
    padding: Padding
    stride_w: int
    stride_h: int
    stride_d: int
    dilation_w_factor: int
    dilation_h_factor: int
    dilation_d_factor: int
    fused_activation_function: ActivationFunctionType

    def __init__(
        self,
        padding: Padding = Padding.SAME,
        stride_w: int = 1,
        stride_h: int = 1,
        stride_d: int = 1,
        dilation_w_factor: int = 1,
        dilation_h_factor: int = 1,
        dilation_d_factor: int = 1,
        fused_activation_function: ActivationFunctionType = ActivationFunctionType.NONE,
    ) -> None:
        super().__init__(BuiltinOptions.Conv3DOptions, BuiltinOperator.CONV_3D)
        self.padding = padding
        self.stride_w = stride_w
        self.stride_h = stride_h
        self.stride_d = stride_d
        self.dilation_w_factor = dilation_w_factor
        self.dilation_h_factor = dilation_h_factor
        self.dilation_d_factor = dilation_d_factor
        self.fused_activation_function = fused_activation_function

    def gen_tflite(self, builder: fb.Builder):
        libConv3DOptions.Start(builder)

        libConv3DOptions.AddPadding(builder, self.padding)

        libConv3DOptions.AddStrideW(builder, self.stride_w)
        libConv3DOptions.AddStrideH(builder, self.stride_h)
        libConv3DOptions.AddStrideD(builder, self.stride_d)

        libConv3DOptions.AddFusedActivationFunction(
            builder, self.fused_activation_function
        )

        libConv3DOptions.AddDilationWFactor(builder, self.dilation_w_factor)
        libConv3DOptions.AddDilationHFactor(builder, self.dilation_h_factor)
        libConv3DOptions.AddDilationDFactor(builder, self.dilation_d_factor)

        return libConv3DOptions.End(builder)
