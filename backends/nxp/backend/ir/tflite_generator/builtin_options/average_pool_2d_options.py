#
# Copyright 2023 Martin Pavella
# Copyright 2024 NXP
#
# License: MIT
# See the LICENSE_MIT for more details.
#
"""
    AveragePool2D

Representation of the TFLite operator 'AveragePool2D'.
"""

import executorch.backends.nxp.backend.ir.lib.tflite.ActivationFunctionType as libActivationFunctionType
import executorch.backends.nxp.backend.ir.lib.tflite.BuiltinOperator as libBuiltinOperator
import executorch.backends.nxp.backend.ir.lib.tflite.BuiltinOptions as libBuiltinOptions
import executorch.backends.nxp.backend.ir.lib.tflite.Padding as libPadding
import executorch.backends.nxp.backend.ir.lib.tflite.Pool2DOptions as libPool2DOptions
import executorch.backends.nxp.backend.ir.tflite_generator.meta.meta as meta
import flatbuffers as fb


class AveragePool2D(meta.BuiltinOptions):
    padding: libPadding.Padding
    stride_w: int
    stride_h: int
    filter_w: int
    filter_h: int
    fused_activation_function: libActivationFunctionType.ActivationFunctionType

    def __init__(
        self,
        padding: libPadding.Padding = libPadding.Padding.SAME,
        stride_w: int = 1,
        stride_h: int = 1,
        filter_w: int = 1,
        filter_h: int = 1,
        fused_activation_function: libActivationFunctionType.ActivationFunctionType = libActivationFunctionType.ActivationFunctionType.NONE,
    ) -> None:
        super().__init__(
            libBuiltinOptions.BuiltinOptions.Pool2DOptions,
            libBuiltinOperator.BuiltinOperator.AVERAGE_POOL_2D,
        )
        self.padding = padding
        self.stride_w = stride_w
        self.stride_h = stride_h
        self.filter_w = filter_w
        self.filter_h = filter_h
        self.fused_activation_function = fused_activation_function

    def gen_tflite(self, builder: fb.Builder):
        libPool2DOptions.Start(builder)

        libPool2DOptions.AddPadding(builder, self.padding)
        libPool2DOptions.AddStrideW(builder, self.stride_w)
        libPool2DOptions.AddStrideH(builder, self.stride_h)
        libPool2DOptions.AddFilterHeight(builder, self.filter_h)
        libPool2DOptions.AddFilterWidth(builder, self.filter_w)
        libPool2DOptions.AddFusedActivationFunction(
            builder, self.fused_activation_function
        )

        return libPool2DOptions.End(builder)
