# Copyright 2023 NXP
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
    depthwise_conv_2d_options

Representation of the TFLite operator 'DepthwiseConv2D'.
"""

import executorch.backends.nxp.backend.ir.lib.tflite.DepthwiseConv2DOptions as libDepthwiseConv2DOptions
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


class DepthwiseConv2D(meta.BuiltinOptions):
    padding: Padding
    stride_w: int = 1
    stride_h: int = 1
    fused_activation_function: ActivationFunctionType
    dilation_w_factor: int = 1
    dilation_h_factor: int = 1
    depth_multiplier: int = 1  # Redundant according to schema.fbs (line 597)

    def __init__(
        self,
        padding: Padding = Padding.SAME,
        stride_w: int = 1,
        stride_h: int = 1,
        dilation_w_factor: int = 1,
        dilation_h_factor: int = 1,
        fused_activation_function: ActivationFunctionType = ActivationFunctionType.NONE,
        depth_multiplier: int = 1,
    ) -> None:
        super().__init__(
            BuiltinOptions.DepthwiseConv2DOptions, BuiltinOperator.DEPTHWISE_CONV_2D
        )
        self.padding = padding
        self.stride_w = stride_w
        self.stride_h = stride_h
        self.fused_activation_function = fused_activation_function
        self.dilation_w_factor = dilation_w_factor
        self.dilation_h_factor = dilation_h_factor
        self.depth_multiplier = depth_multiplier

    def gen_tflite(self, builder: fb.Builder):
        libDepthwiseConv2DOptions.Start(builder)

        libDepthwiseConv2DOptions.AddPadding(builder, self.padding)
        libDepthwiseConv2DOptions.AddStrideW(builder, self.stride_w)
        libDepthwiseConv2DOptions.AddStrideH(builder, self.stride_h)
        libDepthwiseConv2DOptions.AddFusedActivationFunction(
            builder, self.fused_activation_function
        )
        libDepthwiseConv2DOptions.AddDilationWFactor(builder, self.dilation_w_factor)
        libDepthwiseConv2DOptions.AddDilationHFactor(builder, self.dilation_h_factor)

        libDepthwiseConv2DOptions.AddDepthMultiplier(builder, self.depth_multiplier)

        return libDepthwiseConv2DOptions.End(builder)
