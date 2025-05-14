# Copyright 2023 NXP
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import executorch.backends.nxp.backend.ir.lib.tflite.LSTMOptions as libLSTMOptions
import flatbuffers as fb
from executorch.backends.nxp.backend.ir.lib.tflite.ActivationFunctionType import (
    ActivationFunctionType,
)
from executorch.backends.nxp.backend.ir.lib.tflite.BuiltinOperator import (
    BuiltinOperator,
)
from executorch.backends.nxp.backend.ir.lib.tflite.BuiltinOptions import BuiltinOptions
from executorch.backends.nxp.backend.ir.lib.tflite.LSTMKernelType import LSTMKernelType
from executorch.backends.nxp.backend.ir.tflite_generator.meta import meta


class LSTM(meta.BuiltinOptions):
    # LSTM v1+
    fused_activation_function: ActivationFunctionType
    cell_clip: float
    proj_clip: float

    # LSTM v2+
    kernel_type: LSTMKernelType

    # LSTM v4+
    asymmetric_quantize_inputs: bool

    def __init__(
        self,
        cell_clip: float,
        proj_clip: float,
        kernel_type: LSTMKernelType = LSTMKernelType.FULL,
        asymmetric_quantize_inputs: bool = False,
        fused_activation_function: ActivationFunctionType = ActivationFunctionType.NONE,
    ) -> None:
        super().__init__(BuiltinOptions.LSTMOptions, BuiltinOperator.LSTM)

        self.cell_clip = cell_clip
        self.proj_clip = proj_clip
        self.kernel_type = kernel_type
        self.asymmetric_quantize_inputs = asymmetric_quantize_inputs
        self.fused_activation_function = fused_activation_function

    def gen_tflite(self, builder: fb.Builder):
        libLSTMOptions.Start(builder)

        libLSTMOptions.AddFusedActivationFunction(
            builder, self.fused_activation_function
        )
        libLSTMOptions.AddCellClip(builder, self.cell_clip)
        libLSTMOptions.AddProjClip(builder, self.proj_clip)
        libLSTMOptions.AddKernelType(builder, self.kernel_type)
        libLSTMOptions.AddAsymmetricQuantizeInputs(
            builder, self.asymmetric_quantize_inputs
        )

        return libLSTMOptions.End(builder)
