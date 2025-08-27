#
# Copyright 2023 Martin Pavella
# Copyright 2023 NXP
#
# License: MIT
# See the LICENSE_MIT for more details.
#

import executorch.backends.nxp.backend.ir.lib.tflite.FullyConnectedOptions as libFullyConnectedOptions
import executorch.backends.nxp.backend.ir.tflite_generator.meta.meta as meta
import flatbuffers as fb
from executorch.backends.nxp.backend.ir.lib.tflite.ActivationFunctionType import (
    ActivationFunctionType,
)
from executorch.backends.nxp.backend.ir.lib.tflite.BuiltinOperator import (
    BuiltinOperator,
)
from executorch.backends.nxp.backend.ir.lib.tflite.BuiltinOptions import BuiltinOptions
from executorch.backends.nxp.backend.ir.lib.tflite.FullyConnectedOptionsWeightsFormat import (
    FullyConnectedOptionsWeightsFormat,
)


class FullyConnected(meta.BuiltinOptions):
    fused_activation_function: ActivationFunctionType
    weights_format: FullyConnectedOptionsWeightsFormat
    keep_num_dims: bool
    asymmetric_quantize_inputs: bool

    def __init__(
        self,
        fused_activation_function: ActivationFunctionType = ActivationFunctionType.NONE,
        weights_format: FullyConnectedOptionsWeightsFormat = FullyConnectedOptionsWeightsFormat.DEFAULT,
        keep_num_dims: bool = False,
        asymmetric_quantize_inputs: bool = False,
    ) -> None:
        super().__init__(
            BuiltinOptions.FullyConnectedOptions, BuiltinOperator.FULLY_CONNECTED
        )
        self.fused_activation_function = fused_activation_function
        self.weights_format = weights_format
        self.keep_num_dims = keep_num_dims
        self.asymmetric_quantize_inputs = asymmetric_quantize_inputs

    def gen_tflite(self, builder: fb.Builder):
        libFullyConnectedOptions.Start(builder)

        libFullyConnectedOptions.AddFusedActivationFunction(
            builder, self.fused_activation_function
        )
        libFullyConnectedOptions.AddWeightsFormat(builder, self.weights_format)
        libFullyConnectedOptions.AddKeepNumDims(builder, self.keep_num_dims)
        libFullyConnectedOptions.AddAsymmetricQuantizeInputs(
            builder, self.asymmetric_quantize_inputs
        )

        return libFullyConnectedOptions.End(builder)
