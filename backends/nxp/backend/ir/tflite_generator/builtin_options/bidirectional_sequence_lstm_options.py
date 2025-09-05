# Copyright 2024 NXP
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import executorch.backends.nxp.backend.ir.lib.tflite.BidirectionalSequenceLSTMOptions as libBSLSTMOptions
import flatbuffers as fb
from executorch.backends.nxp.backend.ir.lib.tflite.ActivationFunctionType import (
    ActivationFunctionType,
)
from executorch.backends.nxp.backend.ir.lib.tflite.BuiltinOperator import (
    BuiltinOperator,
)
from executorch.backends.nxp.backend.ir.lib.tflite.BuiltinOptions import BuiltinOptions
from executorch.backends.nxp.backend.ir.tflite_generator.meta import meta


class BidirectionalSequenceLSTM(meta.BuiltinOptions):
    fused_activation_function: ActivationFunctionType
    cell_clip: float
    proj_clip: float
    merge_outputs: bool

    # V2+
    time_major: bool  # If True, the first dimension is sequence, otherwise batch.

    # V3+
    asymmetric_quantize_inputs: bool

    def __init__(
        self,
        cell_clip: float,
        proj_clip: float,
        time_major: bool = True,
        merge_outputs: bool = True,
        asymmetric_quantize_inputs: bool = False,
        fused_activation_function: ActivationFunctionType = ActivationFunctionType.NONE,
    ) -> None:
        super().__init__(
            BuiltinOptions.BidirectionalSequenceLSTMOptions,
            BuiltinOperator.BIDIRECTIONAL_SEQUENCE_LSTM,
        )

        self.fused_activation_function = fused_activation_function
        self.cell_clip = cell_clip
        self.proj_clip = proj_clip
        self.merge_outputs = merge_outputs
        self.time_major = time_major
        self.asymmetric_quantize_inputs = asymmetric_quantize_inputs

    def gen_tflite(self, builder: fb.Builder):
        libBSLSTMOptions.Start(builder)

        libBSLSTMOptions.AddFusedActivationFunction(
            builder, self.fused_activation_function
        )
        libBSLSTMOptions.AddCellClip(builder, self.cell_clip)
        libBSLSTMOptions.AddProjClip(builder, self.proj_clip)
        libBSLSTMOptions.AddMergeOutputs(builder, self.merge_outputs)
        libBSLSTMOptions.AddTimeMajor(builder, self.time_major)
        libBSLSTMOptions.AddAsymmetricQuantizeInputs(
            builder, self.asymmetric_quantize_inputs
        )

        return libBSLSTMOptions.End(builder)
