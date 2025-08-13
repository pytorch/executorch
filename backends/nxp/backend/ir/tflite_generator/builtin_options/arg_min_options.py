# Copyright 2024 NXP
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import executorch.backends.nxp.backend.ir.tflite_generator.meta.meta as meta
import flatbuffers as fb
from executorch.backends.nxp.backend.ir.lib.tflite import ArgMinOptions
from executorch.backends.nxp.backend.ir.lib.tflite.BuiltinOperator import (
    BuiltinOperator,
)
from executorch.backends.nxp.backend.ir.lib.tflite.BuiltinOptions import BuiltinOptions
from executorch.backends.nxp.backend.ir.lib.tflite.TensorType import TensorType


class ArgMin(meta.BuiltinOptions):
    output_type: TensorType

    def __init__(self, output_type: TensorType) -> None:
        super().__init__(BuiltinOptions.ArgMinOptions, BuiltinOperator.ARG_MIN)
        self.output_type = output_type

    def gen_tflite(self, builder: fb.Builder):
        ArgMinOptions.Start(builder)

        ArgMinOptions.AddOutputType(builder, self.output_type)

        return ArgMinOptions.End(builder)
