# Copyright 2023 NXP
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


import executorch.backends.nxp.backend.ir.tflite_generator.meta.meta as meta
import flatbuffers as fb
from executorch.backends.nxp.backend.ir.lib.tflite import CastOptions as libCastOptions
from executorch.backends.nxp.backend.ir.lib.tflite.BuiltinOperator import (
    BuiltinOperator,
)
from executorch.backends.nxp.backend.ir.lib.tflite.BuiltinOptions import BuiltinOptions
from executorch.backends.nxp.backend.ir.lib.tflite.TensorType import TensorType


class Cast(meta.BuiltinOptions):
    in_data_type: TensorType
    out_data_type: TensorType

    def __init__(self, in_data_type: TensorType, out_data_type: TensorType) -> None:
        super().__init__(BuiltinOptions.CastOptions, BuiltinOperator.CAST)
        self.in_data_type = in_data_type
        self.out_data_type = out_data_type

    def gen_tflite(self, builder: fb.Builder):
        libCastOptions.Start(builder)

        libCastOptions.AddInDataType(builder, self.in_data_type)
        libCastOptions.AddOutDataType(builder, self.out_data_type)

        return libCastOptions.End(builder)
