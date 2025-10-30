# Copyright 2023 NXP
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
    shape_options

    Representation of a TFLite operator 'Shape'.
"""

import executorch.backends.nxp.backend.ir.lib.tflite.ShapeOptions as libShapeOptions
import executorch.backends.nxp.backend.ir.tflite_generator.meta.meta as meta
import flatbuffers as fb
from executorch.backends.nxp.backend.ir.lib.tflite.BuiltinOperator import (
    BuiltinOperator,
)
from executorch.backends.nxp.backend.ir.lib.tflite.BuiltinOptions import BuiltinOptions
from executorch.backends.nxp.backend.ir.lib.tflite.TensorType import TensorType


class Shape(meta.BuiltinOptions):
    out_type: TensorType

    def __init__(self, out_type: TensorType) -> None:
        super().__init__(BuiltinOptions.ShapeOptions, BuiltinOperator.SHAPE)
        self.out_type = out_type

    def gen_tflite(self, builder: fb.Builder):
        libShapeOptions.Start(builder)

        libShapeOptions.AddOutType(builder, self.out_type)

        return libShapeOptions.End(builder)
