#
# Copyright 2023 Martin Pavella
# Copyright 2024 NXP
#
# License: MIT
# See the LICENSE_MIT for more details.
#
"""
    Reshape

Representation of the TFLite operator 'Reshape'.
"""

from typing import List, Optional

import executorch.backends.nxp.backend.ir.lib.tflite.BuiltinOperator as libBuiltinOperator
import executorch.backends.nxp.backend.ir.lib.tflite.BuiltinOptions as libBuiltinOptions
import executorch.backends.nxp.backend.ir.lib.tflite.ReshapeOptions as libReshapeOptions
import executorch.backends.nxp.backend.ir.tflite_generator.meta.meta as meta

import flatbuffers as fb


class NewShape(meta.IntVector):
    def __init__(self, new_shape: List[int]) -> None:
        super().__init__(new_shape, libReshapeOptions.StartNewShapeVector)


class Reshape(meta.BuiltinOptions):
    new_shape: Optional[NewShape]

    def __init__(self, new_shape: Optional[List[int]]) -> None:
        super().__init__(
            libBuiltinOptions.BuiltinOptions.ReshapeOptions,
            libBuiltinOperator.BuiltinOperator.RESHAPE,
        )
        if new_shape is not None:
            self.new_shape = NewShape(new_shape)
        else:
            self.new_shape = None

    def gen_tflite(self, builder: fb.Builder):
        if self.new_shape is not None:
            tfl_new_shape = self.new_shape.gen_tflite(builder)
        else:
            tfl_new_shape = None

        libReshapeOptions.Start(builder)

        if tfl_new_shape is not None:
            libReshapeOptions.AddNewShape(builder, tfl_new_shape)

        return libReshapeOptions.End(builder)
