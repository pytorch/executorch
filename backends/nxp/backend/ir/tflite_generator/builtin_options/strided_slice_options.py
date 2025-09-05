# Copyright 2023 NXP
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
    slice_options

Representation of the TFLite operator 'Slice'.
"""

import executorch.backends.nxp.backend.ir.tflite_generator.meta.meta as meta
import flatbuffers as fb
from executorch.backends.nxp.backend.ir.lib.tflite import (
    StridedSliceOptions as libStridedSliceOptions,
)
from executorch.backends.nxp.backend.ir.lib.tflite.BuiltinOperator import (
    BuiltinOperator,
)
from executorch.backends.nxp.backend.ir.lib.tflite.BuiltinOptions import BuiltinOptions


class StridedSlice(meta.BuiltinOptions):
    offset: bool

    def __init__(self, offset: bool = False) -> None:
        super().__init__(
            BuiltinOptions.StridedSliceOptions, BuiltinOperator.STRIDED_SLICE
        )
        self.offset = offset

    def gen_tflite(self, builder: fb.Builder):
        libStridedSliceOptions.Start(builder)
        libStridedSliceOptions.AddOffset(builder, self.offset)
        return libStridedSliceOptions.End(builder)
