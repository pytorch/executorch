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
    SliceOptions as libSliceOptions,
)
from executorch.backends.nxp.backend.ir.lib.tflite.BuiltinOperator import (
    BuiltinOperator,
)
from executorch.backends.nxp.backend.ir.lib.tflite.BuiltinOptions import BuiltinOptions


class Slice(meta.BuiltinOptions):
    def __init__(self) -> None:
        super().__init__(BuiltinOptions.SliceOptions, BuiltinOperator.SLICE)

    def gen_tflite(self, builder: fb.Builder):
        libSliceOptions.Start(builder)
        return libSliceOptions.End(builder)
