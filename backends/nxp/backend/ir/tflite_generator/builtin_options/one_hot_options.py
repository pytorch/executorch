# Copyright 2024 NXP
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import executorch.backends.nxp.backend.ir.tflite_generator.meta.meta as meta
import flatbuffers as fb
from executorch.backends.nxp.backend.ir.lib.tflite import OneHotOptions
from executorch.backends.nxp.backend.ir.lib.tflite.BuiltinOperator import (
    BuiltinOperator,
)
from executorch.backends.nxp.backend.ir.lib.tflite.BuiltinOptions import BuiltinOptions


class OneHot(meta.BuiltinOptions):
    axis: int

    def __init__(self, axis: int) -> None:
        super().__init__(BuiltinOptions.OneHotOptions, BuiltinOperator.ONE_HOT)
        self.axis = axis

    def gen_tflite(self, builder: fb.Builder):
        OneHotOptions.Start(builder)

        OneHotOptions.AddAxis(builder, self.axis)

        return OneHotOptions.End(builder)
