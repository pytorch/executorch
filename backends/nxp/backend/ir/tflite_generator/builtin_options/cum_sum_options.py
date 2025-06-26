# Copyright 2024 NXP
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import executorch.backends.nxp.backend.ir.tflite_generator.meta.meta as meta
import flatbuffers as fb
from executorch.backends.nxp.backend.ir.lib.tflite import CumsumOptions
from executorch.backends.nxp.backend.ir.lib.tflite.BuiltinOperator import (
    BuiltinOperator,
)
from executorch.backends.nxp.backend.ir.lib.tflite.BuiltinOptions import BuiltinOptions


class CumSum(meta.BuiltinOptions):
    exclusive: bool
    reverse: bool

    def __init__(self, exclusive: bool, reverse: bool) -> None:
        super().__init__(BuiltinOptions.CumsumOptions, BuiltinOperator.CUMSUM)
        self.exclusive = exclusive
        self.reverse = reverse

    def gen_tflite(self, builder: fb.Builder):
        CumsumOptions.Start(builder)

        CumsumOptions.AddExclusive(builder, self.exclusive)
        CumsumOptions.AddReverse(builder, self.reverse)

        return CumsumOptions.End(builder)
