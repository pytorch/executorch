# Copyright 2024 NXP
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import executorch.backends.nxp.backend.ir.tflite_generator.meta.meta as meta
import flatbuffers as fb
from executorch.backends.nxp.backend.ir.lib.tflite import GeluOptions
from executorch.backends.nxp.backend.ir.lib.tflite.BuiltinOperator import (
    BuiltinOperator,
)
from executorch.backends.nxp.backend.ir.lib.tflite.BuiltinOptions import BuiltinOptions


class Gelu(meta.BuiltinOptions):
    approximate: bool

    def __init__(self, approximate: bool) -> None:
        super().__init__(BuiltinOptions.GeluOptions, BuiltinOperator.GELU)
        self.approximate = approximate

    def gen_tflite(self, builder: fb.Builder):
        GeluOptions.Start(builder)

        GeluOptions.AddApproximate(builder, self.approximate)

        return GeluOptions.End(builder)
