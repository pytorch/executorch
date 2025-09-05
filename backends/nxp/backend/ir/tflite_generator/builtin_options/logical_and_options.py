# Copyright 2024 NXP
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import executorch.backends.nxp.backend.ir.tflite_generator.meta.meta as meta
import flatbuffers as fb
from executorch.backends.nxp.backend.ir.lib.tflite import LogicalAndOptions
from executorch.backends.nxp.backend.ir.lib.tflite.BuiltinOperator import (
    BuiltinOperator,
)
from executorch.backends.nxp.backend.ir.lib.tflite.BuiltinOptions import BuiltinOptions


class LogicalAnd(meta.BuiltinOptions):

    def __init__(self) -> None:
        super().__init__(BuiltinOptions.LogicalAndOptions, BuiltinOperator.LOGICAL_AND)

    def gen_tflite(self, builder: fb.Builder):
        LogicalAndOptions.Start(builder)

        return LogicalAndOptions.End(builder)
