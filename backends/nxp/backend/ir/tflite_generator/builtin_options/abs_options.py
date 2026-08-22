# Copyright 2024 NXP
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import flatbuffers as fb

from executorch.backends.nxp.backend.ir.lib.tflite import AbsOptions

from executorch.backends.nxp.backend.ir.lib.tflite.BuiltinOperator import (
    BuiltinOperator,
)
from executorch.backends.nxp.backend.ir.lib.tflite.BuiltinOptions import BuiltinOptions
from executorch.backends.nxp.backend.ir.tflite_generator.meta import meta


class Abs(meta.BuiltinOptions):

    def __init__(self) -> None:
        super().__init__(BuiltinOptions.AbsOptions, BuiltinOperator.ABS)

    def gen_tflite(self, builder: fb.Builder):
        AbsOptions.Start(builder)

        return AbsOptions.End(builder)
