# Copyright 2024 NXP
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import executorch.backends.nxp.backend.ir.tflite_generator.meta.meta as meta
import flatbuffers as fb
from executorch.backends.nxp.backend.ir.lib.tflite import RandomOptions
from executorch.backends.nxp.backend.ir.lib.tflite.BuiltinOperator import (
    BuiltinOperator,
)
from executorch.backends.nxp.backend.ir.lib.tflite.BuiltinOptions import BuiltinOptions


class Multinomial(meta.BuiltinOptions):
    seed: int
    seed2: int

    def __init__(self, seed: int, seed2: int) -> None:
        super().__init__(BuiltinOptions.RandomOptions, BuiltinOperator.MULTINOMIAL)
        self.seed = seed
        self.seed2 = seed2

    def gen_tflite(self, builder: fb.Builder):
        RandomOptions.Start(builder)

        RandomOptions.AddSeed(builder, self.seed)
        RandomOptions.AddSeed2(builder, self.seed2)

        return RandomOptions.End(builder)
