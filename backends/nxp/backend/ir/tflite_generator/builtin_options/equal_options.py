# Copyright 2023 NXP
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import executorch.backends.nxp.backend.ir.lib.tflite.EqualOptions as libEqualOptions
import flatbuffers as fb
from executorch.backends.nxp.backend.ir.lib.tflite.BuiltinOperator import (
    BuiltinOperator,
)
from executorch.backends.nxp.backend.ir.lib.tflite.BuiltinOptions import BuiltinOptions
from executorch.backends.nxp.backend.ir.tflite_generator.meta import meta


class Equal(meta.BuiltinOptions):

    def __init__(self) -> None:
        super().__init__(BuiltinOptions.EqualOptions, BuiltinOperator.EQUAL)

    def gen_tflite(self, builder: fb.Builder):
        libEqualOptions.Start(builder)

        return libEqualOptions.End(builder)
