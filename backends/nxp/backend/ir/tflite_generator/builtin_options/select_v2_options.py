# Copyright 2023 NXP
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


import executorch.backends.nxp.backend.ir.lib.tflite.BuiltinOperator as libBuiltinOperator
import executorch.backends.nxp.backend.ir.lib.tflite.BuiltinOptions as libBuiltinOptions
import executorch.backends.nxp.backend.ir.lib.tflite.SelectV2Options as libSelectV2Options
import executorch.backends.nxp.backend.ir.tflite_generator.meta.meta as meta
import flatbuffers as fb


class SelectV2(meta.BuiltinOptions):

    def __init__(self) -> None:
        super().__init__(
            libBuiltinOptions.BuiltinOptions.SelectV2Options,
            libBuiltinOperator.BuiltinOperator.SELECT_V2,
        )

    def gen_tflite(self, builder: fb.Builder):
        libSelectV2Options.Start(builder)
        return libSelectV2Options.End(builder)
