# Copyright 2023 NXP
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import flatbuffers

from executorch.backends.nxp.backend.ir.lib.tflite import (
    BuiltinOperator,
    BuiltinOptions,
    PadV2Options,
)
from executorch.backends.nxp.backend.ir.tflite_generator.meta import meta


class PadV2(meta.BuiltinOptions):

    def __init__(self) -> None:
        super().__init__(
            BuiltinOptions.BuiltinOptions.PadV2Options,
            BuiltinOperator.BuiltinOperator.PADV2,
        )

    def gen_tflite(self, builder: flatbuffers.Builder):
        PadV2Options.Start(builder)

        return PadV2Options.End(builder)
