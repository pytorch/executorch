# Copyright 2023 NXP
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import flatbuffers

from executorch.backends.nxp.backend.ir.lib.tflite import (
    BuiltinOperator,
    BuiltinOptions,
    PadOptions,
)
from executorch.backends.nxp.backend.ir.tflite_generator.meta import meta


class Pad(meta.BuiltinOptions):

    def __init__(self) -> None:
        super().__init__(
            BuiltinOptions.BuiltinOptions.PadOptions,
            BuiltinOperator.BuiltinOperator.PAD,
        )

    def gen_tflite(self, builder: flatbuffers.Builder):
        PadOptions.Start(builder)

        return PadOptions.End(builder)
