#
# Copyright 2023 Martin Pavella
# Copyright 2024 NXP
#
# License: MIT
# See the LICENSE_MIT for more details.
#
"""
    AddN

Representation of the TFLite operator 'AddN'.
"""

import executorch.backends.nxp.backend.ir.tflite_generator.meta.meta as meta
import flatbuffers as fb
from executorch.backends.nxp.backend.ir.lib.tflite import (
    AddNOptions as libAddNOptions,
    BuiltinOperator as libBuiltinOperator,
    BuiltinOptions as libBuiltinOptions,
)


class AddN(meta.BuiltinOptions):
    def __init__(self) -> None:
        super().__init__(
            libBuiltinOptions.BuiltinOptions.AddNOptions,
            libBuiltinOperator.BuiltinOperator.ADD_N,
        )

    def gen_tflite(self, builder: fb.Builder):
        libAddNOptions.Start(builder)
        return libAddNOptions.End(builder)
