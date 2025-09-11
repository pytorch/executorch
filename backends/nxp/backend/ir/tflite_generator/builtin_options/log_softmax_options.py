#
# Copyright 2023 Martin Pavella
# Copyright 2024 NXP
#
# License: MIT
# See the LICENSE_MIT for more details.
#
"""
    LogSoftmax

Representation of the TFLite operator 'LogSoftmax'.
"""

import executorch.backends.nxp.backend.ir.lib.tflite.BuiltinOperator as libBuiltinOperator
import executorch.backends.nxp.backend.ir.lib.tflite.BuiltinOptions as libBuiltinOptions
import executorch.backends.nxp.backend.ir.lib.tflite.LogSoftmaxOptions as libLogSoftmaxOptions
import executorch.backends.nxp.backend.ir.tflite_generator.meta.meta as meta
import flatbuffers as fb


class LogSoftmax(meta.BuiltinOptions):
    def __init__(self) -> None:
        super().__init__(
            libBuiltinOptions.BuiltinOptions.LogSoftmaxOptions,
            libBuiltinOperator.BuiltinOperator.LOG_SOFTMAX,
        )

    def gen_tflite(self, builder: fb.Builder):
        libLogSoftmaxOptions.Start(builder)
        return libLogSoftmaxOptions.End(builder)
