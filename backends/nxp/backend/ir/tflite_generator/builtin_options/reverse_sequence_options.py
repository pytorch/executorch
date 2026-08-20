# Copyright 2024 NXP
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import executorch.backends.nxp.backend.ir.tflite_generator.meta.meta as meta
import flatbuffers as fb
from executorch.backends.nxp.backend.ir.lib.tflite import ReverseSequenceOptions
from executorch.backends.nxp.backend.ir.lib.tflite.BuiltinOperator import (
    BuiltinOperator,
)
from executorch.backends.nxp.backend.ir.lib.tflite.BuiltinOptions import BuiltinOptions


class ReverseSequence(meta.BuiltinOptions):
    seq_dim: int
    batch_dim: int

    def __init__(self, seq_dim: int, batch_dim: int) -> None:
        super().__init__(
            BuiltinOptions.ReverseSequenceOptions, BuiltinOperator.REVERSE_SEQUENCE
        )
        self.seq_dim = seq_dim
        self.batch_dim = batch_dim

    def gen_tflite(self, builder: fb.Builder):
        ReverseSequenceOptions.Start(builder)

        ReverseSequenceOptions.AddSeqDim(builder, self.seq_dim)
        ReverseSequenceOptions.AddBatchDim(builder, self.batch_dim)

        return ReverseSequenceOptions.End(builder)
