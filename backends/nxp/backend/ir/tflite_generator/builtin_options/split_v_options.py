# Copyright 2023 NXP
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import executorch.backends.nxp.backend.ir.lib.tflite.SplitVOptions as libSplitVOptions
import flatbuffers as fb
from executorch.backends.nxp.backend.ir.lib.tflite.BuiltinOperator import (
    BuiltinOperator,
)
from executorch.backends.nxp.backend.ir.lib.tflite.BuiltinOptions import BuiltinOptions
from executorch.backends.nxp.backend.ir.tflite_generator.meta import meta


class SplitV(meta.BuiltinOptions):
    num_splits: int

    def __init__(self, num_splits: int) -> None:
        super().__init__(BuiltinOptions.SplitVOptions, BuiltinOperator.SPLIT_V)
        self.num_splits = num_splits

    def gen_tflite(self, builder: fb.Builder):
        libSplitVOptions.Start(builder)

        libSplitVOptions.AddNumSplits(builder, self.num_splits)

        return libSplitVOptions.End(builder)
