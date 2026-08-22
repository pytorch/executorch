# Copyright 2023 NXP
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import executorch.backends.nxp.backend.ir.lib.tflite.GatherOptions as libGatherOptions
import executorch.backends.nxp.backend.ir.tflite_generator.meta.meta as meta
import flatbuffers as fb
from executorch.backends.nxp.backend.ir.lib.tflite.BuiltinOperator import (
    BuiltinOperator,
)
from executorch.backends.nxp.backend.ir.lib.tflite.BuiltinOptions import BuiltinOptions


class Gather(meta.BuiltinOptions):
    axis: int
    batch_dims: int

    def __init__(self, axis: int, batch_dims: int = 0) -> None:
        super().__init__(BuiltinOptions.GatherOptions, BuiltinOperator.GATHER)
        self.axis = axis
        self.batch_dims = batch_dims

    def gen_tflite(self, builder: fb.Builder):
        libGatherOptions.Start(builder)

        libGatherOptions.AddAxis(builder, self.axis)
        libGatherOptions.AddBatchDims(builder, self.batch_dims)

        return libGatherOptions.End(builder)
