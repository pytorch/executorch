# Copyright 2024 NXP
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import executorch.backends.nxp.backend.ir.tflite_generator.meta.meta as meta
import flatbuffers as fb
from executorch.backends.nxp.backend.ir.lib.tflite import SpaceToDepthOptions
from executorch.backends.nxp.backend.ir.lib.tflite.BuiltinOperator import (
    BuiltinOperator,
)
from executorch.backends.nxp.backend.ir.lib.tflite.BuiltinOptions import BuiltinOptions


class SpaceToDepth(meta.BuiltinOptions):
    block_size: int

    def __init__(self, block_size: int) -> None:
        super().__init__(
            BuiltinOptions.SpaceToDepthOptions, BuiltinOperator.SPACE_TO_DEPTH
        )
        self.block_size = block_size

    def gen_tflite(self, builder: fb.Builder):
        SpaceToDepthOptions.Start(builder)

        SpaceToDepthOptions.SpaceToDepthOptionsAddBlockSize(builder, self.block_size)

        return SpaceToDepthOptions.End(builder)
