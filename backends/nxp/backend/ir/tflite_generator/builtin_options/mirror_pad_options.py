# Copyright 2023 NXP
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import flatbuffers

from executorch.backends.nxp.backend.ir.lib.tflite import (
    BuiltinOperator,
    BuiltinOptions,
    MirrorPadOptions,
)
from executorch.backends.nxp.backend.ir.lib.tflite.MirrorPadMode import MirrorPadMode
from executorch.backends.nxp.backend.ir.tflite_generator.meta import meta


# SYMMETRIC pads = [2, 2, 2, 2]
# input:
# [[0. 1.]
#  [2. 3.]]
#
# output:
# [[3. 2. 2. 3. 3. 2.]
#  [1. 0. 0. 1. 1. 0.]
#  [1. 0. 0. 1. 1. 0.]
#  [3. 2. 2. 3. 3. 2.]
#  [3. 2. 2. 3. 3. 2.]
#  [1. 0. 0. 1. 1. 0.]]

# REFLECT pads = [2, 2, 2, 2]
# input:
# [[0. 1.]
#  [2. 3.]]
#
# output: (Doesn't make sense to me, why the first row is all 0. Also last row is weird.)
#         (Also the element [0][0] is sometimes 9.462417e-28, so the computations seems non-deterministic!)
# [[0. 0. 0. 0. 0. 0.]
#  [0. 3. 2. 3. 2. 2.]
#  [2. 1. 0. 1. 0. 0.]
#  [0. 3. 2. 3. 2. 2.]
#  [2. 1. 0. 1. 0. 0.]
#  [2. 1. 0. 1. 0. 0.]]


class MirrorPad(meta.BuiltinOptions):
    mode: MirrorPadMode

    def __init__(self, mode: MirrorPadMode = MirrorPadMode.REFLECT) -> None:
        super().__init__(
            BuiltinOptions.BuiltinOptions.MirrorPadOptions,
            BuiltinOperator.BuiltinOperator.MIRROR_PAD,
        )
        self.mode = mode

    def gen_tflite(self, builder: flatbuffers.Builder):
        MirrorPadOptions.Start(builder)

        MirrorPadOptions.AddMode(builder, self.mode)

        return MirrorPadOptions.End(builder)
