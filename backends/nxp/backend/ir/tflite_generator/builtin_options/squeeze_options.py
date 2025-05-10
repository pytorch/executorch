# Copyright 2023 NXP
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import executorch.backends.nxp.backend.ir.lib.tflite.SqueezeOptions as libSqueezeOptions
import flatbuffers as fb
from executorch.backends.nxp.backend.ir.lib.tflite.BuiltinOperator import (
    BuiltinOperator,
)
from executorch.backends.nxp.backend.ir.lib.tflite.BuiltinOptions import BuiltinOptions
from executorch.backends.nxp.backend.ir.tflite_generator.meta import meta


class SqueezeDims(meta.IntVector):
    def __init__(self, new_shape: list[int]) -> None:
        super().__init__(new_shape, libSqueezeOptions.StartSqueezeDimsVector)


class Squeeze(meta.BuiltinOptions):
    squeeze_dims: SqueezeDims | None

    def __init__(self, squeeze_dims: list[int] | None) -> None:
        super().__init__(BuiltinOptions.SqueezeOptions, BuiltinOperator.SQUEEZE)

        if squeeze_dims is not None:
            self.squeeze_dims = SqueezeDims(squeeze_dims)
        else:
            self.squeeze_dims = None

    def gen_tflite(self, builder: fb.Builder):
        if self.squeeze_dims is not None:
            tfl_squeeze_dims = self.squeeze_dims.gen_tflite(builder)
        else:
            tfl_squeeze_dims = None

        libSqueezeOptions.Start(builder)

        if tfl_squeeze_dims is not None:
            libSqueezeOptions.AddSqueezeDims(builder, tfl_squeeze_dims)

        return libSqueezeOptions.End(builder)
