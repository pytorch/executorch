# Copyright 2024 NXP
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import flatbuffers as fb

from executorch.backends.nxp.backend.ir.lib.tflite import ReducerOptions
from executorch.backends.nxp.backend.ir.lib.tflite.BuiltinOperator import (
    BuiltinOperator,
)
from executorch.backends.nxp.backend.ir.lib.tflite.BuiltinOptions import BuiltinOptions
from executorch.backends.nxp.backend.ir.tflite_generator.meta import meta


class ReduceProd(meta.BuiltinOptions):
    keep_dims: bool

    def __init__(self, keep_dims: bool) -> None:
        super().__init__(BuiltinOptions.ReducerOptions, BuiltinOperator.REDUCE_PROD)
        self.keep_dims = keep_dims

    def gen_tflite(self, builder: fb.Builder):
        ReducerOptions.Start(builder)

        ReducerOptions.AddKeepDims(builder, self.keep_dims)

        return ReducerOptions.End(builder)
