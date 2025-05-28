# Copyright (c) Qualcomm Innovation Center, Inc.
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
from executorch.exir.dialects._ops import ops as exir_ops
from executorch.exir.pass_base import ExportPass


class ConvertUpsampleBicubicWithBilinear(ExportPass):
    """
    Qnn does not support bicubic interpolation, so we need to convert it to bilinear.
    This pass will convert bicubic interpolation to bilinear interpolation.
    """

    bicubic_op_targets = {
        exir_ops.edge.aten.upsample_bicubic2d.vec,
    }
    upsample_bilinear_op = exir_ops.edge.aten.upsample_bilinear2d.default

    def __init__(self):
        super(ConvertUpsampleBicubicWithBilinear, self).__init__()

    def call_operator(self, op, args, kwargs, meta):
        if op not in self.bicubic_op_targets:
            return super().call_operator(op, args, kwargs, meta)
        return super().call_operator(self.upsample_bilinear_op, args[:-1], kwargs, meta)
