# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""``fused_quant`` ops for the ATen ops CMSIS-NN runs that the dialect lacks.

Importing this module registers them, the dialect's own way: each is
``dequantize -> ATen op -> quantize`` with the ATen op's schema, so
``fuse_aten`` and ``DecomposeFusedQuant`` map one onto the other by name, and an
instance the lowering does not take decomposes to the portable kernels.
"""

import torch
from executorch.exir.dialects.fused_quant.ops import _define_dq_op_q

_aten = torch.ops.aten

for _name, _inputs, _extras, _aten_op in (
    ("log", ("inp",), "", _aten.log.default),
    ("log2", ("inp",), "", _aten.log2.default),
    ("log10", ("inp",), "", _aten.log10.default),
    ("log1p", ("inp",), "", _aten.log1p.default),
):
    _define_dq_op_q(_name, _inputs, _extras, _aten_op)
