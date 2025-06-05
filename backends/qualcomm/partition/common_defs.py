# Copyright (c) Qualcomm Innovation Center, Inc.
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
import _operator

import torch

from executorch.exir.dialects._ops import ops as exir_ops

not_supported_operator = [
    exir_ops.edge.aten.clone.default,
    exir_ops.edge.aten.slice_scatter.default,
    exir_ops.edge.aten.copy.default,
    exir_ops.edge.quantized_decomposed.embedding_4bit.dtype,
]

to_be_implemented_operator = []

constant_operator = [
    exir_ops.edge.aten.arange.start_step,
    exir_ops.edge.aten.full.default,
    exir_ops.edge.aten.full_like.default,
    torch.ops.aten.scalar_tensor.default,
]

allow_list_operator = [
    _operator.getitem,
]
