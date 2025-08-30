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
    exir_ops.edge.quantized_decomposed.embedding_4bit.dtype,
]

to_be_implemented_operator = [
    exir_ops.edge.aten._adaptive_avg_pool3d.default,
    exir_ops.edge.aten.adaptive_max_pool2d.default,
    exir_ops.edge.aten.avg_pool3d.default,
    exir_ops.edge.aten.div.Tensor_mode,
    exir_ops.edge.aten.index_select.default,
    exir_ops.edge.aten.log10.default,
    exir_ops.edge.aten.log1p.default,
    exir_ops.edge.aten.log2.default,
    exir_ops.edge.aten.flip.default,
    exir_ops.edge.aten.max_pool3d_with_indices.default,
    exir_ops.edge.aten.median.default,
    exir_ops.edge.aten.median.dim,
    exir_ops.edge.aten.round.decimals,
    exir_ops.edge.aten.le.Scalar,
    exir_ops.edge.aten.trunc.default,
]

constant_operator = [
    exir_ops.edge.aten.arange.start_step,
    exir_ops.edge.aten.full.default,
    exir_ops.edge.aten.full_like.default,
    torch.ops.aten.scalar_tensor.default,
]

allow_list_operator = [
    _operator.getitem,
]
