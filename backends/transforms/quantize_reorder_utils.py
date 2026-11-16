# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe

import torch
from executorch.exir.dialects._ops import ops as exir_ops

# A list of ops that can be trivially quantized
trivially_quantizable_ops_overloadpkt = {
    exir_ops.edge.aten.chunk,
    exir_ops.edge.aten.clone,
    exir_ops.edge.aten.contiguous,
    exir_ops.edge.aten.expand_copy,
    exir_ops.edge.aten.permute_copy,
    exir_ops.edge.aten.select_copy,
    exir_ops.edge.aten.slice_copy,
    exir_ops.edge.aten.squeeze_copy,
    exir_ops.edge.aten.transpose_copy,
    exir_ops.edge.aten.unfold_copy,
    exir_ops.edge.aten.unsqueeze_copy,
    exir_ops.edge.aten.view_copy,
    torch.ops.aten.chunk,
    torch.ops.aten.clone,
    torch.ops.aten.contiguous,
    torch.ops.aten.expand_copy,
    torch.ops.aten.permute,
    torch.ops.aten.permute_copy,
    torch.ops.aten.select_copy,
    torch.ops.aten.slice,
    torch.ops.aten.slice_copy,
    torch.ops.aten.squeeze,
    torch.ops.aten.squeeze_copy,
    torch.ops.aten.transpose,
    torch.ops.aten.transpose_copy,
    torch.ops.aten.unsqueeze,
    torch.ops.aten.unsqueeze_copy,
    torch.ops.aten.view,
    torch.ops.aten.view_copy,
}

# slice-equivalent ops
slice_or_select_overloadpkt = {
    torch.ops.aten.slice_copy,
    torch.ops.aten.select_copy,
    exir_ops.edge.aten.slice_copy,
    exir_ops.edge.aten.select_copy,
}
