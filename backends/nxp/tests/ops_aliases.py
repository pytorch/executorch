# Copyright 2026 NXP
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# This file defines ops aliases for shorter and more readable test description. List is sorted alphabetically.
# When finding a missing alias, add it at the correct place.

import torch
from executorch.exir.dialects._ops import ops as exir_ops

AvgPool2D = exir_ops.edge.aten.avg_pool2d.default
Bmm = exir_ops.edge.aten.bmm.default
Convolution = exir_ops.edge.aten.convolution.default
ExecutorchDelegateCall = torch.ops.higher_order.executorch_call_delegate
HardTanh = exir_ops.edge.aten.hardtanh.default
HardTanh_ = exir_ops.edge.aten.hardtanh_.default
MulTensor = exir_ops.edge.aten.mul.Tensor
Slice = exir_ops.edge.aten.slice.Tensor
SliceCopy = exir_ops.edge.aten.slice_copy.Tensor
Softmax = exir_ops.edge.aten._softmax.default
Squeeze = exir_ops.edge.aten.squeeze.default
SqueezeDim = exir_ops.edge.aten.squeeze.dim
SqueezeDims = exir_ops.edge.aten.squeeze.dims
Unsqueeze = exir_ops.edge.aten.unsqueeze.default
UpsampleBilinear2D = exir_ops.edge.aten.upsample_bilinear2d.vec
UpsampleNearest2D = exir_ops.edge.aten.upsample_nearest2d.vec
ViewCopy = exir_ops.edge.aten.view_copy.default
