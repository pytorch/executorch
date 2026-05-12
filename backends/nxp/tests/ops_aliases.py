# Copyright 2026 NXP
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# This file defines ops aliases for shorter and more readable test description. List is sorted alphabetically.
# When finding a missing alias, add it at the correct place.

import operator

import torch
from executorch.exir.dialects._ops import ops as exir_ops

AvgPool2D = exir_ops.edge.aten.avg_pool2d.default
Bmm = exir_ops.edge.aten.bmm.default
DequantizePerChannel = exir_ops.edge.quantized_decomposed.dequantize_per_channel.default
DequantizePerTensor = exir_ops.edge.quantized_decomposed.dequantize_per_tensor.default
ExecutorchDelegateCall = torch.ops.higher_order.executorch_call_delegate
GetItem = operator.getitem
HardTanh = exir_ops.edge.aten.hardtanh.default
HardTanh_ = exir_ops.edge.aten.hardtanh_.default
MaxPool2DWithIndices = exir_ops.edge.aten.max_pool2d_with_indices.default
QuantizePerChannel = exir_ops.edge.quantized_decomposed.quantize_per_channel.default
QuantizePerTensor = exir_ops.edge.quantized_decomposed.quantize_per_tensor.default
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
