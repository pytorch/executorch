# Copyright 2026 NXP
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# This file defines ops aliases for shorter and more readable test description. List is sorted alphabetically.
# When finding a missing alias, add it at the correct place.

import operator

import torch
from executorch.exir.dialects._ops import ops as exir_ops

Abs = exir_ops.edge.aten.abs.default
AdaptiveAvgPool2D = exir_ops.edge.aten._adaptive_avg_pool2d.default
AddMm = exir_ops.edge.aten.addmm.default
AddTensor = exir_ops.edge.aten.add.Tensor
AvgPool2D = exir_ops.edge.aten.avg_pool2d.default
BMM = exir_ops.edge.aten.bmm.default
Cat = exir_ops.edge.aten.cat.default
Clamp = exir_ops.edge.aten.clamp.default
Clone = exir_ops.edge.aten.clone.default
CloneDimOrder = exir_ops.edge.dim_order_ops._clone_dim_order.default
ConstantPadND = exir_ops.edge.aten.constant_pad_nd.default
Convolution = exir_ops.edge.aten.convolution.default
DequantizePerChannel = exir_ops.edge.quantized_decomposed.dequantize_per_channel.default
DequantizePerTensor = exir_ops.edge.quantized_decomposed.dequantize_per_tensor.default
ExecutorchDelegateCall = torch.ops.higher_order.executorch_call_delegate
Exp = exir_ops.edge.aten.exp.default
GetItem = operator.getitem
HardTanh = exir_ops.edge.aten.hardtanh.default
HardTanh_ = exir_ops.edge.aten.hardtanh_.default
LeakyRelu = exir_ops.edge.aten.leaky_relu.default
Log = exir_ops.edge.aten.log.default
MaxPool2D = exir_ops.edge.aten.max_pool2d.default
MaxPool2DWithIndices = exir_ops.edge.aten.max_pool2d_with_indices.default
MeanDim = exir_ops.edge.aten.mean.dim
MulTensor = exir_ops.edge.aten.mul.Tensor
PermuteCopy = exir_ops.edge.aten.permute_copy.default
QuantizePerChannel = exir_ops.edge.quantized_decomposed.quantize_per_channel.default
QuantizePerTensor = exir_ops.edge.quantized_decomposed.quantize_per_tensor.default
PermuteCopy = exir_ops.edge.aten.permute_copy.default
Relu = exir_ops.edge.aten.relu.default
Sigmoid = exir_ops.edge.aten.sigmoid.default
Slice = exir_ops.edge.aten.slice.Tensor
SliceCopy = exir_ops.edge.aten.slice_copy.Tensor
Softmax = exir_ops.edge.aten._softmax.default
Squeeze = exir_ops.edge.aten.squeeze.default
SqueezeDim = exir_ops.edge.aten.squeeze.dim
SqueezeDims = exir_ops.edge.aten.squeeze.dims
SubTensor = exir_ops.edge.aten.sub.Tensor
Tanh = exir_ops.edge.aten.tanh.default
Tanh_ = exir_ops.edge.aten.tanh_.default
Unsqueeze = exir_ops.edge.aten.unsqueeze.default
UpsampleBilinear2D = exir_ops.edge.aten.upsample_bilinear2d.vec
UpsampleNearest2D = exir_ops.edge.aten.upsample_nearest2d.vec
ViewCopy = exir_ops.edge.aten.view_copy.default
