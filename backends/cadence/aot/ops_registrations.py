# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from math import prod
from typing import Optional, Tuple

import torch
from executorch.backends.cadence.aot.utils import (
    get_conv1d_output_size,
    get_conv2d_output_size,
    get_im2row_output_size,
)
from executorch.exir.scalar_type import ScalarType
from torch._meta_registrations import _linalg_svd_meta
from torch.library import Library, register_fake

lib = Library("cadence", "DEF")

lib.define(
    "quantize_per_tensor(Tensor input, float scale, int zero_point, int quant_min, int quant_max, ScalarType dtype) -> (Tensor Z)"
)
lib.define(
    "quantize_per_tensor.out(Tensor input, float scale, int zero_point, int quant_min, int quant_max, ScalarType dtype, *, Tensor(a!) out) -> Tensor(a!)"
)

lib.define(
    "dequantize_per_tensor(Tensor input, float scale, int zero_point, int quant_min, int quant_max, ScalarType dtype) -> (Tensor Z)"
)
lib.define(
    "dequantize_per_tensor.out(Tensor input, float scale, int zero_point, int quant_min, int quant_max, ScalarType dtype, *, Tensor(a!) out) -> Tensor(a!)"
)

lib.define(
    "quantized_layer_norm(Tensor X, Tensor X_scale, Tensor X_zero_point, int[] normalized_shape, Tensor weight, Tensor bias, float eps, float output_scale, int output_zero_point) -> (Tensor Y)"
)
lib.define(
    "quantized_layer_norm.out(Tensor X, Tensor X_scale, Tensor X_zero_point, int[] normalized_shape, Tensor weight, Tensor bias, float eps, float output_scale, int output_zero_point, *, Tensor(a!) out) -> Tensor (a!)"
)
lib.define(
    "quantized_layer_norm.per_tensor(Tensor X, float X_scale, int X_zero_point, int[] normalized_shape, Tensor weight, Tensor bias, float eps, float output_scale, int output_zero_point) -> (Tensor Y)"
)
lib.define(
    "quantized_layer_norm.per_tensor_out(Tensor X, float X_scale, int X_zero_point, int[] normalized_shape, Tensor weight, Tensor bias, float eps, float output_scale, int output_zero_point, *, Tensor(a!) out) -> Tensor (a!)"
)

lib.define(
    "quantized_linear(Tensor src, Tensor weight, Tensor bias, int src_zero_point, Tensor weight_zero_point, Tensor out_multiplier, Tensor out_shift, int out_zero_point, Tensor? offset) -> (Tensor Z)"
)
lib.define(
    "quantized_linear.out(Tensor src, Tensor weight, Tensor bias, int src_zero_point, Tensor weight_zero_point, Tensor out_multiplier, Tensor out_shift, int out_zero_point, Tensor? offset, *, Tensor(a!) out) ->  Tensor(a!)"
)
lib.define(
    "quantized_linear.per_tensor_out(Tensor src, Tensor weight, Tensor bias, SymInt src_zero_point, SymInt weight_zero_point, SymInt out_multiplier, SymInt out_shift, SymInt out_zero_point, Tensor? offset, *, Tensor(a!) out) -> Tensor(a!)"
)
lib.define(
    "quantized_linear_asym8sxasym8s_asym8s.per_tensor_out(Tensor src, Tensor weight, Tensor bias, int src_zero_point, "
    "int weight_zero_point, int out_multiplier, int out_shift, int out_zero_point, Tensor? offset, *, Tensor(a!) out) -> Tensor(a!)"
)
lib.define(
    "quantized_linear_asym8uxasym8u_asym8u.per_tensor_out(Tensor src, Tensor weight, Tensor bias, int src_zero_point, "
    "int weight_zero_point, int out_multiplier, int out_shift, int out_zero_point, Tensor? offset, *, Tensor(a!) out) -> Tensor(a!)"
)
lib.define(
    "quantized_linear.per_tensor(Tensor src, Tensor weight, Tensor bias, SymInt src_zero_point, "
    "SymInt weight_zero_point, SymInt out_multiplier, SymInt out_shift, SymInt out_zero_point, Tensor? offset) -> Tensor"
)
lib.define(
    "quantized_linear_asym8sxasym8s_asym8s.per_tensor(Tensor src, Tensor weight, Tensor bias, int src_zero_point, "
    "int weight_zero_point, int out_multiplier, int out_shift, int out_zero_point, Tensor? offset) -> (Tensor Z)"
)
lib.define(
    "quantized_linear_asym8uxasym8u_asym8u.per_tensor(Tensor src, Tensor weight, Tensor bias, int src_zero_point, "
    "int weight_zero_point, int out_multiplier, int out_shift, int out_zero_point, Tensor? offset) -> (Tensor Z)"
)

lib.define(
    "quantized_relu(Tensor X, Tensor X_zero_point, int out_zero_point, Tensor out_multiplier, Tensor out_shift) -> (Tensor Y)"
)
lib.define(
    "quantized_relu.out(Tensor X, Tensor X_zero_point, int out_zero_point, Tensor out_multiplier, Tensor out_shift, *, Tensor(a!) out) -> Tensor (a!)"
)

lib.define(
    "quantized_conv_nhwc(Tensor input, Tensor weight, Tensor bias, int[] stride, SymInt[] padding, int[] dilation, int groups, int input_zero_point, Tensor weight_zero_point, Tensor bias_scale, float out_scale, int out_zero_point, Tensor out_multiplier, Tensor out_shift) -> (Tensor Z)"
)
lib.define(
    "quantized_conv_nhwc.out(Tensor input, Tensor weight, Tensor bias, int[] stride, SymInt[] padding, int[] dilation, int groups, int input_zero_point, Tensor weight_zero_point, Tensor bias_scale, float out_scale, int out_zero_point, Tensor out_multiplier, Tensor out_shift, *, Tensor(a!) out) -> Tensor(a!)"
)
lib.define(
    "quantized_conv_nhwc.per_tensor(Tensor input, Tensor weight, Tensor bias, int[] stride, SymInt[] padding, int[] dilation, int groups, int input_zero_point, int weight_zero_point, float bias_scale, float out_scale, int out_zero_point, int out_multiplier, int out_shift) -> (Tensor Z)"
)
lib.define(
    "quantized_conv_nhwc.per_tensor_out(Tensor input, Tensor weight, Tensor bias, int[] stride, SymInt[] padding, int[] dilation, int groups, int input_zero_point, int weight_zero_point, float bias_scale, float out_scale, int out_zero_point, int out_multiplier, int out_shift, *, Tensor(a!) out) -> Tensor(a!)"
)
lib.define(
    "quantized_conv_nchw(Tensor input, Tensor weight, Tensor bias, int[] stride, SymInt[] padding, int[] dilation, int groups, int input_zero_point, Tensor weight_zero_point, Tensor bias_scale, float out_scale, int out_zero_point, Tensor out_multiplier, Tensor out_shift) -> (Tensor Z)"
)
lib.define(
    "quantized_conv_nchw.out(Tensor input, Tensor weight, Tensor bias, int[] stride, SymInt[] padding, int[] dilation, int groups, int input_zero_point, Tensor weight_zero_point, Tensor bias_scale, float out_scale, int out_zero_point, Tensor out_multiplier, Tensor out_shift, *, Tensor(a!) out) -> Tensor(a!)"
)
lib.define(
    "quantized_conv_nchw.per_tensor(Tensor input, Tensor weight, Tensor bias, int[] stride, SymInt[] padding, int[] dilation, int groups, int input_zero_point, int weight_zero_point, float bias_scale, float out_scale, int out_zero_point, int out_multiplier, int out_shift) -> (Tensor Z)"
)
lib.define(
    "quantized_conv_nchw.per_tensor_out(Tensor input, Tensor weight, Tensor bias, int[] stride, SymInt[] padding, int[] dilation, int groups, int input_zero_point, int weight_zero_point, float bias_scale, float out_scale, int out_zero_point, int out_multiplier, int out_shift, *, Tensor(a!) out) -> Tensor(a!)"
)
lib.define(
    "quantized_matmul(Tensor X, int X_zero_point, Tensor Y, int Y_zero_point, Tensor? bias, int out_multiplier, int out_shift, int out_zero_point, bool transposed=False) -> (Tensor Z)"
)
lib.define(
    "quantized_matmul.out(Tensor X, int X_zero_point, Tensor Y, int Y_zero_point, Tensor? bias, int out_multiplier, int out_shift, int out_zero_point, bool transposed=False, *, Tensor(a!) out) -> Tensor(a!)"
)
lib.define(
    "quantized_matmul_asym8sxasym8s_asym8s(Tensor X, int X_zero_point, Tensor Y, int Y_zero_point, Tensor? bias, int out_multiplier, int out_shift, int out_zero_point, bool transposed=False) -> (Tensor Z)"
)
lib.define(
    "quantized_matmul_asym8sxasym8s_asym8s.out(Tensor X, int X_zero_point, Tensor Y, int Y_zero_point, Tensor? bias, int out_multiplier, int out_shift, int out_zero_point, bool transposed=False, *, Tensor(a!) out) -> Tensor(a!)"
)
lib.define(
    "quantized_conv_nchw_asym8sxsym8s_asym8s.per_tensor(Tensor input, Tensor weight, Tensor bias, int[] stride, SymInt[] padding, int[] dilation, int groups, int input_zero_point, int weight_zero_point, float bias_scale, float out_scale, int out_zero_point, int out_multiplier, int out_shift) -> (Tensor Z)"
)
lib.define(
    "quantized_conv_nchw_asym8sxsym8s_asym8s.per_tensor_out(Tensor input, Tensor weight, Tensor bias, int[] stride, SymInt[] padding, int[] dilation, int groups, int input_zero_point, int weight_zero_point, float bias_scale, float out_scale, int out_zero_point, int out_multiplier, int out_shift, *, Tensor(a!) out) -> Tensor(a!)"
)
lib.define(
    "quantized_conv_nchw_asym8uxsym8u_asym8u.per_tensor(Tensor input, Tensor weight, Tensor bias, int[] stride, SymInt[] padding, int[] dilation, int groups, int input_zero_point, int weight_zero_point, float bias_scale, float out_scale, int out_zero_point, int out_multiplier, int out_shift) -> (Tensor Z)"
)
lib.define(
    "quantized_conv_nchw_asym8uxsym8u_asym8u.per_tensor_out(Tensor input, Tensor weight, Tensor bias, int[] stride, SymInt[] padding, int[] dilation, int groups, int input_zero_point, int weight_zero_point, float bias_scale, float out_scale, int out_zero_point, int out_multiplier, int out_shift, *, Tensor(a!) out) -> Tensor(a!)"
)
lib.define(
    "quantized_conv_nhwc_asym8sxsym8s_asym8s.per_tensor(Tensor input, Tensor weight, Tensor bias, int[] stride, SymInt[] padding, int[] dilation, int groups, int input_zero_point, int weight_zero_point, float bias_scale, float out_scale, int out_zero_point, int out_multiplier, int out_shift) -> (Tensor Z)"
)
lib.define(
    "quantized_conv_nhwc_asym8sxsym8s_asym8s.per_tensor_out(Tensor input, Tensor weight, Tensor bias, int[] stride, SymInt[] padding, int[] dilation, int groups, int input_zero_point, int weight_zero_point, float bias_scale, float out_scale, int out_zero_point, int out_multiplier, int out_shift, *, Tensor(a!) out) -> Tensor(a!)"
)
lib.define(
    "quantized_conv_nhwc_asym8uxsym8u_asym8u.per_tensor(Tensor input, Tensor weight, Tensor bias, int[] stride, SymInt[] padding, int[] dilation, int groups, int input_zero_point, int weight_zero_point, float bias_scale, float out_scale, int out_zero_point, int out_multiplier, int out_shift) -> (Tensor Z)"
)
lib.define(
    "quantized_conv_nhwc_asym8uxsym8u_asym8u.per_tensor_out(Tensor input, Tensor weight, Tensor bias, int[] stride, SymInt[] padding, int[] dilation, int groups, int input_zero_point, int weight_zero_point, float bias_scale, float out_scale, int out_zero_point, int out_multiplier, int out_shift, *, Tensor(a!) out) -> Tensor(a!)"
)
lib.define(
    "quantized_conv_nchw_dilated_asym8sxsym8s_asym8s.per_tensor(Tensor input, Tensor weight, Tensor bias, int[] stride, SymInt[] padding, int[] dilation, int groups, int input_zero_point, int weight_zero_point, float bias_scale, float out_scale, int out_zero_point, int out_multiplier, int out_shift) -> (Tensor Z)"
)
lib.define(
    "quantized_conv_nchw_dilated_asym8sxsym8s_asym8s.per_tensor_out(Tensor input, Tensor weight, Tensor bias, int[] stride, SymInt[] padding, int[] dilation, int groups, int input_zero_point, int weight_zero_point, float bias_scale, float out_scale, int out_zero_point, int out_multiplier, int out_shift, *, Tensor(a!) out) -> Tensor(a!)"
)
lib.define(
    "quantized_conv_nchw_dilated_asym8uxsym8u_asym8u.per_tensor(Tensor input, Tensor weight, Tensor bias, int[] stride, SymInt[] padding, int[] dilation, int groups, int input_zero_point, int weight_zero_point, float bias_scale, float out_scale, int out_zero_point, int out_multiplier, int out_shift) -> (Tensor Z)"
)
lib.define(
    "quantized_conv_nchw_dilated_asym8uxsym8u_asym8u.per_tensor_out(Tensor input, Tensor weight, Tensor bias, int[] stride, SymInt[] padding, int[] dilation, int groups, int input_zero_point, int weight_zero_point, float bias_scale, float out_scale, int out_zero_point, int out_multiplier, int out_shift, *, Tensor(a!) out) -> Tensor(a!)"
)
lib.define(
    "quantized_conv_nhwc_dilated_asym8sxsym8s_asym8s.per_tensor(Tensor input, Tensor weight, Tensor bias, int[] stride, SymInt[] padding, int[] dilation, int groups, int input_zero_point, int weight_zero_point, float bias_scale, float out_scale, int out_zero_point, int out_multiplier, int out_shift) -> (Tensor Z)"
)
lib.define(
    "quantized_conv_nhwc_dilated_asym8sxsym8s_asym8s.per_tensor_out(Tensor input, Tensor weight, Tensor bias, int[] stride, SymInt[] padding, int[] dilation, int groups, int input_zero_point, int weight_zero_point, float bias_scale, float out_scale, int out_zero_point, int out_multiplier, int out_shift, *, Tensor(a!) out) -> Tensor(a!)"
)
lib.define(
    "quantized_conv_nhwc_dilated_asym8uxsym8u_asym8u.per_tensor(Tensor input, Tensor weight, Tensor bias, int[] stride, SymInt[] padding, int[] dilation, int groups, int input_zero_point, int weight_zero_point, float bias_scale, float out_scale, int out_zero_point, int out_multiplier, int out_shift) -> (Tensor Z)"
)
lib.define(
    "quantized_conv_nhwc_dilated_asym8uxsym8u_asym8u.per_tensor_out(Tensor input, Tensor weight, Tensor bias, int[] stride, SymInt[] padding, int[] dilation, int groups, int input_zero_point, int weight_zero_point, float bias_scale, float out_scale, int out_zero_point, int out_multiplier, int out_shift, *, Tensor(a!) out) -> Tensor(a!)"
)
lib.define(
    "quantized_conv_nchw_depthwise_asym8sxsym8s_asym8s.per_tensor(Tensor input, Tensor weight, Tensor bias, int[] stride, SymInt[] padding, int[] dilation, int groups, int input_zero_point, int weight_zero_point, float bias_scale, float out_scale, int out_zero_point, int out_multiplier, int out_shift) -> (Tensor Z)"
)
lib.define(
    "quantized_conv_nchw_depthwise_asym8sxsym8s_asym8s.per_tensor_out(Tensor input, Tensor weight, Tensor bias, int[] stride, SymInt[] padding, int[] dilation, int groups, int input_zero_point, int weight_zero_point, float bias_scale, float out_scale, int out_zero_point, int out_multiplier, int out_shift, *, Tensor(a!) out) -> Tensor(a!)"
)
lib.define(
    "quantized_conv_nchw_depthwise_asym8uxsym8u_asym8u.per_tensor(Tensor input, Tensor weight, Tensor bias, int[] stride, SymInt[] padding, int[] dilation, int groups, int input_zero_point, int weight_zero_point, float bias_scale, float out_scale, int out_zero_point, int out_multiplier, int out_shift) -> (Tensor Z)"
)
lib.define(
    "quantized_conv_nchw_depthwise_asym8uxsym8u_asym8u.per_tensor_out(Tensor input, Tensor weight, Tensor bias, int[] stride, SymInt[] padding, int[] dilation, int groups, int input_zero_point, int weight_zero_point, float bias_scale, float out_scale, int out_zero_point, int out_multiplier, int out_shift, *, Tensor(a!) out) -> Tensor(a!)"
)
lib.define(
    "quantized_conv_nhwc_depthwise_asym8sxsym8s_asym8s.per_tensor(Tensor input, Tensor weight, Tensor bias, int[] stride, SymInt[] padding, int[] dilation, int groups, int input_zero_point, int weight_zero_point, float bias_scale, float out_scale, int out_zero_point, int out_multiplier, int out_shift) -> (Tensor Z)"
)
lib.define(
    "quantized_conv_nhwc_depthwise_asym8sxsym8s_asym8s.per_tensor_out(Tensor input, Tensor weight, Tensor bias, int[] stride, SymInt[] padding, int[] dilation, int groups, int input_zero_point, int weight_zero_point, float bias_scale, float out_scale, int out_zero_point, int out_multiplier, int out_shift, *, Tensor(a!) out) -> Tensor(a!)"
)
lib.define(
    "quantized_conv_nhwc_depthwise_asym8uxsym8u_asym8u.per_tensor(Tensor input, Tensor weight, Tensor bias, int[] stride, SymInt[] padding, int[] dilation, int groups, int input_zero_point, int weight_zero_point, float bias_scale, float out_scale, int out_zero_point, int out_multiplier, int out_shift) -> (Tensor Z)"
)
lib.define(
    "quantized_conv_nhwc_depthwise_asym8uxsym8u_asym8u.per_tensor_out(Tensor input, Tensor weight, Tensor bias, int[] stride, SymInt[] padding, int[] dilation, int groups, int input_zero_point, int weight_zero_point, float bias_scale, float out_scale, int out_zero_point, int out_multiplier, int out_shift, *, Tensor(a!) out) -> Tensor(a!)"
)
lib.define(
    "quantized_matmul_asym8uxasym8u_asym8u(Tensor X, int X_zero_point, Tensor Y, int Y_zero_point, Tensor? bias, int out_multiplier, int out_shift, int out_zero_point, bool transposed=False) -> (Tensor Z)"
)
lib.define(
    "quantized_matmul_asym8uxasym8u_asym8u.out(Tensor X, int X_zero_point, Tensor Y, int Y_zero_point, Tensor? bias, int out_multiplier, int out_shift, int out_zero_point, bool transposed=False, *, Tensor(a!) out) -> Tensor(a!)"
)

lib.define(
    "convolution(Tensor input, Tensor weight, Tensor bias, int[] stride, SymInt[] padding, "
    "int[] dilation, int groups, bool channel_last=False) -> (Tensor Y)"
)
lib.define(
    "transposed_convolution(Tensor input, Tensor weight, Tensor bias, int[] stride, SymInt[] padding, "
    "int[] dilation, SymInt[] output_padding, int groups, bool channel_last=False) -> (Tensor Y)"
)
lib.define("dequantize(Tensor X, Tensor X_scale, Tensor X_zero_point) -> (Tensor Y)")
lib.define(
    "quantized_add(Tensor X, Tensor X_scale, Tensor X_zero_point, Tensor Y, Tensor Y_scale, "
    "Tensor Y_zero_point, float out_scale, int out_zero_point) -> (Tensor Z)"
)
lib.define(
    "quantized_add.per_tensor(Tensor X, float X_scale, int X_zero_point, Tensor Y, float Y_scale, "
    "int Y_zero_point, float out_scale, int out_zero_point) -> (Tensor Z)"
)
lib.define(
    "quantized_mul(Tensor X, Tensor X_scale, Tensor X_zero_point, Tensor Y, Tensor Y_scale, "
    "Tensor Y_zero_point, float out_scale, int out_zero_point) -> (Tensor Z)"
)
lib.define(
    "quantized_add_Scalar(Tensor X, Tensor X_scale, Tensor X_zero_point, Scalar Y, "
    "float out_scale, int out_zero_point) -> (Tensor Z)"
)
lib.define(
    "quantized_mul_Scalar(Tensor X, Tensor X_scale, Tensor X_zero_point, Scalar Y, "
    "float out_scale, int out_zero_point) -> (Tensor Z)"
)
lib.define(
    "quantized_embedding_byte(Tensor weight, Tensor weight_scales, Tensor weight_zero_points, "
    "Tensor indices, bool pruned_weights=False) -> (Tensor X)"
)
lib.define(
    "quantized_transposed_conv(Tensor input, Tensor weight, Tensor bias, int[] stride, SymInt[] padding, "
    "int[] dilation, SymInt[] output_padding, int groups, int input_zero_point, Tensor weight_zero_point, "
    "Tensor bias_scale, float out_scale, int out_zero_point, Tensor out_multiplier, Tensor out_shift, bool channel_last=False) -> (Tensor out)"
)
lib.define(
    "avg_pool2d(Tensor input, int[2] kernel_size, int[2] stride=[], int[2] padding=0, bool ceil_mode=False, "
    "bool count_include_pad=True, int? divisor_override=None, Tensor? in_zero_point=None, bool channel_last=False) -> (Tensor out)"
)
lib.define(
    "im2row(Tensor input, int[2] kernel_size, int[2] dilation, int[2] padding, int[2] stride, "
    "Tensor in_zero_point, bool channel_last=False) -> (Tensor out)"
)
lib.define(
    "im2row.per_tensor(Tensor input, int[2] kernel_size, int[2] dilation, int[2] padding, int[2] stride, "
    "int in_zero_point, bool channel_last=False) -> (Tensor out)"
)
lib.define("linalg_vector_norm(Tensor X) -> (Tensor Y)")
lib.define(
    "linalg_svd(Tensor A, bool full_matrices=False, bool compute_uv=True, str? driver=None) -> (Tensor U, Tensor S, Tensor Vh)"
)
lib.define(
    "linalg_svd.out(Tensor A, bool full_matrices=False, bool compute_uv=True, str? driver=None, *, Tensor(a!) U, Tensor(b!) S, Tensor(c!) Vh) -> (Tensor(a!) U, Tensor(b!) S, Tensor(c!) Vh)"
)
lib.define(
    "transposed_im2row(Tensor input, int[2] kernel_size, int[2] dilation, int[2] padding, int[2] stride, "
    "int[2] output_padding, Tensor in_zero_point, bool channel_last=False) -> (Tensor out)"
)
lib.define(
    "requantize(Tensor input, Tensor in_scale, Tensor in_zero_point, Tensor out_scale, "
    "Tensor out_zero_point, ScalarType out_dtype) -> (Tensor Y)"
)
lib.define(
    "requantize.per_tensor(Tensor input, float in_scale, int in_zero_point, float out_scale, "
    "int out_zero_point, ScalarType out_dtype) -> (Tensor Y)"
)
lib.define(
    "fully_connected(Tensor input, Tensor weight, Tensor? bias=None) -> (Tensor out)"
)
lib.define(
    "quantized_fully_connected(Tensor src, Tensor weight, Tensor bias, int src_zero_point, "
    "Tensor weight_zero_point, Tensor out_multiplier, Tensor out_shift, int out_zero_point, Tensor? offset) -> (Tensor Z)"
)
lib.define(
    "quantized_fully_connected.per_tensor(Tensor src, Tensor weight, Tensor bias, int src_zero_point, "
    "int weight_zero_point, int out_multiplier, int out_shift, int out_zero_point, Tensor? offset) -> (Tensor Z)"
)
lib.define(
    "quantized_fully_connected_asym8sxasym8s_asym8s.per_tensor(Tensor src, Tensor weight, Tensor bias, int src_zero_point, "
    "int weight_zero_point, int out_multiplier, int out_shift, int out_zero_point, Tensor? offset) -> (Tensor Z)"
)
lib.define(
    "quantized_fully_connected_asym8uxasym8u_asym8u.per_tensor(Tensor src, Tensor weight, Tensor bias, int src_zero_point, "
    "int weight_zero_point, int out_multiplier, int out_shift, int out_zero_point, Tensor? offset) -> (Tensor Z)"
)
lib.define("where_Scalar(Tensor condition, float self, float other) -> (Tensor Z)")
lib.define(
    "where_Scalar.out(Tensor condition, float self, float other, *, Tensor(a!) out) -> Tensor(a!)"
)

lib.define(
    "rope(Tensor input, Tensor sin_tensor, Tensor cos_tensor, Tensor? pos) -> (Tensor out)"
)
lib.define(
    "rope.out(Tensor input, Tensor sin_tensor, Tensor cos_tensor, Tensor? pos, *, Tensor(a!) out) -> Tensor(a!)"
)

# Load/store with iDMA. These only exist before memory planning.
# Post memory planning, we check that outputs/inputs for the load/store are in
# DTCM and replace idma_load/idma_store with idma_copy.
lib.define("idma_load(Tensor src, int task_num=0, int channel=0) -> Tensor")
lib.define(
    "idma_load.out(Tensor src, int task_num=0, int channel=0, *, Tensor(a!) out) -> Tensor(a!)"
)
lib.define("idma_store(Tensor src, int task_num=0, int channel=0) -> Tensor")
lib.define(
    "idma_store.out(Tensor src, int task_num=0, int channel=0, *, Tensor(a!) out) -> Tensor(a!)"
)

# Non-blocking iDMA copy.
lib.define("idma_copy(Tensor src, int task_num=0, int channel=0) -> Tensor")
lib.define(
    "idma_copy.out(Tensor src, int task_num=0, int channel=0, *, Tensor(a!) out) -> Tensor(a!)"
)
# iDMA wait.
lib.define("idma_wait(Tensor src, int task_num=0) -> Tensor")
lib.define("idma_wait.out(Tensor src, int task_num=0, *, Tensor(a!) out) -> Tensor(a!)")

# ------------------------------------ #
#   Migrated from custom_ops.yaml      #
# ------------------------------------ #
# Migrated from the custom_ops.yaml files containing different operator variants (e.g., .out, .tensor_out)
lib.define(
    "convolution.out(Tensor input, Tensor weight, Tensor bias, int[] stride, SymInt[] padding, int[] dilation, "
    "int groups, bool channel_last=False, *, Tensor(a!) out) -> Tensor(a!)"
)
lib.define(
    "transposed_convolution.out(Tensor input, Tensor weight, Tensor bias, int[] stride, SymInt[] padding, "
    "int[] dilation, SymInt[] output_padding, int groups, bool channel_last=False, *, Tensor(a!) out) -> Tensor(a!)"
)
lib.define(
    "quantized_relu.per_tensor(Tensor X, int X_zero_point, int out_zero_point, int out_multiplier, int out_shift) -> Tensor"
)
lib.define(
    "quantized_relu.per_tensor_out(Tensor X, int X_zero_point, int out_zero_point, int out_multiplier, "
    "int out_shift, *, Tensor(a!) out) -> Tensor(a!)"
)
lib.define(
    "quantized_relu_asym8s_asym8s.per_tensor(Tensor X, int X_zero_point, int out_zero_point, int out_multiplier, int out_shift) -> Tensor"
)
lib.define(
    "quantized_relu_asym8s_asym8s.per_tensor_out(Tensor X, int X_zero_point, int out_zero_point, int out_multiplier, "
    "int out_shift, *, Tensor(a!) out) -> Tensor(a!)"
)
lib.define(
    "quantized_relu_asym8u_asym8u.per_tensor(Tensor X, int X_zero_point, int out_zero_point, int out_multiplier, int out_shift) -> Tensor"
)
lib.define(
    "quantized_relu_asym8u_asym8u.per_tensor_out(Tensor X, int X_zero_point, int out_zero_point, int out_multiplier, "
    "int out_shift, *, Tensor(a!) out) -> Tensor(a!)"
)
lib.define(
    "quantized_add.out(Tensor X, Tensor X_scale, Tensor X_zero_point, Tensor Y, Tensor Y_scale, "
    "Tensor Y_zero_point, float out_scale, int out_zero_point, *, Tensor(a!) out) -> Tensor(a!)"
)
lib.define(
    "quantized_add.per_tensor_out(Tensor X, float X_scale, int X_zero_point, Tensor Y, float Y_scale, "
    "int Y_zero_point, float out_scale, int out_zero_point, *, Tensor(a!) out) -> Tensor(a!)"
)
lib.define(
    "quantized_add_asym8sxasym8s_asym8s.per_tensor(Tensor X, float X_scale, int X_zero_point, Tensor Y, float Y_scale, "
    "int Y_zero_point, float out_scale, int out_zero_point) -> Tensor"
)
lib.define(
    "quantized_add_asym8sxasym8s_asym8s.per_tensor_out(Tensor X, float X_scale, int X_zero_point, Tensor Y, float Y_scale, "
    "int Y_zero_point, float out_scale, int out_zero_point, *, Tensor(a!) out) -> Tensor(a!)"
)
lib.define(
    "quantized_add_asym8uxasym8u_asym8u.per_tensor(Tensor X, float X_scale, int X_zero_point, Tensor Y, float Y_scale, "
    "int Y_zero_point, float out_scale, int out_zero_point) -> Tensor"
)
lib.define(
    "quantized_add_asym8uxasym8u_asym8u.per_tensor_out(Tensor X, float X_scale, int X_zero_point, Tensor Y, float Y_scale, "
    "int Y_zero_point, float out_scale, int out_zero_point, *, Tensor(a!) out) -> Tensor(a!)"
)
lib.define(
    "quantized_mul.out(Tensor X, Tensor X_scale, Tensor X_zero_point, Tensor Y, Tensor Y_scale, "
    "Tensor Y_zero_point, float out_scale, int out_zero_point, *, Tensor(a!) out) -> Tensor(a!)"
)
lib.define(
    "quantized_add_Scalar.out(Tensor X, Tensor X_scale, Tensor X_zero_point, Scalar Y, "
    "float out_scale, int out_zero_point, *, Tensor(a!) out) -> Tensor(a!)"
)
lib.define(
    "quantized_mul_Scalar.out(Tensor X, Tensor X_scale, Tensor X_zero_point, Scalar Y, "
    "float out_scale, int out_zero_point, *, Tensor(a!) out) -> Tensor(a!)"
)
lib.define(
    "fully_connected.out(Tensor input, Tensor weight, Tensor? bias=None, *, Tensor(a!) out) -> Tensor(a!)"
)
lib.define("linalg_vector_norm.out(Tensor X, *, Tensor(a!) out) -> Tensor(a!)")
lib.define(
    "quantized_fully_connected.out(Tensor src, Tensor weight, Tensor bias, int src_zero_point, "
    "Tensor weight_zero_point, Tensor out_multiplier, Tensor out_shift, int out_zero_point, Tensor? offset, *, Tensor(a!) out) -> Tensor(a!)"
)
lib.define(
    "quantized_fully_connected.per_tensor_out(Tensor src, Tensor weight, Tensor bias, int src_zero_point, "
    "int weight_zero_point, int out_multiplier, int out_shift, int out_zero_point, Tensor? offset, *, Tensor(a!) out) -> Tensor(a!)"
)
lib.define(
    "quantized_fully_connected_asym8sxasym8s_asym8s.per_tensor_out(Tensor src, Tensor weight, Tensor bias, int src_zero_point, "
    "int weight_zero_point, int out_multiplier, int out_shift, int out_zero_point, Tensor? offset, *, Tensor(a!) out) -> Tensor(a!)"
)
lib.define(
    "quantized_fully_connected_asym8uxasym8u_asym8u.per_tensor_out(Tensor src, Tensor weight, Tensor bias, int src_zero_point, "
    "int weight_zero_point, int out_multiplier, int out_shift, int out_zero_point, Tensor? offset, *, Tensor(a!) out) -> Tensor(a!)"
)
lib.define(
    "quantized_embedding_byte.out(Tensor weight, Tensor weight_scales, Tensor weight_zero_points, "
    "Tensor indices, bool pruned_weights=False, *, Tensor(a!) out) -> Tensor(a!)"
)

lib.define(
    "quantized_transposed_conv.out(Tensor input, Tensor weight, Tensor bias, int[] stride, "
    "SymInt[] padding, int[] dilation, SymInt[] output_padding, int groups, int input_zero_point, "
    "Tensor weight_zero_point, Tensor bias_scale, float out_scale, int out_zero_point, "
    "Tensor out_multiplier, Tensor out_shift, bool channel_last=False, *, Tensor(a!) out) -> Tensor(a!)"
)
lib.define(
    "avg_pool2d.out(Tensor input, int[2] kernel_size, int[2] stride=[], int[2] padding=0, "
    "bool ceil_mode=False, bool count_include_pad=True, int? divisor_override=None, "
    "Tensor? in_zero_point=None, bool channel_last=False, *, Tensor(a!) out) -> Tensor(a!)"
)
lib.define(
    "im2row.out(Tensor input, int[2] kernel_size, int[2] dilation, int[2] padding, int[2] stride, "
    "Tensor in_zero_point, bool channel_last=False, *, Tensor(a!) out) -> Tensor(a!)"
)
lib.define(
    "im2row.per_tensor_out(Tensor input, int[2] kernel_size, int[2] dilation, int[2] padding, int[2] stride, "
    "int in_zero_point, bool channel_last=False, *, Tensor(a!) out) -> Tensor(a!)"
)
lib.define(
    "transposed_im2row.out(Tensor input, int[2] kernel_size, int[2] dilation, int[2] padding, "
    "int[2] stride, int[2] output_padding, Tensor in_zero_point, bool channel_last=False, *, Tensor(a!) out) -> Tensor(a!)"
)
lib.define(
    "requantize.out(Tensor input, Tensor in_scale, Tensor in_zero_point, Tensor out_scale, "
    "Tensor out_zero_point, ScalarType out_dtype, *, Tensor(a!) out) -> Tensor(a!)"
)
lib.define(
    "requantize.per_tensor_out(Tensor input, float in_scale, int in_zero_point, float out_scale, "
    "int out_zero_point, ScalarType out_dtype, *, Tensor(a!) out) -> Tensor(a!)"
)
lib.define(
    "roi_align_box_processor.out(Tensor rois, int output_size_h, int output_size_w, "
    "int sampling_ratio, bool aligned, *, Tensor(a!) out) -> Tensor(a!)"
)
lib.define(
    "roi_align_box_processor(Tensor rois, int output_size_h, int output_size_w, "
    "int sampling_ratio, bool aligned) -> (Tensor out)"
)
lib.define(
    "_softmax_f32_f32(Tensor self, int dim, bool? half_to_float) -> (Tensor out)"
)
lib.define(
    "_softmax_f32_f32.out(Tensor self, int dim, bool? half_to_float, *, Tensor(a!) out) -> Tensor(a!)"
)

# Custom ops with aten namespace. Need to specify the lib var as FRAGMENT type as aten library is already defined
aten_lib = Library("aten", "FRAGMENT")
aten_lib.define(
    "chunk.out(Tensor self, int chunks, int dim=0, *, Tensor(a!)[] out) -> ()"
)
aten_lib.define(
    "contiguous.out(Tensor self, *, MemoryFormat memory_format=contiguous_format, "
    "Tensor(a!) out) -> Tensor(a!)"
)
aten_lib.define(
    "tensor_split.sections_out(Tensor self, int sections, int dim=0, *, Tensor(a!)[] out) -> ()"
)
aten_lib.define(
    "_slice_copy_nop(Tensor self, int dim=0, SymInt? start=None, SymInt? end=None, "
    "SymInt step=1) -> Tensor(a!)"
)
aten_lib.define(
    "_select_copy_nop.int_out(Tensor self, int dim, SymInt index, *, Tensor(a!) out) -> Tensor(a!)"
)
aten_lib.define(
    "_slice_copy_nop.Tensor_out(Tensor self, int dim=0, SymInt? start=None, SymInt? end=None, "
    "SymInt step=1, *, Tensor(a!) out) -> Tensor(a!)"
)
aten_lib.define("_cat_nop(Tensor[] tensors, int dim=0) -> Tensor(a!)")
aten_lib.define(
    "_cat_nop.out(Tensor[] tensors, int dim=0, *, Tensor(a!) out) -> Tensor(a!)"
)

# Custom ops with cadence_nn_ops namespace
jarvis_nn_lib = Library("jarvis_nn_ops", "DEF")
jarvis_nn_lib.define(
    "attention_mask.out(Tensor input, Tensor start, Tensor stop, *, Tensor(a!) out) -> Tensor(a!)"
)

# Custom ops in aten namespace. RMSNorm is usually decomposed, so having
# an out-variant is non-standard

lib_aten = Library("aten", "FRAGMENT")

lib_aten.define(
    "rms_norm.out(Tensor input, SymInt[] normalized_shape, Tensor? weight=None, float? eps=None, *, Tensor(a!) out) -> Tensor(a!)"
)


@register_fake("cadence::quantize_per_tensor")
def quantize_per_tensor_meta(
    input: torch.Tensor,
    scale: float,
    zero_point: int,
    quant_min: int,
    quant_max: int,
    dtype: torch.dtype,
) -> torch.Tensor:
    return input.new_empty(input.size(), dtype=dtype)


@register_fake("cadence::dequantize_per_tensor")
def dequantize_per_tensor_meta(
    input: torch.Tensor,
    scale: float,
    zero_point: int,
    quant_min: int,
    quant_max: int,
    dtype: torch.dtype,
) -> torch.Tensor:
    return input.new_empty(input.size(), dtype=torch.float)


@register_fake("cadence::quantized_add")
def quantized_add_meta(
    X: torch.Tensor,
    X_scale: torch.Tensor,
    X_zero_point: torch.Tensor,
    Y: torch.Tensor,
    Y_scale: torch.Tensor,
    Y_zero_point: torch.Tensor,
    out_scale: float,
    out_zero_point: int,
) -> torch.Tensor:

    # Determine output shape by broadcasting X and Y
    out_size = torch.broadcast_shapes(X.size(), Y.size())
    return X.new_empty(out_size, dtype=X.dtype)


@register_fake("cadence::quantized_add.per_tensor")
def quantized_add_per_tensor_meta(
    X: torch.Tensor,
    X_scale: float,
    X_zero_point: int,
    Y: torch.Tensor,
    Y_scale: float,
    Y_zero_point: int,
    out_scale: float,
    out_zero_point: int,
) -> torch.Tensor:

    out_size = torch.broadcast_shapes(X.size(), Y.size())
    return X.new_empty(out_size, dtype=X.dtype)


@register_fake("cadence::quantized_add_asym8sxasym8s_asym8s.per_tensor")
def quantized_add_asym8sxasym8s_asym8s_per_tensor_meta(
    X: torch.Tensor,
    X_scale: float,
    X_zero_point: int,
    Y: torch.Tensor,
    Y_scale: float,
    Y_zero_point: int,
    out_scale: float,
    out_zero_point: int,
) -> torch.Tensor:
    out_size = torch.broadcast_shapes(X.size(), Y.size())
    return X.new_empty(out_size, dtype=X.dtype)


@register_fake("cadence::quantized_add_asym8uxasym8u_asym8u.per_tensor")
def quantized_add_asym8uxasym8u_asym8u_per_tensor_meta(
    X: torch.Tensor,
    X_scale: float,
    X_zero_point: int,
    Y: torch.Tensor,
    Y_scale: float,
    Y_zero_point: int,
    out_scale: float,
    out_zero_point: int,
) -> torch.Tensor:
    out_size = torch.broadcast_shapes(X.size(), Y.size())
    return X.new_empty(out_size, dtype=X.dtype)


@register_fake("cadence::quantized_linear")
def quantized_linear_meta(
    src: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    in_zero_point: int,
    weight_zero_point: torch.Tensor,
    out_multiplier: torch.Tensor,
    out_shift: torch.Tensor,
    out_zero_point: int,
    offset: Optional[torch.Tensor],
) -> torch.Tensor:
    # src comes in shape [leading_dims, in_dim]
    # weight comes in shape [out_dim, in_dim]
    # output comes in empty with shape [leading_dims, out_dim]
    out_size = list(src.size())
    weight_size = list(weight.size())
    assert len(weight_size) == 2
    out_size[-1] = weight_size[0]
    return src.new_empty(out_size, dtype=src.dtype)


@register_fake("cadence::quantized_linear.per_tensor")
def quantized_linear_per_tensor_meta(
    src: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    in_zero_point: torch.SymInt,
    weight_zero_point: torch.SymInt,
    out_multiplier: torch.SymInt,
    out_shift: torch.SymInt,
    out_zero_point: torch.SymInt,
    offset: Optional[torch.Tensor],
) -> torch.Tensor:
    # src comes in shape [leading_dims, in_dim]
    # weight comes in shape [out_dim, in_dim]
    # output comes in empty with shape [leading_dims, out_dim]
    out_size = list(src.size())
    weight_size = list(weight.size())
    assert len(weight_size) == 2
    out_size[-1] = weight_size[0]
    return src.new_empty(out_size, dtype=src.dtype)


@register_fake("cadence::quantized_linear_asym8sxasym8s_asym8s.per_tensor")
def quantized_linear_asym8sxasym8s_asym8s_per_tensor_meta(
    src: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    in_zero_point: int,
    weight_zero_point: int,
    out_multiplier: int,
    out_shift: int,
    out_zero_point: int,
    offset: Optional[torch.Tensor],
) -> torch.Tensor:
    # src comes in shape [leading_dims, in_dim]
    # weight comes in shape [out_dim, in_dim]
    # output comes in empty with shape [leading_dims, out_dim]
    out_size = list(src.size())
    weight_size = list(weight.size())
    assert len(weight_size) == 2
    out_size[-1] = weight_size[0]
    return src.new_empty(out_size, dtype=src.dtype)


@register_fake("cadence::quantized_linear_asym8uxasym8u_asym8u.per_tensor")
def quantized_linear_asym8uxasym8u_asym8u_per_tensor_meta(
    src: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    in_zero_point: int,
    weight_zero_point: int,
    out_multiplier: int,
    out_shift: int,
    out_zero_point: int,
    offset: Optional[torch.Tensor],
) -> torch.Tensor:
    # src comes in shape [leading_dims, in_dim]
    # weight comes in shape [out_dim, in_dim]
    # output comes in empty with shape [leading_dims, out_dim]
    out_size = list(src.size())
    weight_size = list(weight.size())
    assert len(weight_size) == 2
    out_size[-1] = weight_size[0]
    return src.new_empty(out_size, dtype=src.dtype)


@register_fake("cadence::quantized_conv_nhwc")
def quantized_conv_nhwc_meta(
    input: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    stride: Tuple[int],
    padding: Tuple[int],
    dilation: Tuple[int],
    groups: int,
    in_zero_point: int,
    weight_zero_point: torch.Tensor,
    bias_scale: torch.Tensor,
    output_scale: float,
    output_zero_point: int,
    out_multiplier: torch.Tensor,
    out_shift: torch.Tensor,
) -> torch.Tensor:
    out_channels, *kernel_size, _ = weight.shape

    in_size = input.shape
    # Assert that the input tensor has at least 3 dimensions, and at most 6
    assert len(in_size) > 2
    assert len(in_size) < 6

    # Compute the output tensor size
    output_size = (
        get_conv1d_output_size(
            in_size,
            out_channels,
            stride[1],
            padding[1],
            dilation[1],
            kernel_size[0],
            True,
        )
        if len(in_size) == 3
        else get_conv2d_output_size(
            in_size, out_channels, stride, padding, dilation, kernel_size, True
        )
    )

    return input.new_empty(output_size, dtype=input.dtype)


@register_fake("cadence::quantized_conv_nchw")
def quantized_conv_nchw_meta(
    input: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    stride: Tuple[int],
    padding: Tuple[int],
    dilation: Tuple[int],
    groups: int,
    in_zero_point: int,
    weight_zero_point: torch.Tensor,
    bias_scale: torch.Tensor,
    output_scale: float,
    output_zero_point: int,
    out_multiplier: torch.Tensor,
    out_shift: torch.Tensor,
) -> torch.Tensor:
    out_channels, _, *kernel_size = weight.shape

    in_size = input.shape
    # Assert that the input tensor has at least 3 dimensions, and at most 6
    assert len(in_size) > 2
    assert len(in_size) < 6

    # Compute the output tensor size
    output_size = (
        get_conv1d_output_size(
            in_size,
            out_channels,
            stride[1],
            padding[1],
            dilation[1],
            kernel_size[0],
            False,
        )
        if len(in_size) == 3
        else get_conv2d_output_size(
            in_size, out_channels, stride, padding, dilation, kernel_size, False
        )
    )

    return input.new_empty(output_size, dtype=input.dtype)


@register_fake("cadence::quantized_conv_nchw.per_tensor")
def quantized_conv_nchw_per_tensor_meta(
    input: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    stride: Tuple[int],
    padding: Tuple[int],
    dilation: Tuple[int],
    groups: int,
    in_zero_point: int,
    weight_zero_point: int,
    bias_scale: float,
    output_scale: float,
    output_zero_point: int,
    out_multiplier: int,
    out_shift: int,
) -> torch.Tensor:
    out_channels, _, *kernel_size = weight.shape

    in_size = input.shape
    # Assert that the input tensor has at least 3 dimensions, and at most 6
    assert len(in_size) > 2
    assert len(in_size) < 6

    # Compute the output tensor size
    output_size = (
        get_conv1d_output_size(
            in_size,
            out_channels,
            stride[1],
            padding[1],
            dilation[1],
            kernel_size[0],
            False,
        )
        if len(in_size) == 3
        else get_conv2d_output_size(
            in_size, out_channels, stride, padding, dilation, kernel_size, False
        )
    )

    return input.new_empty(output_size, dtype=input.dtype)


@register_fake("cadence::quantized_conv_nhwc.per_tensor")
def quantized_conv_nhwc_per_tensor_meta(
    input: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    stride: Tuple[int],
    padding: Tuple[int],
    dilation: Tuple[int],
    groups: int,
    in_zero_point: int,
    weight_zero_point: int,
    bias_scale: float,
    output_scale: float,
    output_zero_point: int,
    out_multiplier: int,
    out_shift: int,
) -> torch.Tensor:
    out_channels, *kernel_size, _ = weight.shape

    in_size = input.shape
    # Assert that the input tensor has at least 3 dimensions, and at most 6
    assert len(in_size) > 2
    assert len(in_size) < 6

    # Compute the output tensor size
    output_size = (
        get_conv1d_output_size(
            in_size,
            out_channels,
            stride[1],
            padding[1],
            dilation[1],
            kernel_size[0],
            True,
        )
        if len(in_size) == 3
        else get_conv2d_output_size(
            in_size, out_channels, stride, padding, dilation, kernel_size, True
        )
    )

    return input.new_empty(output_size, dtype=input.dtype)


@register_fake("cadence::quantized_conv_nchw_asym8sxsym8s_asym8s.per_tensor")
def quantized_conv_nchw_asym8sxsym8s_asym8s_per_tensor_meta(
    input: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    stride: Tuple[int],
    padding: Tuple[int],
    dilation: Tuple[int],
    groups: int,
    in_zero_point: int,
    weight_zero_point: int,
    bias_scale: float,
    output_scale: float,
    output_zero_point: int,
    out_multiplier: int,
    out_shift: int,
) -> torch.Tensor:
    assert (
        input.dtype == torch.int8
        and weight.dtype == torch.int8
        and bias.dtype == torch.int32
    )
    out_channels, _, *kernel_size = weight.shape

    in_size = input.shape
    # Assert that the input tensor has at least 3 dimensions, and at most 6
    assert len(in_size) > 2
    assert len(in_size) < 6

    # Compute the output tensor size
    output_size = (
        get_conv1d_output_size(
            in_size,
            out_channels,
            stride[1],
            padding[1],
            dilation[1],
            kernel_size[0],
            False,
        )
        if len(in_size) == 3
        else get_conv2d_output_size(
            in_size, out_channels, stride, padding, dilation, kernel_size, False
        )
    )

    return input.new_empty(output_size, dtype=input.dtype)


@register_fake("cadence::quantized_conv_nchw_asym8uxsym8u_asym8u.per_tensor")
def quantized_conv_nchw_asym8uxsym8u_asym8u_per_tensor_meta(
    input: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    stride: Tuple[int],
    padding: Tuple[int],
    dilation: Tuple[int],
    groups: int,
    in_zero_point: int,
    weight_zero_point: int,
    bias_scale: float,
    output_scale: float,
    output_zero_point: int,
    out_multiplier: int,
    out_shift: int,
) -> torch.Tensor:
    assert (
        input.dtype == torch.uint8
        and weight.dtype == torch.uint8
        and bias.dtype == torch.int32
    )
    out_channels, _, *kernel_size = weight.shape

    in_size = input.shape
    # Assert that the input tensor has at least 3 dimensions, and at most 6
    assert len(in_size) > 2
    assert len(in_size) < 6

    # Compute the output tensor size
    output_size = (
        get_conv1d_output_size(
            in_size,
            out_channels,
            stride[1],
            padding[1],
            dilation[1],
            kernel_size[0],
            False,
        )
        if len(in_size) == 3
        else get_conv2d_output_size(
            in_size, out_channels, stride, padding, dilation, kernel_size, False
        )
    )

    return input.new_empty(output_size, dtype=input.dtype)


@register_fake("cadence::quantized_conv_nhwc_asym8sxsym8s_asym8s.per_tensor")
def quantized_conv_nhwc_asym8sxsym8s_asym8s_per_tensor_meta(
    input: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    stride: Tuple[int],
    padding: Tuple[int],
    dilation: Tuple[int],
    groups: int,
    in_zero_point: int,
    weight_zero_point: int,
    bias_scale: float,
    output_scale: float,
    output_zero_point: int,
    out_multiplier: int,
    out_shift: int,
) -> torch.Tensor:
    assert (
        input.dtype == torch.int8
        and weight.dtype == torch.int8
        and bias.dtype == torch.int32
    )
    out_channels, *kernel_size, _ = weight.shape

    in_size = input.shape
    # Assert that the input tensor has at least 3 dimensions, and at most 6
    assert len(in_size) > 2
    assert len(in_size) < 6

    # Compute the output tensor size
    output_size = (
        get_conv1d_output_size(
            in_size,
            out_channels,
            stride[1],
            padding[1],
            dilation[1],
            kernel_size[0],
            True,
        )
        if len(in_size) == 3
        else get_conv2d_output_size(
            in_size, out_channels, stride, padding, dilation, kernel_size, True
        )
    )

    return input.new_empty(output_size, dtype=input.dtype)


@register_fake("cadence::quantized_conv_nhwc_asym8uxsym8u_asym8u.per_tensor")
def quantized_conv_nhwc_asym8uxsym8u_asym8u_per_tensor_meta(
    input: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    stride: Tuple[int],
    padding: Tuple[int],
    dilation: Tuple[int],
    groups: int,
    in_zero_point: int,
    weight_zero_point: int,
    bias_scale: float,
    output_scale: float,
    output_zero_point: int,
    out_multiplier: int,
    out_shift: int,
) -> torch.Tensor:
    assert (
        input.dtype == torch.uint8
        and weight.dtype == torch.uint8
        and bias.dtype == torch.int32
    )
    out_channels, *kernel_size, _ = weight.shape

    in_size = input.shape
    # Assert that the input tensor has at least 3 dimensions, and at most 6
    assert len(in_size) > 2
    assert len(in_size) < 6

    # Compute the output tensor size
    output_size = (
        get_conv1d_output_size(
            in_size,
            out_channels,
            stride[1],
            padding[1],
            dilation[1],
            kernel_size[0],
            True,
        )
        if len(in_size) == 3
        else get_conv2d_output_size(
            in_size, out_channels, stride, padding, dilation, kernel_size, True
        )
    )

    return input.new_empty(output_size, dtype=input.dtype)


@register_fake("cadence::quantized_conv_nchw_dilated_asym8sxsym8s_asym8s.per_tensor")
def quantized_conv_nchw_dilated_asym8sxsym8s_asym8s_per_tensor_meta(
    input: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    stride: Tuple[int],
    padding: Tuple[int],
    dilation: Tuple[int],
    groups: int,
    in_zero_point: int,
    weight_zero_point: int,
    bias_scale: float,
    output_scale: float,
    output_zero_point: int,
    out_multiplier: int,
    out_shift: int,
) -> torch.Tensor:
    assert (
        input.dtype == torch.int8
        and weight.dtype == torch.int8
        and bias.dtype == torch.int32
    )
    out_channels, _, *kernel_size = weight.shape

    in_size = input.shape
    # Assert that the input tensor has at least 3 dimensions, and at most 6
    assert len(in_size) > 2
    assert len(in_size) < 6

    # Compute the output tensor size
    output_size = (
        get_conv1d_output_size(
            in_size,
            out_channels,
            stride[1],
            padding[1],
            dilation[1],
            kernel_size[0],
            False,
        )
        if len(in_size) == 3
        else get_conv2d_output_size(
            in_size, out_channels, stride, padding, dilation, kernel_size, False
        )
    )

    return input.new_empty(output_size, dtype=input.dtype)


@register_fake("cadence::quantized_conv_nchw_dilated_asym8uxsym8u_asym8u.per_tensor")
def quantized_conv_nchw_dilated_asym8uxsym8u_asym8u_per_tensor_meta(
    input: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    stride: Tuple[int],
    padding: Tuple[int],
    dilation: Tuple[int],
    groups: int,
    in_zero_point: int,
    weight_zero_point: int,
    bias_scale: float,
    output_scale: float,
    output_zero_point: int,
    out_multiplier: int,
    out_shift: int,
) -> torch.Tensor:
    assert (
        input.dtype == torch.uint8
        and weight.dtype == torch.uint8
        and bias.dtype == torch.int32
    )
    out_channels, _, *kernel_size = weight.shape

    in_size = input.shape
    # Assert that the input tensor has at least 3 dimensions, and at most 6
    assert len(in_size) > 2
    assert len(in_size) < 6

    # Compute the output tensor size
    output_size = (
        get_conv1d_output_size(
            in_size,
            out_channels,
            stride[1],
            padding[1],
            dilation[1],
            kernel_size[0],
            False,
        )
        if len(in_size) == 3
        else get_conv2d_output_size(
            in_size, out_channels, stride, padding, dilation, kernel_size, False
        )
    )

    return input.new_empty(output_size, dtype=input.dtype)


@register_fake("cadence::quantized_conv_nhwc_dilated_asym8sxsym8s_asym8s.per_tensor")
def quantized_conv_nhwc_dilated_asym8sxsym8s_asym8s_per_tensor_meta(
    input: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    stride: Tuple[int],
    padding: Tuple[int],
    dilation: Tuple[int],
    groups: int,
    in_zero_point: int,
    weight_zero_point: int,
    bias_scale: float,
    output_scale: float,
    output_zero_point: int,
    out_multiplier: int,
    out_shift: int,
) -> torch.Tensor:
    assert (
        input.dtype == torch.int8
        and weight.dtype == torch.int8
        and bias.dtype == torch.int32
    )
    out_channels, *kernel_size, _ = weight.shape

    in_size = input.shape
    # Assert that the input tensor has at least 3 dimensions, and at most 6
    assert len(in_size) > 2
    assert len(in_size) < 6

    # Compute the output tensor size
    output_size = (
        get_conv1d_output_size(
            in_size,
            out_channels,
            stride[1],
            padding[1],
            dilation[1],
            kernel_size[0],
            True,
        )
        if len(in_size) == 3
        else get_conv2d_output_size(
            in_size, out_channels, stride, padding, dilation, kernel_size, True
        )
    )

    return input.new_empty(output_size, dtype=input.dtype)


@register_fake("cadence::quantized_conv_nhwc_dilated_asym8uxsym8u_asym8u.per_tensor")
def quantized_conv_nhwc_dilated_asym8uxsym8u_asym8u_per_tensor_meta(
    input: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    stride: Tuple[int],
    padding: Tuple[int],
    dilation: Tuple[int],
    groups: int,
    in_zero_point: int,
    weight_zero_point: int,
    bias_scale: float,
    output_scale: float,
    output_zero_point: int,
    out_multiplier: int,
    out_shift: int,
) -> torch.Tensor:
    assert (
        input.dtype == torch.uint8
        and weight.dtype == torch.uint8
        and bias.dtype == torch.int32
    )
    out_channels, *kernel_size, _ = weight.shape

    in_size = input.shape
    # Assert that the input tensor has at least 3 dimensions, and at most 6
    assert len(in_size) > 2
    assert len(in_size) < 6

    # Compute the output tensor size
    output_size = (
        get_conv1d_output_size(
            in_size,
            out_channels,
            stride[1],
            padding[1],
            dilation[1],
            kernel_size[0],
            True,
        )
        if len(in_size) == 3
        else get_conv2d_output_size(
            in_size, out_channels, stride, padding, dilation, kernel_size, True
        )
    )

    return input.new_empty(output_size, dtype=input.dtype)


@register_fake("cadence::quantized_conv_nchw_depthwise_asym8sxsym8s_asym8s.per_tensor")
def quantized_conv_nchw_depthwise_asym8sxsym8s_asym8s_per_tensor_meta(
    input: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    stride: Tuple[int],
    padding: Tuple[int],
    dilation: Tuple[int],
    groups: int,
    in_zero_point: int,
    weight_zero_point: int,
    bias_scale: float,
    output_scale: float,
    output_zero_point: int,
    out_multiplier: int,
    out_shift: int,
) -> torch.Tensor:
    assert (
        input.dtype == torch.int8
        and weight.dtype == torch.int8
        and bias.dtype == torch.int32
    )
    out_channels, _, *kernel_size = weight.shape

    in_size = input.shape
    # Assert that the input tensor has at least 3 dimensions, and at most 6
    assert len(in_size) > 2
    assert len(in_size) < 6

    # Compute the output tensor size
    output_size = (
        get_conv1d_output_size(
            in_size,
            out_channels,
            stride[1],
            padding[1],
            dilation[1],
            kernel_size[0],
            False,
        )
        if len(in_size) == 3
        else get_conv2d_output_size(
            in_size, out_channels, stride, padding, dilation, kernel_size, False
        )
    )

    return input.new_empty(output_size, dtype=input.dtype)


@register_fake("cadence::quantized_conv_nchw_depthwise_asym8uxsym8u_asym8u.per_tensor")
def quantized_conv_nchw_depthwise_asym8uxsym8u_asym8u_per_tensor_meta(
    input: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    stride: Tuple[int],
    padding: Tuple[int],
    dilation: Tuple[int],
    groups: int,
    in_zero_point: int,
    weight_zero_point: int,
    bias_scale: float,
    output_scale: float,
    output_zero_point: int,
    out_multiplier: int,
    out_shift: int,
) -> torch.Tensor:
    assert (
        input.dtype == torch.uint8
        and weight.dtype == torch.uint8
        and bias.dtype == torch.int32
    )
    out_channels, _, *kernel_size = weight.shape

    in_size = input.shape
    # Assert that the input tensor has at least 3 dimensions, and at most 6
    assert len(in_size) > 2
    assert len(in_size) < 6

    # Compute the output tensor size
    output_size = (
        get_conv1d_output_size(
            in_size,
            out_channels,
            stride[1],
            padding[1],
            dilation[1],
            kernel_size[0],
            False,
        )
        if len(in_size) == 3
        else get_conv2d_output_size(
            in_size, out_channels, stride, padding, dilation, kernel_size, False
        )
    )

    return input.new_empty(output_size, dtype=input.dtype)


@register_fake("cadence::quantized_conv_nhwc_depthwise_asym8sxsym8s_asym8s.per_tensor")
def quantized_conv_nhwc_depthwise_asym8sxsym8s_asym8s_per_tensor_meta(
    input: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    stride: Tuple[int],
    padding: Tuple[int],
    dilation: Tuple[int],
    groups: int,
    in_zero_point: int,
    weight_zero_point: int,
    bias_scale: float,
    output_scale: float,
    output_zero_point: int,
    out_multiplier: int,
    out_shift: int,
) -> torch.Tensor:
    assert (
        input.dtype == torch.int8
        and weight.dtype == torch.int8
        and bias.dtype == torch.int32
    )
    out_channels, *kernel_size, _ = weight.shape

    in_size = input.shape
    # Assert that the input tensor has at least 3 dimensions, and at most 6
    assert len(in_size) > 2
    assert len(in_size) < 6

    # Compute the output tensor size
    output_size = (
        get_conv1d_output_size(
            in_size,
            out_channels,
            stride[1],
            padding[1],
            dilation[1],
            kernel_size[0],
            True,
        )
        if len(in_size) == 3
        else get_conv2d_output_size(
            in_size, out_channels, stride, padding, dilation, kernel_size, True
        )
    )

    return input.new_empty(output_size, dtype=input.dtype)


@register_fake("cadence::quantized_conv_nhwc_depthwise_asym8uxsym8u_asym8u.per_tensor")
def quantized_conv_nhwc_depthwise_asym8uxsym8u_asym8u_per_tensor_meta(
    input: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    stride: Tuple[int],
    padding: Tuple[int],
    dilation: Tuple[int],
    groups: int,
    in_zero_point: int,
    weight_zero_point: int,
    bias_scale: float,
    output_scale: float,
    output_zero_point: int,
    out_multiplier: int,
    out_shift: int,
) -> torch.Tensor:
    assert (
        input.dtype == torch.uint8
        and weight.dtype == torch.uint8
        and bias.dtype == torch.int32
    )
    out_channels, *kernel_size, _ = weight.shape

    in_size = input.shape
    # Assert that the input tensor has at least 3 dimensions, and at most 6
    assert len(in_size) > 2
    assert len(in_size) < 6

    # Compute the output tensor size
    output_size = (
        get_conv1d_output_size(
            in_size,
            out_channels,
            stride[1],
            padding[1],
            dilation[1],
            kernel_size[0],
            True,
        )
        if len(in_size) == 3
        else get_conv2d_output_size(
            in_size, out_channels, stride, padding, dilation, kernel_size, True
        )
    )

    return input.new_empty(output_size, dtype=input.dtype)


@register_fake("cadence::quantized_layer_norm")
def quantized_layer_norm_meta(
    input: torch.Tensor,
    X_scale: torch.Tensor,
    X_zero_point: torch.Tensor,
    normalized_shape: list[int],
    weight: torch.Tensor,
    bias: torch.Tensor,
    eps: float,
    output_scale: float,
    output_zero_point: int,
) -> torch.Tensor:
    return input.new_empty(input.size(), dtype=input.dtype)


@register_fake("cadence::quantized_layer_norm.per_tensor")
def quantized_layer_norm_per_tensor_meta(
    input: torch.Tensor,
    X_scale: float,
    X_zero_point: int,
    normalized_shape: list[int],
    weight: torch.Tensor,
    bias: torch.Tensor,
    eps: float,
    output_scale: float,
    output_zero_point: int,
) -> torch.Tensor:
    return input.new_empty(input.size(), dtype=input.dtype)


@register_fake("cadence::quantized_relu")
def quantized_relu_meta(
    X: torch.Tensor,
    X_zero_point: torch.Tensor,
    out_zero_point: int,
    out_multiplier: torch.Tensor,
    out_shift: torch.Tensor,
) -> torch.Tensor:
    return X.new_empty(X.size(), dtype=X.dtype)


@register_fake("cadence::quantized_matmul")
def quantized_matmul_meta(
    X: torch.Tensor,
    X_zero_point: int,
    Y: torch.Tensor,
    Y_zero_point: int,
    bias: Optional[torch.Tensor],
    out_multiplier: int,
    out_shift: int,
    out_zero_point: int,
    transposed: bool = False,
) -> torch.Tensor:
    X_size = list(X.size())
    Y_size = list(Y.size())

    # Get the batch dimensions for both tensors
    X_batch_dims = X_size[:-2]
    Y_batch_dims = Y_size[:-2]

    # If they don't match, check that they're compatible
    if X_batch_dims != Y_batch_dims:
        assert prod(X_batch_dims) == prod(
            Y_batch_dims
        ), f"Batch dimensions of X and Y do not match: {X_batch_dims} vs {Y_batch_dims}"

    # Get the matmul output size
    if transposed:
        assert X_size[-1] == Y_size[-1], "matrices cannot be multiplied"
        mat_size = [X_size[-2], Y_size[-2]]
    else:
        assert X_size[-1] == Y_size[-2], "matrices cannot be multiplied"
        mat_size = [X_size[-2], Y_size[-1]]

    # Combine the larger batch dimensions with the matmul output size
    out_size = (
        X_batch_dims + mat_size
        if len(X_batch_dims) > len(Y_batch_dims)
        else Y_batch_dims + mat_size
    )

    return X.new_empty(out_size, dtype=X.dtype)


@register_fake("cadence::quantized_matmul_asym8sxasym8s_asym8s")
def quantized_matmul_asym8sxasym8s_asym8s_meta(
    X: torch.Tensor,
    X_zero_point: int,
    Y: torch.Tensor,
    Y_zero_point: int,
    bias: Optional[torch.Tensor],
    out_multiplier: int,
    out_shift: int,
    out_zero_point: int,
    transposed: bool = False,
) -> torch.Tensor:
    X_size = list(X.size())
    Y_size = list(Y.size())

    # Get the batch dimensions for both tensors
    X_batch_dims = X_size[:-2]
    Y_batch_dims = Y_size[:-2]

    # If they don't match, check that they're compatible
    if X_batch_dims != Y_batch_dims:
        assert prod(X_batch_dims) == prod(
            Y_batch_dims
        ), f"Batch dimensions of X and Y do not match: {X_batch_dims} vs {Y_batch_dims}"

    # Get the matmul output size
    if transposed:
        assert X_size[-1] == Y_size[-1], "matrices cannot be multiplied"
        mat_size = [X_size[-2], Y_size[-2]]
    else:
        assert X_size[-1] == Y_size[-2], "matrices cannot be multiplied"
        mat_size = [X_size[-2], Y_size[-1]]

    # Combine the larger batch dimensions with the matmul output size
    out_size = (
        X_batch_dims + mat_size
        if len(X_batch_dims) > len(Y_batch_dims)
        else Y_batch_dims + mat_size
    )

    return X.new_empty(out_size, dtype=X.dtype)


@register_fake("cadence::quantized_matmul_asym8uxasym8u_asym8u")
def quantized_matmul_asym8uxasym8u_asym8u_meta(
    X: torch.Tensor,
    X_zero_point: int,
    Y: torch.Tensor,
    Y_zero_point: int,
    bias: Optional[torch.Tensor],
    out_multiplier: int,
    out_shift: int,
    out_zero_point: int,
    transposed: bool = False,
) -> torch.Tensor:
    X_size = list(X.size())
    Y_size = list(Y.size())

    # Get the batch dimensions for both tensors
    X_batch_dims = X_size[:-2]
    Y_batch_dims = Y_size[:-2]

    # If they don't match, check that they're compatible
    if X_batch_dims != Y_batch_dims:
        assert prod(X_batch_dims) == prod(
            Y_batch_dims
        ), f"Batch dimensions of X and Y do not match: {X_batch_dims} vs {Y_batch_dims}"

    # Get the matmul output size
    if transposed:
        assert X_size[-1] == Y_size[-1], "matrices cannot be multiplied"
        mat_size = [X_size[-2], Y_size[-2]]
    else:
        assert X_size[-1] == Y_size[-2], "matrices cannot be multiplied"
        mat_size = [X_size[-2], Y_size[-1]]

    # Combine the larger batch dimensions with the matmul output size
    out_size = (
        X_batch_dims + mat_size
        if len(X_batch_dims) > len(Y_batch_dims)
        else Y_batch_dims + mat_size
    )

    return X.new_empty(out_size, dtype=X.dtype)


@register_fake("cadence::im2row")
def im2row_meta(
    input: torch.Tensor,
    kernel_size: Tuple[int],
    dilation: Tuple[int],
    padding: Tuple[int],
    stride: Tuple[int],
    in_zero_point: torch.Tensor,
    channel_last: bool = False,
) -> torch.Tensor:
    output_size = get_im2row_output_size(
        input, kernel_size, dilation, padding, stride, channel_last
    )
    return input.new_empty(output_size, dtype=input.dtype)


@register_fake("cadence::im2row.per_tensor")
def im2row_per_tensor_meta(
    input: torch.Tensor,
    kernel_size: Tuple[int],
    dilation: Tuple[int],
    padding: Tuple[int],
    stride: Tuple[int],
    in_zero_point: int,
    channel_last: bool = False,
) -> torch.Tensor:
    output_size = get_im2row_output_size(
        input, kernel_size, dilation, padding, stride, channel_last
    )
    return input.new_empty(output_size, dtype=input.dtype)


# Define the abstract implementations of the operators as required
@register_fake("cadence::linalg_vector_norm")
def linalg_vector_norm_meta(
    X: torch.Tensor,
) -> torch.Tensor:
    # Output of norm is a scalar, so we return a [] tensor
    return X.new_empty([], dtype=X.dtype)


@register_fake("cadence::linalg_svd")
def linalg_svd_meta(
    A: torch.Tensor,
    full_matrices: bool = False,
    compute_uv: bool = True,
    driver: Optional[str] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    # Based on the _linalg_svd meta implementation, but ensuring contiguous strides

    # Get the shapes from the original meta function
    U, S, Vh = _linalg_svd_meta(A, full_matrices, compute_uv, driver)

    # Create new tensors with contiguous strides to fix the non-contiguous issue
    U_contiguous = A.new_empty(U.shape, dtype=A.dtype).contiguous()
    S_contiguous = A.new_empty(S.shape, dtype=A.dtype).contiguous()
    Vh_contiguous = A.new_empty(Vh.shape, dtype=A.dtype).contiguous()

    return U_contiguous, S_contiguous, Vh_contiguous


@register_fake("cadence::requantize")
def requantize_meta(
    input: torch.Tensor,
    in_scale: torch.Tensor,
    in_zero_point: torch.Tensor,
    out_scale: torch.Tensor,
    out_zero_point: torch.Tensor,
    dtype: ScalarType,
) -> torch.Tensor:
    return input.new_empty(
        input.size(),
        # pyre-ignore[6]: Incompatible type
        dtype=dtype,
    )


@register_fake("cadence::requantize.per_tensor")
def requantize_per_tensor_meta(
    input: torch.Tensor,
    in_scale: float,
    in_zero_point: int,
    out_scale: float,
    out_zero_point: int,
    dtype: ScalarType,
) -> torch.Tensor:
    return input.new_empty(
        input.size(),
        # pyre-ignore[6]: Incompatible type
        dtype=dtype,
    )


@register_fake("cadence::quantized_relu.per_tensor")
def quantized_relu_per_tensor_meta(
    input: torch.Tensor,
    in_zero_point: int,
    out_zero_point: int,
    out_multiplier: int,
    out_shift: int,
) -> torch.Tensor:
    return input.new_empty(input.size(), dtype=input.dtype)


@register_fake("cadence::quantized_relu_asym8s_asym8s.per_tensor")
def quantized_relu_asym8s_asym8s_per_tensor_meta(
    input: torch.Tensor,
    in_zero_point: int,
    out_zero_point: int,
    out_multiplier: int,
    out_shift: int,
) -> torch.Tensor:
    return input.new_empty(input.size(), dtype=input.dtype)


@register_fake("cadence::quantized_relu_asym8u_asym8u.per_tensor")
def quantized_relu_asym8u_asym8u_per_tensor_meta(
    input: torch.Tensor,
    in_zero_point: int,
    out_zero_point: int,
    out_multiplier: int,
    out_shift: int,
) -> torch.Tensor:
    return input.new_empty(input.size(), dtype=input.dtype)


@register_fake("cadence::fully_connected")
def fully_connected_meta(
    src: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
) -> torch.Tensor:
    # src comes in shape [leading_dims, in_dim]
    # weight comes in shape [out_dim, in_dim]
    # output comes in empty with shape [leading_dims, out_dim]
    out_size = list(src.size())
    weight_size = list(weight.size())
    assert len(weight_size) == 2
    out_size[-1] = weight_size[0]
    return src.new_empty(out_size, dtype=src.dtype)


@register_fake("cadence::quantized_fully_connected")
def quantized_fully_connected_meta(
    src: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    in_zero_point: int,
    weight_zero_point: torch.Tensor,
    out_multiplier: torch.Tensor,
    out_shift: torch.Tensor,
    out_zero_point: int,
    offset: Optional[torch.Tensor],
) -> torch.Tensor:
    # src comes in shape [leading_dims, in_dim]
    # weight comes in shape [out_dim, in_dim]
    # output comes in empty with shape [leading_dims, out_dim]
    assert src.shape[0] == 1
    out_size = list(src.size())
    weight_size = list(weight.size())
    assert len(weight_size) == 2
    out_size[-1] = weight_size[0]
    return src.new_empty(out_size, dtype=src.dtype)


@register_fake("cadence::quantized_fully_connected.per_tensor")
def quantized_fully_connected_per_tensor_meta(
    src: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    in_zero_point: int,
    weight_zero_point: int,
    out_multiplier: int,
    out_shift: int,
    out_zero_point: int,
    offset: Optional[torch.Tensor],
) -> torch.Tensor:
    # src comes in shape [leading_dims, in_dim]
    # weight comes in shape [out_dim, in_dim]
    # output comes in empty with shape [leading_dims, out_dim]
    assert src.shape[0] == 1
    out_size = list(src.size())
    weight_size = list(weight.size())
    assert len(weight_size) == 2
    out_size[-1] = weight_size[0]
    return src.new_empty(out_size, dtype=src.dtype)


@register_fake("cadence::quantized_fully_connected_asym8sxasym8s_asym8s.per_tensor")
def quantized_fully_connected_asym8sxasym8s_asym8s_per_tensor_meta(
    src: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    in_zero_point: int,
    weight_zero_point: int,
    out_multiplier: int,
    out_shift: int,
    out_zero_point: int,
    offset: Optional[torch.Tensor],
) -> torch.Tensor:
    # src comes in shape [leading_dims, in_dim]
    # weight comes in shape [out_dim, in_dim]
    # output comes in empty with shape [leading_dims, out_dim]
    assert src.shape[0] == 1
    out_size = list(src.size())
    weight_size = list(weight.size())
    assert len(weight_size) == 2
    out_size[-1] = weight_size[0]
    return src.new_empty(out_size, dtype=src.dtype)


@register_fake("cadence::quantized_fully_connected_asym8uxasym8u_asym8u.per_tensor")
def quantized_fully_connected_asym8uxasym8u_asym8u_per_tensor_meta(
    src: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    in_zero_point: int,
    weight_zero_point: int,
    out_multiplier: int,
    out_shift: int,
    out_zero_point: int,
    offset: Optional[torch.Tensor],
) -> torch.Tensor:
    # src comes in shape [leading_dims, in_dim]
    # weight comes in shape [out_dim, in_dim]
    # output comes in empty with shape [leading_dims, out_dim]
    assert src.shape[0] == 1
    out_size = list(src.size())
    weight_size = list(weight.size())
    assert len(weight_size) == 2
    out_size[-1] = weight_size[0]
    return src.new_empty(out_size, dtype=src.dtype)


@register_fake("cadence::convolution")
def convolution_meta(
    input: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    stride: Tuple[int],
    padding: Tuple[int],
    dilation: Tuple[int],
    groups: int,
    channel_last: bool = False,
) -> torch.Tensor:
    if channel_last:
        out_channels, *kernel_size, _ = weight.shape
    else:
        out_channels, _, *kernel_size = weight.shape
    in_size = input.shape
    # Assert that the input tensor has at least 3 dimensions, and at most 6
    assert len(in_size) > 2
    assert len(in_size) < 6

    # Compute the output tensor size
    output_size = (
        get_conv1d_output_size(
            in_size,
            out_channels,
            stride[0],
            padding[0],
            dilation[0],
            kernel_size[0],
            channel_last,
        )
        if len(in_size) == 3
        else get_conv2d_output_size(
            in_size, out_channels, stride, padding, dilation, kernel_size, channel_last
        )
    )

    return input.new_empty(output_size, dtype=input.dtype)


@register_fake("cadence::transposed_convolution")
def transposed_convolution_meta(
    input: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    stride: Tuple[int],
    padding: Tuple[int],
    dilation: Tuple[int],
    output_padding: Tuple[int],
    groups: int,
    channel_last: bool = False,
) -> torch.Tensor:
    # The native definition of torch transposed conv will have weight shape as
    # (in_channels, out_channels/groups, *kernel_size).
    # However, the two channel position is flipped in the Cadence pass of replacing it
    # with cadence::transposed_convolution here: https://fburl.com/code/d2s7pkyy
    out_channels, _input_channels, *kernel_size = weight.shape
    out_channels *= groups
    in_size = input.shape

    # Get the output size of a transposed 1D convolution given the input size and parameters
    def get_conv_transpose1d_output_size(
        in_size: torch.Size,
        kernel_size: list[int],
        out_channels: int,
        stride: Tuple[int],
        padding: Tuple[int],
        dilation: Tuple[int],
        output_padding: Tuple[int],
        channel_last: bool = False,
    ) -> torch.Size:
        assert len(in_size) == 3
        if channel_last:
            N, L, C = in_size
        else:
            N, C, L = in_size

        # Reference: https://pytorch.org/docs/stable/generated/torch.nn.ConvTranspose1d.html
        lout = (
            (L - 1) * stride[0]
            - 2 * padding[0]
            + dilation[0] * (kernel_size[0] - 1)
            + output_padding[0]
            + 1
        )

        if channel_last:
            return torch.Size((in_size[0], lout, out_channels))
        else:
            return torch.Size((in_size[0], out_channels, lout))

    def get_conv_transpose2d_output_size(
        in_size: torch.Size,
        kernel_size: list[int],
        out_channels: int,
        stride: Tuple[int],
        padding: Tuple[int],
        dilation: Tuple[int],
        output_padding: Tuple[int],
        channel_last: bool = False,
    ) -> torch.Size:
        assert len(in_size) == 4
        if channel_last:
            N, H, W, C = in_size
        else:
            N, C, H, W = in_size

        # Reference: https://pytorch.org/docs/stable/generated/torch.nn.ConvTranspose2d.html
        hout = (
            (H - 1) * stride[0]
            - 2 * padding[0]
            + dilation[0] * (kernel_size[0] - 1)
            + output_padding[0]
            + 1
        )
        wout = (
            (W - 1) * stride[1]
            - 2 * padding[1]
            + dilation[1] * (kernel_size[1] - 1)
            + output_padding[1]
            + 1
        )

        if channel_last:
            return torch.Size((in_size[0], hout, wout, out_channels))
        else:
            return torch.Size((in_size[0], out_channels, hout, wout))

    # Compute the output tensor size
    if len(in_size) == 3:
        output_size = get_conv_transpose1d_output_size(
            in_size,
            kernel_size,
            out_channels,
            stride,
            padding,
            dilation,
            output_padding,
            channel_last,
        )
    elif len(in_size) == 4:
        output_size = get_conv_transpose2d_output_size(
            in_size,
            kernel_size,
            out_channels,
            stride,
            padding,
            dilation,
            output_padding,
            channel_last,
        )
    else:
        raise NotImplementedError(
            f"transposed_convolution meta is not implemented for input tensor with {len(in_size)} dimensions"
        )

    return input.new_empty(output_size, dtype=input.dtype)


@register_fake("cadence::avg_pool2d")
def avg_pool2d_meta(
    input: torch.Tensor,
    kernel_size: Tuple[int],
    stride: Tuple[int],
    padding: Tuple[int],
    ceil_mode: bool,
    count_include_pad: Optional[bool] = True,
    divisor_override: Optional[int] = None,
    in_zero_point: Optional[int] = None,
    channel_last: bool = False,
) -> torch.Tensor:
    # Use torch native meta kernels when operator semantics are similar
    return torch._meta_registrations.meta_avg_pool2d(
        input,
        kernel_size,
        stride,
        padding,
        ceil_mode,
        count_include_pad,
        divisor_override,
    )


@register_fake("cadence::transposed_im2row")
def transposed_im2row_meta(
    input: torch.Tensor,
    kernel_size: Tuple[int],
    dilation: Tuple[int],
    padding: Tuple[int],
    stride: Tuple[int],
    output_padding: Tuple[int],
    in_zero_point: torch.Tensor,
    channel_last: bool = False,
) -> torch.Tensor:
    if len(input.shape) == 3:
        height_dim = 1 if channel_last else 2
        input = input.unsqueeze(height_dim)

    batch_size = input.shape[0]
    n_input_plane = input.shape[3] if channel_last else input.shape[1]
    input_height = input.shape[1] if channel_last else input.shape[2]
    input_width = input.shape[2] if channel_last else input.shape[3]
    output_height = (
        (input_height - 1) * stride[0]
        - 2 * padding[0]
        + dilation[0] * (kernel_size[0] - 1)
        + output_padding[0]
        + 1
    )
    output_width = (
        (input_width - 1) * stride[1]
        - 2 * padding[1]
        + dilation[1] * (kernel_size[1] - 1)
        + output_padding[1]
        + 1
    )
    n_output_plane = n_input_plane * kernel_size[0] * kernel_size[1]
    output_length = output_height * output_width
    output_size = torch.Size((batch_size, output_length, n_output_plane))

    return input.new_empty(output_size, dtype=input.dtype)


@register_fake("cadence::where_Scalar")
def where_Scalar_meta(
    condition: torch.Tensor,
    self: float,
    other: float,
) -> torch.Tensor:
    return condition.new_empty(condition.size(), dtype=torch.float32)


@register_fake("cadence::rope")
def rope_meta(
    input: torch.Tensor,
    sin_tensor: torch.Tensor,
    cos_tensor: torch.Tensor,
    pos: Optional[torch.Tensor],
) -> torch.Tensor:
    input_shape = list(input.shape)
    assert (
        len(input_shape) in (4, 5) and input_shape[0] == 1
    ), f"input shape {input_shape} must be (1, seq, h, hd) or (1, seq, h, hd / 2, 2)"
    seq = input_shape[1]
    h = input_shape[2]
    hd = prod(input_shape) / (seq * h)
    sin_shape = list(sin_tensor.shape)
    cos_shape = list(cos_tensor.shape)
    assert sin_shape == cos_shape, f"{sin_shape=} must be same as {cos_shape}"
    assert (
        len(sin_shape) == 2 and sin_shape[-1] == hd // 2
    ), f"{sin_shape=} must be [seq, hd/2]"
    if pos is not None:
        assert (
            len(pos.shape) == 1 and pos.shape[0] == seq
        ), f"{pos.shape} must be [{seq}]"
    return input.new_empty(input.shape, dtype=input.dtype)


@register_fake("cadence::idma_copy")
def copy_idma_copy_impl(
    src: torch.Tensor,
    task_num: int = 0,
    channel: int = 0,
) -> torch.Tensor:
    return src.new_empty(*src.shape, dtype=src.dtype)


@register_fake("cadence::idma_wait")
def copy_idma_wait_impl(
    src: torch.Tensor,
    task_num: int = 0,
) -> torch.Tensor:
    return src.new_empty(*src.shape, dtype=src.dtype)


@register_fake("cadence::idma_load")
def idma_load_impl(
    src: torch.Tensor,
    task_num: int = 0,
    channel: int = 0,
) -> torch.Tensor:
    return copy_idma_copy_impl(src, task_num, channel)


@register_fake("cadence::idma_store")
def idma_store_impl(
    src: torch.Tensor,
    task_num: int = 0,
    channel: int = 0,
) -> torch.Tensor:
    return copy_idma_copy_impl(src, task_num, channel)


@register_fake("cadence::roi_align_box_processor")
def roi_align_box_processor_meta(
    rois: torch.Tensor,
    output_size_h: int,
    output_size_w: int,
    sampling_ratio: int,
    aligned: bool,
) -> torch.Tensor:
    return rois.new_empty((rois.shape[0], 80), dtype=torch.uint8)


@register_fake("cadence::_softmax_f32_f32")
def softmax_f32_f32_meta(
    self: torch.Tensor,
    dim: int,
    dtype: torch.dtype,
    half_to_float: Optional[bool] = None,
) -> torch.Tensor:
    return self.new_empty(self.size(), dtype=self.dtype)
