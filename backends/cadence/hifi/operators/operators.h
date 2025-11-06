/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include "executorch/runtime/core/exec_aten/exec_aten.h"
#include "executorch/runtime/kernel/kernel_runtime_context.h"

#define ET_FORALL_CADENCE_QUANTIZED_TYPES(_) \
  _(uint8_t, Byte)                           \
  _(int8_t, Char)

namespace impl {
namespace HiFi {
namespace native {

void dequantize_per_tensor_out(
    ::executorch::runtime::KernelRuntimeContext& ctx,
    const ::executorch::aten::Tensor& input,
    double scale,
    int64_t zero_point,
    int64_t quant_min,
    int64_t quant_max,
    ::executorch::aten::ScalarType dtype,
    ::executorch::aten::Tensor& out);

// Quantize the input tensor (PT2 version). Note that quant_<min,max> are not
// used in any computation.
void quantize_per_tensor_out(
    ::executorch::runtime::KernelRuntimeContext& ctx,
    const ::executorch::aten::Tensor& input,
    double scale,
    int64_t zero_point,
    int64_t quant_min,
    int64_t quant_max,
    ::executorch::aten::ScalarType dtype,
    ::executorch::aten::Tensor& out);

::executorch::aten::Tensor& div_out_mode(
    ::executorch::runtime::KernelRuntimeContext& ctx,
    const ::executorch::aten::Tensor& a,
    const ::executorch::aten::Tensor& b,
    std::optional<std::string_view> mode,
    ::executorch::aten::Tensor& out);

void quantized_relu_out(
    ::executorch::runtime::KernelRuntimeContext& ctx,
    const ::executorch::aten::Tensor& input,
    const ::executorch::aten::Tensor& in_zero_point,
    const int64_t out_zero_point,
    const ::executorch::aten::Tensor& out_multiplier,
    const ::executorch::aten::Tensor& out_shift,
    ::executorch::aten::Tensor& output);

void quantized_linear_out(
    ::executorch::runtime::KernelRuntimeContext& ctx,
    const ::executorch::aten::Tensor& in,
    const ::executorch::aten::Tensor& weight,
    const ::executorch::aten::Tensor& bias,
    int64_t in_zero_point,
    const ::executorch::aten::Tensor& weight_zero_point,
    const ::executorch::aten::Tensor& out_multiplier,
    const ::executorch::aten::Tensor& out_shift,
    int64_t out_zero_point,
    const ::executorch::aten::optional<::executorch::aten::Tensor>& offset,
    ::executorch::aten::Tensor& out);

void quantized_linear_per_tensor_out(
    ::executorch::runtime::KernelRuntimeContext& ctx,
    const ::executorch::aten::Tensor& in,
    const ::executorch::aten::Tensor& weight,
    const ::executorch::aten::Tensor& bias,
    int64_t in_zero_point,
    int64_t weight_zero_point,
    int64_t out_multiplier,
    int64_t out_shift,
    int64_t out_zero_point,
    const ::executorch::aten::optional<::executorch::aten::Tensor>& offset,
    ::executorch::aten::Tensor& out);

void quantized_conv2d_nhwc_out(
    ::executorch::runtime::KernelRuntimeContext& ctx,
    const ::executorch::aten::Tensor& input,
    const ::executorch::aten::Tensor& weight,
    const ::executorch::aten::Tensor& bias,
    ::executorch::aten::IntArrayRef stride,
    ::executorch::aten::IntArrayRef padding,
    ::executorch::aten::IntArrayRef dilation,
    int64_t groups,
    int64_t in_zero_point,
    const ::executorch::aten::Tensor& weight_zero_point,
    const ::executorch::aten::Tensor& bias_scale,
    double output_scale,
    int64_t output_zero_point,
    const ::executorch::aten::Tensor& out_multiplier,
    const ::executorch::aten::Tensor& out_shift,
    ::executorch::aten::Tensor& out);

void quantized_conv2d_nchw_out(
    ::executorch::runtime::KernelRuntimeContext& ctx,
    const ::executorch::aten::Tensor& input,
    const ::executorch::aten::Tensor& weight,
    const ::executorch::aten::Tensor& bias,
    ::executorch::aten::IntArrayRef stride,
    ::executorch::aten::IntArrayRef padding,
    ::executorch::aten::IntArrayRef dilation,
    int64_t groups,
    int64_t in_zero_point,
    const ::executorch::aten::Tensor& weight_zero_point,
    const ::executorch::aten::Tensor& bias_scale,
    double output_scale,
    int64_t output_zero_point,
    const ::executorch::aten::Tensor& out_multiplier,
    const ::executorch::aten::Tensor& out_shift,
    ::executorch::aten::Tensor& out);

void quantized_conv2d_nchw_per_tensor_out(
    ::executorch::runtime::KernelRuntimeContext& ctx,
    const ::executorch::aten::Tensor& input,
    const ::executorch::aten::Tensor& weight,
    const ::executorch::aten::Tensor& bias,
    ::executorch::aten::IntArrayRef stride,
    ::executorch::aten::IntArrayRef padding,
    ::executorch::aten::IntArrayRef dilation,
    int64_t groups,
    int64_t in_zero_point,
    int64_t weight_zero_point,
    double bias_scale,
    double output_scale,
    int64_t output_zero_point,
    int64_t out_multiplier,
    int64_t out_shift,
    ::executorch::aten::Tensor& out);

void quantized_conv2d_nhwc_per_tensor_out(
    ::executorch::runtime::KernelRuntimeContext& ctx,
    const ::executorch::aten::Tensor& input,
    const ::executorch::aten::Tensor& weight,
    const ::executorch::aten::Tensor& bias,
    ::executorch::aten::IntArrayRef stride,
    ::executorch::aten::IntArrayRef padding,
    ::executorch::aten::IntArrayRef dilation,
    int64_t groups,
    int64_t in_zero_point,
    int64_t weight_zero_point,
    double bias_scale,
    double output_scale,
    int64_t output_zero_point,
    int64_t out_multiplier,
    int64_t out_shift,
    ::executorch::aten::Tensor& out);

::executorch::aten::Tensor& cat_out(
    ::executorch::runtime::KernelRuntimeContext& ctx,
    ::executorch::aten::ArrayRef<::executorch::aten::Tensor> tensors,
    int64_t dim,
    ::executorch::aten::Tensor& out);

::executorch::aten::Tensor& permute_copy_out(
    ::executorch::runtime::KernelRuntimeContext& ctx,
    const ::executorch::aten::Tensor& in,
    ::executorch::aten::IntArrayRef dims,
    ::executorch::aten::Tensor& out);

void quantized_add_asym8sxasym8s_asym8s_per_tensor_out(
    ::executorch::runtime::KernelRuntimeContext& ctx,
    const ::executorch::aten::Tensor& X,
    double X_scale,
    int64_t X_zero_point,
    const ::executorch::aten::Tensor& Y,
    double Y_scale,
    int64_t Y_zero_point,
    double out_scale,
    int64_t out_zero_point,
    ::executorch::aten::Tensor& out);

void quantized_add_asym8uxasym8u_asym8u_per_tensor_out(
    ::executorch::runtime::KernelRuntimeContext& ctx,
    const ::executorch::aten::Tensor& X,
    double X_scale,
    int64_t X_zero_point,
    const ::executorch::aten::Tensor& Y,
    double Y_scale,
    int64_t Y_zero_point,
    double out_scale,
    int64_t out_zero_point,
    ::executorch::aten::Tensor& out);

} // namespace native
} // namespace HiFi
} // namespace impl
