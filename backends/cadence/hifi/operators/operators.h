/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#define ET_FORALL_CADENCE_QUANTIZED_TYPES(_) \
  _(uint8_t, Byte)                           \
  _(int8_t, Char)

using ::executorch::aten::IntArrayRef;
using ::executorch::aten::optional;
using ::executorch::aten::ScalarType;
using ::executorch::aten::Tensor;
using ::executorch::runtime::KernelRuntimeContext;

namespace cadence {
namespace impl {
namespace HiFi {
namespace native {

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
    ::executorch::aten::optional<::executorch::aten::string_view> mode,
    ::executorch::aten::Tensor& out);

void quantized_linear_out(
    __ET_UNUSED KernelRuntimeContext& ctx,
    const Tensor& in,
    const Tensor& weight,
    const Tensor& bias,
    int64_t in_zero_point,
    const Tensor& weight_zero_point,
    const Tensor& out_multiplier,
    const Tensor& out_shift,
    int64_t out_zero_point,
    __ET_UNUSED const optional<Tensor>& offset,
    Tensor& out);

void quantized_linear_per_tensor_out(
    __ET_UNUSED KernelRuntimeContext& ctx,
    const Tensor& in,
    const Tensor& weight,
    const Tensor& bias,
    int64_t in_zero_point,
    int64_t weight_zero_point,
    int64_t out_multiplier,
    int64_t out_shift,
    int64_t out_zero_point,
    __ET_UNUSED const optional<Tensor>& offset,
    Tensor& out);

void quantized_conv_out(
    __ET_UNUSED KernelRuntimeContext& ctx,
    const Tensor& input,
    const Tensor& weight,
    const Tensor& bias,
    IntArrayRef stride,
    IntArrayRef padding,
    IntArrayRef dilation,
    int64_t groups,
    int64_t in_zero_point,
    const Tensor& weight_zero_point,
    const Tensor& bias_scale,
    double output_scale,
    int64_t output_zero_point,
    __ET_UNUSED const Tensor& out_multiplier,
    __ET_UNUSED const Tensor& out_shift,
    bool channel_last,
    Tensor& out);

void quantized_conv_per_tensor_out(
    __ET_UNUSED KernelRuntimeContext& ctx,
    const Tensor& input,
    const Tensor& weight,
    const Tensor& bias,
    IntArrayRef stride,
    IntArrayRef padding,
    IntArrayRef dilation,
    int64_t groups,
    int64_t in_zero_point,
    int64_t weight_zero_point,
    double bias_scale,
    double output_scale,
    int64_t output_zero_point,
    __ET_UNUSED int64_t out_multiplier,
    __ET_UNUSED int64_t out_shift,
    bool channel_last,
    Tensor& out);

} // namespace native
} // namespace HiFi
} // namespace impl
} // namespace cadence
