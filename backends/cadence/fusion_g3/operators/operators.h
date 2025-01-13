/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <executorch/runtime/core/exec_aten/exec_aten.h>
#include <executorch/runtime/kernel/kernel_runtime_context.h>

namespace cadence {
namespace impl {
namespace G3 {
namespace native {

::executorch::aten::Tensor& _softmax_out(
    ::executorch::runtime::KernelRuntimeContext& ctx,
    const ::executorch::aten::Tensor& in,
    int64_t dim,
    bool half_to_float,
    ::executorch::aten::Tensor& out);

::executorch::aten::Tensor& add_out(
    ::executorch::runtime::KernelRuntimeContext& ctx,
    const ::executorch::aten::Tensor& a,
    const ::executorch::aten::Tensor& b,
    const ::executorch::aten::Scalar& alpha,
    ::executorch::aten::Tensor& out);

::executorch::aten::Tensor& add_scalar_out(
    ::executorch::runtime::KernelRuntimeContext& ctx,
    const ::executorch::aten::Tensor& a,
    const ::executorch::aten::Scalar& b,
    const ::executorch::aten::Scalar& alpha,
    ::executorch::aten::Tensor& out);

::executorch::aten::Tensor& cat_out(
    ::executorch::runtime::KernelRuntimeContext& ctx,
    ::executorch::aten::ArrayRef<::executorch::aten::Tensor> tensors,
    int64_t dim,
    ::executorch::aten::Tensor& out);

::executorch::aten::Tensor& dequantize_per_channel_out(
    ::executorch::runtime::KernelRuntimeContext& context,
    const ::executorch::aten::Tensor& input,
    const ::executorch::aten::Tensor& scale,
    const ::executorch::aten::optional<::executorch::aten::Tensor>&
        opt_zero_points,
    int64_t axis,
    int64_t quant_min,
    int64_t quant_max,
    ::executorch::aten::ScalarType dtype,
    ::executorch::aten::optional<::executorch::aten::ScalarType> out_dtype,
    ::executorch::aten::Tensor& out);

::executorch::aten::Tensor& dequantize_per_tensor_out(
    ::executorch::runtime::KernelRuntimeContext& context,
    const ::executorch::aten::Tensor& input,
    double scale,
    int64_t zero_point,
    int64_t quant_min,
    int64_t quant_max,
    ::executorch::aten::ScalarType dtype,
    ::executorch::aten::optional<::executorch::aten::ScalarType> out_dtype,
    ::executorch::aten::Tensor& out);

::executorch::aten::Tensor& div_out(
    ::executorch::runtime::KernelRuntimeContext& ctx,
    const ::executorch::aten::Tensor& a,
    const ::executorch::aten::Tensor& b,
    ::executorch::aten::Tensor& out);

::executorch::aten::Tensor& div_out_mode(
    ::executorch::runtime::KernelRuntimeContext& ctx,
    const ::executorch::aten::Tensor& a,
    const ::executorch::aten::Tensor& b,
    ::executorch::aten::optional<::executorch::aten::string_view> mode,
    ::executorch::aten::Tensor& out);

::executorch::aten::Tensor& div_scalar_out(
    ::executorch::runtime::KernelRuntimeContext& ctx,
    const ::executorch::aten::Tensor& a,
    const ::executorch::aten::Scalar& b,
    ::executorch::aten::Tensor& out);

::executorch::aten::Tensor& div_scalar_mode_out(
    ::executorch::runtime::KernelRuntimeContext& ctx,
    const ::executorch::aten::Tensor& a,
    const ::executorch::aten::Scalar& b,
    ::executorch::aten::optional<::executorch::aten::string_view> mode,
    ::executorch::aten::Tensor& out);

::executorch::aten::Tensor& exp_out(
    ::executorch::runtime::KernelRuntimeContext& ctx,
    const ::executorch::aten::Tensor& in,
    ::executorch::aten::Tensor& out);

::executorch::aten::Tensor& mean_dim_out(
    ::executorch::runtime::KernelRuntimeContext& ctx,
    const ::executorch::aten::Tensor& in,
    ::executorch::aten::optional<::executorch::aten::ArrayRef<int64_t>>
        dim_list,
    bool keepdim,
    ::executorch::aten::optional<::executorch::aten::ScalarType> dtype,
    ::executorch::aten::Tensor& out);

::executorch::aten::Tensor& mul_out(
    ::executorch::runtime::KernelRuntimeContext& ctx,
    const ::executorch::aten::Tensor& a,
    const ::executorch::aten::Tensor& b,
    ::executorch::aten::Tensor& out);

::executorch::aten::Tensor& mul_scalar_out(
    ::executorch::runtime::KernelRuntimeContext& ctx,
    const ::executorch::aten::Tensor& a,
    const ::executorch::aten::Scalar& b,
    ::executorch::aten::Tensor& out);

std::tuple<
    ::executorch::aten::Tensor&,
    ::executorch::aten::Tensor&,
    ::executorch::aten::Tensor&>
native_layer_norm_out(
    ::executorch::runtime::KernelRuntimeContext& ctx,
    const ::executorch::aten::Tensor& input,
    ::executorch::aten::IntArrayRef normalized_shape,
    const ::executorch::aten::optional<::executorch::aten::Tensor>& weight,
    const ::executorch::aten::optional<::executorch::aten::Tensor>& bias,
    double eps,
    ::executorch::aten::Tensor& out,
    ::executorch::aten::Tensor& mean_out,
    ::executorch::aten::Tensor& rstd_out);

::executorch::aten::Tensor& permute_copy_out(
    ::executorch::runtime::KernelRuntimeContext& ctx,
    const ::executorch::aten::Tensor& in,
    ::executorch::aten::IntArrayRef dims,
    ::executorch::aten::Tensor& out);

::executorch::aten::Tensor& quantize_per_channel_out(
    ::executorch::runtime::KernelRuntimeContext& context,
    const ::executorch::aten::Tensor& input,
    const ::executorch::aten::Tensor& scale,
    const ::executorch::aten::Tensor& zero_point,
    int64_t axis,
    int64_t quant_min,
    int64_t quant_max,
    ::executorch::aten::ScalarType dtype,
    ::executorch::aten::Tensor& out);

::executorch::aten::Tensor& quantize_per_tensor_out(
    ::executorch::runtime::KernelRuntimeContext& context,
    const ::executorch::aten::Tensor& input,
    double scale,
    int64_t zero_point,
    int64_t quant_min,
    int64_t quant_max,
    ::executorch::aten::ScalarType dtype,
    ::executorch::aten::Tensor& out);

::executorch::aten::Tensor& slice_copy_Tensor_out(
    ::executorch::runtime::KernelRuntimeContext& ctx,
    const ::executorch::aten::Tensor& in,
    int64_t dim,
    ::executorch::aten::optional<int64_t> start_val,
    ::executorch::aten::optional<int64_t> end_val,
    int64_t step,
    ::executorch::aten::Tensor& out);

::executorch::aten::Tensor& sub_out(
    ::executorch::runtime::KernelRuntimeContext& ctx,
    const ::executorch::aten::Tensor& a,
    const ::executorch::aten::Tensor& b,
    const ::executorch::aten::Scalar& alpha,
    ::executorch::aten::Tensor& out);

::executorch::aten::Tensor& sub_scalar_out(
    ::executorch::runtime::KernelRuntimeContext& ctx,
    const ::executorch::aten::Tensor& a,
    const ::executorch::aten::Scalar& b,
    const ::executorch::aten::Scalar& alpha,
    ::executorch::aten::Tensor& out);

} // namespace native
} // namespace G3
} // namespace impl
} // namespace cadence
