/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/backends/cadence/hifi/operators/operators.h>

#include <algorithm>
#include <cmath>
#include <optional>

#include <xa_nnlib_api.h>
#include <xtensa/tie/xt_datacache.h>

#include <executorch/backends/cadence/generic/operators/op_quantized_linear.h>
#include <executorch/backends/cadence/hifi/kernels/kernels.h>
#include <executorch/runtime/kernel/kernel_includes.h>

namespace impl::HiFi::native {

using ::executorch::aten::ScalarType;
using ::executorch::aten::Tensor;
using ::executorch::runtime::getLeadingDims;
using ::executorch::runtime::KernelRuntimeContext;
using std::optional;

// The nnlib kernel to compute quantized linear via matmul.

void _quantized_linear_asym8u(
    const Tensor& in,
    const Tensor& weight,
    const Tensor& bias,
    int64_t in_zero_point,
    const Tensor& weight_zero_point,
    const Tensor& out_multiplier,
    const Tensor& out_shift,
    int64_t out_zero_point,
    __ET_UNUSED const optional<Tensor>& offset,
    Tensor& out) {
  const int64_t leading_dims = getLeadingDims(in, in.dim() - 1);
  const int64_t out_dim = weight.size(0); // = out_dim
  const int64_t in_dim = weight.size(1); // = in_dim
  const uint8_t* __restrict__ in_data = in.const_data_ptr<uint8_t>();
  const uint8_t* __restrict__ weight_data = weight.const_data_ptr<uint8_t>();
  const int32_t* __restrict__ bias_data = bias.const_data_ptr<int32_t>();
  uint8_t* __restrict__ out_data = out.mutable_data_ptr<uint8_t>();
  int32_t ret = xa_nn_matmul_asym8uxasym8u_asym8u(
      out_data,
      weight_data,
      in_data,
      bias_data,
      out_dim,
      in_dim,
      in_dim,
      leading_dims,
      in_dim,
      out_dim,
      1,
      -weight_zero_point.const_data_ptr<int32_t>()[0], // mat1_zero_bias
      -in_zero_point, // mat2_zero_bias
      out_multiplier.const_data_ptr<int32_t>()[0],
      out_shift.const_data_ptr<int32_t>()[0],
      out_zero_point);
  ET_DCHECK_MSG(ret == 0, "HiFi quantized::linear failed");
}

void inline _quantized_linear_asym8s(
    const Tensor& in,
    const Tensor& weight,
    const Tensor& bias,
    int64_t in_zero_point,
    const Tensor& weight_zero_point,
    const Tensor& out_multiplier,
    const Tensor& out_shift,
    int64_t out_zero_point,
    __ET_UNUSED const optional<Tensor>& offset,
    Tensor& out) {
  // input comes in shape [leading_dims, in_dim]
  // weight comes in shape [out_dim, in_dim]
  // output comes in empty with shape [leading_dims, out_dim]
  // Perform matrix multiply (M x N) x (N x P)' => M x P
  const int64_t leading_dims = getLeadingDims(in, in.dim() - 1);
  const int64_t out_dim = weight.size(0); // = out_dim
  const int64_t in_dim = weight.size(1); // = in_dim

  const int8_t* __restrict__ in_data = in.const_data_ptr<int8_t>();
  const int8_t* __restrict__ weight_data = weight.const_data_ptr<int8_t>();
  const int32_t* __restrict__ bias_data = bias.const_data_ptr<int32_t>();
  int8_t* __restrict__ out_data = out.mutable_data_ptr<int8_t>();

  // The nnlib kernel to compute quantized linear via matmul.
  int32_t ret = xa_nn_matmul_asym8sxasym8s_asym8s(
      out_data, // p_out
      weight_data, // p_mat1,
      in_data, // p_mat2,
      bias_data, // p_bias
      out_dim, // rows of p_mat1
      in_dim, // cols of p_mat1
      in_dim, // row_stride of p_mat1
      leading_dims, // vec_count, i.e., rows of p_mat2
      in_dim, // vec_offset of p_mat2.
      out_dim, // out_offset, i.e., offset of next output element written
      1, // out_stride, i.e., stride to go to next output row
      -weight_zero_point.const_data_ptr<int32_t>()[0], // mat1_zero_bias
      -in_zero_point, // mat2_zero_bias
      out_multiplier.const_data_ptr<int32_t>()[0], // out_multiplier
      out_shift.const_data_ptr<int32_t>()[0], // out_shift
      out_zero_point); // out_zero_bias
  ET_DCHECK_MSG(ret == 0, "HiFi quantized::linear failed");
}

void inline _quantized_linear_per_tensor_asym8u(
    const Tensor& in,
    const Tensor& weight,
    const Tensor& bias,
    int64_t in_zero_point,
    int64_t weight_zero_point,
    int64_t out_multiplier,
    int64_t out_shift,
    int64_t out_zero_point,
    __ET_UNUSED const optional<Tensor>& offset,
    Tensor& out) {
  // input comes in shape [leading_dims, in_dim]
  // weight comes in shape [out_dim, in_dim]
  // output comes in empty with shape [leading_dims, out_dim]
  // Perform matrix multiply (M x N) x (N x P)' => M x P
  const int64_t leading_dims = getLeadingDims(in, in.dim() - 1);
  const int64_t out_dim = weight.size(0); // = out_dim
  const int64_t in_dim = weight.size(1); // = in_dim

  const uint8_t* __restrict__ in_data = in.const_data_ptr<uint8_t>();
  const uint8_t* __restrict__ weight_data = weight.const_data_ptr<uint8_t>();
  const int32_t* __restrict__ bias_data = bias.const_data_ptr<int32_t>();
  uint8_t* __restrict__ out_data = out.mutable_data_ptr<uint8_t>();

  const int32_t out_multipler_int32 = static_cast<int32_t>(out_multiplier);
  const int32_t out_shift_int32 = static_cast<int32_t>(out_shift);

  // The nnlib kernel to compute quantized linear via matmul.
  const int32_t ret = xa_nn_matmul_asym8uxasym8u_asym8u(
      out_data, // p_out
      weight_data, // p_mat1,
      in_data, // p_mat2,
      bias_data, // p_bias
      out_dim, // rows of p_mat1
      in_dim, // cols of p_mat1
      in_dim, // row_stride of p_mat1
      leading_dims, // vec_count, i.e., rows of p_mat2
      in_dim, // vec_offset of p_mat2.
      out_dim, // out_offset, i.e., offset of next output element written
      1, // out_stride, i.e., stride to go to next output row
      -weight_zero_point, // mat1_zero_bias
      -in_zero_point, // mat2_zero_bias
      out_multipler_int32, // out_multiplier
      out_shift_int32, // out_shift
      out_zero_point); // out_zero_bias
  ET_DCHECK_MSG(ret == 0, "HiFi quantized::linear_per_tensor failed");
}

void inline _quantized_linear_per_tensor_asym8s(
    const Tensor& in,
    const Tensor& weight,
    const Tensor& bias,
    int64_t in_zero_point,
    int64_t weight_zero_point,
    int64_t out_multiplier,
    int64_t out_shift,
    int64_t out_zero_point,
    __ET_UNUSED const optional<Tensor>& offset,
    Tensor& out) {
  // input comes in shape [leading_dims, in_dim]
  // weight comes in shape [out_dim, in_dim]
  // output comes in empty with shape [leading_dims, out_dim]
  // Perform matrix multiply (M x N) x (N x P)' => M x P
  const int64_t leading_dims = getLeadingDims(in, in.dim() - 1);
  const int64_t out_dim = weight.size(0); // = out_dim
  const int64_t in_dim = weight.size(1); // = in_dim

  const int8_t* __restrict__ in_data = in.const_data_ptr<int8_t>();
  const int8_t* __restrict__ weight_data = weight.const_data_ptr<int8_t>();
  const int32_t* __restrict__ bias_data = bias.const_data_ptr<int32_t>();
  int8_t* __restrict__ out_data = out.mutable_data_ptr<int8_t>();

  const int32_t out_multipler_int32 = static_cast<int32_t>(out_multiplier);
  const int32_t out_shift_int32 = static_cast<int32_t>(out_shift);

  // The nnlib kernel to compute quantized linear via matmul.
  const int32_t ret = xa_nn_matmul_asym8sxasym8s_asym8s(
      out_data, // p_out
      weight_data, // p_mat1,
      in_data, // p_mat2,
      bias_data, // p_bias
      out_dim, // rows of p_mat1
      in_dim, // cols of p_mat1
      in_dim, // row_stride of p_mat1
      leading_dims, // vec_count, i.e., rows of p_mat2
      in_dim, // vec_offset of p_mat2.
      out_dim, // out_offset, i.e., offset of next output element written
      1, // out_stride, i.e., stride to go to next output row
      -weight_zero_point, // mat1_zero_bias
      -in_zero_point, // mat2_zero_bias
      out_multipler_int32, // out_multiplier
      out_shift_int32, // out_shift
      out_zero_point); // out_zero_bias
  ET_DCHECK_MSG(ret == 0, "HiFi quantized::linear_per_tensor failed");
}

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
    Tensor& out) {
  if (out.scalar_type() == ::executorch::aten::ScalarType::Short &&
      in.scalar_type() == ::executorch::aten::ScalarType::Short &&
      weight.scalar_type() == ::executorch::aten::ScalarType::Char) {
    ::impl::generic::native::quantized_linear_out(
        ctx,
        in,
        weight,
        bias,
        in_zero_point,
        weight_zero_point,
        out_multiplier,
        out_shift,
        out_zero_point,
        offset,
        out);
  } else if (out.scalar_type() == executorch::aten::ScalarType::Byte) {
    _quantized_linear_asym8u(
        in,
        weight,
        bias,
        in_zero_point,
        weight_zero_point,
        out_multiplier,
        out_shift,
        out_zero_point,
        offset,
        out);
  } else if (out.scalar_type() == executorch::aten::ScalarType::Char) {
    _quantized_linear_asym8s(
        in,
        weight,
        bias,
        in_zero_point,
        weight_zero_point,
        out_multiplier,
        out_shift,
        out_zero_point,
        offset,
        out);
  } else {
    ET_CHECK_MSG(
        false, "quantized linear only supported for uint8 and int8 dtypes");
  }
}

void quantized_linear_per_tensor_out(
    KernelRuntimeContext& ctx,
    const Tensor& in,
    const Tensor& weight,
    const Tensor& bias,
    const int64_t in_zero_point,
    const int64_t weight_zero_point,
    const int64_t out_multiplier,
    const int64_t out_shift,
    const int64_t out_zero_point,
    const optional<Tensor>& offset,
    Tensor& out) {
  if (out.scalar_type() == ::executorch::aten::ScalarType::Short &&
      in.scalar_type() == ::executorch::aten::ScalarType::Short &&
      weight.scalar_type() == ::executorch::aten::ScalarType::Char) {
    ::impl::generic::native::quantized_linear_per_tensor_out(
        ctx,
        in,
        weight,
        bias,
        in_zero_point,
        weight_zero_point,
        out_multiplier,
        out_shift,
        out_zero_point,
        offset,
        out);
  } else if (out.scalar_type() == executorch::aten::ScalarType::Byte) {
    _quantized_linear_per_tensor_asym8u(
        in,
        weight,
        bias,
        in_zero_point,
        weight_zero_point,
        out_multiplier,
        out_shift,
        out_zero_point,
        offset,
        out);
  } else if (out.scalar_type() == executorch::aten::ScalarType::Char) {
    _quantized_linear_per_tensor_asym8s(
        in,
        weight,
        bias,
        in_zero_point,
        weight_zero_point,
        out_multiplier,
        out_shift,
        out_zero_point,
        offset,
        out);
  } else {
    ET_CHECK_MSG(
        false, "quantized linear only supported for uint8 and int8 dtypes");
  }
}

} // namespace impl::HiFi::native
