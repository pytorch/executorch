/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/backends/cadence/hifi/kernels/kernels.h>
#include <executorch/runtime/kernel/kernel_includes.h>
#include <algorithm>
#include <cmath>

namespace cadence {
namespace impl {
namespace HiFi {
namespace native {

using executorch::aten::Tensor;
using executorch::runtime::getLeadingDims;
using executorch::runtime::KernelRuntimeContext;

void quantized_linear_out(
    __ET_UNUSED KernelRuntimeContext& ctx,
    const Tensor& src,
    const Tensor& weight,
    const Tensor& bias,
    int64_t src_zero_point,
    const Tensor& weight_zero_point_t,
    const Tensor& out_multiplier,
    const Tensor& out_shift,
    int64_t out_zero_point,
    __ET_UNUSED const executorch::aten::optional<Tensor>& offset,
    Tensor& out) {
  int64_t leading_dims = getLeadingDims(src, src.dim() - 1);
  int64_t out_dim = weight.size(0);
  int64_t in_dim = weight.size(1);

  if (out.scalar_type() == executorch::aten::ScalarType::Byte) {
    const uint8_t* __restrict__ in_data = src.const_data_ptr<uint8_t>();
    const uint8_t* __restrict__ weight_data = weight.const_data_ptr<uint8_t>();
    const int32_t* __restrict__ bias_data = bias.const_data_ptr<int32_t>();
    uint8_t* __restrict__ out_data = out.mutable_data_ptr<uint8_t>();

    // The nnlib kernel to compute quantized linear via matmul.
    xa_nn_matmul_asym8uxasym8u_asym8u(
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
        -weight_zero_point_t.const_data_ptr<int32_t>()[0],
        -src_zero_point,
        out_multiplier.const_data_ptr<int32_t>()[0],
        out_shift.const_data_ptr<int32_t>()[0],
        out_zero_point);
  } else if (out.scalar_type() == executorch::aten::ScalarType::Char) {
    const int8_t* __restrict__ in_data = src.const_data_ptr<int8_t>();
    const int8_t* __restrict__ weight_data = weight.const_data_ptr<int8_t>();
    const int32_t* __restrict__ bias_data = bias.const_data_ptr<int32_t>();
    int8_t* __restrict__ out_data = out.mutable_data_ptr<int8_t>();

    xa_nn_matmul_asym8sxasym8s_asym8s(
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
        -weight_zero_point_t.const_data_ptr<int32_t>()[0],
        -src_zero_point,
        out_multiplier.const_data_ptr<int32_t>()[0],
        out_shift.const_data_ptr<int32_t>()[0],
        out_zero_point);
  } else {
    ET_CHECK_MSG(
        false,
        "Unhandled input dtype %hhd",
        static_cast<int8_t>(src.scalar_type()));
  }
}

}; // namespace native
}; // namespace HiFi
}; // namespace impl
}; // namespace cadence
