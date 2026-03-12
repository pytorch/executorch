/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "cortex_m_ops_common.h"

extern "C" {
#include "arm_nnfunctions.h"
}

namespace cortex_m {
namespace native {

using KernelRuntimeContext = torch::executor::KernelRuntimeContext;

namespace {

bool validate_batch_matmul_arguments(
    KernelRuntimeContext& context,
    const Tensor& lhs,
    const Tensor& rhs_transposed,
    const Tensor& out) {
  if (lhs.scalar_type() != ScalarType::Char ||
      rhs_transposed.scalar_type() != ScalarType::Char ||
      out.scalar_type() != ScalarType::Char) {
    ET_LOG(Error, "quantized_batch_matmul: all tensors must be int8");
    context.fail(Error::InvalidArgument);
    return false;
  }

  if (lhs.dim() != 3 || rhs_transposed.dim() != 3 || out.dim() != 3) {
    ET_LOG(Error, "quantized_batch_matmul: all tensors must be 3-D");
    context.fail(Error::InvalidArgument);
    return false;
  }

  if (lhs.size(0) != rhs_transposed.size(0)) {
    ET_LOG(Error, "quantized_batch_matmul: batch dims must match");
    context.fail(Error::InvalidArgument);
    return false;
  }

  if (lhs.size(2) != rhs_transposed.size(2)) {
    ET_LOG(Error, "quantized_batch_matmul: inner dims must match");
    context.fail(Error::InvalidArgument);
    return false;
  }

  if (out.size(0) != lhs.size(0) || out.size(1) != lhs.size(1) ||
      out.size(2) != rhs_transposed.size(1)) {
    ET_LOG(Error, "quantized_batch_matmul: output shape mismatch");
    context.fail(Error::InvalidArgument);
    return false;
  }

  return true;
}

} // namespace

Tensor& quantized_batch_matmul_out(
    KernelRuntimeContext& context,
    const Tensor& lhs,
    int64_t lhs_offset,
    const Tensor& rhs_transposed,
    int64_t rhs_offset,
    int64_t output_offset,
    int64_t output_multiplier,
    int64_t output_shift,
    Tensor& out) {
  if (!validate_batch_matmul_arguments(context, lhs, rhs_transposed, out)) {
    return out;
  }

  const int32_t batch = static_cast<int32_t>(lhs.size(0));
  const int32_t lhs_rows = static_cast<int32_t>(lhs.size(1));
  const int32_t inner = static_cast<int32_t>(lhs.size(2));
  const int32_t rhs_cols = static_cast<int32_t>(rhs_transposed.size(1));

  const cmsis_nn_dims lhs_dims = {1, batch, lhs_rows, inner};
  const cmsis_nn_dims rhs_dims = {1, batch, rhs_cols, inner};
  const cmsis_nn_dims out_dims = {1, batch, lhs_rows, rhs_cols};

  const cmsis_nn_bmm_params bmm_params = {
      /* adj_x */ false,
      /* adj_y */ false,
      /* fc_params */
      {static_cast<int32_t>(lhs_offset),
       static_cast<int32_t>(rhs_offset),
       static_cast<int32_t>(output_offset),
       /* activation */
       {std::numeric_limits<int8_t>::min(),
        std::numeric_limits<int8_t>::max()}}};

  cmsis_nn_per_tensor_quant_params quant_params;
  quant_params.multiplier = static_cast<int32_t>(output_multiplier);
  quant_params.shift = static_cast<int32_t>(output_shift);

  const int32_t buf_size = arm_fully_connected_s8_get_buffer_size(&out_dims);

  cmsis_nn_context ctx;
  ctx.buf = nullptr;
  ctx.size = 0;

  if (buf_size > 0) {
    auto buffer_or_error = context.allocate_temp(buf_size);
    if (!buffer_or_error.ok()) {
      ET_LOG(
          Error,
          "quantized_batch_matmul: failed to allocate scratch buffer (%d bytes)",
          buf_size);
      context.fail(buffer_or_error.error());
      return out;
    }
    ctx.buf = buffer_or_error.get();
    ctx.size = buf_size;
  }

  const arm_cmsis_nn_status status = arm_batch_matmul_s8(
      &ctx,
      &bmm_params,
      &quant_params,
      &lhs_dims,
      lhs.const_data_ptr<int8_t>(),
      &rhs_dims,
      rhs_transposed.const_data_ptr<int8_t>(),
      &out_dims,
      out.mutable_data_ptr<int8_t>());

  if (status != ARM_CMSIS_NN_SUCCESS) {
    ET_LOG(
        Error,
        "quantized_batch_matmul: arm_batch_matmul_s8 failed with status [%d]",
        status);
    context.fail(Error::Internal);
  }

  return out;
}

} // namespace native
} // namespace cortex_m
