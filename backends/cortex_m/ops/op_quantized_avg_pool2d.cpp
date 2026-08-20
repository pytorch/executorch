/*
 * Copyright 2025-2026 Arm Limited and/or its affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "cortex_m_ops_common.h"

namespace cortex_m {
namespace native {

using KernelRuntimeContext = torch::executor::KernelRuntimeContext;

namespace {

int32_t pooling_output_size(
    int32_t input,
    int32_t kernel,
    int32_t stride,
    int32_t padding,
    bool ceil_mode) {
  const int32_t numerator = input + 2 * padding - kernel;
  int32_t output = numerator / stride + 1;
  if (ceil_mode && numerator % stride != 0) {
    output += 1;
  }
  if (ceil_mode && (output - 1) * stride >= input + padding) {
    output -= 1;
  }
  return output;
}

bool validate_avg_pool2d_output_size(
    KernelRuntimeContext& context,
    const CmsisPool2DConfig& pool_config,
    bool ceil_mode) {
  const int32_t expected_h = pooling_output_size(
      pool_config.input_dims.h,
      pool_config.filter_dims.h,
      pool_config.pool_params.stride.h,
      pool_config.pool_params.padding.h,
      ceil_mode);
  const int32_t expected_w = pooling_output_size(
      pool_config.input_dims.w,
      pool_config.filter_dims.w,
      pool_config.pool_params.stride.w,
      pool_config.pool_params.padding.w,
      ceil_mode);

  if (pool_config.output_dims.h != expected_h ||
      pool_config.output_dims.w != expected_w) {
    ET_LOG(
        Error,
        "quantized_avg_pool2d_out: output shape mismatch - actual: (%d, %d) expected: (%d, %d)",
        pool_config.output_dims.h,
        pool_config.output_dims.w,
        expected_h,
        expected_w);
    context.fail(Error::InvalidArgument);
    return false;
  }
  return true;
}

} // namespace

// cppcheck-suppress unusedFunction
Tensor& quantized_avg_pool2d_out(
    KernelRuntimeContext& context,
    const Tensor& input,
    const Int64ArrayRef kernel_size,
    const Int64ArrayRef stride,
    const Int64ArrayRef padding,
    const bool ceil_mode,
    const int64_t zero_point,
    const int64_t multiplier,
    const int64_t shift,
    const Tensor& scratch,
    Tensor& out) {
  constexpr int32_t activation_min = std::numeric_limits<int8_t>::min();
  constexpr int32_t activation_max = std::numeric_limits<int8_t>::max();

  const int64_t dilation_values[2] = {1, 1};
  const Int64ArrayRef dilation(dilation_values, 2);
  CmsisPool2DConfig pool_config;
  if (!prepare_cmsis_pool2d_config(
          context,
          "quantized_avg_pool2d_out",
          input,
          out,
          kernel_size,
          stride,
          padding,
          dilation,
          ceil_mode,
          activation_min,
          activation_max,
          pool_config,
          true,
          true)) {
    return out;
  }

  if (!validate_avg_pool2d_output_size(context, pool_config, ceil_mode)) {
    return out;
  }

  cmsis_nn_context cmsis_ctx;
  cmsis_ctx.buf = nullptr;
  cmsis_ctx.size = scratch.nbytes();
  if (cmsis_ctx.size > 0) {
    cmsis_ctx.buf = scratch.mutable_data_ptr<int8_t>();
  }

#ifdef CORTEX_M_ENABLE_RUNTIME_CHECKS
  const int32_t runtime_buffer_bytes = arm_avgpool_s8_get_buffer_size(
      pool_config.output_dims.w, pool_config.input_dims.c);
  if (scratch.nbytes() != static_cast<size_t>(runtime_buffer_bytes)) {
    ET_LOG(
        Error,
        "quantized_avg_pool2d_out: scratch buffer size incorrect - actual: (%d) needed: (%d)",
        static_cast<int>(scratch.nbytes()),
        static_cast<int>(runtime_buffer_bytes));
    context.fail(Error::Internal);
    return out;
  }
#endif

  const int8_t* input_data = input.const_data_ptr<int8_t>();
  int8_t* output_data = out.mutable_data_ptr<int8_t>();

  const arm_cmsis_nn_status status = arm_avgpool_s8(
      &cmsis_ctx,
      &pool_config.pool_params,
      &pool_config.input_dims,
      input_data,
      &pool_config.filter_dims,
      &pool_config.output_dims,
      output_data);
  if (status != ARM_CMSIS_NN_SUCCESS) {
    ET_LOG(
        Error,
        "quantized_avg_pool2d_out: arm_avgpool_s8 failed with status [%d]",
        status);
    context.fail(Error::Internal);
  }

  (void)zero_point;
  (void)multiplier;
  (void)shift;

  return out;
}

} // namespace native
} // namespace cortex_m
