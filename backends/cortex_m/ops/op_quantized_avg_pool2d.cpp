/*
 * Copyright 2025-2026 Arm Limited and/or its affiliates.
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

Tensor& quantized_avg_pool2d_out(
    KernelRuntimeContext& context,
    const Tensor& input,
    const Int64ArrayRef kernel_size,
    const Int64ArrayRef stride,
    const Int64ArrayRef padding,
    const int64_t zero_point,
    const int64_t multiplier,
    const int64_t shift,
    Tensor& out) {
  constexpr int32_t activation_min = std::numeric_limits<int8_t>::min();
  constexpr int32_t activation_max = std::numeric_limits<int8_t>::max();

  const int64_t dilation_values[2] = {1, 1};
  const Int64ArrayRef dilation(dilation_values, 2);
  const bool ceil_mode = false;

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
          pool_config)) {
    return out;
  }

  cmsis_nn_context cmsis_ctx;
  cmsis_ctx.buf = nullptr;
  cmsis_ctx.size = 0;

  const int8_t* input_data = input.const_data_ptr<int8_t>();
  int8_t* output_data = out.mutable_data_ptr<int8_t>();

  arm_cmsis_nn_status status = arm_avgpool_s8(
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
