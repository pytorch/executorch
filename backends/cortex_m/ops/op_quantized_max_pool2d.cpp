/*
 * Copyright 2026 Arm Limited and/or its affiliates.
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

Tensor& quantized_max_pool2d_out(
    KernelRuntimeContext& context,
    const Tensor& input,
    const Int64ArrayRef kernel_size,
    const Int64ArrayRef stride,
    const Int64ArrayRef padding,
    const Int64ArrayRef dilation,
    const bool ceil_mode,
    const int64_t input_zero_point,
    const int64_t output_zero_point,
    const int64_t activation_min,
    const int64_t activation_max,
    Tensor& out) {
  CmsisPool2DConfig pool_config;
  if (!prepare_cmsis_pool2d_config(
          context,
          "quantized_max_pool2d_out",
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

  auto validate_int8_zero_point = [&](int64_t zp, const char* name) -> bool {
    if (zp < std::numeric_limits<int8_t>::min() ||
        zp > std::numeric_limits<int8_t>::max()) {
      ET_LOG(
          Error,
          "quantized_max_pool2d_out: %s must be int8, got %ld",
          name,
          zp);
      context.fail(Error::InvalidArgument);
      return false;
    }
    return true;
  };

  if (!validate_int8_zero_point(input_zero_point, "input zero point") ||
      !validate_int8_zero_point(output_zero_point, "output zero point")) {
    return out;
  }

  if (input_zero_point != output_zero_point) {
    ET_LOG(
        Error,
        "quantized_max_pool2d_out: input and output zero points must match");
    context.fail(Error::InvalidArgument);
    return out;
  }

  cmsis_nn_context cmsis_context;
  cmsis_context.buf = nullptr;
  cmsis_context.size = 0;

  const int8_t* input_data = input.const_data_ptr<int8_t>();
  int8_t* output_data = out.mutable_data_ptr<int8_t>();

  const arm_cmsis_nn_status status = arm_max_pool_s8(
      &cmsis_context,
      &pool_config.pool_params,
      &pool_config.input_dims,
      input_data,
      &pool_config.filter_dims,
      &pool_config.output_dims,
      output_data);

  if (status != ARM_CMSIS_NN_SUCCESS) {
    ET_LOG(
        Error,
        "quantized_max_pool2d_out: arm_max_pool_s8 failed with status %d",
        status);
    context.fail(Error::Internal);
  }

  return out;
}

} // namespace native
} // namespace cortex_m
