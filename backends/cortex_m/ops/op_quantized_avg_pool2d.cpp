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
  if (input.dim() != 4 || out.dim() != 4) {
    ET_LOG(Error, "quantized_avg_pool2d_out: tensors must be 4-D");
    context.fail(Error::InvalidArgument);
    return out;
  }
  int32_t batch = static_cast<int32_t>(input.size(0));
  int32_t channels = static_cast<int32_t>(input.size(1));
  int32_t input_h = static_cast<int32_t>(input.size(2));
  int32_t input_w = static_cast<int32_t>(input.size(3));
  int32_t kernel_h = static_cast<int32_t>(kernel_size[0]);
  int32_t kernel_w = static_cast<int32_t>(kernel_size[1]);
  int32_t stride_h = static_cast<int32_t>(stride[0]);
  int32_t stride_w = static_cast<int32_t>(stride[1]);
  int32_t pad_h = static_cast<int32_t>(padding[0]);
  int32_t pad_w = static_cast<int32_t>(padding[1]);
  int32_t output_h = static_cast<int32_t>(out.size(2));
  int32_t output_w = static_cast<int32_t>(out.size(3));
  const int32_t activation_min = std::numeric_limits<int8_t>::min();
  const int32_t activation_max = std::numeric_limits<int8_t>::max();

  const int8_t* input_data = input.const_data_ptr<int8_t>();
  int8_t* output_data = out.mutable_data_ptr<int8_t>();

  cmsis_nn_context cmsis_ctx;
  cmsis_ctx.buf = nullptr;
  cmsis_ctx.size = 0;
  cmsis_nn_pool_params pool_params;
  pool_params.stride.h = stride_h;
  pool_params.stride.w = stride_w;
  pool_params.padding.h = pad_h;
  pool_params.padding.w = pad_w;
  pool_params.activation.min = activation_min;
  pool_params.activation.max = activation_max;

  cmsis_nn_dims input_dims{batch, input_h, input_w, channels};
  cmsis_nn_dims filter_dims{1, kernel_h, kernel_w, 1};
  cmsis_nn_dims output_dims{batch, output_h, output_w, channels};

  arm_cmsis_nn_status status = arm_avgpool_s8(
      &cmsis_ctx,
      &pool_params,
      &input_dims,
      input_data,
      &filter_dims,
      &output_dims,
      output_data);
  if (status != ARM_CMSIS_NN_SUCCESS) {
    ET_LOG(
        Error,
        "quantized_avg_pool2d_out: arm_avgpool_s8 failed with status [%d]",
        status);
    context.fail(Error::Internal);
  }
  return out;
}

} // namespace native
} // namespace cortex_m