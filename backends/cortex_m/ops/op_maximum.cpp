/*
 * Copyright 2025 Arm Limited and/or its affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "cortex_m_ops_common.h"

// Include CMSIS-NN headers with C linkage
extern "C" {
#include "arm_nnfunctions.h"
}

namespace cortex_m {
namespace native {

using KernelRuntimeContext = torch::executor::KernelRuntimeContext;

Tensor& maximum_out(
    KernelRuntimeContext& context,
    const Tensor& input1,
    const Tensor& input2,
    Tensor& out) {
  validate_cmsis_nn_tensor_requirements(
      input1,
      input2,
      out,
      ScalarType::Char,
      /*require_channels_last=*/false,
      /*require_same_sizes=*/false);

  auto resize_error = resize_to_broadcast_target_size(input1, input2, out);
  if (resize_error != Error::Ok) {
    ET_LOG(Error, "maximum_out: broadcast shape mismatch between inputs");
    context.fail(resize_error);
    return out;
  }

  const int8_t* input1_data = input1.const_data_ptr<int8_t>();
  const int8_t* input2_data = input2.const_data_ptr<int8_t>();
  int8_t* output_data = out.mutable_data_ptr<int8_t>();

  // Create CMSIS-NN dims directly from tensor sizes
  const auto input1_rank = input1.dim();
  const auto input1_sizes = input1.sizes();
  const cmsis_nn_dims input1_dims{
      static_cast<int32_t>(
          input1_rank >= 4 ? input1_sizes[input1_rank - 4] : 1),
      static_cast<int32_t>(
          input1_rank >= 3 ? input1_sizes[input1_rank - 3] : 1),
      static_cast<int32_t>(
          input1_rank >= 2 ? input1_sizes[input1_rank - 2] : 1),
      static_cast<int32_t>(
          input1_rank >= 1 ? input1_sizes[input1_rank - 1] : 1)};

  const auto input2_rank = input2.dim();
  const auto input2_sizes = input2.sizes();
  const cmsis_nn_dims input2_dims{
      static_cast<int32_t>(
          input2_rank >= 4 ? input2_sizes[input2_rank - 4] : 1),
      static_cast<int32_t>(
          input2_rank >= 3 ? input2_sizes[input2_rank - 3] : 1),
      static_cast<int32_t>(
          input2_rank >= 2 ? input2_sizes[input2_rank - 2] : 1),
      static_cast<int32_t>(
          input2_rank >= 1 ? input2_sizes[input2_rank - 1] : 1)};

  const auto output_rank = out.dim();
  const auto output_sizes = out.sizes();
  const cmsis_nn_dims output_dims{
      static_cast<int32_t>(
          output_rank >= 4 ? output_sizes[output_rank - 4] : 1),
      static_cast<int32_t>(
          output_rank >= 3 ? output_sizes[output_rank - 3] : 1),
      static_cast<int32_t>(
          output_rank >= 2 ? output_sizes[output_rank - 2] : 1),
      static_cast<int32_t>(
          output_rank >= 1 ? output_sizes[output_rank - 1] : 1)};

  const arm_cmsis_nn_status status = arm_maximum_s8(
      /* ctx */ nullptr,
      input1_data,
      &input1_dims,
      input2_data,
      &input2_dims,
      output_data,
      &output_dims);

  if (status != ARM_CMSIS_NN_SUCCESS) {
    ET_LOG(
        Error,
        "maximum_out: arm_maximum_s8 failed with status [%d]",
        static_cast<int>(status));
    context.fail(Error::Internal);
  }

  return out;
}

} // namespace native
} // namespace cortex_m
