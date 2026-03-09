/*
 * Copyright 2025 Arm Limited and/or its affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "cortex_m_ops_common.h"

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

  for (int32_t n = 0; n < output_dims.n; ++n) {
    for (int32_t h = 0; h < output_dims.h; ++h) {
      for (int32_t w = 0; w < output_dims.w; ++w) {
        for (int32_t c = 0; c < output_dims.c; ++c) {
          const int32_t n1 = (input1_dims.n == 1) ? 0 : n;
          const int32_t h1 = (input1_dims.h == 1) ? 0 : h;
          const int32_t w1 = (input1_dims.w == 1) ? 0 : w;
          const int32_t c1 = (input1_dims.c == 1) ? 0 : c;
          const int32_t n2 = (input2_dims.n == 1) ? 0 : n;
          const int32_t h2 = (input2_dims.h == 1) ? 0 : h;
          const int32_t w2 = (input2_dims.w == 1) ? 0 : w;
          const int32_t c2 = (input2_dims.c == 1) ? 0 : c;
          const int32_t idx1 =
              ((n1 * input1_dims.h + h1) * input1_dims.w + w1) * input1_dims.c +
              c1;
          const int32_t idx2 =
              ((n2 * input2_dims.h + h2) * input2_dims.w + w2) * input2_dims.c +
              c2;
          const int32_t out_idx =
              ((n * output_dims.h + h) * output_dims.w + w) * output_dims.c + c;
          output_data[out_idx] = input1_data[idx1] > input2_data[idx2]
              ? input1_data[idx1]
              : input2_data[idx2];
        }
      }
    }
  }

  return out;
}

} // namespace native
} // namespace cortex_m
