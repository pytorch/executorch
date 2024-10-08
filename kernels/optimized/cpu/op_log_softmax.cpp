/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#ifdef __aarch64__
#include <arm_neon.h>
#include <sleef.h>
#endif

#include <cmath>
#include <type_traits>

#include <executorch/kernels/portable/cpu/util/activation_ops_util.h>
#include <executorch/runtime/kernel/kernel_includes.h>

// `_log_softmax_out` Applies the Log_Softmax function to an n-dimensional input
// Tensor rescaling them so that the elements of the n-dimensional output
// Tensor.

namespace torch {
namespace executor {
namespace native {

using Tensor = exec_aten::Tensor;
namespace {

template <typename IN_T, typename OUT_T>
void log_softmax_kernel(const Tensor& input, int64_t dim, Tensor& out) {
  const IN_T* __restrict__ input_data_base = input.const_data_ptr<IN_T>();
  OUT_T* __restrict__ output_data_base = out.mutable_data_ptr<OUT_T>();

  if (input.dim() == 0) {
    output_data_base[0] = 0;
    return;
  }

  int64_t dim_size = input.size(dim);

  int64_t outer_size = 1;
  int64_t inner_size = 1;
  for (int64_t i = 0; i < dim; ++i) {
    outer_size *= input.size(i);
  }
  for (int64_t i = dim + 1; i < input.dim(); ++i) {
    inner_size *= input.size(i);
  }

  int64_t dim_stride = inner_size;
  int64_t outer_stride = dim_size * dim_stride;

  for (size_t outer_idx = 0; outer_idx < outer_size; ++outer_idx) {
    for (size_t inner_idx = 0; inner_idx < inner_size; ++inner_idx) {
      const IN_T* input_data =
          input_data_base + outer_idx * outer_stride + inner_idx;
      OUT_T* output_data =
          output_data_base + outer_idx * outer_stride + inner_idx;

      // calculate max in softmax dim
      IN_T max_input = input_data[0];
      for (auto d = 0; d < dim_size; ++d) {
        max_input = std::max(max_input, input_data[d * dim_stride]);
      }
      // calculate sum and exponential in softmax dim
      OUT_T temp_sum = 0;
#ifndef __aarch64__
      for (auto d = 0; d < dim_size; ++d) {
        output_data[d * dim_stride] =
            std::exp(input_data[d * dim_stride] - max_input);
        temp_sum += output_data[d * dim_stride];
      }
#else
      auto d = 0;
      for (; d + 4 < dim_size; d += 4) {
        auto index = d * dim_stride;
        float32x4_t in =
            vld1q_f32(static_cast<const float*>(&input_data[index]));
        float32x4_t out_ =
            Sleef_expf4_u10(vsubq_f32(in, vmovq_n_f32(max_input)));
        vst1q_f32(static_cast<float*>(&output_data[index]), out_);
        temp_sum += vaddvq_f32(out_);
      }

      for (; d < dim_size; ++d) {
        output_data[d * dim_stride] =
            std::exp(input_data[d * dim_stride] - max_input);
        temp_sum += output_data[d * dim_stride];
      }
#endif // __aarch64__

      temp_sum = std::log(temp_sum);

      for (auto dd = 0; dd < dim_size; ++dd) {
        output_data[dd * dim_stride] =
            input_data[dd * dim_stride] - max_input - temp_sum;
      }
    }
  }
}

// OUT_T is the corresponding C++ type for out.scalar_type(). Only takes float
// or double.
template <
    typename OUT_T,
    std::enable_if_t<std::is_floating_point<OUT_T>::value, bool> = true>
void log_softmax_wrapper(const Tensor& X, int64_t dim, Tensor& out) {
  auto input_scalar_type = X.scalar_type();
  switch (input_scalar_type) {
    // TODO: support Double as well
    case ScalarType::Float:
      log_softmax_kernel<float, OUT_T>(X, dim, out);
      break;
    default:
      ET_CHECK_MSG(
          false,
          "Unhandled input dtype %" PRId8,
          static_cast<int8_t>(input_scalar_type));
  }
}
} // namespace

// _log_softmax.out(Tensor self, int dim, bool half_to_float, *, Tensor(a!) out)
// -> Tensor(a!)
Tensor& opt_log_softmax_out(
    KernelRuntimeContext& context,
    const Tensor& self,
    int64_t dim,
    bool half_to_float,
    Tensor& out) {
  (void)context;

  ET_KERNEL_CHECK(
      context,
      check_log_softmax_args(self, dim, half_to_float, out),
      InvalidArgument,
      out);

  ET_KERNEL_CHECK(
      context,
      resize_tensor(out, self.sizes()) == Error::Ok,
      InvalidArgument,
      out);

  dim = dim < 0 ? dim + nonzero_dim(self) : dim;

  auto out_scalar_type = out.scalar_type();
  switch (out_scalar_type) {
    // TODO: support Double as well
    case ScalarType::Float:
      log_softmax_wrapper<float>(self, dim, out);
      break;
    default:
      ET_CHECK_MSG(
          false,
          "Unhandled out dtype %" PRId8,
          static_cast<int8_t>(out_scalar_type));
  }
  return out;
}

} // namespace native
} // namespace executor
} // namespace torch
