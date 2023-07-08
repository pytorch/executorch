// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#ifdef __aarch64__
#include <arm_neon.h>
#include <sleef.h>
#endif

#include <cmath>
#include <type_traits>

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
  const IN_T* __restrict__ input_data_base = input.data_ptr<IN_T>();
  OUT_T* __restrict__ output_data_base = out.data_ptr<OUT_T>();

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

      for (auto d = 0; d < dim_size; ++d) {
        output_data[d * dim_stride] =
            input_data[d * dim_stride] - max_input - temp_sum;
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
      ET_CHECK_MSG(false, "Unhandled input dtype %hhd", input_scalar_type);
  }
}
} // namespace

void opt_log_soft_max_check_preconditions(
    const Tensor& self,
    int64_t dim,
    bool half_to_float,
    Tensor& out) {
  // Ensure half_to_float is not true
  ET_CHECK_MSG(
      !half_to_float,
      "softmax with half to float conversion is not supported on CPU");
  // Ensure self has value
  ET_CHECK_MSG(self.numel() > 0, "self.numel() %zd <= 0", self.numel());
  // Ensure dim is valid
  ET_CHECK_MSG(
      dim >= 0 && dim < self.dim(),
      "dim %" PRId64 " >= 0 && dim %" PRId64 " < self.dim() %zd",
      dim,
      dim,
      self.dim());
  // Ensure self and out have the same shape
  ET_CHECK_SAME_SHAPE2(self, out);
  // Ensure self and out are float
  auto out_scalar_type = out.scalar_type();
  ET_CHECK_MSG(
      out_scalar_type == ScalarType::Float,
      "out.scalar_type() %hhd is not Float",
      out_scalar_type);
  auto input_scalar_type = self.scalar_type();
  ET_CHECK_MSG(
      input_scalar_type == ScalarType::Float,
      "self.scalar_type() %hhd is not Float",
      input_scalar_type);
}

// _log_softmax.out(Tensor self, int dim, bool half_to_float, *, Tensor(a!) out)
// -> Tensor(a!)
Tensor& opt_log_softmax_out(
    RuntimeContext& context,
    const Tensor& self,
    int64_t dim,
    bool half_to_float,
    Tensor& out) {
  (void)context;
  dim = dim < 0 ? dim + self.dim() : dim;
  Tensor::SizesType expected_output_size[16];
  for (size_t i = 0; i < out.dim(); ++i) {
    expected_output_size[i] = self.size(i);
  }
  auto error = resize_tensor(
      out, {expected_output_size, static_cast<size_t>(out.dim())});
  // TODO: Construct error message with requested output sizes.
  ET_CHECK_MSG(error == Error::Ok, "Failed to resize output tensor.");
  opt_log_soft_max_check_preconditions(self, dim, half_to_float, out);
  auto out_scalar_type = out.scalar_type();
  switch (out_scalar_type) {
    // TODO: support Double as well
    case ScalarType::Float:
      log_softmax_wrapper<float>(self, dim, out);
      break;
    default:
      ET_CHECK_MSG(false, "Unhandled out dtype %hhd", out_scalar_type);
  }
  return out;
}

} // namespace native
} // namespace executor
} // namespace torch
