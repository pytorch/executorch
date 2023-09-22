/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/runtime/kernel/kernel_includes.h>
#include <cmath>
#include <tuple>

#include <executorch/kernels/optimized/cpu/moments_utils.h>
#include <executorch/kernels/optimized/vec/functional.h>
#include <executorch/kernels/optimized/vec/vec.h>

namespace torch {
namespace executor {
namespace native {

using Tensor = exec_aten::Tensor;

namespace {

template <typename CTYPE>
void layer_norm(
    const Tensor& input,
    const Tensor& gamma,
    const Tensor& beta,
    const size_t M,
    const size_t N,
    CTYPE eps,
    Tensor& output,
    Tensor& mean,
    Tensor& rstd) {
  using Vec = executorch::vec::Vectorized<CTYPE>;

  const CTYPE* __restrict__ input_data = input.data_ptr<CTYPE>();
  const CTYPE* __restrict__ gamma_data = gamma.data_ptr<CTYPE>();
  const CTYPE* __restrict__ beta_data = beta.data_ptr<CTYPE>();
  CTYPE* __restrict__ output_data = output.data_ptr<CTYPE>();

  const bool gamma_null = gamma_data == nullptr;
  const bool beta_null = beta_data == nullptr;

  for (size_t i = 0; i < M; ++i) {
    const CTYPE* src_ptr = input_data + i * N;
    CTYPE* dst_ptr = output_data + i * N;

    CTYPE mean_val;
    CTYPE rstd_val;
    std::tie(mean_val, rstd_val) = RowwiseMoments(src_ptr, N);
    rstd_val = CTYPE(1) / std::sqrt(rstd_val + eps);

    const CTYPE scale = rstd_val;
    const CTYPE bias = -rstd_val * mean_val;

    if (gamma_null || beta_null) {
      for (size_t j = 0; j < N; ++j) {
        const CTYPE gamma_v = gamma_null ? CTYPE(1) : gamma_data[j];
        const CTYPE beta_v = beta_null ? CTYPE(0) : beta_data[j];
        dst_ptr[j] = (src_ptr[j] * scale + bias) * gamma_v + beta_v;
      }
    } else {
      executorch::vec::map3<CTYPE>(
          [scale, bias](Vec x, Vec gamma, Vec beta) {
            return (x * Vec(scale) + Vec(bias)) * gamma + beta;
          },
          dst_ptr,
          src_ptr,
          gamma_data,
          beta_data,
          N);
    }
  }

  // Assign NAN to mean and rstd. They are not used in seen examples.
  // Use NAN to make the error more obvious in case they are used.
  mean.data_ptr<CTYPE>()[0] = NAN;
  rstd.data_ptr<CTYPE>()[0] = NAN;
}

} // namespace

// native_layer_norm.out(Tensor input, int[] normalized_shape, Tensor? weight,
// Tensor? bias, float eps, *, Tensor(a!) out, Tensor(b!) mean_out, Tensor(c!)
// rstd_out) -> (Tensor(a!), Tensor(b!), Tensor(c!))
//
// Unlike the ATen implementation of native_layer_norm, mean_out and rstd_out
// are not filled with any meaningful data. Instead, they are set to NAN to
// easily detect if they are being used.
std::tuple<Tensor&, Tensor&, Tensor&> opt_native_layer_norm_out(
    RuntimeContext& context,
    const Tensor& input,
    IntArrayRef normalized_shape,
    const exec_aten::optional<Tensor>& gamma,
    const exec_aten::optional<Tensor>& beta,
    double eps,
    Tensor& out,
    Tensor& mean_out,
    Tensor& rstd_out) {
  (void)context;

  ET_CHECK_MSG(
      normalized_shape.size() == 1,
      "normalize_shape.size() must be 1 but saw %zd",
      normalized_shape.size());
  ET_CHECK_MSG(
      input.scalar_type() == out.scalar_type(),
      "out and input must have the same type.");
  ET_CHECK_MSG(
      input.dim() == out.dim(),
      "out and input must have the same number of dimensions");
  ET_CHECK_MSG(
      input.scalar_type() == mean_out.scalar_type(),
      "mean_out and input must have the same type.");
  ET_CHECK_MSG(
      input.scalar_type() == rstd_out.scalar_type(),
      "rstd_out and input must have the same type.");

  if (input.sizes() == out.sizes()) {
    ET_CHECK_MSG(
        normalized_shape[0] == input.sizes()[input.dim() - 1],
        "Normalized shape value must match the size of input.");
  } else {
    // If we need to resize out to support dynamic input shapes, we can't count
    // on normalized_shape matching the shape of the input or output. But we
    // don't need to modify normalized_shape because it's not used in this
    // function besides some checks
    torch::executor::Error err = resize_tensor(out, input.sizes());
    ET_CHECK_MSG(
        err == torch::executor::Error::Ok,
        "Failed to resize out Tensor in opt_native_layer_norm_out");
  }

  const size_t input_ndim = input.dim();
  const size_t normalized_ndim = normalized_shape.size();

  const size_t axis = input_ndim - normalized_ndim;

  const size_t M = getLeadingDims(input, axis);
  const size_t N = getTrailingDims(input, axis - 1);

// helper for generating the cases for different data types
#define LAYER_NORM(ctype, dtype) \
  case ScalarType::dtype:        \
    layer_norm<ctype>(           \
        input,                   \
        gamma.value(),           \
        beta.value(),            \
        M,                       \
        N,                       \
        eps,                     \
        out,                     \
        mean_out,                \
        rstd_out);               \
    break;

  switch (input.scalar_type()) {
    // TODO support bfloat16
    ET_FORALL_FLOAT_TYPES(LAYER_NORM)
    default:
      ET_CHECK_MSG(
          false,
          "Unhandled dtype %hhd",
          static_cast<int8_t>(input.scalar_type()));
  }
#undef LAYER_NORM
  return {out, mean_out, rstd_out};
}

} // namespace native
} // namespace executor
} // namespace torch
