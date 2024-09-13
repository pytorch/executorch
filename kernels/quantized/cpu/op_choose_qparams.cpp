/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/kernels/portable/cpu/vec_ops.h>
#include <executorch/runtime/kernel/kernel_includes.h>
#include <algorithm>
#include <cinttypes>
#include <cmath>
#include <tuple>
/**
 * For an input tensor, use the scale and zero_point arguments to quantize it.
 */
namespace torch {
namespace executor {
namespace native {

using Tensor = exec_aten::Tensor;
using Scalar = exec_aten::Scalar;
using ScalarType = exec_aten::ScalarType;

namespace {

constexpr float SMALL_SCALE_THRESHOLD = 6.1e-5f;

/**
 * Asserts that the parameters are valid.
 */
void check_quantize_per_tensor_args(
    const Tensor& input,
    int64_t qmin,
    int64_t qmax,
    ScalarType dtype,
    Tensor& scale_out,
    Tensor& zero_point_out) {
  (void)dtype;
  ET_CHECK_MSG(
      qmin < qmax,
      "qmin should be less than qmax, but received min: %" PRId64
      ", max %" PRId64,
      qmin,
      qmax);
  ET_CHECK_MSG(
      input.scalar_type() == ScalarType::Float,
      "Expected input to be Float tensor received: %" PRId8,
      static_cast<int8_t>(input.scalar_type()));
  ET_CHECK_MSG(
      scale_out.scalar_type() == ScalarType::Double,
      "Expected scale to be Double tensor received: %" PRId8,
      static_cast<int8_t>(scale_out.scalar_type()));
  ET_CHECK_MSG(
      zero_point_out.scalar_type() == ScalarType::Long,
      "Expected scale to be Long tensor received: %" PRId8,
      static_cast<int8_t>(zero_point_out.scalar_type()));
  ET_CHECK_MSG(
      scale_out.numel() == 1,
      "Exepcted scale to only have one element received: %zd",
      ssize_t(scale_out.numel()));
  ET_CHECK_MSG(
      zero_point_out.numel() == 1,
      "Exepcted zero_point to only have one element received: %zd",
      ssize_t(zero_point_out.numel()));
}

void choose_qparams(
    const Tensor& input,
    int32_t qmin,
    int32_t qmax,
    Tensor& scale_out,
    Tensor& zero_point_out) {
  const float* x_fp32 = input.const_data_ptr<float>();
  // Compute x_min, x_max and q_params (scale, zero_point)
  float min = torch::executor::vec_minf(x_fp32, input.numel());
  float max = torch::executor::vec_maxf(x_fp32, input.numel());

  // We extend the [min, max] interval to ensure that it contains 0.
  // Otherwise, we would not meet the requirement that 0 be an exactly
  // representable value.
  min = std::min(min, 0.f);
  max = std::max(max, 0.f);

  // Use double precision for intermediate computation but use single precision
  // in final number to reflect the actual number used during quantization.
  double scale = (static_cast<double>(max) - min) / (qmax - qmin);
  // If scale is 0 or too small so its reciprocal is infinity, we arbitrary
  // adjust the scale to 0.1 . We want to avoid scale's reciprocal being
  // infinity because some of fbgemm code pre-computes scale's reciprocal to do
  // multiplication instead of division in the time critical part of code.
  if (float(scale) == 0.0f || std::isinf(1.0f / float(scale))) {
    scale = 0.1;
  }
  ET_CHECK_MSG(scale > 0, "quantization scale should be > 0");

  // Cut off small scale
  if (scale < SMALL_SCALE_THRESHOLD) {
    float org_scale = scale;
    scale = SMALL_SCALE_THRESHOLD;
    // Adjust the min and max based on the new scale
    if (min == 0.0f) {
      max = SMALL_SCALE_THRESHOLD * (qmax - qmin);
    } else if (max == 0.0f) {
      min = -SMALL_SCALE_THRESHOLD * (qmax - qmin);
    } else {
      float amplifier = SMALL_SCALE_THRESHOLD / org_scale;
      min *= amplifier;
      max *= amplifier;
    }
  }

  // Zero-point computation.
  // First the initial floating-point computation. The zero-point can be
  // determined from solving an affine equation for any known pair
  // (real value, corresponding quantized value).
  // We know two such pairs: (rmin, qmin) and (rmax, qmax).
  // The arithmetic error on the zero point computed from either pair
  // will be roughly machine_epsilon * (sum of absolute values of terms)
  // so we want to use the variant that adds the smaller terms.
  double zero_point_from_min = qmin - min / static_cast<double>(scale);
  double zero_point_from_max = qmax - max / static_cast<double>(scale);
  double zero_point_from_min_error =
      std::abs(qmin) - std::abs(min / static_cast<double>(scale));
  double zero_point_from_max_error =
      std::abs(qmax) - std::abs(max / static_cast<double>(scale));
  double initial_zero_point =
      zero_point_from_min_error < zero_point_from_max_error
      ? zero_point_from_min
      : zero_point_from_max;

  // Now we need to nudge the zero point to be an integer
  // (our zero points are integer, and this is motivated by the requirement
  // to be able to represent the real value "0" exactly as a quantized value,
  // which is required in multiple places, for example in Im2col with zero
  // padding).
  int32_t nudged_zero_point = 0;
  if (initial_zero_point < qmin) {
    nudged_zero_point = qmin;
  } else if (initial_zero_point > qmax) {
    nudged_zero_point = qmax;
  } else {
    nudged_zero_point = nearbyint(static_cast<float>(initial_zero_point));
  }

  scale_out.mutable_data_ptr<double>()[0] = scale;
  zero_point_out.mutable_data_ptr<int64_t>()[0] = nudged_zero_point;
}
} // namespace

std::tuple<Tensor&, Tensor&> choose_qparams_tensor_out(
    const Tensor& input,
    int64_t quant_min,
    int64_t quant_max,
    ET_UNUSED double eps,
    ScalarType dtype,
    Tensor& scale_out,
    Tensor& zero_point_out) {
  check_quantize_per_tensor_args(
      input, quant_min, quant_max, dtype, scale_out, zero_point_out);

  choose_qparams(input, quant_min, quant_max, scale_out, zero_point_out);
  return {scale_out, zero_point_out};
}

::std::tuple<Tensor&, Tensor&> choose_qparams_tensor_out(
    KernelRuntimeContext& context,
    const Tensor& input,
    int64_t quant_min,
    int64_t quant_max,
    double eps,
    ScalarType dtype,
    Tensor& scale_out,
    Tensor& zero_point_out) {
  // TODO(larryliu): Add a context arg to the real op function and remove this
  // wrapper
  (void)context;
  return choose_qparams_tensor_out(
      input, quant_min, quant_max, eps, dtype, scale_out, zero_point_out);
}

} // namespace native
} // namespace executor
} // namespace torch
