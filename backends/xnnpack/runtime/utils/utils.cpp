/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/backends/xnnpack/runtime/utils/utils.h>
#include <executorch/runtime/platform/assert.h>
#include <cinttypes>

namespace executorch {
namespace backends {
namespace xnnpack {
namespace utils {

using executorch::aten::ScalarType;
using executorch::aten::Tensor;
using executorch::runtime::Error;

constexpr float SMALL_SCALE_THRESHOLD = 6.1e-5f;

Error ChooseQuantizationParams(
    float min,
    float max,
    int32_t qmin,
    int32_t qmax,
    QuantizationParams& result,
    bool preserve_sparsity = false,
    bool force_scale_power_of_two = false,
    bool reduce_range = false) {
  ET_CHECK_OR_RETURN_ERROR(
      min <= max,
      Internal,
      "In ChooseQuantizationParams, min should be less than or equal to max. min: %f, max: %f",
      min,
      max);

  if (reduce_range) {
    qmin = qmin / 2;
    qmax = qmax / 2;
  }
  if (min < 0 && max > 0 && preserve_sparsity) {
    int symmetric_qmin = -((qmax - qmin) / 2 + 1);
    int symmetric_qmax = (qmax - qmin) / 2;
    double max_scale =
        std::max(fabs(min / symmetric_qmin), fabs(max / symmetric_qmax));
    min = max_scale * symmetric_qmin;
    max = max_scale * symmetric_qmax;
  }

  // We extend the [min, max] interval to ensure that it contains 0.
  // Otherwise, we would not meet the requirement that 0 be an exactly
  // representable value.
  min = std::min(min, 0.f);
  max = std::max(max, 0.f);

  ET_CHECK_OR_RETURN_ERROR(
      qmin < qmax,
      Internal,
      "In ChooseQuantizationParams, qmin should be less than qmax");

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
  ET_CHECK_OR_RETURN_ERROR(
      scale > 0, Internal, "quantization scale should be > 0");

  if (force_scale_power_of_two) {
    if (scale < 1) {
      scale = 1.0 / (1 << static_cast<int>(floor(log(1.0 / scale) / log(2))));
    } else {
      scale = 1 << static_cast<int>(ceil(log(scale) / log(2)));
    }
  }

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

  // for symmetric quantization (preserve_sparsity == true), we force zero_point
  // to be a middle value between qmin and qmax.
  // If either min or max is 0, then we just use 0 as zero_point.
  if (min < 0 && max > 0 && preserve_sparsity) {
    initial_zero_point = static_cast<double>(qmin + qmax) / 2;
  }

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
    nudged_zero_point = nearbyint(initial_zero_point);
  }

  result.scale = scale;
  result.zero_point = nudged_zero_point;
  return Error::Ok;
}

Error GenerateRequantizationScale(
    const Tensor& weight_scales,
    float input_scale,
    float output_scale,
    std::vector<float>& requant_scales) {
  // Since weight scale is allocated with padding
  // weight_scales.numel() gives us padded num elements.
  const auto num_output_channels_padded = weight_scales.numel();
  const float* weight_scales_data = weight_scales.const_data_ptr<float>();
  if (static_cast<int64_t>(requant_scales.size()) <
      num_output_channels_padded) {
    requant_scales.resize(num_output_channels_padded);
  }
  for (int i = 0; i < num_output_channels_padded; ++i) {
    const auto inverse_output_scale = 1.f / output_scale;
    requant_scales[i] =
        (weight_scales_data[i] * input_scale) * inverse_output_scale;
    ET_CHECK_OR_RETURN_ERROR(
        requant_scales[i] > 0.0f && std::isnormal(requant_scales[i]),
        Internal,
        "failed to create op with requantization scale");
  }
  return Error::Ok;
}

std::pair<float, float> GetMinMax(const Tensor& ft) {
  float min = std::numeric_limits<float>::max();
  float max = -std::numeric_limits<float>::max();
  ET_CHECK_MSG(
      ft.scalar_type() == ScalarType::Float,
      "Expected float tensor but got %" PRId8,
      static_cast<int8_t>(ft.scalar_type()));
  const float* d = ft.const_data_ptr<float>();
  for (int i = 0; i < ft.numel(); ++i) {
    min = (d[i] < min) ? d[i] : min;
    max = (d[i] > max) ? d[i] : max;
  }
  return std::pair<float, float>(min, max);
}

#ifdef __aarch64__
template <>
uint8x8_t vqmov<uint8x8_t>(int16x8_t vraw) {
  return vqmovun_s16(vraw);
}

template <>
int8x8_t vqmov<int8x8_t>(int16x8_t vraw) {
  return vqmovn_s16(vraw);
}

template <>
void vst1<uint8_t, uint8x8_t>(uint8_t* out, uint8x8_t vout) {
  vst1_u8(out, vout);
}

template <>
void vst1<int8_t, int8x8_t>(int8_t* out, int8x8_t vout) {
  vst1_s8(out, vout);
}

template <>
void quantize_tensor_arm64_q8_wrapper<uint8_t>(
    const float* __restrict__ in,
    uint8_t* __restrict__ out,
    const int64_t N,
    const float scale,
    const int32_t zero_point) {
  quantize_tensor_arm64_q8<uint8_t, uint8x8_t>(in, out, N, scale, zero_point);
}

template <>
void quantize_tensor_arm64_q8_wrapper<int8_t>(
    const float* __restrict__ in,
    int8_t* __restrict__ out,
    const int64_t N,
    const float scale,
    const int32_t zero_point) {
  quantize_tensor_arm64_q8<int8_t, int8x8_t>(in, out, N, scale, zero_point);
}
#endif

} // namespace utils
} // namespace xnnpack
} // namespace backends
} // namespace executorch
