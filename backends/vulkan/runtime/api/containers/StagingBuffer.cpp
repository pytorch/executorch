/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/backends/vulkan/runtime/api/containers/StagingBuffer.h>

namespace vkcompute {
namespace api {

namespace {

//
// The following fp16<->fp32 conversion functions are adapted from:
// executorch/runtime/core/portable_type/c10/torch/headeronly/util/Half.h
// (fp16_ieee_to_fp32_value and fp16_ieee_from_fp32_value)
//

inline float fp32_from_bits(uint32_t bits) {
  float result;
  std::memcpy(&result, &bits, sizeof(result));
  return result;
}

inline uint32_t fp32_to_bits(float f) {
  uint32_t bits;
  std::memcpy(&bits, &f, sizeof(bits));
  return bits;
}

/*
 * Convert a 16-bit floating-point number in IEEE half-precision format, in bit
 * representation, to a 32-bit floating-point number in IEEE single-precision
 * format.
 */
float half_to_float(uint16_t h) {
  /*
   * Extend the half-precision floating-point number to 32 bits and shift to the
   * upper part of the 32-bit word:
   *      +---+-----+------------+-------------------+
   *      | S |EEEEE|MM MMMM MMMM|0000 0000 0000 0000|
   *      +---+-----+------------+-------------------+
   * Bits  31  26-30    16-25            0-15
   */
  const uint32_t w = (uint32_t)h << 16;
  /*
   * Extract the sign of the input number into the high bit of the 32-bit word:
   */
  const uint32_t sign = w & UINT32_C(0x80000000);
  /*
   * Extract mantissa and biased exponent of the input number into the high bits
   * of the 32-bit word:
   */
  const uint32_t two_w = w + w;

  /*
   * Shift mantissa and exponent into bits 23-28 and bits 13-22 so they become
   * mantissa and exponent of a single-precision floating-point number:
   *
   * The exponent needs to be corrected by the difference in exponent bias
   * between single-precision and half-precision formats (0x7F - 0xF = 0x70).
   * We use 0xE0 initially and then scale by 2^(-112) to handle Inf/NaN.
   */
  constexpr uint32_t exp_offset = UINT32_C(0xE0) << 23;
  constexpr uint32_t scale_bits = (uint32_t)15 << 23;
  float exp_scale_val = 0;
  std::memcpy(&exp_scale_val, &scale_bits, sizeof(exp_scale_val));
  const float exp_scale = exp_scale_val;
  const float normalized_value =
      fp32_from_bits((two_w >> 4) + exp_offset) * exp_scale;

  /*
   * Convert denormalized half-precision inputs into single-precision results
   * (always normalized). Zero inputs are also handled here.
   */
  constexpr uint32_t magic_mask = UINT32_C(126) << 23;
  constexpr float magic_bias = 0.5f;
  const float denormalized_value =
      fp32_from_bits((two_w >> 17) | magic_mask) - magic_bias;

  /*
   * Choose either results of conversion of input as a normalized number, or
   * as a denormalized number, depending on the input exponent.
   */
  constexpr uint32_t denormalized_cutoff = UINT32_C(1) << 27;
  const uint32_t result = sign |
      (two_w < denormalized_cutoff ? fp32_to_bits(denormalized_value)
                                   : fp32_to_bits(normalized_value));
  return fp32_from_bits(result);
}

/*
 * Convert a 32-bit floating-point number in IEEE single-precision format to a
 * 16-bit floating-point number in IEEE half-precision format, in bit
 * representation.
 */
uint16_t float_to_half(float f) {
  constexpr uint32_t scale_to_inf_bits = (uint32_t)239 << 23;
  constexpr uint32_t scale_to_zero_bits = (uint32_t)17 << 23;
  float scale_to_inf_val = 0, scale_to_zero_val = 0;
  std::memcpy(&scale_to_inf_val, &scale_to_inf_bits, sizeof(scale_to_inf_val));
  std::memcpy(
      &scale_to_zero_val, &scale_to_zero_bits, sizeof(scale_to_zero_val));
  const float scale_to_inf = scale_to_inf_val;
  const float scale_to_zero = scale_to_zero_val;

  float base = (fabsf(f) * scale_to_inf) * scale_to_zero;

  const uint32_t w = fp32_to_bits(f);
  const uint32_t shl1_w = w + w;
  const uint32_t sign = w & UINT32_C(0x80000000);
  uint32_t bias = shl1_w & UINT32_C(0xFF000000);
  if (bias < UINT32_C(0x71000000)) {
    bias = UINT32_C(0x71000000);
  }

  base = fp32_from_bits((bias >> 1) + UINT32_C(0x07800000)) + base;
  const uint32_t bits = fp32_to_bits(base);
  const uint32_t exp_bits = (bits >> 13) & UINT32_C(0x00007C00);
  const uint32_t mantissa_bits = bits & UINT32_C(0x00000FFF);
  const uint32_t nonsign = exp_bits + mantissa_bits;
  return static_cast<uint16_t>(
      (sign >> 16) |
      (shl1_w > UINT32_C(0xFF000000) ? UINT16_C(0x7E00) : nonsign));
}

} // namespace

StagingBuffer::StagingBuffer(
    Context* context_p,
    const vkapi::ScalarType dtype,
    const size_t numel,
    const vkapi::CopyDirection direction)
    : context_p_(context_p),
      dtype_(get_staging_dtype(context_p, dtype)),
      vulkan_buffer_(context_p_->adapter_ptr()->vma().create_staging_buffer(
          element_size(dtype_) * numel,
          direction)),
      mapped_data_(nullptr) {}

vkapi::ScalarType get_staging_dtype(
    Context* context_p,
    vkapi::ScalarType dtype) {
  if (dtype == vkapi::kHalf &&
      !context_p->adapter_ptr()->has_full_float16_buffers_support()) {
    return vkapi::kFloat;
  }
  return dtype;
}

void StagingBuffer::cast_half_to_float_and_copy_from(
    const uint16_t* src,
    const size_t numel) {
  VK_CHECK_COND(numel <= this->numel());
  float* dst = reinterpret_cast<float*>(data());
  for (size_t i = 0; i < numel; ++i) {
    dst[i] = half_to_float(src[i]);
  }
}

void StagingBuffer::cast_float_to_half_and_copy_to(
    uint16_t* dst,
    const size_t numel) {
  VK_CHECK_COND(numel <= this->numel());
  vmaInvalidateAllocation(
      vulkan_buffer_.vma_allocator(),
      vulkan_buffer_.allocation(),
      0u,
      VK_WHOLE_SIZE);
  const float* src = reinterpret_cast<const float*>(data());
  for (size_t i = 0; i < numel; ++i) {
    dst[i] = float_to_half(src[i]);
  }
}

} // namespace api
} // namespace vkcompute
