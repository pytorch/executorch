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

// Convert IEEE 16-bit half to IEEE 32-bit float
float half_to_float(uint16_t h) {
  const uint32_t sign = (h >> 15) & 0x1;
  const uint32_t exp = (h >> 10) & 0x1F;
  const uint32_t mant = h & 0x3FF;

  uint32_t f;
  if (exp == 0) {
    if (mant == 0) {
      // Signed zero
      f = sign << 31;
    } else {
      // Subnormal half -> normalized float
      uint32_t e = 0;
      uint32_t m = mant;
      while ((m & 0x400) == 0) {
        m <<= 1;
        e++;
      }
      f = (sign << 31) | ((127 - 15 - e) << 23) | ((m & 0x3FF) << 13);
    }
  } else if (exp == 31) {
    // Inf or NaN
    f = (sign << 31) | 0x7F800000 | (mant << 13);
  } else {
    // Normalized
    f = (sign << 31) | ((exp - 15 + 127) << 23) | (mant << 13);
  }

  float result;
  memcpy(&result, &f, sizeof(float));
  return result;
}

// Convert IEEE 32-bit float to IEEE 16-bit half
int16_t float_to_half(float val) {
  uint32_t f;
  memcpy(&f, &val, sizeof(float));
  const uint32_t sign = (f >> 31) & 0x1;
  const int32_t exp = ((f >> 23) & 0xFF) - 127 + 15; // Rebias exponent
  const uint32_t mant = f & 0x7FFFFF;

  uint16_t h;
  if ((f & 0x7FFFFFFF) == 0) {
    // Signed zero
    h = static_cast<uint16_t>(sign << 15);
  } else if (exp <= 0) {
    // Underflow to zero or subnormal
    if (exp < -10) {
      // Too small, flush to zero
      h = static_cast<uint16_t>(sign << 15);
    } else {
      // Subnormal half
      const uint32_t m = (mant | 0x800000) >> (1 - exp);
      h = static_cast<uint16_t>((sign << 15) | (m >> 13));
    }
  } else if (exp >= 31) {
    // Overflow to infinity, or preserve NaN
    if (exp == 143 && mant != 0) {
      // NaN - preserve some mantissa bits
      h = static_cast<uint16_t>((sign << 15) | 0x7C00 | (mant >> 13));
      // Ensure NaN mantissa is non-zero
      if ((h & 0x03FF) == 0) {
        h |= 0x0001;
      }
    } else {
      // Infinity
      h = static_cast<uint16_t>((sign << 15) | 0x7C00);
    }
  } else {
    // Normalized
    h = static_cast<uint16_t>((sign << 15) | (exp << 10) | (mant >> 13));
  }

  return static_cast<int16_t>(h);
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
    const int16_t* src,
    const size_t numel) {
  VK_CHECK_COND(numel <= this->numel());
  float* dst = reinterpret_cast<float*>(data());
  for (size_t i = 0; i < numel; ++i) {
    dst[i] = half_to_float(static_cast<uint16_t>(src[i]));
  }
}

void StagingBuffer::cast_float_to_half_and_copy_to(
    int16_t* dst,
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
