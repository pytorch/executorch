/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/backends/cadence/hifi/kernels/kernels.h>
#include <xa_nnlib_common.h>
#include <xa_nnlib_common_macros.h>

namespace impl {
namespace HiFi {
namespace kernels {

__attribute__((always_inline)) void
memcpy(void* dst, const void* src, size_t num_bytes) {
  MEMCPY_8b(dst, src, num_bytes);
}

void* allocate_temp_memory(KernelRuntimeContext& ctx, size_t size) {
  constexpr size_t kAlignment =
      16; // 16-byte alignment for vectorized operations
  Result<void*> temp_mem_res = ctx.allocate_temp(size, kAlignment);
  if (temp_mem_res.ok()) {
    void* ptr = temp_mem_res.get();
    return ptr;
  } else {
    ET_LOG(
        Error,
        "Failed to allocate temp memory, error: 0x%x",
        static_cast<uint32_t>(temp_mem_res.error()));
    return nullptr;
  }
}

// Quantize a fp32 value to an int8_t/uint8_t value
template <typename T>
__attribute__((always_inline)) T
quantize(const float x, float scale, int32_t zero_point) {
  constexpr float min_val = std::numeric_limits<T>::min();
  constexpr float max_val = std::numeric_limits<T>::max();
  float tmp = roundf(x * scale + zero_point);
  return std::max(std::min(tmp, max_val), min_val);
}

// Quantize a fp32 array to an int8_t/uint8_t/int16_t array
template <typename T>
void quantize(
    T* __restrict__ y,
    const float* __restrict__ x,
    float scale,
    int32_t zero_point,
    size_t size) {
  xtfloatx2 scale_vec = (xtfloatx2)scale;
  xtfloatx2 zero_vec = XT_FLOAT_SX2(zero_point, 0);

  constexpr float min_val = std::numeric_limits<T>::min();
  constexpr float max_val = std::numeric_limits<T>::max();

  const xtfloatx2* __restrict__ p0 = (const xtfloatx2* __restrict__)x;
  ae_valign va0 = XT_LASX2PP(p0);

  size_t i = 0;
  // Vectorize by 2
  for (; i < (size & ~1); i += 2) {
    xtfloatx2 in_vec;
    XT_LASX2IP(in_vec, va0, p0);
    xtfloatx2 acc = zero_vec;
    XT_MADD_SX2(acc, scale_vec, in_vec);
    xtfloatx2 t0 = XT_FIROUND_SX2(acc);
    ae_int32x2 t1 =
        XT_UTRUNC_SX2(XT_MAX_SX2(XT_MIN_SX2(t0, max_val), min_val), 0);
    y[i] = AE_MOVAD32_H(t1);
    y[i + 1] = AE_MOVAD32_L(t1);
  }
  // Handle residual iteration
  if (i < size) {
    y[i] = quantize<T>(x[i], scale, zero_point);
  }
}

// Dequantize an int8_t/uint8_t value to a fp32 value
template <typename T>
__attribute__((always_inline)) float
dequantize(const T x, float scale, int32_t zero_point) {
  return scale * (x - zero_point);
}

// Deuantize an int8_t/uint8_t/int16_t array to an fp32 array
template <typename T>
void dequantize(
    float* __restrict__ y,
    const T* __restrict__ x,
    float scale,
    int32_t zero_point,
    size_t size) {
  xtfloatx2 scale_vec = (xtfloatx2)scale;
  xtfloatx2 zero_vec = XT_FLOAT_SX2(zero_point, 0);

  xtfloatx2* __restrict__ p0 = (xtfloatx2*)y;
  ae_valign va0 = AE_ZALIGN64();
  size_t i = 0;
  // Vectorize by 2
  for (size_t e = (size >> 1) << 1; i < e; i += 2) {
    xtfloatx2 in_vec = {(float)x[i], (float)x[i + 1]};
    xtfloatx2 t0 = XT_SUB_SX2(in_vec, zero_vec);
    xtfloatx2 t1 = XT_MUL_SX2(t0, scale_vec);
    XT_SASX2IP(t1, va0, p0);
  }
  // Flush the output stream
  XT_SASX2POSFP(va0, p0);

  // Handle residual iteration
  if (i < size) {
    y[i] = dequantize<T>(x[i], scale, zero_point);
  }
}

// explicit template instantiation

#define typed_quantize_val(dtype)                         \
  template __attribute__((always_inline)) dtype quantize( \
      const float x, float inv_scale, int32_t zero_point);
typed_quantize_val(int8_t);
typed_quantize_val(uint8_t);
typed_quantize_val(int16_t);
typed_quantize_val(uint16_t);
typed_quantize_val(int32_t);
#undef typed_quantize_val

#define typed_quantize_vec(dtype)  \
  template void quantize(          \
      dtype* __restrict__ y,       \
      const float* __restrict__ x, \
      float inv_scale,             \
      int32_t zero_point,          \
      size_t size);
typed_quantize_vec(int8_t);
typed_quantize_vec(uint8_t);
typed_quantize_vec(int16_t);
typed_quantize_vec(uint16_t);
typed_quantize_vec(int32_t);
#undef typed_quantize_vec

#define typed_dequantize_val(dtype)                         \
  template __attribute__((always_inline)) float dequantize( \
      const dtype x, float scale, int32_t zero_point);
typed_dequantize_val(int8_t);
typed_dequantize_val(uint8_t);
typed_dequantize_val(int16_t);
typed_dequantize_val(uint16_t);
typed_dequantize_val(int32_t);
#undef typed_dequantize_val

#define typed_dequantize_vec(dtype) \
  template void dequantize(         \
      float* __restrict__ y,        \
      const dtype* __restrict__ x,  \
      float scale,                  \
      int32_t zero_point,           \
      size_t size);
typed_dequantize_vec(int8_t);
typed_dequantize_vec(uint8_t);
typed_dequantize_vec(int16_t);
typed_dequantize_vec(uint16_t);
typed_dequantize_vec(int32_t);
#undef typed_dequantize_vec

} // namespace kernels
} // namespace HiFi
} // namespace impl
