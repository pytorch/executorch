/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "cortex_m_ops_common.h"

#include <cstring>

#if defined(__ARM_FEATURE_MVE) && (__ARM_FEATURE_MVE & 1)
#include <arm_mve.h>
#define HAS_HELIUM_SIMD 1
#endif

#if defined(ARM_MATH_DSP) && !defined(HAS_HELIUM_SIMD)
#include <arm_acle.h>
#define HAS_DSP_PACKED_LUT 1
#endif

namespace cortex_m {
namespace native {

#if defined(HAS_DSP_PACKED_LUT)
// Local 4-byte read/write helpers. We deliberately don't include
// `arm_nnsupportfunctions.h` for the equivalent CMSIS-NN `arm_nn_read_s8x4_ia`
// / `arm_nn_write_s8x4_ia` -- the header is public but pulls in the entire
// CMSIS-NN support surface (~1500 lines) just for two memcpy wrappers.
static inline uint32_t read_u8x4_ia(const int8_t** in) {
  uint32_t val;
  std::memcpy(&val, *in, 4);
  *in += 4;
  return val;
}

static inline void write_u8x4_ia(int8_t** out, uint32_t val) {
  std::memcpy(*out, &val, 4);
  *out += 4;
}
#endif

// cppcheck-suppress unusedFunction
Tensor& quantized_activation_out(
    KernelRuntimeContext& context,
    const Tensor& input,
    const Tensor& lut,
    Tensor& out) {
  ET_CHECK_MSG(
      input.scalar_type() == ScalarType::Char,
      "quantized_activation: input must be int8");
  ET_CHECK_MSG(
      out.scalar_type() == ScalarType::Char,
      "quantized_activation: output must be int8");
  ET_CHECK_MSG(
      lut.scalar_type() == ScalarType::Char,
      "quantized_activation: lut must be int8");
  ET_CHECK_MSG(
      lut.numel() == 256,
      "quantized_activation: lut must have 256 entries, got %" PRId64,
      static_cast<int64_t>(lut.numel()));
  ET_CHECK_MSG(
      input.numel() == out.numel(),
      "quantized_activation: input and output must have the same numel");

  const int8_t* in_data = input.const_data_ptr<int8_t>();
  const int8_t* lut_data = lut.const_data_ptr<int8_t>();
  int8_t* out_data = out.mutable_data_ptr<int8_t>();

  // The LUT is precomputed AoT from the input/output qparams and the
  // activation function (sigmoid / tanh / silu / ...), so the kernel does not
  // need to know which activation it is implementing. The signed int8 input
  // is biased by 128 to use it as an unsigned [0, 255] table index.
  const int64_t n = input.numel();
  int64_t i = 0;

#if defined(HAS_HELIUM_SIMD)
  // M55/M85: 16 lanes per iteration. Reinterpret the int8 input as uint8
  // (bit-identical load), add 128 mod 256 to produce a uint8 LUT index, then
  // gather-load the int8 result from the LUT.
  for (; i + 15 < n; i += 16) {
    uint8x16_t in_u8 =
        vldrbq_u8(reinterpret_cast<const uint8_t*>(in_data + i));
    uint8x16_t idx = vaddq_n_u8(in_u8, 128);
    int8x16_t result = vldrbq_gather_offset_s8(lut_data, idx);
    vstrbq_s8(out_data + i, result);
  }
#elif defined(HAS_DSP_PACKED_LUT)
  // M4/M7 (DSP, no MVE): process 4 bytes per iteration. The DSP win comes from
  // (a) folding 4 byte-loads into one word-load, (b) batching the +128 bias
  // with `__uadd8`, and (c) folding 4 byte-stores into one word-store. The
  // LUT lookups themselves still hit memory four times per word -- no DSP
  // gather instruction exists on M-class.
  const int8_t* in_ptr = in_data;
  int8_t* out_ptr = out_data;
  const int64_t word_iters = n >> 2;
  for (int64_t w = 0; w < word_iters; ++w) {
    const uint32_t in_word = read_u8x4_ia(&in_ptr);
    const uint32_t idx_word = __uadd8(in_word, 0x80808080u);
    const uint32_t out_word =
        static_cast<uint32_t>(static_cast<uint8_t>(lut_data[idx_word & 0xFFu])) |
        (static_cast<uint32_t>(static_cast<uint8_t>(lut_data[(idx_word >> 8) & 0xFFu]))
         << 8) |
        (static_cast<uint32_t>(static_cast<uint8_t>(lut_data[(idx_word >> 16) & 0xFFu]))
         << 16) |
        (static_cast<uint32_t>(static_cast<uint8_t>(lut_data[(idx_word >> 24) & 0xFFu]))
         << 24);
    write_u8x4_ia(&out_ptr, out_word);
  }
  i = word_iters << 2;
#endif

  // 4x-unrolled scalar tail. On M-class cores without MVE or DSP the unroll
  // lets the compiler issue independent LUT loads; on the MVE / DSP paths
  // above this only runs for the < 16- (or < 4-) element remainder.
  for (; i + 3 < n; i += 4) {
    out_data[i + 0] = lut_data[static_cast<uint8_t>(in_data[i + 0] + 128)];
    out_data[i + 1] = lut_data[static_cast<uint8_t>(in_data[i + 1] + 128)];
    out_data[i + 2] = lut_data[static_cast<uint8_t>(in_data[i + 2] + 128)];
    out_data[i + 3] = lut_data[static_cast<uint8_t>(in_data[i + 3] + 128)];
  }
  for (; i < n; ++i) {
    out_data[i] = lut_data[static_cast<uint8_t>(in_data[i] + 128)];
  }

  return out;
}

} // namespace native
} // namespace cortex_m
