/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/kernels/portable/cpu/util/copy_ops_util.h>
#include <executorch/runtime/kernel/kernel_includes.h>
#include <executorch/runtime/kernel/thread_parallel_interface.h>

#if defined(__aarch64__)
#include <arm_neon.h>

#include <cstring>
#endif

#include <optional>
#include <type_traits>

namespace torch {
namespace executor {
namespace native {

using BFloat16 = executorch::aten::BFloat16;
using MemoryFormat = executorch::aten::MemoryFormat;
using ScalarType = executorch::aten::ScalarType;
using Tensor = executorch::aten::Tensor;

Tensor& to_copy_out(
    KernelRuntimeContext& ctx,
    const Tensor& self,
    bool non_blocking,
    std::optional<MemoryFormat> memory_format,
    Tensor& out);

namespace {

#if defined(__aarch64__)
static_assert(sizeof(BFloat16) == sizeof(uint16_t));
static_assert(std::is_trivially_copyable_v<BFloat16>);

void float_to_bfloat16_range(
    const float* const input,
    BFloat16* const output,
    const int64_t begin,
    const int64_t end) {
  constexpr int64_t kVectorWidth = 8;
  const uint32x4_t mantissa_lsb_mask = vdupq_n_u32(1);
  const uint32x4_t rounding_bias = vdupq_n_u32(0x7FFF);
  const uint32x4_t magnitude_mask = vdupq_n_u32(0x7FFFFFFF);
  const uint32x4_t infinity = vdupq_n_u32(0x7F800000);
  const uint32x4_t canonical_nan = vdupq_n_u32(0x7FC00000);

  int64_t i = begin;
  for (; i + kVectorWidth <= end; i += kVectorWidth) {
    const uint32x4_t low_bits = vreinterpretq_u32_f32(vld1q_f32(input + i));
    const uint32x4_t high_bits =
        vreinterpretq_u32_f32(vld1q_f32(input + i + 4));

    const auto round_and_canonicalize = [&](const uint32x4_t bits) {
      const uint32x4_t mantissa_lsb =
          vandq_u32(vshrq_n_u32(bits, 16), mantissa_lsb_mask);
      const uint32x4_t rounded =
          vaddq_u32(bits, vaddq_u32(rounding_bias, mantissa_lsb));
      const uint32x4_t is_nan =
          vcgtq_u32(vandq_u32(bits, magnitude_mask), infinity);
      return vbslq_u32(is_nan, canonical_nan, rounded);
    };

    const uint16x4_t low = vshrn_n_u32(round_and_canonicalize(low_bits), 16);
    const uint16x8_t result =
        vshrn_high_n_u32(low, round_and_canonicalize(high_bits), 16);
    std::memcpy(output + i, &result, sizeof(result));
  }

  for (; i < end; ++i) {
    output[i] = static_cast<BFloat16>(input[i]);
  }
}

void bfloat16_to_float_range(
    const BFloat16* const input,
    float* const output,
    const int64_t begin,
    const int64_t end) {
  constexpr int64_t kVectorWidth = 8;

  int64_t i = begin;
  for (; i + kVectorWidth <= end; i += kVectorWidth) {
    uint16x8_t input_bits;
    std::memcpy(&input_bits, input + i, sizeof(input_bits));
    const uint32x4_t low_bits = vshll_n_u16(vget_low_u16(input_bits), 16);
    const uint32x4_t high_bits = vshll_high_n_u16(input_bits, 16);
    vst1q_f32(output + i, vreinterpretq_f32_u32(low_bits));
    vst1q_f32(output + i + 4, vreinterpretq_f32_u32(high_bits));
  }

  for (; i < end; ++i) {
    output[i] = static_cast<float>(input[i]);
  }
}

template <typename Input, typename Output>
void convert_contiguous(const Tensor& self, Tensor& out) {
  const auto numel = self.numel();
  if (numel == 0) {
    return;
  }

  const auto* const input = self.const_data_ptr<Input>();
  auto* const output = out.mutable_data_ptr<Output>();
  const auto convert_range = [&](const auto begin, const auto end) {
    if constexpr (std::is_same_v<Input, float>) {
      float_to_bfloat16_range(input, output, begin, end);
    } else {
      bfloat16_to_float_range(input, output, begin, end);
    }
  };

  if (numel > ::executorch::extension::internal::GRAIN_SIZE) {
    ::executorch::extension::parallel_for(
        0, numel, ::executorch::extension::internal::GRAIN_SIZE, convert_range);
    return;
  }
  convert_range(0, numel);
}
#endif

} // namespace

Tensor& opt_to_copy_out(
    KernelRuntimeContext& ctx,
    const Tensor& self,
    bool non_blocking,
    std::optional<MemoryFormat> memory_format,
    Tensor& out) {
#if defined(__aarch64__)
  const bool float_to_bfloat16 = self.scalar_type() == ScalarType::Float &&
      out.scalar_type() == ScalarType::BFloat16;
  const bool bfloat16_to_float = self.scalar_type() == ScalarType::BFloat16 &&
      out.scalar_type() == ScalarType::Float;
  const bool supported_memory_format = !memory_format.has_value() ||
      memory_format.value() == MemoryFormat::Contiguous;
  const bool can_use_optimized_kernel =
      (float_to_bfloat16 || bfloat16_to_float) && !non_blocking &&
      supported_memory_format && tensor_is_default_dim_order(self) &&
      tensor_is_default_dim_order(out);
  if (can_use_optimized_kernel) {
    ET_KERNEL_CHECK(
        ctx,
        check_to_copy_args(self, non_blocking, memory_format, out),
        InvalidArgument,
        out);
    ET_KERNEL_CHECK(
        ctx,
        resize_tensor(out, self.sizes()) == Error::Ok,
        InvalidArgument,
        out);
    ET_KERNEL_CHECK(
        ctx, tensors_have_same_dim_order(self, out), InvalidArgument, out);
    ET_KERNEL_CHECK(
        ctx, tensor_is_default_dim_order(self), InvalidArgument, out);

    if (float_to_bfloat16) {
      convert_contiguous<float, BFloat16>(self, out);
    } else {
      convert_contiguous<BFloat16, float>(self, out);
    }
    return out;
  }
#endif

  return to_copy_out(ctx, self, non_blocking, memory_format, out);
}

} // namespace native
} // namespace executor
} // namespace torch
