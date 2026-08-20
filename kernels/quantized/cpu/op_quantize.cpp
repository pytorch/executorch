/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/runtime/kernel/kernel_includes.h>
#include <executorch/runtime/kernel/thread_parallel_interface.h>
#include <algorithm>
#include <cinttypes>
#include <cmath>

#if defined(__aarch64__) || defined(__ARM_NEON__)
#include <arm_neon.h>
#endif

/**
 * For an input tensor, use the scale and zero_point arguments to quantize it.
 */
namespace torch {
namespace executor {
namespace native {

using Tensor = executorch::aten::Tensor;
using Scalar = executorch::aten::Scalar;
using ScalarType = executorch::aten::ScalarType;
using KernelRuntimeContext = torch::executor::KernelRuntimeContext;

namespace {

/**
 * Asserts that the parameters are valid.
 */
void check_quantize_per_tensor_args(
    const Tensor& input,
    int64_t quant_min,
    int64_t quant_max,
    ScalarType dtype,
    Tensor& out) {
  // Ensure self and out has the same shape
  ET_CHECK_MSG(
      torch::executor::isFloatingType(input.scalar_type()),
      "input.scalar_type() %" PRId8 " is not floating type",
      static_cast<int8_t>(input.scalar_type()));

  int32_t quant_min_lower_bound = 0, quant_max_upper_bound = 0;
  ScalarType out_dtype = out.scalar_type();
  ET_CHECK_MSG(
      out_dtype == dtype,
      "out.scalar_type() %" PRId8 " is not matching dtype argument %" PRId8,
      static_cast<int8_t>(out_dtype),
      static_cast<int8_t>(dtype));
  if (out_dtype == ScalarType::Byte) {
    quant_min_lower_bound =
        static_cast<int32_t>(std::numeric_limits<uint8_t>::min());
    quant_max_upper_bound =
        static_cast<int32_t>(std::numeric_limits<uint8_t>::max());
  } else if (dtype == ScalarType::Char) {
    quant_min_lower_bound =
        static_cast<int32_t>(std::numeric_limits<int8_t>::min());
    quant_max_upper_bound =
        static_cast<int32_t>(std::numeric_limits<int8_t>::max());
  } else if (dtype == ScalarType::Bits16 || dtype == ScalarType::UInt16) {
    quant_min_lower_bound = std::numeric_limits<uint16_t>::min();
    quant_max_upper_bound = std::numeric_limits<uint16_t>::max();
  } else if (dtype == ScalarType::Short) {
    quant_min_lower_bound = std::numeric_limits<int16_t>::min();
    quant_max_upper_bound = std::numeric_limits<int16_t>::max();
  } else if (dtype == ScalarType::Int) {
    quant_min_lower_bound = std::numeric_limits<int32_t>::min();
    quant_max_upper_bound = std::numeric_limits<int32_t>::max();
  } else {
    ET_CHECK_MSG(
        false, "Unsupported dtype: %" PRId8, static_cast<int8_t>(out_dtype));
  }
  ET_CHECK_MSG(
      quant_min >= quant_min_lower_bound,
      "quant_min out of bound for dtype, expected quant_min_lower_bound: %" PRId32
      " actual quant_min: %" PRId64,
      quant_min_lower_bound,
      quant_min);

  ET_CHECK_MSG(
      quant_max <= quant_max_upper_bound,
      "quant_max out of bound for dtype, expected quant_max_upper_bound: %" PRId32
      " actual quant_max: %" PRId64,
      quant_max_upper_bound,
      quant_max);
}

} // namespace

template <typename T, typename K>
T quantize_val(
    double scale,
    int64_t zero_point,
    K value,
    int64_t quant_min,
    int64_t quant_max) {
  int64_t qvalue;
  float inv_scale = 1.0f / static_cast<float>(scale);
  qvalue = static_cast<int64_t>(
      static_cast<int32_t>(zero_point) +
      std::nearbyint(static_cast<float>(inv_scale * value)));

  qvalue = std::max<int64_t>(qvalue, quant_min);
  qvalue = std::min<int64_t>(qvalue, quant_max);
  return static_cast<T>(qvalue);
}

#if defined(__aarch64__) || defined(__ARM_NEON__)

// Traits for type-specific NEON operations
template <typename T>
struct NeonQuantizeTraits;

template <>
struct NeonQuantizeTraits<uint8_t> {
  // Narrow int16x8 to uint8x8 with saturation (unsigned)
  static inline uint8x8_t narrow_and_saturate(int16x8_t v) {
    return vqmovun_s16(v);
  }

  // Store uint8x8 to memory
  static inline void store(uint8_t* ptr, uint8x8_t v) {
    vst1_u8(ptr, v);
  }

  // Scalar clamping for uint8
  static inline uint8_t clamp_scalar(int32_t val) {
    return static_cast<uint8_t>(std::min(255, std::max(0, val)));
  }
};

template <>
struct NeonQuantizeTraits<int8_t> {
  // Narrow int16x8 to int8x8 with saturation (signed)
  static inline int8x8_t narrow_and_saturate(int16x8_t v) {
    return vqmovn_s16(v);
  }

  // Store int8x8 to memory
  static inline void store(int8_t* ptr, int8x8_t v) {
    vst1_s8(ptr, v);
  }

  // Scalar clamping for int8
  static inline int8_t clamp_scalar(int32_t val) {
    return static_cast<int8_t>(std::min(127, std::max(-128, val)));
  }
};

// Unified ARM NEON optimized quantization for contiguous blocks
// Processes N elements with a single scale/zero_point pair
// Used for both per-tensor (entire tensor) and per-channel (one block per
// channel)
template <typename T>
void quantize_arm(
    const float* __restrict__ in,
    T* __restrict__ out,
    const int64_t N,
    const float inv_scale,
    const int32_t zero_point,
    const int32_t quant_min,
    const int32_t quant_max) {
  using Traits = NeonQuantizeTraits<T>;
  const float32x4_t vinv_scale = vdupq_n_f32(inv_scale);

#if defined(__aarch64__)
  // ARMv8: Use vcvtnq_s32_f32 for rounding
  const int16x8_t vzero_point = vdupq_n_s16(static_cast<int16_t>(zero_point));
  const int16x8_t vquant_min = vdupq_n_s16(static_cast<int16_t>(quant_min));
  const int16x8_t vquant_max = vdupq_n_s16(static_cast<int16_t>(quant_max));

  int64_t i = 0;
  // Process 8 elements at a time
  for (; i + 8 <= N; i += 8) {
    const float32x4_t vin0123 = vld1q_f32(in + i);
    const float32x4_t vin4567 = vld1q_f32(in + i + 4);

    // Multiply by inv_scale and round
    const int32x4_t v0123_rounded =
        vcvtnq_s32_f32(vmulq_f32(vin0123, vinv_scale));
    const int32x4_t v4567_rounded =
        vcvtnq_s32_f32(vmulq_f32(vin4567, vinv_scale));

    // Combine to int16 and add zero_point
    int16x8_t v01234567_packed = vqaddq_s16(
        vqmovn_high_s32(vqmovn_s32(v0123_rounded), v4567_rounded), vzero_point);

    // Clamp to quant_min/quant_max
    v01234567_packed = vmaxq_s16(v01234567_packed, vquant_min);
    v01234567_packed = vminq_s16(v01234567_packed, vquant_max);

    // Convert to T (int8/uint8) with saturation using type-specific operation
    const auto vout01234567 = Traits::narrow_and_saturate(v01234567_packed);
    Traits::store(out + i, vout01234567);
  }

  // Handle remaining elements with proper quant_min/quant_max clamping
  for (; i < N; ++i) {
    float val = in[i] * inv_scale;
    int32_t qval = static_cast<int32_t>(std::nearbyint(val)) + zero_point;
    qval = std::max(quant_min, std::min(quant_max, qval));
    out[i] = static_cast<T>(qval);
  }

#else
  // ARMv7: Use magic float rounding
  const int32x4_t voffset = vdupq_n_s32(zero_point - 0x4B400000);
  const float32x4_t vmagic_float = vdupq_n_f32(12582912.0f);

  int64_t i = 0;
  // Process 8 elements at a time
  for (; i + 8 <= N; i += 8) {
    const float32x4_t vin0123 = vld1q_f32(in + i);
    const float32x4_t vin4567 = vld1q_f32(in + i + 4);

    const int32x4_t vraw0123 = vaddq_s32(
        voffset,
        vreinterpretq_s32_f32(
            vaddq_f32(vmagic_float, vmulq_f32(vin0123, vinv_scale))));
    const int32x4_t vraw4567 = vaddq_s32(
        voffset,
        vreinterpretq_s32_f32(
            vaddq_f32(vmagic_float, vmulq_f32(vin4567, vinv_scale))));

    const int16x8_t vraw01234567 =
        vcombine_s16(vqmovn_s32(vraw0123), vqmovn_s32(vraw4567));

    // Convert to T (int8/uint8) with saturation using type-specific operation
    const auto vout01234567 = Traits::narrow_and_saturate(vraw01234567);
    Traits::store(out + i, vout01234567);
  }

  // Handle remaining elements with proper quant_min/quant_max clamping
  for (; i < N; ++i) {
    float val = in[i] * inv_scale;
    int32_t qval = static_cast<int32_t>(std::nearbyint(val)) + zero_point;
    qval = std::max(quant_min, std::min(quant_max, qval));
    out[i] = static_cast<T>(qval);
  }
#endif
}

#endif // defined(__aarch64__) || defined(__ARM_NEON__)

Tensor& quantize_per_tensor_out(
    const Tensor& input,
    double scale,
    int64_t zero_point,
    int64_t quant_min,
    int64_t quant_max,
    ScalarType dtype,
    Tensor& out) {
  torch::executor::Error err = resize_tensor(out, input.sizes());
  ET_CHECK_MSG(
      err == torch::executor::Error::Ok,
      "Failed to resize out Tensor in quantize_per_tensor_out");

  check_quantize_per_tensor_args(input, quant_min, quant_max, dtype, out);

  // Try ARM NEON optimized path for float->int8/uint8 quantization
#if defined(__aarch64__) || defined(__ARM_NEON__)
  if (input.scalar_type() == ScalarType::Float) {
    if (dtype == ScalarType::Byte) {
      quantize_arm<uint8_t>(
          input.const_data_ptr<float>(),
          out.mutable_data_ptr<uint8_t>(),
          input.numel(),
          1.0f / static_cast<float>(scale),
          static_cast<int32_t>(zero_point),
          static_cast<int32_t>(quant_min),
          static_cast<int32_t>(quant_max));
      return out;
    } else if (dtype == ScalarType::Char) {
      quantize_arm<int8_t>(
          input.const_data_ptr<float>(),
          out.mutable_data_ptr<int8_t>(),
          input.numel(),
          1.0f / static_cast<float>(scale),
          static_cast<int32_t>(zero_point),
          static_cast<int32_t>(quant_min),
          static_cast<int32_t>(quant_max));
      return out;
    }
  }
#endif

  // Fallback scalar implementation for all other cases
#define QUANTIZE_IMPL(IN_CTYPE, OUT_CTYPE, out_dtype)              \
  case ScalarType::out_dtype: {                                    \
    auto* out_data_ptr = out.mutable_data_ptr<OUT_CTYPE>();        \
    const auto* input_data_ptr = input.const_data_ptr<IN_CTYPE>(); \
    const auto input_numel = input.numel();                        \
    for (size_t i = 0; i < input_numel; i++) {                     \
      IN_CTYPE value = input_data_ptr[i];                          \
      out_data_ptr[i] = quantize_val<OUT_CTYPE, IN_CTYPE>(         \
          scale, zero_point, value, quant_min, quant_max);         \
    }                                                              \
  } break;
#define CALCULATE_FLOAT_TYPE(IN_CTYPE, in_dtype)         \
  case ScalarType::in_dtype:                             \
    switch (out.scalar_type()) {                         \
      ET_FORALL_INT_TYPES_WITH(IN_CTYPE, QUANTIZE_IMPL); \
      QUANTIZE_IMPL(IN_CTYPE, uint16_t, Bits16)          \
      QUANTIZE_IMPL(IN_CTYPE, uint16_t, UInt16)          \
      default:                                           \
        ET_CHECK_MSG(                                    \
            false,                                       \
            "Unhandled output dtype %" PRId8,            \
            static_cast<int8_t>(out.scalar_type()));     \
    }                                                    \
    break;

  switch (input.scalar_type()) {
    ET_FORALL_FLOATH_TYPES(CALCULATE_FLOAT_TYPE);
    default:
      ET_CHECK_MSG(
          false,
          "Unhandled input dtype %" PRId8,
          static_cast<int8_t>(input.scalar_type()));
  }
#undef CALCULATE_FLOAT_TYPE
#undef QUANTIZE_IMPL
  return out;
}

Tensor& quantize_per_tensor_tensor_args_out(
    KernelRuntimeContext& context,
    const Tensor& input,
    const Tensor& scale,
    const Tensor& zero_point,
    int64_t quant_min,
    int64_t quant_max,
    ScalarType dtype,
    Tensor& out) {
  // Temporary change to allow not fatal failure for now to unblock some
  // expected failure tests that are dying instead of failure. Will revisit
  // after ET_KERNEL_CHECK is fully implemented and properly allows non fatal
  // failures.
  if (scale.scalar_type() != ScalarType::Double) {
    context.fail(torch::executor::Error::InvalidArgument);
    return out;
  }
  ET_CHECK_MSG(
      scale.scalar_type() == ScalarType::Double,
      "Expected scale to be Double tensor received: %" PRId8,
      static_cast<int8_t>(scale.scalar_type()));
  ET_CHECK_MSG(
      zero_point.scalar_type() == ScalarType::Long,
      "Expected zero_point to be Long tensor received: %" PRId8,
      static_cast<int8_t>(zero_point.scalar_type()));
  ET_CHECK_MSG(
      scale.numel() == 1,
      "Exepcted scale to only have one element received: %zd",
      ssize_t(scale.numel()));
  ET_CHECK_MSG(
      zero_point.numel() == 1,
      "Exepcted zero_point to only have one element received: %zd",
      ssize_t(zero_point.numel()));

  quantize_per_tensor_out(
      input,
      scale.const_data_ptr<double>()[0],
      zero_point.const_data_ptr<int64_t>()[0],
      quant_min,
      quant_max,
      dtype,
      out);
  return out;
}

Tensor& quantize_per_tensor_tensor_args_out(
    const Tensor& input,
    const Tensor& scale,
    const Tensor& zero_point,
    int64_t quant_min,
    int64_t quant_max,
    ScalarType dtype,
    Tensor& out) {
  auto context = KernelRuntimeContext();
  auto& res = quantize_per_tensor_tensor_args_out(
      context, input, scale, zero_point, quant_min, quant_max, dtype, out);
  ET_CHECK(context.failure_state() == Error::Ok);
  return res;
}

Tensor& quantize_per_tensor_out(
    KernelRuntimeContext& context,
    const Tensor& input,
    double scale,
    int64_t zero_point,
    int64_t quant_min,
    int64_t quant_max,
    ScalarType dtype,
    Tensor& out) {
  // TODO(larryliu): Add a context arg to the real op function and remove this
  // wrapper
  (void)context;
  return quantize_per_tensor_out(
      input, scale, zero_point, quant_min, quant_max, dtype, out);
}

Tensor& quantize_per_channel_out(
    const Tensor& input,
    const Tensor& scale,
    const Tensor& zero_point,
    int64_t axis,
    int64_t quant_min,
    int64_t quant_max,
    ScalarType dtype,
    Tensor& out) {
  // normalize axis
  ET_CHECK_MSG(
      tensor_has_dim(input, axis),
      "axis %zd is not legal it should be -input.dim() <= axis < input.dim() %zd",
      ssize_t(axis),
      ssize_t(input.dim()));

  if (axis < 0) {
    axis += nonzero_dim(input);
  }

  ET_CHECK_MSG(
      scale.scalar_type() == ScalarType::Double,
      "scale.scalar_type() %" PRId8 " is not double type",
      static_cast<int8_t>(scale.scalar_type()));

  ET_CHECK_MSG(
      scale.numel() == input.size(axis),
      "scale.numel() %zd != input.size(axis) %zd",
      scale.numel(),
      input.size(axis));

  ET_CHECK_MSG(
      zero_point.scalar_type() == ScalarType::Long,
      "zero_point.scalar_type() %" PRId8 " is not integer type",
      static_cast<int8_t>(zero_point.scalar_type()));

  ET_CHECK_MSG(
      zero_point.numel() == input.size(axis),
      "zero_point.numel() %zd != input.size(axis) %zd",
      zero_point.numel(),
      input.size(axis));

  check_quantize_per_tensor_args(input, quant_min, quant_max, dtype, out);

  const double* scale_data = scale.const_data_ptr<double>();
  const int64_t* zero_point_data = zero_point.const_data_ptr<int64_t>();

  // Calculate the block size for each channel
  int64_t axis_block_size = 1;
  for (int64_t i = axis + 1; i < input.dim(); i++) {
    axis_block_size *= input.size(i);
  }
  const int64_t axis_size = input.size(axis);

  // Try ARM NEON optimized path for float->int8/uint8 quantization
#if defined(__aarch64__) || defined(__ARM_NEON__)
  if (input.scalar_type() == ScalarType::Float) {
    const int64_t num_blocks = input.numel() / axis_block_size;
    const int64_t total_elements = input.numel();
    constexpr int64_t MIN_ELEMENTS_FOR_PARALLEL = 512;
    const bool use_parallel = (total_elements >= MIN_ELEMENTS_FOR_PARALLEL);

    if (dtype == ScalarType::Byte) {
      auto* out_data_ptr = out.mutable_data_ptr<uint8_t>();
      const auto* input_data_ptr = input.const_data_ptr<float>();

      if (use_parallel) {
        ::executorch::extension::parallel_for(
            0, num_blocks, 1, [&](const int64_t begin, const int64_t end) {
              for (int64_t block = begin; block < end; ++block) {
                int64_t channel_idx = block % axis_size;
                float inv_scale =
                    1.0f / static_cast<float>(scale_data[channel_idx]);
                int32_t zp = static_cast<int32_t>(zero_point_data[channel_idx]);

                const float* in_ptr = input_data_ptr + block * axis_block_size;
                uint8_t* out_ptr = out_data_ptr + block * axis_block_size;

                quantize_arm<uint8_t>(
                    in_ptr,
                    out_ptr,
                    axis_block_size,
                    inv_scale,
                    zp,
                    static_cast<int32_t>(quant_min),
                    static_cast<int32_t>(quant_max));
              }
            });
      } else {
        // Process each contiguous block (which shares the same
        // scale/zero_point)
        for (int64_t block = 0; block < num_blocks; ++block) {
          int64_t channel_idx = block % axis_size;
          float inv_scale = 1.0f / static_cast<float>(scale_data[channel_idx]);
          int32_t zp = static_cast<int32_t>(zero_point_data[channel_idx]);

          const float* in_ptr = input_data_ptr + block * axis_block_size;
          uint8_t* out_ptr = out_data_ptr + block * axis_block_size;

          quantize_arm<uint8_t>(
              in_ptr,
              out_ptr,
              axis_block_size,
              inv_scale,
              zp,
              static_cast<int32_t>(quant_min),
              static_cast<int32_t>(quant_max));
        }
      }
      return out;
    } else if (dtype == ScalarType::Char) {
      auto* out_data_ptr = out.mutable_data_ptr<int8_t>();
      const auto* input_data_ptr = input.const_data_ptr<float>();

      if (use_parallel) {
        ::executorch::extension::parallel_for(
            0, num_blocks, 1, [&](const int64_t begin, const int64_t end) {
              for (int64_t block = begin; block < end; ++block) {
                int64_t channel_idx = block % axis_size;
                float inv_scale =
                    1.0f / static_cast<float>(scale_data[channel_idx]);
                int32_t zp = static_cast<int32_t>(zero_point_data[channel_idx]);

                const float* in_ptr = input_data_ptr + block * axis_block_size;
                int8_t* out_ptr = out_data_ptr + block * axis_block_size;

                quantize_arm<int8_t>(
                    in_ptr,
                    out_ptr,
                    axis_block_size,
                    inv_scale,
                    zp,
                    static_cast<int32_t>(quant_min),
                    static_cast<int32_t>(quant_max));
              }
            });
      } else {
        // Process each contiguous block (which shares the same
        // scale/zero_point)
        for (int64_t block = 0; block < num_blocks; ++block) {
          int64_t channel_idx = block % axis_size;
          float inv_scale = 1.0f / static_cast<float>(scale_data[channel_idx]);
          int32_t zp = static_cast<int32_t>(zero_point_data[channel_idx]);

          const float* in_ptr = input_data_ptr + block * axis_block_size;
          int8_t* out_ptr = out_data_ptr + block * axis_block_size;

          quantize_arm<int8_t>(
              in_ptr,
              out_ptr,
              axis_block_size,
              inv_scale,
              zp,
              static_cast<int32_t>(quant_min),
              static_cast<int32_t>(quant_max));
        }
      }
      return out;
    }
  }
#endif

  // Fallback scalar implementation
#define QUANTIZE_IMPL(CTYPE_IN, CTYPE_OUT, out_dtype)                    \
  case ScalarType::out_dtype: {                                          \
    auto* out_data_ptr = out.mutable_data_ptr<CTYPE_OUT>();              \
    const auto* input_data_ptr = input.const_data_ptr<CTYPE_IN>();       \
    const int64_t input_numel = input.numel();                           \
    /* Single loop over all elements */                                  \
    for (int64_t i = 0; i < input_numel; i++) {                          \
      /* Calculate which channel this element belongs to */              \
      int64_t channel_idx = (i / axis_block_size) % axis_size;           \
      /* Get quantization parameters for this channel */                 \
      double _scale = scale_data[channel_idx];                           \
      int64_t _zero_point = zero_point_data[channel_idx];                \
      /* Apply quantization */                                           \
      out_data_ptr[i] = quantize_val<CTYPE_OUT, CTYPE_IN>(               \
          _scale, _zero_point, input_data_ptr[i], quant_min, quant_max); \
    }                                                                    \
  } break;

#define CALCULATE_FLOAT_TYPE(CTYPE_IN, in_dtype)         \
  case ScalarType::in_dtype:                             \
    switch (out.scalar_type()) {                         \
      ET_FORALL_INT_TYPES_WITH(CTYPE_IN, QUANTIZE_IMPL); \
      QUANTIZE_IMPL(CTYPE_IN, uint16_t, Bits16)          \
      QUANTIZE_IMPL(CTYPE_IN, uint16_t, UInt16)          \
      default:                                           \
        ET_CHECK_MSG(                                    \
            false,                                       \
            "Unhandled output dtype %" PRId8,            \
            static_cast<int8_t>(out.scalar_type()));     \
    }                                                    \
    break;

  switch (input.scalar_type()) {
    ET_FORALL_FLOATH_TYPES(CALCULATE_FLOAT_TYPE);
    default:
      ET_CHECK_MSG(
          false,
          "Unhandled input dtype %" PRId8,
          static_cast<int8_t>(input.scalar_type()));
  }
#undef CALCULATE_FLOAT_TYPE
#undef QUANTIZE_IMPL

  return out;
}

Tensor& quantize_per_channel_out(
    KernelRuntimeContext& context,
    const Tensor& input,
    const Tensor& scale,
    const Tensor& zero_point,
    int64_t axis,
    int64_t quant_min,
    int64_t quant_max,
    ScalarType dtype,
    Tensor& out) {
  (void)context;
  torch::executor::Error err = resize_tensor(out, input.sizes());
  ET_CHECK_MSG(
      err == torch::executor::Error::Ok,
      "Failed to resize out Tensor in quantize_per_channel_out");

  return quantize_per_channel_out(
      input, scale, zero_point, axis, quant_min, quant_max, dtype, out);
}

Tensor& quantize_per_token_out(
    const Tensor& input,
    const Tensor& scale,
    const Tensor& zero_point,
    int64_t quant_min,
    int64_t quant_max,
    ScalarType dtype,
    Tensor& out) {
  size_t num_tokens = 1;
  for (size_t i = 0; i < input.dim() - 1; i++) {
    num_tokens *= input.size(i);
  }
// This unfortunate change is needed because we compile op_quantize for aten
// mode as well
#ifdef USE_ATEN_LIB
  std::vector<int64_t> sizes(2);
  sizes[0] = num_tokens;
  sizes[1] = input.size(input.dim() - 1);
  Tensor reshaped_input = at::from_blob(
      input.mutable_data_ptr(), sizes, at::TensorOptions(input.scalar_type()));
#else
  std::array<executorch::aten::DimOrderType, 2> input_dim_order{0, 1};
  std::array<executorch::aten::SizesType, 2> input_sizes;
  input_sizes[0] = num_tokens;
  input_sizes[1] = input.size(input.dim() - 1);
  std::array<executorch::aten::StridesType, 2> input_strides;
  dim_order_to_stride_nocheck(
      input_sizes.data(), input_dim_order.data(), 2, input_strides.data());
  void* input_data = input.mutable_data_ptr();
  TensorImpl reshaped_input_impl = TensorImpl(
      input.scalar_type(),
      2,
      input_sizes.data(),
      input_data,
      input_dim_order.data(),
      input_strides.data(),
      TensorShapeDynamism::STATIC);
  Tensor reshaped_input(&reshaped_input_impl);
  torch::executor::Error err = resize_tensor(out, input.sizes());
  ET_CHECK_MSG(
      err == torch::executor::Error::Ok,
      "Failed to resize out Tensor in quantize_per_channel_out");
#endif

  return quantize_per_channel_out(
      reshaped_input, scale, zero_point, 0, quant_min, quant_max, dtype, out);
}

Tensor& quantize_per_token_out(
    RuntimeContext& context,
    const Tensor& input,
    const Tensor& scale,
    const Tensor& zero_point,
    int64_t quant_min,
    int64_t quant_max,
    ScalarType dtype,
    Tensor& out) {
  (void)context;
  return quantize_per_token_out(
      input, scale, zero_point, quant_min, quant_max, dtype, out);
}
} // namespace native
} // namespace executor
} // namespace torch
