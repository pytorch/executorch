/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/backends/cadence/fusion_g3/operators/operators.h>

#include <algorithm>
#include <cinttypes>
#include <cmath>

#include <xa_nnlib_kernels_api.h>

#include <executorch/backends/cadence/common/xt_macros.h>
#include <executorch/kernels/portable/cpu/util/reduce_util.h>
#include <executorch/runtime/kernel/kernel_includes.h>

using ::executorch::aten::ScalarType;
using ::executorch::aten::Tensor;
using ::executorch::runtime::Error;
using ::executorch::runtime::KernelRuntimeContext;

template <typename T>
using optional = std::optional<T>;
/* ScalarType in Executorch do not have support for below data types.
 * So, creating a placeholder for these data types. Once, ScalarTypes is
 * updated to have support for below data types, these can be removed and
 * operator need to be updated accordingly
 */

#ifdef G3_ENABLE_4BIT_QUANTIZATION
enum datatype { Bits4u = 21, Bits4 = 22 };
#endif

/**
 * For an input tensor, use the scale and zero_point arguments to quantize it.
 */
namespace impl {
namespace G3 {
namespace native {

namespace {

/**
 * Asserts that the parameters are valid.
 */
[[maybe_unused]] void check_dequantize_per_tensor_args(
    const Tensor& input,
    int64_t quant_min,
    int64_t quant_max,
    ScalarType dtype,
    std::optional<ScalarType>& out_dtype,
    Tensor& out) {
  ET_CHECK_MSG(
      input.scalar_type() == ScalarType::Byte ||
          input.scalar_type() == ScalarType::Char ||
          input.scalar_type() == ScalarType::UInt16 ||
          input.scalar_type() == ScalarType::Short ||
#ifdef G3_ENABLE_4BIT_QUANTIZATION
          input.scalar_type() == (ScalarType)Bits4 ||
          input.scalar_type() == (ScalarType)Bits4u ||
#endif
          input.scalar_type() == ScalarType::Int,

      "input.scalar_type() %" PRId8 " is not supported:",
      static_cast<int8_t>(input.scalar_type()));

  ET_CHECK_MSG(
      input.scalar_type() == dtype,
      "input.scalar_type() %s is not matching dtype arguments:",
      ::executorch::runtime::toString(input.scalar_type()));

  if (out_dtype.has_value()) {
    ET_CHECK_MSG(
        out.scalar_type() == out_dtype.value(),
        "output_dtype must match the dtype of the out tensor");
  }

  ET_CHECK_MSG(
      quant_min <= quant_max,
      "quant min: %" PRId64 " is greater than quant max: %" PRId64,
      quant_min,
      quant_max);
}

// PDX aligned vector load/store (PDX_LV*) require 16-byte (128-bit) aligned
// buffers on Fusion G3; an unaligned access raises LoadStoreAlignmentCause. The
// inline SIMD fast path below is only safe when both buffers are aligned;
// otherwise the caller falls back to the alignment-safe NNLib kernel.
[[maybe_unused]] constexpr uintptr_t kSimdAlignmentBytes = 16;
[[maybe_unused]] inline bool dequant_simd_aligned(
    const void* in_ptr,
    const void* out_ptr) {
  return (reinterpret_cast<uintptr_t>(in_ptr) % kSimdAlignmentBytes == 0) &&
      (reinterpret_cast<uintptr_t>(out_ptr) % kSimdAlignmentBytes == 0);
}

} // namespace

/* Local function which calls the kernels based on the input datatype */
Tensor& dequantize_impl(
    KernelRuntimeContext& ctx,
    Tensor& out,
    const Tensor& input,
    float* scale_data,
    int* zero_point_data,
    int* axis,
    std::optional<ScalarType> out_dtype) {
  const ::executorch::aten::ArrayRef<Tensor::SizesType> input_size =
      input.sizes();

  int kTensorDimensionLimit = 5;

  int inp_shape[kTensorDimensionLimit];

  for (auto i = 0; i < input_size.size(); i++) {
    inp_shape[i] = input_size[i];
  }

  bool is_asym_dequant = 0;

  if (zero_point_data != NULL) // asymmetric dequant
  {
    if (axis != NULL) // channel
    {
      for (int i = 0; i < input.size(*axis); i++) {
        if (zero_point_data[i] != 0) {
          is_asym_dequant |= 1;
        }
      }
    } else {
      if (*zero_point_data != 0) // tensor
      {
        is_asym_dequant |= 1;
      }
    }
  }
  float* out_data = out.mutable_data_ptr<float>();

  bool optimized = true;

  if (out.scalar_type() != ScalarType::Float) {
    optimized = false;
  }

  if (is_asym_dequant) {
    if ((input.scalar_type() == ScalarType::Byte) && (optimized)) {
      const uint8_t* input_data = input.const_data_ptr<uint8_t>();
      XT_KERNEL_CHECK(
          ctx,
          out,
          xa_nn_elm_dequantize_asym8u_f32,
          out_data,
          input_data,
          inp_shape,
          input.dim(),
          axis,
          zero_point_data,
          scale_data);
    } else if ((input.scalar_type() == ScalarType::Char) && (optimized)) {
      const int8_t* input_data = input.const_data_ptr<int8_t>();
      XT_KERNEL_CHECK(
          ctx,
          out,
          xa_nn_elm_dequantize_asym8_f32,
          out_data,
          input_data,
          inp_shape,
          input.dim(),
          axis,
          zero_point_data,
          scale_data);
    } else if ((input.scalar_type() == ScalarType::UInt16) && (optimized)) {
      const uint16_t* input_data = input.const_data_ptr<uint16_t>();
      XT_KERNEL_CHECK(
          ctx,
          out,
          xa_nn_elm_dequantize_asym16u_f32,
          out_data,
          input_data,
          inp_shape,
          input.dim(),
          axis,
          zero_point_data,
          scale_data);
    } else if ((input.scalar_type() == ScalarType::Short) && (optimized)) {
      const int16_t* input_data = input.const_data_ptr<int16_t>();
      XT_KERNEL_CHECK(
          ctx,
          out,
          xa_nn_elm_dequantize_asym16_f32,
          out_data,
          input_data,
          inp_shape,
          input.dim(),
          axis,
          zero_point_data,
          scale_data);
#ifdef G3_ENABLE_4BIT_QUANTIZATION
    } else if ((input.scalar_type() == (ScalarType)Bits4u) && (optimized)) {
      const uint8_t* input_data = input.const_data_ptr<uint8_t>();
      XT_KERNEL_CHECK(
          ctx,
          out,
          xa_nn_elm_dequantize_asym4u_f32,
          out_data,
          input_data,
          inp_shape,
          input.dim(),
          axis,
          zero_point_data,
          scale_data);
    } else if ((input.scalar_type() == (ScalarType)Bits4) && (optimized)) {
      const int8_t* input_data = input.const_data_ptr<int8_t>();
      XT_KERNEL_CHECK(
          ctx,
          out,
          xa_nn_elm_dequantize_asym4_f32,
          out_data,
          input_data,
          inp_shape,
          input.dim(),
          axis,
          zero_point_data,
          scale_data);
#endif
    } else {
      if (axis == NULL) {
// calculate the dequantized output, cast scale to float to match fbgemm
// behavior
#define ASYM_DEQUANTIZE_IMPL_TENSOR(IN_CTYPE, OUT_CTYPE, out_dtype)            \
  case ScalarType::out_dtype: {                                                \
    /* Hoist these function calls out of our inner loop because they might not \
     * get inlined without LTO, particularly in ATen mode. */                  \
    auto* out_data_ptr = out.mutable_data_ptr<OUT_CTYPE>();                    \
    const auto* input_data_ptr = input.const_data_ptr<IN_CTYPE>();             \
    const auto input_numel = input.numel();                                    \
    for (size_t i = 0; i < input_numel; i++) {                                 \
      out_data_ptr[i] = static_cast<OUT_CTYPE>(                                \
          (input_data_ptr[i] - static_cast<int32_t>(*zero_point_data)) *       \
          static_cast<float>(*scale_data));                                    \
    }                                                                          \
  } break;
#define ASYM_CALCULATE_INT_TYPE_TENSOR(IN_CTYPE, in_dtype)               \
  case ScalarType::in_dtype:                                             \
    switch (out.scalar_type()) {                                         \
      ET_FORALL_FLOAT_TYPES_WITH(IN_CTYPE, ASYM_DEQUANTIZE_IMPL_TENSOR); \
      default:                                                           \
        ET_CHECK_MSG(                                                    \
            false,                                                       \
            "Unhandled output dtype %" PRId8,                            \
            static_cast<int8_t>(out.scalar_type()));                     \
    }                                                                    \
    break;
        switch (input.scalar_type()) {
          ET_FORALL_INT_TYPES(ASYM_CALCULATE_INT_TYPE_TENSOR);
          ASYM_CALCULATE_INT_TYPE_TENSOR(uint16_t, UInt16);
          default:
            ET_CHECK_MSG(
                false,
                "Unhandled input dtype %" PRId8,
                static_cast<int8_t>(input.scalar_type()));
        }
#undef ASYM_CALCULATE_INT_TYPE_TENSOR
#undef ASYM_DEQUANTIZE_IMPL_TENSOR
      } else {
        // a list contains all dimensions except axis
        int64_t dims[input.dim() - 1];
        for (int64_t i = 0; i < input.dim() - 1; i++) {
          if (i < *axis) {
            dims[i] = i;
          } else {
            dims[i] = i + 1;
          }
        }

        std::optional<::executorch::aten::ArrayRef<int64_t>> optional_dim_list{
            ::executorch::aten::ArrayRef<int64_t>{
                dims, size_t(input.dim() - 1)}};

// Actual dequantization logic
// input, out are the input and output tensors
// channel_ix is the index along the axis dimension. 0 <= channel_ix <
// input.size(axis).
//   i.e. if the tensor has shape (N,C,H,W), axis being 1, then channel_ix
//   will be 0, 1, 2, ... C-1
// in_ix is the flat index of the element you are dequantizing.
//   in other words you are dequantizing in_data[in_ix]
#define ASYM_DEQUANTIZE_IMPL_CHANNEL(CTYPE_IN, CTYPE_OUT, out_dtype)          \
  case ScalarType::out_dtype:                                                 \
    if (input.dim() == 1) {                                                   \
      auto* out_data_ptr = out.mutable_data_ptr<CTYPE_OUT>();                 \
      const auto* input_data_ptr = input.const_data_ptr<CTYPE_IN>();          \
      ET_CHECK_MSG(                                                           \
          *axis == 0, "Axis must be 0 for a single dimensional tensors");     \
      const optional<int64_t> dim;                                            \
      torch::executor::apply_over_dim(                                        \
          [input_data_ptr, out_data_ptr, zero_point_data, scale_data](        \
              size_t numel, size_t stride, size_t base_ix) {                  \
            for (size_t i = 0; i < numel; i++) {                              \
              size_t current_ix = base_ix * stride + i;                       \
              float _scale = scale_data[current_ix];                          \
              int64_t zero_point = 0;                                         \
              if (zero_point_data != nullptr) {                               \
                zero_point = zero_point_data[current_ix];                     \
              }                                                               \
              out_data_ptr[current_ix] = static_cast<CTYPE_OUT>(              \
                  (input_data_ptr[current_ix] - zero_point) * _scale);        \
            }                                                                 \
          },                                                                  \
          input,                                                              \
          dim);                                                               \
      break;                                                                  \
    }                                                                         \
    for (size_t channel_ix = 0; channel_ix < input.size(*axis);               \
         ++channel_ix) {                                                      \
      float _scale = scale_data[channel_ix];                                  \
      int64_t _zero_point = 0;                                                \
      if (zero_point_data != nullptr) {                                       \
        _zero_point = zero_point_data[channel_ix];                            \
      }                                                                       \
      auto* out_data_ptr = out.mutable_data_ptr<CTYPE_OUT>();                 \
      const auto* input_data_ptr = input.const_data_ptr<CTYPE_IN>();          \
      torch::executor::apply_over_dim_list(                                   \
          [input_data_ptr, out_data_ptr, _scale, _zero_point](size_t in_ix) { \
            out_data_ptr[in_ix] = static_cast<CTYPE_OUT>(                     \
                (input_data_ptr[in_ix] - _zero_point) * _scale);              \
          },                                                                  \
          input,                                                              \
          optional_dim_list,                                                  \
          channel_ix);                                                        \
    }                                                                         \
    break;
#define ASYM_CALCULATE_INT_TYPE_CHANNEL(IN_CTYPE, in_dtype)               \
  case ScalarType::in_dtype:                                              \
    switch (out.scalar_type()) {                                          \
      ET_FORALL_FLOAT_TYPES_WITH(IN_CTYPE, ASYM_DEQUANTIZE_IMPL_CHANNEL); \
      default:                                                            \
        ET_CHECK_MSG(                                                     \
            false,                                                        \
            "Unhandled output dtype %" PRId8,                             \
            static_cast<int8_t>(out.scalar_type()));                      \
    }                                                                     \
    break;
        switch (input.scalar_type()) {
          ET_FORALL_INT_TYPES(ASYM_CALCULATE_INT_TYPE_CHANNEL);
          ASYM_CALCULATE_INT_TYPE_CHANNEL(uint16_t, UInt16);
          default:
            ET_CHECK_MSG(
                false,
                "Unhandled input dtype %" PRId8,
                static_cast<int8_t>(input.scalar_type()));
        }
#undef ASYM_CALCULATE_INT_TYPE_CHANNEL
#undef ASYM_DEQUANTIZE_IMPL_CHANNEL
      }
    }
  } else {
    if ((input.scalar_type() == ScalarType::Byte) && (optimized)) {
      const uint8_t* input_data = input.const_data_ptr<uint8_t>();
      XT_KERNEL_CHECK(
          ctx,
          out,
          xa_nn_elm_dequantize_sym8u_f32,
          out_data,
          input_data,
          inp_shape,
          input.dim(),
          axis,
          scale_data);
    } else if ((input.scalar_type() == ScalarType::Char) && (optimized)) {
      const int8_t* input_data = input.const_data_ptr<int8_t>();
      XT_KERNEL_CHECK(
          ctx,
          out,
          xa_nn_elm_dequantize_sym8_f32,
          out_data,
          input_data,
          inp_shape,
          input.dim(),
          axis,
          scale_data);
    } else if ((input.scalar_type() == ScalarType::UInt16) && (optimized)) {
      const uint16_t* input_data = input.const_data_ptr<uint16_t>();
      XT_KERNEL_CHECK(
          ctx,
          out,
          xa_nn_elm_dequantize_sym16u_f32,
          out_data,
          input_data,
          inp_shape,
          input.dim(),
          axis,
          scale_data);
    } else if ((input.scalar_type() == ScalarType::Short) && (optimized)) {
      const int16_t* input_data = input.const_data_ptr<int16_t>();
      XT_KERNEL_CHECK(
          ctx,
          out,
          xa_nn_elm_dequantize_sym16_f32,
          out_data,
          input_data,
          inp_shape,
          input.dim(),
          axis,
          scale_data);
#ifdef G3_ENABLE_4BIT_QUANTIZATION
    } else if ((input.scalar_type() == (ScalarType)Bits4u) && (optimized)) {
      const uint8_t* input_data = input.const_data_ptr<uint8_t>();
      XT_KERNEL_CHECK(
          ctx,
          out,
          xa_nn_elm_dequantize_sym4u_f32,
          out_data,
          input_data,
          inp_shape,
          input.dim(),
          axis,
          scale_data);
    } else if ((input.scalar_type() == (ScalarType)Bits4) && (optimized)) {
      const int8_t* input_data = input.const_data_ptr<int8_t>();
      XT_KERNEL_CHECK(
          ctx,
          out,
          xa_nn_elm_dequantize_sym4_f32,
          out_data,
          input_data,
          inp_shape,
          input.dim(),
          axis,
          scale_data);
#endif
    } else {
      if (axis == NULL) {
// calculate the dequantized output, cast scale to float to match fbgemm
// behavior
#define SYM_DEQUANTIZE_IMPL_TESNOR(IN_CTYPE, OUT_CTYPE, out_dtype)             \
  case ScalarType::out_dtype: {                                                \
    /* Hoist these function calls out of our inner loop because they might not \
     * get inlined without LTO, particularly in ATen mode. */                  \
    auto* out_data_ptr = out.mutable_data_ptr<OUT_CTYPE>();                    \
    const auto* input_data_ptr = input.const_data_ptr<IN_CTYPE>();             \
    const auto input_numel = input.numel();                                    \
    for (size_t i = 0; i < input_numel; i++) {                                 \
      out_data_ptr[i] = static_cast<OUT_CTYPE>(                                \
          (input_data_ptr[i] - static_cast<int32_t>(*zero_point_data)) *       \
          static_cast<float>(*scale_data));                                    \
    }                                                                          \
  } break;
#define SYM_CALCULATE_INT_TYPE_TENSOR(IN_CTYPE, in_dtype)               \
  case ScalarType::in_dtype:                                            \
    switch (out.scalar_type()) {                                        \
      ET_FORALL_FLOAT_TYPES_WITH(IN_CTYPE, SYM_DEQUANTIZE_IMPL_TESNOR); \
      default:                                                          \
        ET_CHECK_MSG(                                                   \
            false,                                                      \
            "Unhandled output dtype %" PRId8,                           \
            static_cast<int8_t>(out.scalar_type()));                    \
    }                                                                   \
    break;
        switch (input.scalar_type()) {
          ET_FORALL_INT_TYPES(SYM_CALCULATE_INT_TYPE_TENSOR);
          SYM_CALCULATE_INT_TYPE_TENSOR(uint16_t, UInt16);
          default:
            ET_CHECK_MSG(
                false,
                "Unhandled input dtype %" PRId8,
                static_cast<int8_t>(input.scalar_type()));
        }
#undef SYM_DEQUANTIZE_IMPL_TESNOR
#undef SYM_CALCULATE_INT_TYPE_TENSOR
      } else {
        // a list contains all dimensions except axis
        int64_t dims[input.dim() - 1];
        for (int64_t i = 0; i < input.dim() - 1; i++) {
          if (i < *axis) {
            dims[i] = i;
          } else {
            dims[i] = i + 1;
          }
        }

        std::optional<::executorch::aten::ArrayRef<int64_t>> optional_dim_list{
            ::executorch::aten::ArrayRef<int64_t>{
                dims, size_t(input.dim() - 1)}};

// Actual dequantization logic
// input, out are the input and output tensors
// channel_ix is the index along the axis dimension. 0 <= channel_ix <
// input.size(axis).
//   i.e. if the tensor has shape (N,C,H,W), axis being 1, then channel_ix
//   will be 0, 1, 2, ... C-1
// in_ix is the flat index of the element you are dequantizing.
//   in other words you are dequantizing in_data[in_ix]
#define SYM_DEQUANTIZE_IMPL_CHANNEL(CTYPE_IN, CTYPE_OUT, out_dtype)           \
  case ScalarType::out_dtype:                                                 \
    if (input.dim() == 1) {                                                   \
      auto* out_data_ptr = out.mutable_data_ptr<CTYPE_OUT>();                 \
      const auto* input_data_ptr = input.const_data_ptr<CTYPE_IN>();          \
      ET_CHECK_MSG(                                                           \
          *axis == 0, "Axis must be 0 for a single dimensional tensors");     \
      const optional<int64_t> dim;                                            \
      torch::executor::apply_over_dim(                                        \
          [input_data_ptr, out_data_ptr, zero_point_data, scale_data](        \
              size_t numel, size_t stride, size_t base_ix) {                  \
            for (size_t i = 0; i < numel; i++) {                              \
              size_t current_ix = base_ix * stride + i;                       \
              float _scale = scale_data[current_ix];                          \
              int64_t zero_point = 0;                                         \
              if (zero_point_data != nullptr) {                               \
                zero_point = zero_point_data[current_ix];                     \
              }                                                               \
              out_data_ptr[current_ix] = static_cast<CTYPE_OUT>(              \
                  (input_data_ptr[current_ix] - zero_point) * _scale);        \
            }                                                                 \
          },                                                                  \
          input,                                                              \
          dim);                                                               \
      break;                                                                  \
    }                                                                         \
    for (size_t channel_ix = 0; channel_ix < input.size(*axis);               \
         ++channel_ix) {                                                      \
      float _scale = scale_data[channel_ix];                                  \
      int64_t _zero_point = 0;                                                \
      if (zero_point_data != nullptr) {                                       \
        _zero_point = zero_point_data[channel_ix];                            \
      }                                                                       \
      auto* out_data_ptr = out.mutable_data_ptr<CTYPE_OUT>();                 \
      const auto* input_data_ptr = input.const_data_ptr<CTYPE_IN>();          \
      torch::executor::apply_over_dim_list(                                   \
          [input_data_ptr, out_data_ptr, _scale, _zero_point](size_t in_ix) { \
            out_data_ptr[in_ix] = static_cast<CTYPE_OUT>(                     \
                (input_data_ptr[in_ix] - _zero_point) * _scale);              \
          },                                                                  \
          input,                                                              \
          optional_dim_list,                                                  \
          channel_ix);                                                        \
    }                                                                         \
    break;
#define SYM_CALCULATE_INT_TYPE_CHANNEL(IN_CTYPE, in_dtype)               \
  case ScalarType::in_dtype:                                             \
    switch (out.scalar_type()) {                                         \
      ET_FORALL_FLOAT_TYPES_WITH(IN_CTYPE, SYM_DEQUANTIZE_IMPL_CHANNEL); \
      default:                                                           \
        ET_CHECK_MSG(                                                    \
            false,                                                       \
            "Unhandled output dtype %" PRId8,                            \
            static_cast<int8_t>(out.scalar_type()));                     \
    }                                                                    \
    break;
        switch (input.scalar_type()) {
          ET_FORALL_INT_TYPES(SYM_CALCULATE_INT_TYPE_CHANNEL);
          SYM_CALCULATE_INT_TYPE_CHANNEL(uint16_t, UInt16);
          default:
            ET_CHECK_MSG(
                false,
                "Unhandled input dtype %" PRId8,
                static_cast<int8_t>(input.scalar_type()));
        }
#undef SYM_DEQUANTIZE_IMPL_CHANNEL
#undef SYM_CALCULATE_INT_TYPE_CHANNEL
      }
    }
  }
  return out;
}

/**
 * Dequantizes the input tensor according to the formula (input - zero_point) *
 * scale
 *
 * NOTE: quant_min and quant_max are not used in computation, but rather
 * metadata that is passed around which can be useful for pattern matching. See
 * https://github.com/pytorch/pytorch/pull/87093#discussion_r1000841181 for more
 * info.
 */
Tensor& dequantize_per_tensor_out(
    KernelRuntimeContext& context,
    const Tensor& input,
    double scale,
    int64_t zero_point,
    __ET_UNUSED int64_t quant_min,
    __ET_UNUSED int64_t quant_max,
    ScalarType dtype,
    Tensor& out) {
  constexpr ScalarType out_dtype = ScalarType::Float;
#ifdef OP_ARG_CHECK
  torch::executor::Error err = resize_tensor(out, input.sizes());
  ET_CHECK_MSG(
      err == torch::executor::Error::Ok,
      "Failed to resize out Tensor in dequantize_per_tensor_out");
  check_dequantize_per_tensor_args(
      input, quant_min, quant_max, dtype, out_dtype, out);
#endif
  float scale_data = (float)scale;
  int zero_point_data = (int)zero_point;
  // Fast path: bypass dequantize_impl dispatch overhead for the most common
  // per-tensor cases (int8/uint8 -> float32). This avoids the inp_shape copy
  // loop, is_asym_dequant check, optimized flag check, and the multi-branch
  // type dispatch in dequantize_impl.
  if (out.scalar_type() == ScalarType::Float) {
    float* __restrict__ out_data = out.mutable_data_ptr<float>();
    // For per-tensor dequantization (axis=NULL), flatten to 1D shape.
    // This avoids the shape copy loop and lets the NNLIB kernel skip
    // multi-dimensional stride calculations internally.
    int inp_shape[1] = {static_cast<int>(input.numel())};
    constexpr int num_inp_dims = 1;
    if (input.scalar_type() == ScalarType::Char) {
      const int8_t* __restrict__ input_data = input.const_data_ptr<int8_t>();
#if defined(__XTENSA__)
      // Direct inline PDX SIMD dequantization for per-tensor int8->float32.
      // The zero-point subtract is kept in the int32 domain (PDX_SUB_MX32) so
      // it issues on the integer unit and overlaps the float pipe; the widened
      // int32 codes feed straight into the mixed-type PDX_MUL_MXF32. That
      // intrinsic takes an xb_vecMxf32, so the compiler still emits an
      // int->float convert -- writing it mixed-type only keeps the source
      // concise, it does not remove the convert. q - zp cannot overflow int32
      // (inputs are <=16-bit and zp is in the same quant range) and converts
      // exactly to float, so the result is bit-identical to
      // float(q) - float(zp). 4x-unrolled to hide load-to-use latency.
      if (dequant_simd_aligned(input_data, out_data)) {
        const int numel = inp_shape[0];
        auto vIn = reinterpret_cast<const xb_vecMx8*>(input_data);
        auto vOut = reinterpret_cast<xb_vecMxf32*>(out_data);
        const xb_vecMxf32 v_scale{
            scale_data, scale_data, scale_data, scale_data};
        int i = 0;
        // 4x unrolled main loop: 16 elements per iteration
        const int e16 = (numel >> 4) << 4;
        if (zero_point_data != 0) {
          const xb_vecMx32 v_zp{
              zero_point_data,
              zero_point_data,
              zero_point_data,
              zero_point_data};
          const float zp_f = static_cast<float>(zero_point_data);
          for (; i < e16; i += 16) {
            xb_vecMx32 vP0 = PDX_LV32_MX8_I(vIn, 0);
            xb_vecMx32 vP1 = PDX_LV32_MX8_I(vIn + 1, 0);
            xb_vecMx32 vP2 = PDX_LV32_MX8_I(vIn + 2, 0);
            xb_vecMx32 vP3 = PDX_LV32_MX8_I(vIn + 3, 0);
            vOut[0] = PDX_MUL_MXF32(PDX_SUB_MX32(vP0, v_zp), v_scale);
            vOut[1] = PDX_MUL_MXF32(PDX_SUB_MX32(vP1, v_zp), v_scale);
            vOut[2] = PDX_MUL_MXF32(PDX_SUB_MX32(vP2, v_zp), v_scale);
            vOut[3] = PDX_MUL_MXF32(PDX_SUB_MX32(vP3, v_zp), v_scale);
            vIn += 4;
            vOut += 4;
          }
          // Scalar residual for remaining 0-15 elements
          for (; i < numel; ++i) {
            out_data[i] =
                (static_cast<float>(input_data[i]) - zp_f) * scale_data;
          }
        } else {
          for (; i < e16; i += 16) {
            xb_vecMx32 vP0 = PDX_LV32_MX8_I(vIn, 0);
            xb_vecMx32 vP1 = PDX_LV32_MX8_I(vIn + 1, 0);
            xb_vecMx32 vP2 = PDX_LV32_MX8_I(vIn + 2, 0);
            xb_vecMx32 vP3 = PDX_LV32_MX8_I(vIn + 3, 0);
            vOut[0] = PDX_MUL_MXF32(vP0, v_scale);
            vOut[1] = PDX_MUL_MXF32(vP1, v_scale);
            vOut[2] = PDX_MUL_MXF32(vP2, v_scale);
            vOut[3] = PDX_MUL_MXF32(vP3, v_scale);
            vIn += 4;
            vOut += 4;
          }
          // Scalar residual for remaining 0-15 elements
          for (; i < numel; ++i) {
            out_data[i] = static_cast<float>(input_data[i]) * scale_data;
          }
        }
        return out;
      }
#endif
      if (zero_point_data != 0) {
        XT_KERNEL_CHECK(
            context,
            out,
            xa_nn_elm_dequantize_asym8_f32,
            out_data,
            input_data,
            inp_shape,
            num_inp_dims,
            NULL,
            &zero_point_data,
            &scale_data);
      } else {
        XT_KERNEL_CHECK(
            context,
            out,
            xa_nn_elm_dequantize_sym8_f32,
            out_data,
            input_data,
            inp_shape,
            num_inp_dims,
            NULL,
            &scale_data);
      }
      return out;
    } else if (input.scalar_type() == ScalarType::Byte) {
      const uint8_t* __restrict__ input_data = input.const_data_ptr<uint8_t>();
#if defined(__XTENSA__)
      // Direct inline PDX SIMD dequantization for per-tensor uint8->float32.
      // The zero-point subtract is kept in the int32 domain (PDX_SUB_MX32) so
      // it issues on the integer unit and overlaps the float pipe; the widened
      // int32 codes feed straight into the mixed-type PDX_MUL_MXF32. That
      // intrinsic takes an xb_vecMxf32, so the compiler still emits an
      // int->float convert -- writing it mixed-type only keeps the source
      // concise, it does not remove the convert. q - zp cannot overflow int32
      // (inputs are <=16-bit and zp is in the same quant range) and converts
      // exactly to float, so the result is bit-identical to
      // float(q) - float(zp). 4x-unrolled to hide load-to-use latency.
      if (dequant_simd_aligned(input_data, out_data)) {
        const int numel = inp_shape[0];
        auto vIn = reinterpret_cast<const xb_vecMxu8*>(input_data);
        auto vOut = reinterpret_cast<xb_vecMxf32*>(out_data);
        const xb_vecMxf32 v_scale{
            scale_data, scale_data, scale_data, scale_data};
        int i = 0;
        // 4x unrolled main loop: 16 elements per iteration
        const int e16 = (numel >> 4) << 4;
        if (zero_point_data != 0) {
          const xb_vecMxu32 v_zp{
              static_cast<uint32_t>(zero_point_data),
              static_cast<uint32_t>(zero_point_data),
              static_cast<uint32_t>(zero_point_data),
              static_cast<uint32_t>(zero_point_data)};
          const float zp_f = static_cast<float>(zero_point_data);
          for (; i < e16; i += 16) {
            xb_vecMxu32 vP0 = PDX_LVU32_MX8_I(vIn, 0);
            xb_vecMxu32 vP1 = PDX_LVU32_MX8_I(vIn + 1, 0);
            xb_vecMxu32 vP2 = PDX_LVU32_MX8_I(vIn + 2, 0);
            xb_vecMxu32 vP3 = PDX_LVU32_MX8_I(vIn + 3, 0);
            vOut[0] = PDX_MUL_MXF32(PDX_SUB_MX32(vP0, v_zp), v_scale);
            vOut[1] = PDX_MUL_MXF32(PDX_SUB_MX32(vP1, v_zp), v_scale);
            vOut[2] = PDX_MUL_MXF32(PDX_SUB_MX32(vP2, v_zp), v_scale);
            vOut[3] = PDX_MUL_MXF32(PDX_SUB_MX32(vP3, v_zp), v_scale);
            vIn += 4;
            vOut += 4;
          }
          // Scalar residual for remaining 0-15 elements
          for (; i < numel; ++i) {
            out_data[i] =
                (static_cast<float>(input_data[i]) - zp_f) * scale_data;
          }
        } else {
          for (; i < e16; i += 16) {
            xb_vecMxu32 vP0 = PDX_LVU32_MX8_I(vIn, 0);
            xb_vecMxu32 vP1 = PDX_LVU32_MX8_I(vIn + 1, 0);
            xb_vecMxu32 vP2 = PDX_LVU32_MX8_I(vIn + 2, 0);
            xb_vecMxu32 vP3 = PDX_LVU32_MX8_I(vIn + 3, 0);
            vOut[0] = PDX_MUL_MXF32(vP0, v_scale);
            vOut[1] = PDX_MUL_MXF32(vP1, v_scale);
            vOut[2] = PDX_MUL_MXF32(vP2, v_scale);
            vOut[3] = PDX_MUL_MXF32(vP3, v_scale);
            vIn += 4;
            vOut += 4;
          }
          // Scalar residual for remaining 0-15 elements
          for (; i < numel; ++i) {
            out_data[i] = static_cast<float>(input_data[i]) * scale_data;
          }
        }
        return out;
      }
#endif
      if (zero_point_data != 0) {
        XT_KERNEL_CHECK(
            context,
            out,
            xa_nn_elm_dequantize_asym8u_f32,
            out_data,
            input_data,
            inp_shape,
            num_inp_dims,
            NULL,
            &zero_point_data,
            &scale_data);
      } else {
        XT_KERNEL_CHECK(
            context,
            out,
            xa_nn_elm_dequantize_sym8u_f32,
            out_data,
            input_data,
            inp_shape,
            num_inp_dims,
            NULL,
            &scale_data);
      }
      return out;
    } else if (input.scalar_type() == ScalarType::Short) {
      const int16_t* __restrict__ input_data = input.const_data_ptr<int16_t>();
#if defined(__XTENSA__)
      // Direct inline PDX SIMD dequantization for per-tensor int16->float32.
      // The zero-point subtract is kept in the int32 domain (PDX_SUB_MX32) so
      // it issues on the integer unit and overlaps the float pipe; the widened
      // int32 codes feed straight into the mixed-type PDX_MUL_MXF32. That
      // intrinsic takes an xb_vecMxf32, so the compiler still emits an
      // int->float convert -- writing it mixed-type only keeps the source
      // concise, it does not remove the convert. q - zp cannot overflow int32
      // (inputs are <=16-bit and zp is in the same quant range) and converts
      // exactly to float, so the result is bit-identical to
      // float(q) - float(zp). 4x-unrolled to hide load-to-use latency.
      if (dequant_simd_aligned(input_data, out_data)) {
        const int numel = inp_shape[0];
        auto vIn = reinterpret_cast<const xb_vecMx16*>(input_data);
        auto vOut = reinterpret_cast<xb_vecMxf32*>(out_data);
        const xb_vecMxf32 v_scale{
            scale_data, scale_data, scale_data, scale_data};
        int i = 0;
        // 4x unrolled main loop: 16 elements per iteration
        const int e16 = (numel >> 4) << 4;
        if (zero_point_data != 0) {
          const xb_vecMx32 v_zp{
              zero_point_data,
              zero_point_data,
              zero_point_data,
              zero_point_data};
          const float zp_f = static_cast<float>(zero_point_data);
          for (; i < e16; i += 16) {
            xb_vecMx32 vP0 = PDX_LV32_MX16_I(vIn, 0);
            xb_vecMx32 vP1 = PDX_LV32_MX16_I(vIn + 1, 0);
            xb_vecMx32 vP2 = PDX_LV32_MX16_I(vIn + 2, 0);
            xb_vecMx32 vP3 = PDX_LV32_MX16_I(vIn + 3, 0);
            vOut[0] = PDX_MUL_MXF32(PDX_SUB_MX32(vP0, v_zp), v_scale);
            vOut[1] = PDX_MUL_MXF32(PDX_SUB_MX32(vP1, v_zp), v_scale);
            vOut[2] = PDX_MUL_MXF32(PDX_SUB_MX32(vP2, v_zp), v_scale);
            vOut[3] = PDX_MUL_MXF32(PDX_SUB_MX32(vP3, v_zp), v_scale);
            vIn += 4;
            vOut += 4;
          }
          // Scalar residual for remaining 0-15 elements
          for (; i < numel; ++i) {
            out_data[i] =
                (static_cast<float>(input_data[i]) - zp_f) * scale_data;
          }
        } else {
          for (; i < e16; i += 16) {
            xb_vecMx32 vP0 = PDX_LV32_MX16_I(vIn, 0);
            xb_vecMx32 vP1 = PDX_LV32_MX16_I(vIn + 1, 0);
            xb_vecMx32 vP2 = PDX_LV32_MX16_I(vIn + 2, 0);
            xb_vecMx32 vP3 = PDX_LV32_MX16_I(vIn + 3, 0);
            vOut[0] = PDX_MUL_MXF32(vP0, v_scale);
            vOut[1] = PDX_MUL_MXF32(vP1, v_scale);
            vOut[2] = PDX_MUL_MXF32(vP2, v_scale);
            vOut[3] = PDX_MUL_MXF32(vP3, v_scale);
            vIn += 4;
            vOut += 4;
          }
          // Scalar residual for remaining 0-15 elements
          for (; i < numel; ++i) {
            out_data[i] = static_cast<float>(input_data[i]) * scale_data;
          }
        }
        return out;
      }
#endif
      if (zero_point_data != 0) {
        XT_KERNEL_CHECK(
            context,
            out,
            xa_nn_elm_dequantize_asym16_f32,
            out_data,
            input_data,
            inp_shape,
            num_inp_dims,
            NULL,
            &zero_point_data,
            &scale_data);
      } else {
        XT_KERNEL_CHECK(
            context,
            out,
            xa_nn_elm_dequantize_sym16_f32,
            out_data,
            input_data,
            inp_shape,
            num_inp_dims,
            NULL,
            &scale_data);
      }
      return out;
    } else if (input.scalar_type() == ScalarType::UInt16) {
      const uint16_t* __restrict__ input_data =
          input.const_data_ptr<uint16_t>();
#if defined(__XTENSA__)
      // Direct inline PDX SIMD dequantization for per-tensor uint16->float32.
      // The zero-point subtract is kept in the int32 domain (PDX_SUB_MX32) so
      // it issues on the integer unit and overlaps the float pipe; the widened
      // int32 codes feed straight into the mixed-type PDX_MUL_MXF32. That
      // intrinsic takes an xb_vecMxf32, so the compiler still emits an
      // int->float convert -- writing it mixed-type only keeps the source
      // concise, it does not remove the convert. q - zp cannot overflow int32
      // (inputs are <=16-bit and zp is in the same quant range) and converts
      // exactly to float, so the result is bit-identical to
      // float(q) - float(zp). 4x-unrolled to hide load-to-use latency.
      if (dequant_simd_aligned(input_data, out_data)) {
        const int numel = inp_shape[0];
        auto vIn = reinterpret_cast<const xb_vecMxu16*>(input_data);
        auto vOut = reinterpret_cast<xb_vecMxf32*>(out_data);
        const xb_vecMxf32 v_scale{
            scale_data, scale_data, scale_data, scale_data};
        int i = 0;
        // 4x unrolled main loop: 16 elements per iteration
        const int e16 = (numel >> 4) << 4;
        if (zero_point_data != 0) {
          const xb_vecMxu32 v_zp{
              static_cast<uint32_t>(zero_point_data),
              static_cast<uint32_t>(zero_point_data),
              static_cast<uint32_t>(zero_point_data),
              static_cast<uint32_t>(zero_point_data)};
          const float zp_f = static_cast<float>(zero_point_data);
          for (; i < e16; i += 16) {
            xb_vecMxu32 vP0 = PDX_LVU32_MX16_I(vIn, 0);
            xb_vecMxu32 vP1 = PDX_LVU32_MX16_I(vIn + 1, 0);
            xb_vecMxu32 vP2 = PDX_LVU32_MX16_I(vIn + 2, 0);
            xb_vecMxu32 vP3 = PDX_LVU32_MX16_I(vIn + 3, 0);
            vOut[0] = PDX_MUL_MXF32(PDX_SUB_MX32(vP0, v_zp), v_scale);
            vOut[1] = PDX_MUL_MXF32(PDX_SUB_MX32(vP1, v_zp), v_scale);
            vOut[2] = PDX_MUL_MXF32(PDX_SUB_MX32(vP2, v_zp), v_scale);
            vOut[3] = PDX_MUL_MXF32(PDX_SUB_MX32(vP3, v_zp), v_scale);
            vIn += 4;
            vOut += 4;
          }
          // Scalar residual for remaining 0-15 elements
          for (; i < numel; ++i) {
            out_data[i] =
                (static_cast<float>(input_data[i]) - zp_f) * scale_data;
          }
        } else {
          for (; i < e16; i += 16) {
            xb_vecMxu32 vP0 = PDX_LVU32_MX16_I(vIn, 0);
            xb_vecMxu32 vP1 = PDX_LVU32_MX16_I(vIn + 1, 0);
            xb_vecMxu32 vP2 = PDX_LVU32_MX16_I(vIn + 2, 0);
            xb_vecMxu32 vP3 = PDX_LVU32_MX16_I(vIn + 3, 0);
            vOut[0] = PDX_MUL_MXF32(vP0, v_scale);
            vOut[1] = PDX_MUL_MXF32(vP1, v_scale);
            vOut[2] = PDX_MUL_MXF32(vP2, v_scale);
            vOut[3] = PDX_MUL_MXF32(vP3, v_scale);
            vIn += 4;
            vOut += 4;
          }
          // Scalar residual for remaining 0-15 elements
          for (; i < numel; ++i) {
            out_data[i] = static_cast<float>(input_data[i]) * scale_data;
          }
        }
        return out;
      }
#endif
      if (zero_point_data != 0) {
        XT_KERNEL_CHECK(
            context,
            out,
            xa_nn_elm_dequantize_asym16u_f32,
            out_data,
            input_data,
            inp_shape,
            num_inp_dims,
            NULL,
            &zero_point_data,
            &scale_data);
      } else {
        XT_KERNEL_CHECK(
            context,
            out,
            xa_nn_elm_dequantize_sym16u_f32,
            out_data,
            input_data,
            inp_shape,
            num_inp_dims,
            NULL,
            &scale_data);
      }
      return out;
    }
    // Fall through to generic path for other input types
  }
  dequantize_impl(
      context, out, input, &scale_data, &zero_point_data, NULL, out_dtype);
  return out;
}
Tensor& dequantize_per_tensor_tensor_args_out(
    KernelRuntimeContext& context,
    const Tensor& input,
    const Tensor& scale,
    const Tensor& zero_point,
    int64_t quant_min,
    int64_t quant_max,
    ScalarType dtype,
    std::optional<ScalarType> out_dtype,
    Tensor& out) {
#ifdef OP_ARG_CHECK
  ET_CHECK_MSG(
      scale.scalar_type() == ScalarType::Double,
      "Expected scale to be Double tensor received: %" PRId8,
      static_cast<int8_t>(scale.scalar_type()));
  ET_CHECK_MSG(
      zero_point.scalar_type() == ScalarType::Long,
      "Expected scale to be Long tensor received: %" PRId8,
      static_cast<int8_t>(zero_point.scalar_type()));
  ET_CHECK_MSG(
      scale.numel() == 1,
      "Exepcted scale to only have one element received: %zd",
      ssize_t(scale.numel()));
  ET_CHECK_MSG(
      zero_point.numel() == 1,
      "Exepcted zero_point to only have one element received: %zd",
      ssize_t(zero_point.numel()));
#endif

  dequantize_per_tensor_out(
      context,
      input,
      scale.const_data_ptr<double>()[0],
      zero_point.const_data_ptr<int64_t>()[0],
      quant_min,
      quant_max,
      dtype,
      out_dtype,
      out);

  return out;
}

Tensor& dequantize_per_channel_out(
    KernelRuntimeContext& context,
    const Tensor& input,
    const Tensor& scale,
    const std::optional<Tensor>& opt_zero_points,
    int64_t axis,
    int64_t quant_min,
    int64_t quant_max,
    ScalarType dtype,
    std::optional<ScalarType> out_dtype,
    Tensor& out) {
  if (axis < 0) {
    axis += executorch::runtime::nonzero_dim(input);
  }
  /* if the arguments are passed properly to the operator disable the Macro -
   * "OP_ARG_CHECK" if not the case, enable the Macro - "OP_ARG_CHECK", to have
   * the checks only in operator level(As there are no checks in kernel).
   */
#ifdef OP_ARG_CHECK
  torch::executor::Error err = resize_tensor(out, input.sizes());

  // normalize axis
  ET_CHECK_MSG(
      executorch::runtime::tensor_has_dim(input, axis),
      "axis %zd is not legal it should be -input.dim() <= axis < input.dim() %zd",
      ssize_t(axis),
      ssize_t(input.dim()));

  ET_CHECK_MSG(
      err == torch::executor::Error::Ok,
      "Failed to resize out Tensor in dequantize_per_channel_out");

  ET_CHECK_MSG(
      scale.scalar_type() == ScalarType::Double,
      "scale.scalar_type() %" PRId8 " is not double type",
      static_cast<int8_t>(scale.scalar_type()));

  ET_CHECK_MSG(
      scale.numel() == input.size(axis),
      "scale.numel() %zd != input.size(axis) %zd",
      ssize_t(scale.numel()),
      ssize_t(input.size(axis)));

  if (opt_zero_points.has_value()) {
    auto zero_point = opt_zero_points.value();
    ET_CHECK_MSG(
        zero_point.scalar_type() == ScalarType::Long,
        "zero_point.scalar_type() %" PRId8 " is not integer type",
        static_cast<int8_t>(zero_point.scalar_type()));

    ET_CHECK_MSG(
        zero_point.numel() == input.size(axis),
        "zero_point.numel() %zd != input.size(axis) %zd",
        ssize_t(zero_point.numel()),
        ssize_t(input.size(axis)));
  }
  check_dequantize_per_tensor_args(
      input, quant_min, quant_max, dtype, out_dtype, out);
#endif

  int* axis_ptr = (int*)&axis;

  const double* scale_dt = scale.const_data_ptr<double>();
  const int64_t* zero_point_dt;
  int zero_point_data[input.size(axis)];
  int* zero_point_ptr;
  if (opt_zero_points.has_value()) {
    zero_point_dt = opt_zero_points.value().const_data_ptr<int64_t>();
    zero_point_ptr = &zero_point_data[0];
    for (int i = 0; i < scale.numel(); i++) {
      zero_point_ptr[i] = (int)zero_point_dt[i];
    }
  } else {
    zero_point_ptr = nullptr;
  }
  float scale_data[input.size(axis)];
  for (int i = 0; i < scale.numel(); i++) {
    scale_data[i] = (float)scale_dt[i];
  }
  dequantize_impl(
      context, out, input, scale_data, zero_point_ptr, axis_ptr, out_dtype);

  return out;
}

Tensor& dequantize_per_token_out(
    KernelRuntimeContext& context,
    const Tensor& input,
    const Tensor& scale,
    const Tensor& zero_points,
    int64_t quant_min,
    int64_t quant_max,
    ScalarType dtype,
    ScalarType out_dtype,
    Tensor& out) {
  // Refactor this into a util
  size_t num_channels = 1;
  for (size_t i = 0; i < input.dim() - 1; i++) {
    num_channels *= input.size(i);
  }
  // This unfortunate change is needed because we compile op_quantize for aten
  // mode as well
  std::array<::executorch::aten::SizesType, 2> input_sizes;
  input_sizes[0] = static_cast<::executorch::aten::SizesType>(num_channels);
  input_sizes[1] =
      static_cast<::executorch::aten::SizesType>(input.size(input.dim() - 1));
#ifdef USE_ATEN_LIB
  Tensor reshaped_input = at::from_blob(
      input.mutable_data_ptr(),
      input_sizes,
      at::TensorOptions(input.scalar_type()));
#else
  std::array<::executorch::aten::DimOrderType, 2> input_dim_order{0, 1};
  std::array<::executorch::aten::StridesType, 2> input_strides;
  executorch::runtime::dim_order_to_stride_nocheck(
      input_sizes.data(), input_dim_order.data(), 2, input_strides.data());
  void* input_data = input.mutable_data_ptr();
  torch::executor::TensorImpl reshaped_input_impl =
      executorch::runtime::etensor::TensorImpl(
          input.scalar_type(),
          2,
          input_sizes.data(),
          input_data,
          input_dim_order.data(),
          input_strides.data(),
          executorch::runtime::TensorShapeDynamism::STATIC);
  Tensor reshaped_input(&reshaped_input_impl);
  torch::executor::Error err = resize_tensor(out, input.sizes());
  ET_CHECK_MSG(
      err == torch::executor::Error::Ok,
      "Failed to resize out Tensor in dequantize_per_channel_out");
#endif

  return dequantize_per_channel_out(
      context,
      reshaped_input,
      scale,
      zero_points,
      0, /* axis */
      quant_min,
      quant_max,
      dtype,
      out_dtype,
      out);
}

} // namespace native
} // namespace G3
} // namespace impl
