/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/kernels/portable/cpu/util/reduce_util.h>
#include <executorch/runtime/kernel/kernel_includes.h>
#include <algorithm>
#include <cinttypes>
#include <cmath>
#if defined(__aarch64__) || defined(__ARM_NEON)
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
using StridesType = executorch::aten::StridesType;
using SizesType = executorch::aten::SizesType;

namespace {

/**
 * Asserts that the parameters are valid.
 */
void check_dequantize_per_tensor_args(
    const Tensor& input,
    int64_t quant_min,
    int64_t quant_max,
    ScalarType dtype,
    std::optional<ScalarType>& out_dtype,
    Tensor& out) {
  ET_CHECK_MSG(
      input.scalar_type() == ScalarType::Byte ||
          input.scalar_type() == ScalarType::Char ||
          input.scalar_type() == ScalarType::Bits16 ||
          input.scalar_type() == ScalarType::UInt16 ||
          input.scalar_type() == ScalarType::Short ||
          input.scalar_type() == ScalarType::Int,
      "input.scalar_type() %" PRId8 " is not supported:",
      static_cast<int8_t>(input.scalar_type()));

  ET_CHECK_MSG(
      input.scalar_type() == dtype,
      "input.scalar_type() %" PRId8 " is not matching dtype argumenta:",
      static_cast<int8_t>(input.scalar_type()));

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

/**
 * Useful to reduce a tensor `in` over a given dimension `dim` using the
 * reduce function `fn`, which should have the following signature:
 * void fn(const size_t size, const size_t stride, const size_t base_ix)
 * where `size` and `stride` are the size and stride of the dimension being
 * reduced and `base_ix` is the index of the first element of the reduction.
 */
template <typename Fn>
void apply_over_unpacked_dim(
    const Fn& fn,
    const executorch::aten::Tensor& in,
    const int64_t& dim) {
  if (in.numel() == 0) {
    return;
  }

  ET_CHECK_MSG(in.dim() > 0, "Input tensor must have at least one dimension");
  ET_CHECK_VALID_DIM(dim, in.dim());

  const size_t d = ET_NORMALIZE_IX(dim, in.dim());
  const size_t dim_size = in.size(d);
  const size_t outer_size = getLeadingDims(in, d);
  const size_t inner_size = getTrailingDims(in, d);
  // Loop through all outer dimensions
  for (size_t outer_idx = 0; outer_idx < outer_size; ++outer_idx) {
    // Loop through dim
    for (size_t unpacked_dim_idx = 0; unpacked_dim_idx < dim_size;
         ++unpacked_dim_idx) {
      fn(inner_size, outer_idx, unpacked_dim_idx);
    }
  }
}

void dequantize_optimized(
    const int8_t* in,
    const double scale,
    const int64_t zero_point,
    float* out,
    int64_t quant_min,
    int64_t quant_max,
    size_t numel) {
  ET_CHECK_MSG(
      zero_point >= quant_min,
      "zero_point must be %" PRId64 " <= quant_min %" PRId64,
      zero_point,
      quant_min);
  ET_CHECK_MSG(
      zero_point <= quant_max,
      "zero_point must be %" PRId64 " >= quant_max %" PRId64,
      zero_point,
      quant_max);
  size_t i = 0;
#if defined(__aarch64__) || defined(__ARM_NEON)
  int8x8_t zero_point_vec = vdup_n_s8(zero_point);
  float32x4_t scales = vdupq_n_f32(static_cast<float>(scale));
  constexpr int32_t kVecSize = 16;
  const size_t num_vecs = numel / kVecSize;
  const int8_t* in_copy = in;
  float* out_copy = out;
  for (; i < num_vecs; i++) {
    int8x16_t in_vec = vld1q_s8(in_copy);
    int16x8_t sub_vec_0_7 = vsubl_s8(vget_low_s8(in_vec), zero_point_vec);
    int32x4_t sub_vec_0_3 = vmovl_s16(vget_low_s16(sub_vec_0_7));
    int32x4_t sub_vec_4_7 = vmovl_s16(vget_high_s16(sub_vec_0_7));
    float32x4_t out_vec_0_3 = vmulq_f32(vcvtq_f32_s32(sub_vec_0_3), scales);
    float32x4_t out_vec_4_7 = vmulq_f32(vcvtq_f32_s32(sub_vec_4_7), scales);

    int16x8_t sub_vec_8_15 = vsubl_s8(vget_high_s8(in_vec), zero_point_vec);
    int32x4_t sub_vec_8_11 = vmovl_s16(vget_low_s16(sub_vec_8_15));
    int32x4_t sub_vec_12_15 = vmovl_s16(vget_high_s16(sub_vec_8_15));
    float32x4_t out_vec_8_11 = vmulq_f32(vcvtq_f32_s32(sub_vec_8_11), scales);
    float32x4_t out_vec_12_15 = vmulq_f32(vcvtq_f32_s32(sub_vec_12_15), scales);
    vst1q_f32(out_copy + 0, out_vec_0_3);
    vst1q_f32(out_copy + 4, out_vec_4_7);
    vst1q_f32(out_copy + 8, out_vec_8_11);
    vst1q_f32(out_copy + 12, out_vec_12_15);
    in_copy += kVecSize;
    out_copy += kVecSize;
  }
  i = i * kVecSize;
#endif
  for (; i < numel; i++) {
    out[i] = (in[i] - zero_point) * scale;
  }
}

float get_scale(const Tensor& scale, size_t channel_ix) {
  ET_CHECK_MSG(
      (scale.scalar_type() == ScalarType::Double) ||
          (scale.scalar_type() == ScalarType::Float),
      "scale.scalar_type() %" PRId8 " is not double or float type",
      static_cast<int8_t>(scale.scalar_type()));
  if (scale.scalar_type() == ScalarType::Double) {
    return static_cast<float>(scale.const_data_ptr<double>()[channel_ix]);
  } else {
    return scale.const_data_ptr<float>()[channel_ix];
  }
}

bool can_use_optimized_dequantize_per_channel(
    const Tensor& in,
    const ScalarType in_dtype,
    std::optional<ScalarType>& out_dtype) {
  bool is_contiguous = false;
#ifdef USE_ATEN_LIB
  is_contiguous = in.is_contiguous();
#else
  is_contiguous = executorch::runtime::is_contiguous_dim_order(
      in.dim_order().data(), in.dim());
#endif
  if (!is_contiguous || (in_dtype != ScalarType::Char) ||
      (out_dtype.has_value() && out_dtype.value() != ScalarType::Float)) {
    return false;
  }
  return true;
}

void dequantize_per_channel_optimized(
    const Tensor& in,
    const Tensor& scales,
    const std::optional<Tensor>& opt_zero_points,
    Tensor& out,
    int64_t axis,
    int64_t quant_min,
    int64_t quant_max,
    ScalarType in_dtype,
    std::optional<ScalarType>& out_dtype) {
  check_dequantize_per_tensor_args(
      in, quant_min, quant_max, in_dtype, out_dtype, out);
  ET_CHECK_MSG(
      in_dtype == ScalarType::Char,
      "in.scalar_type() %" PRId8 " is not supported:",
      static_cast<int8_t>(in.scalar_type()));
  if (out_dtype.has_value()) {
    ET_CHECK_MSG(
        out_dtype.value() == ScalarType::Float,
        "Only float output is supported");
  }
  const int8_t* in_data = in.const_data_ptr<int8_t>();
  float* out_data = out.mutable_data_ptr<float>();
  const int64_t* zero_points_data = nullptr;
  if (opt_zero_points.has_value()) {
    zero_points_data = opt_zero_points.value().const_data_ptr<int64_t>();
  }
  const StridesType axis_stride = in.strides()[axis];
  const StridesType outer_stride = in.size(axis) * axis_stride;
  apply_over_unpacked_dim(
      [in_data,
       out_data,
       &scales,
       zero_points_data,
       axis_stride,
       outer_stride,
       quant_min,
       quant_max](
          SizesType numel, SizesType outer_idx, SizesType unpacked_dim_idx) {
        const int8_t* in_data_local =
            in_data + outer_idx * outer_stride + unpacked_dim_idx * axis_stride;
        const double scale = get_scale(scales, unpacked_dim_idx);
        const int64_t zero_point = zero_points_data != nullptr
            ? zero_points_data[unpacked_dim_idx]
            : 0;
        float* out_data_local = out_data + outer_idx * outer_stride +
            unpacked_dim_idx * axis_stride;
        dequantize_optimized(
            in_data_local,
            scale,
            zero_point,
            out_data_local,
            quant_min,
            quant_max,
            numel);
      },
      in,
      axis);
}

} // namespace

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
    const Tensor& input,
    double scale,
    int64_t zero_point,
    int64_t quant_min,
    int64_t quant_max,
    ScalarType dtype,
    std::optional<ScalarType> out_dtype,
    Tensor& out) {
  torch::executor::Error err = resize_tensor(out, input.sizes());
  ET_CHECK_MSG(
      err == torch::executor::Error::Ok,
      "Failed to resize out Tensor in dequantize_per_tensor_out");

  check_dequantize_per_tensor_args(
      input, quant_min, quant_max, dtype, out_dtype, out);

  // calculate the dequantized output, cast scale to float to match fbgemm
  // behavior
#define DEQUANTIZE_IMPL(IN_CTYPE, OUT_CTYPE, out_dtype)                        \
  case ScalarType::out_dtype: {                                                \
    /* Hoist these function calls out of our inner loop because they might not \
     * get inlined without LTO, particularly in ATen mode. */                  \
    auto* out_data_ptr = out.mutable_data_ptr<OUT_CTYPE>();                    \
    const auto* input_data_ptr = input.const_data_ptr<IN_CTYPE>();             \
    const auto input_numel = input.numel();                                    \
    for (size_t i = 0; i < input_numel; i++) {                                 \
      out_data_ptr[i] = static_cast<OUT_CTYPE>(                                \
          (input_data_ptr[i] - static_cast<int32_t>(zero_point)) *             \
          static_cast<float>(scale));                                          \
    }                                                                          \
  } break;
#define CALCULATE_INT_TYPE(IN_CTYPE, in_dtype)                \
  case ScalarType::in_dtype:                                  \
    switch (out.scalar_type()) {                              \
      ET_FORALL_FLOATH_TYPES_WITH(IN_CTYPE, DEQUANTIZE_IMPL); \
      default:                                                \
        ET_CHECK_MSG(                                         \
            false,                                            \
            "Unhandled output dtype %" PRId8,                 \
            static_cast<int8_t>(out.scalar_type()));          \
    }                                                         \
    break;

  switch (input.scalar_type()) {
    ET_FORALL_INT_TYPES(CALCULATE_INT_TYPE);
    CALCULATE_INT_TYPE(uint16_t, Bits16);
    CALCULATE_INT_TYPE(uint16_t, UInt16);
    default:
      ET_CHECK_MSG(
          false,
          "Unhandled input dtype %" PRId8,
          static_cast<int8_t>(input.scalar_type()));
  }

#undef CALCULATE_FLOAT_TYPE
#undef DEQUANTIZE_IMPL
  return out;
}

Tensor& dequantize_per_tensor_tensor_args_out(
    const Tensor& input,
    const Tensor& scale,
    const Tensor& zero_point,
    int64_t quant_min,
    int64_t quant_max,
    ScalarType dtype,
    std::optional<ScalarType> out_dtype,
    Tensor& out) {
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

  dequantize_per_tensor_out(
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
    const Tensor& input,
    const Tensor& scale,
    const std::optional<Tensor>& opt_zero_points,
    int64_t axis,
    int64_t quant_min,
    int64_t quant_max,
    ScalarType dtype,
    std::optional<ScalarType> out_dtype,
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
      scale.numel() == input.size(axis),
      "scale.numel() %zd != input.size(axis) %zd",
      ssize_t(scale.numel()),
      ssize_t(input.size(axis)));

  if (opt_zero_points.has_value()) {
    auto zero_point = opt_zero_points.value();
    ET_CHECK_MSG(
        zero_point.scalar_type() == ScalarType::Int ||
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

  if (can_use_optimized_dequantize_per_channel(input, dtype, out_dtype)) {
    dequantize_per_channel_optimized(
        input,
        scale,
        opt_zero_points,
        out,
        axis,
        quant_min,
        quant_max,
        dtype,
        out_dtype);
    return out;
  }

  // a list contains all dimensions except axis
  int64_t dims[kTensorDimensionLimit];
  for (int64_t i = 0; i < input.dim() - 1; i++) {
    if (i < axis) {
      dims[i] = i;
    } else {
      dims[i] = i + 1;
    }
  }
  const int64_t* zero_point_data;
  if (opt_zero_points.has_value()) {
    zero_point_data = opt_zero_points.value().const_data_ptr<int64_t>();
  } else {
    zero_point_data = nullptr;
  }

  std::optional<executorch::aten::ArrayRef<int64_t>> optional_dim_list{
      executorch::aten::ArrayRef<int64_t>{dims, size_t(input.dim() - 1)}};

  // Actual dequantization logic
  // input, out are the input and output tensors
  // channel_ix is the index along the axis dimension. 0 <= channel_ix <
  // input.size(axis).
  //   i.e. if the tensor has shape (N,C,H,W), axis being 1, then channel_ix
  //   will be 0, 1, 2, ... C-1
  // in_ix is the flat index of the element you are dequantizing.
  //   in other words you are dequantizing in_data[in_ix]
#define DEQUANTIZE_IMPL(CTYPE_IN, CTYPE_OUT, out_dtype)                        \
  case ScalarType::out_dtype:                                                  \
    if (input.dim() == 1) {                                                    \
      auto* out_data_ptr = out.mutable_data_ptr<CTYPE_OUT>();                  \
      const auto* input_data_ptr = input.const_data_ptr<CTYPE_IN>();           \
      ET_CHECK_MSG(                                                            \
          axis == 0, "Axis must be 0 for a single dimensional tensors");       \
      const std::optional<int64_t> dim;                                        \
      apply_over_dim(                                                          \
          [input_data_ptr, out_data_ptr, zero_point_data, &scale](             \
              size_t numel, size_t stride, size_t base_ix) {                   \
            for (size_t i = 0; i < numel; i++) {                               \
              size_t current_ix = base_ix * stride + i;                        \
              float _scale = get_scale(scale, current_ix);                     \
              int64_t zero_point = 0;                                          \
              if (zero_point_data != nullptr) {                                \
                zero_point = zero_point_data[current_ix];                      \
              }                                                                \
              out_data_ptr[current_ix] =                                       \
                  static_cast<CTYPE_OUT>(                                      \
                      input_data_ptr[current_ix] -                             \
                      static_cast<int32_t>(zero_point)) *                      \
                  _scale;                                                      \
            }                                                                  \
          },                                                                   \
          input,                                                               \
          dim);                                                                \
      break;                                                                   \
    }                                                                          \
    for (size_t channel_ix = 0; channel_ix < input.size(axis); ++channel_ix) { \
      float _scale = get_scale(scale, channel_ix);                             \
      int64_t _zero_point = 0;                                                 \
      if (zero_point_data != nullptr) {                                        \
        _zero_point = zero_point_data[channel_ix];                             \
      }                                                                        \
      auto* out_data_ptr = out.mutable_data_ptr<CTYPE_OUT>();                  \
      const auto* input_data_ptr = input.const_data_ptr<CTYPE_IN>();           \
      apply_over_dim_list(                                                     \
          [input_data_ptr, out_data_ptr, _scale, _zero_point](size_t in_ix) {  \
            out_data_ptr[in_ix] = static_cast<CTYPE_OUT>(                      \
                (input_data_ptr[in_ix] - static_cast<int32_t>(_zero_point)) *  \
                _scale);                                                       \
          },                                                                   \
          input,                                                               \
          optional_dim_list,                                                   \
          channel_ix);                                                         \
    }                                                                          \
    break;
#define CALCULATE_FLOAT_TYPE(CTYPE_IN, in_dtype)              \
  case ScalarType::in_dtype:                                  \
    switch (out.scalar_type()) {                              \
      ET_FORALL_FLOATH_TYPES_WITH(CTYPE_IN, DEQUANTIZE_IMPL); \
      default:                                                \
        ET_CHECK_MSG(                                         \
            false,                                            \
            "Unhandled output dtype %" PRId8,                 \
            static_cast<int8_t>(out.scalar_type()));          \
    }                                                         \
    break;

  switch (input.scalar_type()) {
    ET_FORALL_INT_TYPES(CALCULATE_FLOAT_TYPE);
    CALCULATE_INT_TYPE(uint16_t, Bits16);
    CALCULATE_INT_TYPE(uint16_t, UInt16);
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
  (void)context;
  torch::executor::Error err = resize_tensor(out, input.sizes());
  ET_CHECK_MSG(
      err == torch::executor::Error::Ok,
      "Failed to resize out Tensor in dequantize_per_channel_out");

  return dequantize_per_channel_out(
      input,
      scale,
      opt_zero_points,
      axis,
      quant_min,
      quant_max,
      dtype,
      out_dtype,
      out);
}

Tensor& dequantize_per_tensor_out(
    KernelRuntimeContext& context,
    const Tensor& input,
    double scale,
    int64_t zero_point,
    int64_t quant_min,
    int64_t quant_max,
    ScalarType dtype,
    std::optional<ScalarType> out_dtype,
    Tensor& out) {
  // TODO(larryliu): Add a context arg to the real op function and remove this
  // wrapper
  (void)context;
  return dequantize_per_tensor_out(
      input, scale, zero_point, quant_min, quant_max, dtype, out_dtype, out);
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
  // TODO(larryliu): Add a context arg to the real op function and remove this
  // wrapper
  (void)context;
  return dequantize_per_tensor_tensor_args_out(
      input, scale, zero_point, quant_min, quant_max, dtype, out_dtype, out);
}

Tensor& dequantize_per_token_out(
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
  std::array<executorch::aten::SizesType, 2> input_sizes;
  input_sizes[0] = static_cast<executorch::aten::SizesType>(num_channels);
  input_sizes[1] =
      static_cast<executorch::aten::SizesType>(input.size(input.dim() - 1));
#ifdef USE_ATEN_LIB
  Tensor reshaped_input = at::from_blob(
      input.mutable_data_ptr(),
      input_sizes,
      at::TensorOptions(input.scalar_type()));
#else
  std::array<executorch::aten::DimOrderType, 2> input_dim_order{0, 1};
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
      "Failed to resize out Tensor in dequantize_per_channel_out");
#endif

  return dequantize_per_channel_out(
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

Tensor& dequantize_per_token_out(
    RuntimeContext& context,
    const Tensor& input,
    const Tensor& scale,
    const Tensor& zero_points,
    int64_t quant_min,
    int64_t quant_max,
    ScalarType dtype,
    ScalarType out_dtype,
    Tensor& out) {
  (void)context;
  return dequantize_per_token_out(
      input, scale, zero_points, quant_min, quant_max, dtype, out_dtype, out);
}

} // namespace native
} // namespace executor
} // namespace torch
