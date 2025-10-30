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

/* ScalarType in Executorch do not have support for below data types.
 * So, creating a placeholder for these data types. Once, ScalarTypes is
 * updated to have support for below data types, these can be removed and
 * operator need to be updated accordingly
 */
enum datatype { Bits4u = 21, Bits4 = 22 };

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
  } else if (dtype == ScalarType::UInt16) {
    quant_min_lower_bound = std::numeric_limits<uint16_t>::min();
    quant_max_upper_bound = std::numeric_limits<uint16_t>::max();
  } else if (dtype == ScalarType::Short) {
    quant_min_lower_bound = std::numeric_limits<int16_t>::min();
    quant_max_upper_bound = std::numeric_limits<int16_t>::max();
  } else if (dtype == (ScalarType)Bits4u) {
    quant_min_lower_bound = std::numeric_limits<uint8_t>::min();
    quant_max_upper_bound = std::numeric_limits<uint8_t>::max();
    /* Minimum and maximum values fo unsigned 4-bit data type */
    quant_min_lower_bound = quant_min_lower_bound >> 4;
    quant_max_upper_bound = quant_max_upper_bound >> 4;
  } else if (dtype == (ScalarType)Bits4) {
    quant_min_lower_bound = std::numeric_limits<int8_t>::min();
    quant_max_upper_bound = std::numeric_limits<int8_t>::max();
    /* Minimum and maximum values fo signed 4-bit data type */
    quant_min_lower_bound = quant_min_lower_bound >> 4;
    quant_max_upper_bound = quant_max_upper_bound >> 4;
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
} /* check_quantize_per_tensor_args */

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

/* Local function which calls the kernels based on the output datatype */
Tensor& quantize_impl(
    KernelRuntimeContext& ctx,
    Tensor& out,
    const Tensor& input,
    float* scale_data,
    int* zero_point_data,
    int* axis,
    int quant_min,
    int quant_max) {
  const ::executorch::aten::ArrayRef<Tensor::SizesType> input_size =
      input.sizes();

  int kTensorDimensionLimit = 5;

  int inp_shape[kTensorDimensionLimit];

  for (auto i = 0; i < input_size.size(); i++) {
    inp_shape[i] = input_size[i];
  }

  const float* input_data = input.const_data_ptr<float>();

  bool is_asym_quant = 0;

  bool optimized = true;

  if (input.scalar_type() != ScalarType::Float) {
    optimized = false;
  }

  if (zero_point_data != NULL) // asymmetric quant
  {
    if (axis != NULL) // channel
    {
      for (int i = 0; i < input.size(*axis); i++) {
        if (zero_point_data[i] != 0) {
          is_asym_quant |= 1;
        }
      }
    } else {
      if (*zero_point_data != 0) // tensor
      {
        is_asym_quant |= 1;
      }
    }
  }

  if (is_asym_quant) {
    if ((out.scalar_type() == ScalarType::Byte) && (optimized)) {
      uint8_t* out_data = out.mutable_data_ptr<uint8_t>();
      XT_KERNEL_CHECK(
          ctx,
          out,
          xa_nn_elm_quantize_f32_asym8u,
          out_data,
          input_data,
          inp_shape,
          input.dim(),
          axis,
          scale_data,
          zero_point_data,
          quant_min,
          quant_max);
    } else if ((out.scalar_type() == ScalarType::Char) && (optimized)) {
      int8_t* out_data = out.mutable_data_ptr<int8_t>();

      XT_KERNEL_CHECK(
          ctx,
          out,
          xa_nn_elm_quantize_f32_asym8,
          out_data,
          input_data,
          inp_shape,
          input.dim(),
          axis,
          scale_data,
          zero_point_data,
          quant_min,
          quant_max);
    } else if ((out.scalar_type() == ScalarType::UInt16) && (optimized)) {
      uint16_t* out_data = out.mutable_data_ptr<uint16_t>();
      XT_KERNEL_CHECK(
          ctx,
          out,
          xa_nn_elm_quantize_f32_asym16u,
          out_data,
          input_data,
          inp_shape,
          input.dim(),
          axis,
          scale_data,
          zero_point_data,
          quant_min,
          quant_max);
    } else if ((out.scalar_type() == ScalarType::Short) && (optimized)) {
      int16_t* out_data = out.mutable_data_ptr<int16_t>();
      XT_KERNEL_CHECK(
          ctx,
          out,
          xa_nn_elm_quantize_f32_asym16,
          out_data,
          input_data,
          inp_shape,
          input.dim(),
          axis,
          scale_data,
          zero_point_data,
          quant_min,
          quant_max);
    } else if ((out.scalar_type() == (ScalarType)Bits4u) && (optimized)) {
      uint8_t* out_data = out.mutable_data_ptr<uint8_t>();
      XT_KERNEL_CHECK(
          ctx,
          out,
          xa_nn_elm_quantize_f32_asym4u,
          out_data,
          input_data,
          inp_shape,
          input.dim(),
          axis,
          scale_data,
          zero_point_data,
          quant_min,
          quant_max);
    } else if ((out.scalar_type() == (ScalarType)Bits4) && (optimized)) {
      int8_t* out_data = out.mutable_data_ptr<int8_t>();
      XT_KERNEL_CHECK(
          ctx,
          out,
          xa_nn_elm_quantize_f32_asym4,
          out_data,
          input_data,
          inp_shape,
          input.dim(),
          axis,
          scale_data,
          zero_point_data,
          quant_min,
          quant_max);
    } else {
      if (axis == NULL) {
        // Vector quantization
// calculate the quantized input
#define ASYM_QUANTIZE_IMPL_TENSOR(IN_CTYPE, OUT_CTYPE, out_dtype)              \
  case ScalarType::out_dtype: {                                                \
    /* Hoist these function calls out of our inner loop because they might not \
     * get inlined without LTO, particularly in ATen mode. */                  \
    auto* out_data_ptr = out.mutable_data_ptr<OUT_CTYPE>();                    \
    const auto* input_data_ptr = input.const_data_ptr<IN_CTYPE>();             \
    const auto input_numel = input.numel();                                    \
    for (size_t i = 0; i < input_numel; i++) {                                 \
      IN_CTYPE value = input_data_ptr[i];                                      \
      out_data_ptr[i] = quantize_val<OUT_CTYPE, IN_CTYPE>(                     \
          (double)*scale_data,                                                 \
          (int64_t) * zero_point_data,                                         \
          value,                                                               \
          (int64_t)quant_min,                                                  \
          (int64_t)quant_max);                                                 \
    }                                                                          \
  } break;
#define ASYM_CALCULATE_FLOAT_TYPE_TENSOR(IN_CTYPE, in_dtype)         \
  case ScalarType::in_dtype:                                         \
    switch (out.scalar_type()) {                                     \
      ET_FORALL_INT_TYPES_WITH(IN_CTYPE, ASYM_QUANTIZE_IMPL_TENSOR); \
      ASYM_QUANTIZE_IMPL_TENSOR(IN_CTYPE, uint16_t, UInt16)          \
      default:                                                       \
        ET_CHECK_MSG(                                                \
            false,                                                   \
            "Unhandled output dtype %" PRId8,                        \
            static_cast<int8_t>(out.scalar_type()));                 \
    }                                                                \
    break;

        switch (input.scalar_type()) {
          ET_FORALL_FLOAT_TYPES(ASYM_CALCULATE_FLOAT_TYPE_TENSOR);
          default:
            ET_CHECK_MSG(
                false,
                "Unhandled input dtype %" PRId8,
                static_cast<int8_t>(input.scalar_type()));
        }

      } else {
        // Channel based quantization
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

// Actual quantization logic
// input, out are the input and output tensors
// channel_ix is the index along the axis dimension. 0 <= channel_ix <
// input.size(axis).
//   i.e. if the tensor has shape (N,C,H,W), axis being 1, then channel_ix
//   will be 0, 1, 2, ... C-1
// in_ix is the flat index of the element you are quantizing.
//   in other words you are quantizing in_data[in_ix]
#define ASYM_QUANTIZE_IMPL_CHANNEL(CTYPE_IN, CTYPE_OUT, out_dtype)   \
  case ScalarType::out_dtype:                                        \
    for (size_t channel_ix = 0; channel_ix < input.size(*axis);      \
         ++channel_ix) {                                             \
      double _scale = (double)scale_data[channel_ix];                \
      int64_t _zero_point = (int64_t)zero_point_data[channel_ix];    \
      auto* out_data_ptr = out.mutable_data_ptr<CTYPE_OUT>();        \
      const auto* input_data_ptr = input.const_data_ptr<CTYPE_IN>(); \
      torch::executor::apply_over_dim_list(                          \
          [input_data_ptr,                                           \
           out_data_ptr,                                             \
           _scale,                                                   \
           _zero_point,                                              \
           quant_min,                                                \
           quant_max](size_t in_ix) {                                \
            out_data_ptr[in_ix] = quantize_val<CTYPE_OUT, CTYPE_IN>( \
                _scale,                                              \
                _zero_point,                                         \
                input_data_ptr[in_ix],                               \
                quant_min,                                           \
                quant_max);                                          \
          },                                                         \
          input,                                                     \
          optional_dim_list,                                         \
          channel_ix);                                               \
    }                                                                \
    break;
#define ASYM_CALCULATE_FLOAT_TYPE_CHANNEL(CTYPE_IN, in_dtype)         \
  case ScalarType::in_dtype:                                          \
    switch (out.scalar_type()) {                                      \
      ET_FORALL_INT_TYPES_WITH(CTYPE_IN, ASYM_QUANTIZE_IMPL_CHANNEL); \
      ASYM_QUANTIZE_IMPL_CHANNEL(CTYPE_IN, uint16_t, UInt16)          \
      default:                                                        \
        ET_CHECK_MSG(                                                 \
            false,                                                    \
            "Unhandled output dtype %" PRId8,                         \
            static_cast<int8_t>(out.scalar_type()));                  \
    }                                                                 \
    break;

        switch (input.scalar_type()) {
          ET_FORALL_FLOAT_TYPES(ASYM_CALCULATE_FLOAT_TYPE_CHANNEL);
          default:
            ET_CHECK_MSG(
                false,
                "Unhandled input dtype %" PRId8,
                static_cast<int8_t>(input.scalar_type()));
        }
      }

#undef ASYM_CALCULATE_FLOAT_TYPE_TENSOR
#undef ASYM_CALCULATE_FLOAT_TYPE_CHANNEL
#undef ASYM_QUANTIZE_IMPL_TENSOR
#undef ASYM_QUANTIZE_IMPL_CHANNEL
    }
  } else {
    if ((out.scalar_type() == ScalarType::Byte) && (optimized)) {
      uint8_t* out_data = out.mutable_data_ptr<uint8_t>();
      XT_KERNEL_CHECK(
          ctx,
          out,
          xa_nn_elm_quantize_f32_sym8u,
          out_data,
          input_data,
          inp_shape,
          input.dim(),
          axis,
          scale_data,
          quant_min,
          quant_max);
    } else if ((out.scalar_type() == ScalarType::Char) && (optimized)) {
      int8_t* out_data = out.mutable_data_ptr<int8_t>();
      XT_KERNEL_CHECK(
          ctx,
          out,
          xa_nn_elm_quantize_f32_sym8,
          out_data,
          input_data,
          inp_shape,
          input.dim(),
          axis,
          scale_data,
          quant_min,
          quant_max);
    } else if ((out.scalar_type() == ScalarType::UInt16) && (optimized)) {
      uint16_t* out_data = out.mutable_data_ptr<uint16_t>();
      XT_KERNEL_CHECK(
          ctx,
          out,
          xa_nn_elm_quantize_f32_sym16u,
          out_data,
          input_data,
          inp_shape,
          input.dim(),
          axis,
          scale_data,
          quant_min,
          quant_max);
    } else if ((out.scalar_type() == ScalarType::Short) && (optimized)) {
      int16_t* out_data = out.mutable_data_ptr<int16_t>();
      XT_KERNEL_CHECK(
          ctx,
          out,
          xa_nn_elm_quantize_f32_sym16,
          out_data,
          input_data,
          inp_shape,
          input.dim(),
          axis,
          scale_data,
          quant_min,
          quant_max);
    } else if ((out.scalar_type() == (ScalarType)Bits4u) && (optimized)) {
      uint8_t* out_data = out.mutable_data_ptr<uint8_t>();
      XT_KERNEL_CHECK(
          ctx,
          out,
          xa_nn_elm_quantize_f32_sym4u,
          out_data,
          input_data,
          inp_shape,
          input.dim(),
          axis,
          scale_data,
          quant_min,
          quant_max);
    } else if ((out.scalar_type() == (ScalarType)Bits4) && (optimized)) {
      int8_t* out_data = out.mutable_data_ptr<int8_t>();
      XT_KERNEL_CHECK(
          ctx,
          out,
          xa_nn_elm_quantize_f32_sym4,
          out_data,
          input_data,
          inp_shape,
          input.dim(),
          axis,
          scale_data,
          quant_min,
          quant_max);
    } else {
      if (axis == NULL) {
        // calculate the quantized input
#define SYM_QUANTIZE_IMPL_TENSOR(IN_CTYPE, OUT_CTYPE, out_dtype)               \
  case ScalarType::out_dtype: {                                                \
    /* Hoist these function calls out of our inner loop because they might not \
     * get inlined without LTO, particularly in ATen mode. */                  \
    auto* out_data_ptr = out.mutable_data_ptr<OUT_CTYPE>();                    \
    const auto* input_data_ptr = input.const_data_ptr<IN_CTYPE>();             \
    const auto input_numel = input.numel();                                    \
    for (size_t i = 0; i < input_numel; i++) {                                 \
      IN_CTYPE value = input_data_ptr[i];                                      \
      out_data_ptr[i] = quantize_val<OUT_CTYPE, IN_CTYPE>(                     \
          (double)*scale_data,                                                 \
          (int64_t) * zero_point_data,                                         \
          value,                                                               \
          (int64_t)quant_min,                                                  \
          (int64_t)quant_max);                                                 \
    }                                                                          \
  } break;
#define SYM_CALCULATE_FLOAT_TYPE_TENSOR(IN_CTYPE, in_dtype)         \
  case ScalarType::in_dtype:                                        \
    switch (out.scalar_type()) {                                    \
      ET_FORALL_INT_TYPES_WITH(IN_CTYPE, SYM_QUANTIZE_IMPL_TENSOR); \
      SYM_QUANTIZE_IMPL_TENSOR(IN_CTYPE, uint16_t, UInt16)          \
      default:                                                      \
        ET_CHECK_MSG(                                               \
            false,                                                  \
            "Unhandled output dtype %" PRId8,                       \
            static_cast<int8_t>(out.scalar_type()));                \
    }                                                               \
    break;

        switch (input.scalar_type()) {
          ET_FORALL_FLOAT_TYPES(SYM_CALCULATE_FLOAT_TYPE_TENSOR);
          default:
            ET_CHECK_MSG(
                false,
                "Unhandled input dtype %" PRId8,
                static_cast<int8_t>(input.scalar_type()));
        }

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

// Actual quantization logic
// input, out are the input and output tensors
// channel_ix is the index along the axis dimension. 0 <= channel_ix <
// input.size(axis).
//   i.e. if the tensor has shape (N,C,H,W), axis being 1, then channel_ix
//   will be 0, 1, 2, ... C-1
// in_ix is the flat index of the element you are quantizing.
//   in other words you are quantizing in_data[in_ix]
#define SYM_QUANTIZE_IMPL_CHANNEL(CTYPE_IN, CTYPE_OUT, out_dtype)    \
  case ScalarType::out_dtype:                                        \
    for (size_t channel_ix = 0; channel_ix < input.size(*axis);      \
         ++channel_ix) {                                             \
      double _scale = (double)scale_data[channel_ix];                \
      int64_t _zero_point = (int64_t)zero_point_data[channel_ix];    \
      auto* out_data_ptr = out.mutable_data_ptr<CTYPE_OUT>();        \
      const auto* input_data_ptr = input.const_data_ptr<CTYPE_IN>(); \
      torch::executor::apply_over_dim_list(                          \
          [input_data_ptr,                                           \
           out_data_ptr,                                             \
           _scale,                                                   \
           _zero_point,                                              \
           quant_min,                                                \
           quant_max](size_t in_ix) {                                \
            out_data_ptr[in_ix] = quantize_val<CTYPE_OUT, CTYPE_IN>( \
                _scale,                                              \
                _zero_point,                                         \
                input_data_ptr[in_ix],                               \
                quant_min,                                           \
                quant_max);                                          \
          },                                                         \
          input,                                                     \
          optional_dim_list,                                         \
          channel_ix);                                               \
    }                                                                \
    break;
#define SYM_CALCULATE_FLOAT_TYPE_CHANNEL(CTYPE_IN, in_dtype)         \
  case ScalarType::in_dtype:                                         \
    switch (out.scalar_type()) {                                     \
      ET_FORALL_INT_TYPES_WITH(CTYPE_IN, SYM_QUANTIZE_IMPL_CHANNEL); \
      SYM_QUANTIZE_IMPL_CHANNEL(CTYPE_IN, uint16_t, UInt16)          \
      default:                                                       \
        ET_CHECK_MSG(                                                \
            false,                                                   \
            "Unhandled output dtype %" PRId8,                        \
            static_cast<int8_t>(out.scalar_type()));                 \
    }                                                                \
    break;

        switch (input.scalar_type()) {
          ET_FORALL_FLOAT_TYPES(SYM_CALCULATE_FLOAT_TYPE_CHANNEL);
          default:
            ET_CHECK_MSG(
                false,
                "Unhandled input dtype %" PRId8,
                static_cast<int8_t>(input.scalar_type()));
        }
      }
#undef SYM_CALCULATE_FLOAT_TYPE_TENSOR
#undef SYM_CALCULATE_FLOAT_TYPE_CHANNEL
#undef SYM_QUANTIZE_IMPL_TENSOR
#undef SYM_QUANTIZE_IMPL_CHANNEL
    }
  }
  return out;
}

// Quantize the input tensor
Tensor& quantize_per_tensor_out(
    KernelRuntimeContext& context,
    const Tensor& input,
    double scale,
    int64_t zero_point,
    int64_t quant_min,
    int64_t quant_max,
    ScalarType dtype,
    Tensor& out) {
#ifdef OP_ARG_CHECK
  Error err = resize_tensor(out, input.sizes());
  ET_CHECK_MSG(
      err == Error::Ok,
      "Failed to resize out Tensor in quantize_per_tensor_out");
  check_quantize_per_tensor_args(input, quant_min, quant_max, dtype, out);
#endif

  float scale_data = (float)scale;
  int zero_point_data = (int)zero_point;
  quantize_impl(
      context,
      out,
      input,
      &scale_data,
      &zero_point_data,
      NULL,
      (int)quant_min,
      (int)quant_max);

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
    context.fail(Error::InvalidArgument);
    return out;
  }
#ifdef OP_ARG_CHECK
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
#endif

  quantize_per_tensor_out(
      context,
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
  auto context = torch::executor::RuntimeContext();
  auto& res = quantize_per_tensor_tensor_args_out(
      context, input, scale, zero_point, quant_min, quant_max, dtype, out);
  ET_CHECK(context.failure_state() == Error::Ok);
  return res;
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
  if (axis < 0) {
    axis += executorch::runtime::nonzero_dim(input);
  }

#ifdef OP_ARG_CHECK
  Error err = resize_tensor(out, input.sizes());
  // normalize axis
  ET_CHECK_MSG(
      executorch::runtime::tensor_has_dim(input, axis),
      "axis %zd is not legal it should be -input.dim() <= axis < input.dim() %zd",
      ssize_t(axis),
      ssize_t(input.dim()));

  ET_CHECK_MSG(
      err == Error::Ok,
      "Failed to resize out Tensor in quantize_per_channel_out");

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
#endif

  const double* scale_dt = scale.const_data_ptr<double>();
  const int64_t* zero_point_dt = zero_point.const_data_ptr<int64_t>();

  float scale_data[input.size(axis)];
  int zero_point_data[input.size(axis)];

  for (int i = 0; i < scale.numel(); i++) {
    scale_data[i] = (float)scale_dt[i];
    zero_point_data[i] = (int)zero_point_dt[i];
  }

  int* axis_ptr = (int*)&axis;

  quantize_impl(
      context,
      out,
      input,
      scale_data,
      zero_point_data,
      axis_ptr,
      (int)quant_min,
      (int)quant_max);

  return out;
}

Tensor& quantize_per_token_out(
    KernelRuntimeContext& context,
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
  std::array<::executorch::aten::DimOrderType, 2> input_dim_order{0, 1};
  std::array<::executorch::aten::SizesType, 2> input_sizes;
  input_sizes[0] = num_tokens;
  input_sizes[1] = input.size(input.dim() - 1);
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
  Error err = resize_tensor(out, input.sizes());
  ET_CHECK_MSG(
      err == Error::Ok,
      "Failed to resize out Tensor in quantize_per_channel_out");
#endif

  return quantize_per_channel_out(
      context,
      reshaped_input,
      scale,
      zero_point,
      0,
      quant_min,
      quant_max,
      dtype,
      out);
}

} // namespace native
} // namespace G3
} // namespace impl
