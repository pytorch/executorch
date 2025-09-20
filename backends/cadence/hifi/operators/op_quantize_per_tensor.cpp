/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <xa_type_def.h>

#include <xa_nnlib_kernels_api.h>

#include <executorch/backends/cadence/hifi/kernels/kernels.h>
#include <executorch/runtime/core/exec_aten/exec_aten.h>
#include <executorch/runtime/core/exec_aten/util/scalar_type_util.h>
#include <executorch/runtime/kernel/kernel_includes.h>
#include <executorch/runtime/kernel/kernel_runtime_context.h>

namespace impl {
namespace HiFi {
namespace native {

namespace {

using ::executorch::aten::ScalarType;
using ::executorch::aten::Tensor;
using ::executorch::runtime::KernelRuntimeContext;
using ::impl::HiFi::kernels::quantize;

// Add checks for dtype quant min/max bounds.
template <typename T>
void check_quant_min_and_max(
    KernelRuntimeContext& ctx,
    const int64_t quant_min,
    const int64_t quant_max) {
  ET_KERNEL_CHECK(
      ctx,
      std::numeric_limits<T>::min() == quant_min &&
          std::numeric_limits<T>::max() == quant_max,
      InvalidArgument, );
}

} // namespace

// Quantize the input tensor (PT2 version). Note that quant_<min,max> are not
// used in any computation.
void quantize_per_tensor_out(
    KernelRuntimeContext& ctx,
    const Tensor& input,
    double scale,
    const int64_t zero_point,
    __ET_UNUSED int64_t quant_min,
    __ET_UNUSED int64_t quant_max,
    const ScalarType dtype,
    Tensor& out) {
  // Check for input scalar type.
  ET_KERNEL_CHECK_MSG(
      ctx,
      input.scalar_type() == ScalarType::Float,
      InvalidType,
      ,
      "Input tensor for quantize_per_tensor.out should be type %s, but got %s",
      ::torch::executor::toString(ScalarType::Float),
      ::torch::executor::toString(input.scalar_type()));

  // Check quant min/max for output types.
  switch (out.scalar_type()) {
    case ScalarType::Byte:
      check_quant_min_and_max<uint8_t>(ctx, quant_min, quant_max);
      break;
    case ScalarType::Char:
      check_quant_min_and_max<int8_t>(ctx, quant_min, quant_max);
      break;
    case ScalarType::Short:
      check_quant_min_and_max<int16_t>(ctx, quant_min, quant_max);
      break;
    case ScalarType::Bits16:
    case ScalarType::UInt16:
      check_quant_min_and_max<uint16_t>(ctx, quant_min, quant_max);
      break;
    case ScalarType::Int:
      check_quant_min_and_max<int32_t>(ctx, quant_min, quant_max);
      break;
    default:
      ET_KERNEL_CHECK_MSG(
          ctx,
          false,
          InvalidType,
          ,
          "Unhandled output dtype %s",
          ::torch::executor::toString(out.scalar_type()));
  }

  const float* input_data = input.const_data_ptr<float>();
  const size_t numel = out.numel();
  if (out.scalar_type() == ScalarType::Byte) {
    uint8_t* out_data = out.mutable_data_ptr<uint8_t>();
    quantize<uint8_t>(out_data, input_data, 1. / scale, zero_point, numel);
  } else if (out.scalar_type() == ScalarType::Char) {
    int8_t* out_data = out.mutable_data_ptr<int8_t>();
    xa_nn_elm_quantize_f32_asym8s(
        out_data, input_data, scale, zero_point, numel);
  } else if (out.scalar_type() == ScalarType::Short) {
    int16_t* out_data = out.mutable_data_ptr<int16_t>();
    quantize<int16_t>(out_data, input_data, 1. / scale, zero_point, numel);
  } else if (
      out.scalar_type() == ScalarType::Bits16 ||
      out.scalar_type() == ScalarType::UInt16) {
    uint16_t* out_data = out.mutable_data_ptr<uint16_t>();
    quantize<uint16_t>(out_data, input_data, 1. / scale, zero_point, numel);
  } else {
    ET_KERNEL_CHECK_MSG(
        ctx,
        false,
        InvalidType,
        ,
        "Unhandled output dtype %s",
        ::torch::executor::toString(out.scalar_type()));
  }
}

void quantize_per_tensor_asym8u_out(
    KernelRuntimeContext& context,
    const Tensor& input,
    double scale,
    int64_t zero_point,
    int64_t quant_min,
    int64_t quant_max,
    ScalarType dtype,
    Tensor& out) {
  const float* input_data = input.const_data_ptr<float>();
  size_t numel = out.numel();
  uint8_t* out_data = out.mutable_data_ptr<uint8_t>();
  quantize<uint8_t>(out_data, input_data, 1. / scale, zero_point, numel);
}

void quantize_per_tensor_asym16s_out(
    KernelRuntimeContext& context,
    const Tensor& input,
    double scale,
    int64_t zero_point,
    int64_t quant_min,
    int64_t quant_max,
    ScalarType dtype,
    Tensor& out) {
  const float* input_data = input.const_data_ptr<float>();
  size_t numel = out.numel();
  int16_t* out_data = out.mutable_data_ptr<int16_t>();
  quantize<int16_t>(out_data, input_data, 1. / scale, zero_point, numel);
}

void quantize_per_tensor_asym16u_out(
    KernelRuntimeContext& context,
    const Tensor& input,
    double scale,
    int64_t zero_point,
    int64_t quant_min,
    int64_t quant_max,
    ScalarType dtype,
    Tensor& out) {
  const float* input_data = input.const_data_ptr<float>();
  size_t numel = out.numel();
  uint16_t* out_data = out.mutable_data_ptr<uint16_t>();
  quantize<uint16_t>(out_data, input_data, 1. / scale, zero_point, numel);
}

}; // namespace native
}; // namespace HiFi
}; // namespace impl
