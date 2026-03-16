/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/backends/cadence/generic/operators/op_quantized_linear.h>

#include <executorch/backends/cadence/generic/kernels/kernels.h>
#include <executorch/backends/cadence/generic/operators/cadence_type_util.h>
#include <executorch/backends/cadence/generic/operators/quantized_linear.h>
#include <executorch/runtime/core/exec_aten/util/scalar_type_util.h>
#include <executorch/runtime/core/exec_aten/util/tensor_util.h>

namespace impl {
namespace generic {
namespace native {

using ::executorch::aten::optional;
using ::executorch::aten::ScalarType;
using ::executorch::aten::Tensor;
using ::executorch::runtime::KernelRuntimeContext;
using ::executorch::runtime::toString;

Tensor& quantized_linear_out(
    ET_UNUSED KernelRuntimeContext& ctx,
    const Tensor& src,
    const Tensor& weight,
    const Tensor& bias,
    int64_t src_zero_point,
    const Tensor& weight_zero_point_t,
    const Tensor& out_multiplier,
    const Tensor& out_shift,
    int64_t out_zero_point,
    ET_UNUSED const optional<Tensor>& offset,
    Tensor& out) {
#define typed_quantized_linear(ctype, dtype)              \
  case ScalarType::dtype: {                               \
    ::impl::generic::quantized::quantized_linear_<ctype>( \
        src,                                              \
        weight,                                           \
        bias,                                             \
        src_zero_point,                                   \
        weight_zero_point_t,                              \
        out_multiplier,                                   \
        out_shift,                                        \
        out_zero_point,                                   \
        out);                                             \
    break;                                                \
  }

  ScalarType dtype = out.scalar_type();

  // Handle W8A16 heterogeneous type (int16_t activations, int8_t weights)
  if (dtype == ScalarType::Short && src.scalar_type() == ScalarType::Short &&
      weight.scalar_type() == ScalarType::Char) {
    ::impl::generic::quantized::quantized_linear_<int16_t, int8_t>(
        src,
        weight,
        bias,
        src_zero_point,
        weight_zero_point_t,
        out_multiplier,
        out_shift,
        out_zero_point,
        out);
    return out;
  }

  switch (dtype) {
    ET_FORALL_CADENCE_QUANTIZED_TYPES_WITH_INT16(typed_quantized_linear);
    default:
      ET_DCHECK_MSG(false, "Unhandled dtype %s", toString(dtype));
  }
#undef typed_quantized_linear
  return out;
}

Tensor& quantized_linear_per_tensor_out(
    KernelRuntimeContext& ctx,
    const Tensor& src,
    const Tensor& weight,
    const Tensor& bias,
    const int64_t src_zero_point,
    const int64_t weight_zero_point,
    const int64_t out_multiplier,
    const int64_t out_shift,
    const int64_t out_zero_point,
    ET_UNUSED const optional<Tensor>& offset,
    Tensor& out) {
#define typed_quantized_linear_per_tensor(ctype, dtype)              \
  case ScalarType::dtype: {                                          \
    ::impl::generic::quantized::quantized_linear_per_tensor_<ctype>( \
        src,                                                         \
        weight,                                                      \
        bias,                                                        \
        src_zero_point,                                              \
        weight_zero_point,                                           \
        out_multiplier,                                              \
        out_shift,                                                   \
        out_zero_point,                                              \
        out);                                                        \
    break;                                                           \
  }

  ScalarType dtype = out.scalar_type();
  // Handle W8A16 heterogeneous type (int16_t activations, int8_t weights)
  if (dtype == ScalarType::Short && src.scalar_type() == ScalarType::Short &&
      weight.scalar_type() == ScalarType::Char) {
    ::impl::generic::quantized::quantized_linear_per_tensor_<int16_t, int8_t>(
        src,
        weight,
        bias,
        src_zero_point,
        weight_zero_point,
        out_multiplier,
        out_shift,
        out_zero_point,
        out);
    return out;
  }
  switch (dtype) {
    ET_FORALL_CADENCE_QUANTIZED_TYPES_WITH_INT16(
        typed_quantized_linear_per_tensor);
    default:
      ET_KERNEL_CHECK_MSG(
          ctx,
          false,
          InvalidArgument,
          out,
          "Unhandled dtype %s",
          toString(dtype));
  }
#undef typed_quantized_linear_per_tensor
  return out;
}

Tensor& quantized_linear_asym8sxasym8s_asym8s_per_tensor_out(
    ET_UNUSED KernelRuntimeContext& ctx,
    const Tensor& src,
    const Tensor& weight,
    const Tensor& bias,
    const int64_t src_zero_point,
    const int64_t weight_zero_point,
    const int64_t out_multiplier,
    const int64_t out_shift,
    const int64_t out_zero_point,
    ET_UNUSED const std::optional<Tensor>& offset,
    Tensor& out) {
#define typed_quantized_linear_per_tensor(ctype, dtype)              \
  case ScalarType::dtype: {                                          \
    ::impl::generic::quantized::quantized_linear_per_tensor_<ctype>( \
        src,                                                         \
        weight,                                                      \
        bias,                                                        \
        src_zero_point,                                              \
        weight_zero_point,                                           \
        out_multiplier,                                              \
        out_shift,                                                   \
        out_zero_point,                                              \
        out);                                                        \
    break;                                                           \
  }

  ScalarType dtype = out.scalar_type();
  switch (dtype) {
    ET_FORALL_CADENCE_QUANTIZED_TYPES_WITH_INT16(
        typed_quantized_linear_per_tensor);
    default:
      ET_DCHECK_MSG(
          false, "Unhandled dtype %s", executorch::runtime::toString(dtype));
  }
#undef typed_quantized_linear_per_tensor
  return out;
}

Tensor& quantized_linear_asym8uxasym8u_asym8u_per_tensor_out(
    ET_UNUSED KernelRuntimeContext& ctx,
    const Tensor& src,
    const Tensor& weight,
    const Tensor& bias,
    const int64_t src_zero_point,
    const int64_t weight_zero_point,
    const int64_t out_multiplier,
    const int64_t out_shift,
    const int64_t out_zero_point,
    ET_UNUSED const std::optional<Tensor>& offset,
    Tensor& out) {
#define typed_quantized_linear_per_tensor(ctype, dtype)              \
  case ScalarType::dtype: {                                          \
    ::impl::generic::quantized::quantized_linear_per_tensor_<ctype>( \
        src,                                                         \
        weight,                                                      \
        bias,                                                        \
        src_zero_point,                                              \
        weight_zero_point,                                           \
        out_multiplier,                                              \
        out_shift,                                                   \
        out_zero_point,                                              \
        out);                                                        \
    break;                                                           \
  }

  ScalarType dtype = out.scalar_type();
  switch (dtype) {
    ET_FORALL_CADENCE_QUANTIZED_TYPES_WITH_INT16(
        typed_quantized_linear_per_tensor);
    default:
      ET_DCHECK_MSG(
          false, "Unhandled dtype %s", executorch::runtime::toString(dtype));
  }
#undef typed_quantized_linear_per_tensor
  return out;
}

} // namespace native
} // namespace generic
} // namespace impl
