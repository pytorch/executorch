/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
#include <executorch/backends/cadence/vision/kernels/kernels.h>
#include <executorch/backends/cadence/vision/operators/operators.h>
#include <executorch/backends/cadence/vision/operators/quantized_ops.h>

namespace impl {
namespace vision {
namespace native {

using ::executorch::aten::optional;
using ::executorch::aten::ScalarType;
using ::executorch::aten::Tensor;
using ::executorch::runtime::KernelRuntimeContext;

void quantized_fully_connected_out(
    __ET_UNUSED KernelRuntimeContext& ctx,
    const Tensor& in,
    const Tensor& weight,
    const Tensor& bias,
    int64_t in_zero_point,
    const Tensor& weight_zero_point_t,
    const Tensor& out_multiplier,
    const Tensor& out_shift,
    int64_t out_zero_point,
    __ET_UNUSED const optional<Tensor>& offset,
    Tensor& out) {
#define typed_quantized_linear(ctype, dtype) \
  case ScalarType::dtype: {                  \
    quantized_linear_<ctype>(                \
        in,                                  \
        weight,                              \
        bias,                                \
        in_zero_point,                       \
        weight_zero_point_t,                 \
        out_multiplier,                      \
        out_shift,                           \
        out_zero_point,                      \
        out);                                \
    break;                                   \
  }

  ScalarType dtype = out.scalar_type();
  switch (dtype) {
    ET_FORALL_CADENCE_QUANTIZED_TYPES(typed_quantized_linear);
    default:
      ET_DCHECK_MSG(
          false, "Unhandled dtype %s", torch::executor::toString(dtype));
  }
#undef typed_quantized_linear
}

void quantized_fully_connected_per_tensor_out(
    __ET_UNUSED KernelRuntimeContext& ctx,
    const Tensor& in,
    const Tensor& weight,
    const Tensor& bias,
    int64_t in_zero_point,
    int64_t weight_zero_point,
    int64_t out_multiplier,
    int64_t out_shift,
    int64_t out_zero_point,
    __ET_UNUSED const optional<Tensor>& offset,
    Tensor& out) {
#define typed_quantized_linear(ctype, dtype) \
  case ScalarType::dtype: {                  \
    quantized_linear_per_tensor_<ctype>(     \
        in,                                  \
        weight,                              \
        bias,                                \
        in_zero_point,                       \
        weight_zero_point,                   \
        out_multiplier,                      \
        out_shift,                           \
        out_zero_point,                      \
        out);                                \
    break;                                   \
  }

  ScalarType dtype = out.scalar_type();
  switch (dtype) {
    ET_FORALL_CADENCE_QUANTIZED_TYPES(typed_quantized_linear);
    default:
      ET_DCHECK_MSG(
          false, "Unhandled dtype %s", torch::executor::toString(dtype));
  }
#undef typed_quantized_linear
}

}; // namespace native
}; // namespace vision
}; // namespace impl
