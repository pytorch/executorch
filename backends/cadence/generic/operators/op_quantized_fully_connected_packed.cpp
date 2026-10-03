/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/backends/cadence/generic/operators/op_quantized_fully_connected_packed.h>

#include <executorch/backends/cadence/generic/operators/cadence_type_util.h>
#include <executorch/backends/cadence/generic/operators/quantized_linear.h>
#include <executorch/backends/cadence/generic/operators/quantized_linear_packed.h>
#include <executorch/runtime/core/exec_aten/util/scalar_type_util.h>

namespace impl {
namespace generic {
namespace native {

using ::executorch::aten::ScalarType;
using ::executorch::aten::Tensor;
using ::executorch::runtime::KernelRuntimeContext;
using std::optional;

Tensor& quantized_fully_connected_packed_out(
    ET_UNUSED KernelRuntimeContext& ctx,
    const Tensor& in,
    const Tensor& weight,
    const Tensor& bias,
    int64_t in_dim,
    int64_t weight_bits,
    int64_t in_zero_point,
    const optional<Tensor>& weight_zero_point_t,
    const Tensor& out_multiplier,
    const Tensor& out_shift,
    int64_t out_zero_point,
    ET_UNUSED const optional<Tensor>& offset,
    Tensor& out) {
  // Mirrors quantized_linear_: the generic kernels carry a per-channel
  // multiplier and shift but a single weight zero point.
  const int64_t weight_zero_point =
      ::impl::generic::quantized::resolve_weight_zero_point(weight_zero_point_t);

  const ScalarType dtype = out.scalar_type();

#define typed_quantized_linear_packed(ctype, name)                           \
  if (dtype == ScalarType::name) {                                           \
    ::impl::generic::quantized::quantized_linear_packed_per_channel_<ctype>( \
        in,                                                                  \
        weight,                                                              \
        bias,                                                                \
        in_dim,                                                              \
        weight_bits,                                                         \
        in_zero_point,                                                       \
        weight_zero_point,                                                   \
        out_multiplier,                                                      \
        out_shift,                                                           \
        out_zero_point,                                                      \
        out);                                                                \
    return out;                                                              \
  }

  ET_FORALL_CADENCE_QUANTIZED_TYPES_WITH_INT16(typed_quantized_linear_packed)
#undef typed_quantized_linear_packed

  ET_DCHECK_MSG(false, "Unhandled dtype %s", torch::executor::toString(dtype));
  return out;
}

} // namespace native
} // namespace generic
} // namespace impl
