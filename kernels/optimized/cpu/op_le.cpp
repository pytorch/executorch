/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <ATen/cpu/vec/functional.h>
#include <ATen/cpu/vec/vec.h>
#include <executorch/kernels/optimized/cpu/binary_ops.h>
#include <executorch/kernels/portable/cpu/pattern/comparison_op.h>
#include <executorch/kernels/portable/cpu/scalar_utils.h>
#include <executorch/kernels/portable/cpu/util/broadcast_util.h>
#include <executorch/runtime/kernel/kernel_includes.h>
#include <executorch/runtime/platform/assert.h>

namespace torch {
namespace executor {
namespace native {

using Tensor = executorch::aten::Tensor;
using ScalarType = executorch::aten::ScalarType;

Tensor& opt_le_tensor_out(
    KernelRuntimeContext& ctx,
    const Tensor& a,
    const Tensor& b,
    Tensor& out) {
  ScalarType a_type = a.scalar_type();
  ScalarType out_type = out.scalar_type();

  ET_KERNEL_CHECK(
      ctx, tensors_have_same_dim_order(a, b, out), InvalidArgument, out);

  ET_KERNEL_CHECK(
      ctx,
      resize_to_broadcast_target_size(a, b, out) == Error::Ok,
      InvalidArgument,
      out);

  // @lint-ignore CLANGTIDY facebook-hte-CArray
  static constexpr const char op_name[] = "le.Tensor_out";

  // Check for optimized broadcast paths
  auto selected_optimized_path = select_optimized_path(a, b, out);
  if (selected_optimized_path == ElementwiseOptimizedPath::kTreatAs1d) {
    ET_SWITCH_REALB_TYPES(a_type, ctx, op_name, CTYPE, [&]() {
      using Vec = at::vec::Vectorized<CTYPE>;
      at::vec::map2<CTYPE>(
          [](Vec x, Vec y) { return x.le(y); },
          out.mutable_data_ptr<CTYPE>(),
          a.const_data_ptr<CTYPE>(),
          b.const_data_ptr<CTYPE>(),
          out.numel());
    });
  } else if (selected_optimized_path != ElementwiseOptimizedPath::kNone) {
    // Handle optimized broadcast cases
    ET_SWITCH_REALB_TYPES(out_type, ctx, op_name, CTYPE, [&]() {
      auto le_lambda = [](auto x, auto y) { return x.le(y); };
      torch::executor::handle_broadcast_elementwise<CTYPE>(
          ctx, le_lambda, a, b, out, selected_optimized_path);
    });
  } else {
    internal::comparison_tensor_out<std::less_equal, op_name>(ctx, a, b, out);
  }

  return out;
}

Tensor& opt_le_scalar_out(
    KernelRuntimeContext& ctx,
    const Tensor& a,
    const Scalar& b,
    Tensor& out) {
  ScalarType a_type = a.scalar_type();
  ScalarType b_type = utils::get_scalar_dtype(b);
  ScalarType common_type = utils::promote_type_with_scalar(a_type, b);
  ScalarType out_type = out.scalar_type();

  ET_KERNEL_CHECK(
      ctx, tensors_have_same_dim_order(a, out), InvalidArgument, out);

  ET_KERNEL_CHECK(
      ctx, resize_tensor(out, a.sizes()) == Error::Ok, InvalidArgument, out);

  // @lint-ignore CLANGTIDY facebook-hte-CArray
  static constexpr const char op_name[] = "le.Scalar_out";

  if (a_type == common_type && a_type == out_type &&
      a_type != ScalarType::Half && a_type != ScalarType::BFloat16) {
    ET_SWITCH_REALB_TYPES(a_type, ctx, op_name, CTYPE, [&]() {
      ET_SWITCH_REALB_TYPES(b_type, ctx, op_name, CTYPE_B, [&]() {
        CTYPE_B b_val = 0;
        ET_EXTRACT_SCALAR(b, b_val);
        CTYPE b_casted = static_cast<CTYPE>(b_val);
        using Vec = at::vec::Vectorized<CTYPE>;
        at::vec::map<CTYPE>(
            [b_casted](Vec x) { return x.le(Vec(b_casted)); },
            out.mutable_data_ptr<CTYPE>(),
            a.const_data_ptr<CTYPE>(),
            a.numel());
      });
    });
  } else {
    internal::comparison_scalar_out<std::less_equal, op_name>(ctx, a, b, out);
  }

  return out;
}

} // namespace native
} // namespace executor
} // namespace torch
