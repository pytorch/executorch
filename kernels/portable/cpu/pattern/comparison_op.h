/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <executorch/kernels/portable/cpu/scalar_utils.h>
#include <executorch/kernels/portable/cpu/util/elementwise_util.h>
#include <executorch/runtime/kernel/kernel_includes.h>

namespace torch {
namespace executor {
namespace native {
namespace internal {

#define DEFINE_BINARY_OPERATOR_TEMPLATE(name, op) \
  template <typename T>                           \
  T name(const T val_a, const T val_b) {          \
    return val_a op val_b;                        \
  }

DEFINE_BINARY_OPERATOR_TEMPLATE(eq, ==)
DEFINE_BINARY_OPERATOR_TEMPLATE(ne, !=)
DEFINE_BINARY_OPERATOR_TEMPLATE(ge, >=)
DEFINE_BINARY_OPERATOR_TEMPLATE(le, <=)
DEFINE_BINARY_OPERATOR_TEMPLATE(gt, >)
DEFINE_BINARY_OPERATOR_TEMPLATE(lt, <)

template <typename T>
using comparison_fn = T (*)(const T, const T);

template <typename T, const char* op_name>
constexpr comparison_fn<T> get_comparison_fn() {
  std::string_view op = op_name;
  if (op == "eq.Tensor_out" || op == "eq.Scalar_out") {
    return eq;
  }
  if (op == "ne.Tensor_out" || op == "ne.Scalar_out") {
    return ne;
  }
  if (op == "ge.Tensor_out" || op == "ge.Scalar_out") {
    return ge;
  }
  if (op == "le.Tensor_out" || op == "le.Scalar_out") {
    return le;
  }
  if (op == "gt.Tensor_out" || op == "gt.Scalar_out") {
    return gt;
  }
  if (op == "lt.Tensor_out" || op == "lt.Scalar_out") {
    return lt;
  }
  return nullptr;
};

template <typename T, const char* op_name>
struct ComparisonFnForOp {
  static constexpr auto value = get_comparison_fn<T, op_name>();
  static_assert(value != nullptr, "unknown op_name!");
};

template <const char* op_name>
Tensor& comparison_tensor_out(
    KernelRuntimeContext& ctx,
    const Tensor& a,
    const Tensor& b,
    Tensor& out) {
  // Common Dtype
  ScalarType common_type = promoteTypes(a.scalar_type(), b.scalar_type());
  if (executorch::runtime::isFloatingType(common_type) &&
      a.scalar_type() != b.scalar_type()) {
    common_type = ScalarType::Float;
  }

  // Check Dim Order
  ET_KERNEL_CHECK(
      ctx, tensors_have_same_dim_order(a, b, out), InvalidArgument, out);

  // Resize
  ET_KERNEL_CHECK(
      ctx,
      resize_to_broadcast_target_size(a, b, out) == Error::Ok,
      InvalidArgument,
      out);

  // Compute Dtype
  ScalarType compute_type = utils::get_compute_type(common_type);

  ET_SWITCH_REALB_TYPES(compute_type, ctx, op_name, CTYPE_COMPUTE, [&]() {
    utils::apply_bitensor_elementwise_fn<CTYPE_COMPUTE, op_name>(
        ComparisonFnForOp<CTYPE_COMPUTE, op_name>::value,
        ctx,
        a,
        utils::SupportedTensorDtypes::REALHBBF16,
        b,
        utils::SupportedTensorDtypes::REALHBBF16,
        out,
        utils::SupportedTensorDtypes::REALHBBF16);
  });

  return out;
}

template <const char* op_name>
Tensor& comparison_scalar_out(
    KernelRuntimeContext& ctx,
    const Tensor& a,
    const Scalar& b,
    Tensor& out) {
  // Common Dtype
  ScalarType common_type = utils::promote_type_with_scalar(a.scalar_type(), b);

  // Check Dim Order
  ET_KERNEL_CHECK(
      ctx, tensors_have_same_dim_order(a, out), InvalidArgument, out);

  // Resize
  ET_KERNEL_CHECK(
      ctx, resize_tensor(out, a.sizes()) == Error::Ok, InvalidArgument, out);

  // Compute Dtype
  ScalarType compute_type = utils::get_compute_type(common_type);

  ET_SWITCH_REALB_TYPES(compute_type, ctx, op_name, CTYPE_COMPUTE, [&]() {
    const CTYPE_COMPUTE val_b = utils::scalar_to<CTYPE_COMPUTE>(b);
    utils::apply_unitensor_elementwise_fn<CTYPE_COMPUTE, op_name>(
        [val_b](const CTYPE_COMPUTE val_a) {
          return ComparisonFnForOp<CTYPE_COMPUTE, op_name>::value(val_a, val_b);
        },
        ctx,
        a,
        utils::SupportedTensorDtypes::REALHBBF16,
        out,
        utils::SupportedTensorDtypes::REALHBBF16);
  });

  return out;
}

} // namespace internal
} // namespace native
} // namespace executor
} // namespace torch
