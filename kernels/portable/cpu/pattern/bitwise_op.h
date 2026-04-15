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

DEFINE_BINARY_OPERATOR_TEMPLATE(bitwise_and, &)
DEFINE_BINARY_OPERATOR_TEMPLATE(bitwise_or, |)
DEFINE_BINARY_OPERATOR_TEMPLATE(bitwise_xor, ^)

// Functor wrappers for shift operations (similar to std::bit_and, etc.)
template <typename T = void>
struct bit_lshift {
  constexpr T operator()(const T& lhs, const T& rhs) const {
    return static_cast<T>(lhs << rhs);
  }
};

template <typename T = void>
struct bit_rshift {
  constexpr T operator()(const T& lhs, const T& rhs) const {
    return static_cast<T>(lhs >> rhs);
  }
};

template <typename T>
using bitwise_fn = T (*)(const T, const T);

template <typename T, const char* op_name>
constexpr bitwise_fn<T> get_bitwise_fn() {
  std::string_view op = op_name;
  if (op == "bitwise_and.Tensor_out" || op == "bitwise_and.Scalar_out") {
    return bitwise_and;
  }
  if (op == "bitwise_or.Tensor_out" || op == "bitwise_or.Scalar_out") {
    return bitwise_or;
  }
  if (op == "bitwise_xor.Tensor_out" || op == "bitwise_xor.Scalar_out") {
    return bitwise_xor;
  }
  return nullptr;
};

template <typename T, const char* op_name>
struct BitwiseFnForOp {
  static constexpr auto get_value() {
    return get_bitwise_fn<T, op_name>();
  }
  static_assert(get_value() != nullptr, "unknown op_name!");
};

template <template <typename> class BitOp, const char* op_name>
Tensor& bitwise_tensor_out(
    RuntimeContext& ctx,
    const Tensor& a,
    const Tensor& b,
    Tensor& out) {
  // Common Dtype
  ScalarType common_type = promoteTypes(a.scalar_type(), b.scalar_type());

  // Check Common Dtype
  ET_KERNEL_CHECK(
      ctx, canCast(common_type, out.scalar_type()), InvalidArgument, out);

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

  ET_SWITCH_INT_TYPES_AND(
      Bool, compute_type, ctx, op_name, CTYPE_COMPUTE, [&]() {
        utils::apply_bitensor_elementwise_fn<
            CTYPE_COMPUTE,
            op_name,
            utils::SupportedTensorDtypes::REALHBBF16>(
            // TODO: rewrite this to be vectorization-capable.
            BitOp<CTYPE_COMPUTE>(),
            ctx,
            a,
            utils::SupportedTensorDtypes::INTB,
            b,
            utils::SupportedTensorDtypes::INTB,
            out);
      });

  return out;
}

template <template <typename> class BitOp, const char* op_name>
Tensor& bitwise_scalar_out(
    RuntimeContext& ctx,
    const Tensor& a,
    const Scalar& b,
    Tensor& out) {
  // Common Dtype
  ScalarType common_type = utils::promote_type_with_scalar(a.scalar_type(), b);

  // Check Common Dtype
  ET_KERNEL_CHECK(
      ctx, canCast(common_type, out.scalar_type()), InvalidArgument, out);

  // Check Dim Order
  ET_KERNEL_CHECK(
      ctx, tensors_have_same_dim_order(a, out), InvalidArgument, out);

  // Resize
  ET_KERNEL_CHECK(
      ctx, resize_tensor(out, a.sizes()) == Error::Ok, InvalidArgument, out);

  // Compute Dtype
  ScalarType compute_type = utils::get_compute_type(common_type);

  ET_SWITCH_INT_TYPES_AND(
      Bool, compute_type, ctx, op_name, CTYPE_COMPUTE, [&]() {
        const CTYPE_COMPUTE val_b = utils::scalar_to<CTYPE_COMPUTE>(b);
        utils::apply_unitensor_elementwise_fn<
            CTYPE_COMPUTE,
            op_name,
            utils::SupportedTensorDtypes::REALHBBF16>(
            [val_b](const CTYPE_COMPUTE val_a) {
              // TODO: rewrite this to be vectorization-capable.
              return BitOp<CTYPE_COMPUTE>()(val_a, val_b);
            },
            ctx,
            a,
            utils::SupportedTensorDtypes::INTB,
            out);
      });

  return out;
}

} // namespace internal
} // namespace native
} // namespace executor
} // namespace torch
