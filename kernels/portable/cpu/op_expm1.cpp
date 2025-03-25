/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/kernels/portable/cpu/pattern/pattern.h>
#include <executorch/kernels/portable/cpu/util/elementwise_util.h>
#include <executorch/runtime/kernel/kernel_includes.h>
#include <cmath>

namespace torch {
namespace executor {
namespace native {

// REVIEW: I'm not entirely sure what the best way to implement this
// namespace is. Some options:
// 1) All in one file, with or without an `IMPLEMENT_VECTORIZED_MATH_OP` macro.
// 2) Include in each `unary_ufunc_*` op_foo.cpp, with or without an
// `IMPLEMENT_VECTORIZED_MATH_OP` macro.
//
// I think my preferred option would be (2) with a macro, but I've
// left the macro out for ease of reading this PoC PR.
namespace math {
using std::expm1;
#ifdef ET_USE_PYTORCH_HEADERS
template <typename T>
auto expm1(at::vec::Vectorized<T> x) {
  // ATen knows to do this conversion because the TensorIterator for this op
  // (and lots of similar ones in aten/src/ATen/native/UnaryOps.cpp) is created
  // with build_borrowing_unary_float_op.
  if constexpr (!executorch::runtime::is_floating_point<T>::value) {
    return at::vec::convert<float>(x).expm1();
  } else {
    return x.expm1();
  }
}
#endif
} // namespace math
Tensor& expm1_out(KernelRuntimeContext& ctx, const Tensor& in, Tensor& out) {
  ET_KERNEL_CHECK(ctx, tensor_is_floating_type(out), InvalidArgument, out);

  // Resize for dynamic shape
  ET_KERNEL_CHECK_MSG(
      ctx,
      resize_tensor(out, in.sizes()) == Error::Ok,
      InvalidArgument,
      out,
      "Failed to resize output tensor.");

  ET_KERNEL_CHECK(
      ctx, tensors_have_same_dim_order(in, out), InvalidArgument, out);

  static constexpr const char op_name[] = "expm1.out";
  ET_SWITCH_REALHBBF16_TYPES(in.scalar_type(), ctx, op_name, CTYPE_IN, [&] {
    utils::apply_unitensor_elementwise_fn<
        CTYPE_IN,
        op_name,
        utils::SupportedTensorDtypes::FLOATHBF16>(
        [](auto x) { return math::expm1(x); },
        ctx,
        in,
        utils::SupportedTensorDtypes::REALHBBF16,
        out);
  });

  return out;
}

} // namespace native
} // namespace executor
} // namespace torch
