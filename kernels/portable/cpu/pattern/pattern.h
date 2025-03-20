/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

/* Naming Convention
--------------------
Each pattern name should be all lowercase words separated by underscores.

It should start with unary/binary/ternary depending on the number of input
tensors.

It should then specify the type of function that it is. E.g: ufunc, reduce, etc
Ufunc is a concept used in NumPy and PyTorch (short for universal function)
See NumPy's definition here: https://numpy.org/doc/stable/reference/ufuncs.html

It should then specify the sets of dtypes accepted as inputs and output.
We currently support the following sets of dtypes:
 - bool   {Bool}
 - int    {Byte, Char, Short, Int, Long}
 - intb   {Byte, Char, Short, Int, Long, Bool}
 - float  {Float, Double}
 - floath {Half, Float, Double}
 - floatb {Float, Double, Bool}
 - real   {Byte, Char, Short, Int, Long, Float, Double}
 - realb  {Byte, Char, Short, Int, Long, Float, Double, Bool}
 - realh  {Byte, Char, Short, Int, Long, Half, Float, Double}
 - realhb {Byte, Char, Short, Int, Long, Half, Float, Double, Bool}

Input types are separated from output types by the "to" word.
Input types are separated by underscores. Output types as well, in the cases
when there are multiple outputs.
In the case of ops that only allow the same dtype (for example abs/neg where
the input and output dtype must be the same, only that dtype is specified.
E.g, there is a difference between:
 - unary_ufunc_real_to_real
     (output and input types must be real, but might be different)
 - unary_ufunc_real
     (output must be same type than input, which must be real)

A pattern named using this convention will be quite generic. If the pattern in
question is a bit more specific, then add a descriptive sufix. */

#pragma once

#include <executorch/kernels/portable/cpu/util/elementwise_util.h>
#include <executorch/runtime/kernel/kernel_includes.h>

namespace torch {
namespace executor {
namespace native {
namespace internal {

// Implementation detail for the other helpers in this header. Returns
// true on success, false on failure.
bool check_and_resize_inputs(
    KernelRuntimeContext& ctx,
    const Tensor& in,
    Tensor& out);

/**
 * Implements an op pattern for ops that take a single input tensor of any
 * realh dtype, no additional arguments, and outputs a tensor of the same size
 * and dtype. The function fn specifies the math operation which is applied to
 * the input tensor element-wise.
 */
template <const char* op_name, typename Op>
Tensor& unary_ufunc_realh(
    const Op& fn,
    KernelRuntimeContext& ctx,
    const Tensor& in,
    Tensor& out) {
  if (!check_and_resize_inputs(ctx, in, out)) {
    return out;
  }
  ET_KERNEL_CHECK(
      ctx, tensors_have_same_shape_and_dtype(in, out), InvalidArgument, out);

  ET_SWITCH_REALH_TYPES(in.scalar_type(), ctx, op_name, CTYPE, [&] {
    utils::apply_unitensor_elementwise_fn<CTYPE, op_name>(
        fn,
        ctx,
        in,
        utils::SupportedTensorDtypes::REALH,
        out,
        utils::SupportedTensorDtypes::SAME_AS_COMMON);
  });
  return out;
}

/**
 * Implements an op pattern for ops that take a single input tensor of any
 * realhb dtype (real, half and boolean), no additional arguments, and outputs a
 * boolean tensor of the same size. The function fn specifies the math
 * operation which is applied to the input tensor element-wise.
 */
template <const char* op_name, typename Op>
Tensor& unary_ufunc_realhb_to_bool(
    const Op& fn,
    KernelRuntimeContext& ctx,
    const Tensor& in,
    Tensor& out) {
  if (!check_and_resize_inputs(ctx, in, out)) {
    return out;
  }
  ET_SWITCH_REALHBBF16_TYPES(in.scalar_type(), ctx, op_name, CTYPE_IN, [&] {
    utils::apply_unitensor_elementwise_fn<CTYPE_IN, op_name>(
        [fn](const CTYPE_IN val_in) { return fn(val_in); },
        ctx,
        in,
        utils::SupportedTensorDtypes::REALHBBF16,
        out,
        utils::SupportedTensorDtypes::BOOL);
  });

  return out;
}

/**
 * Implements an op pattern for ops that take a single input tensor of any
 * realhbbf16 dtype (real/half/bool/bfloat16), no additional arguments, and
 * outputs a floating point tensor of the same size. The function fn specifies
 * the math operation which is applied to the input tensor element-wise.
 */
template <const char* op_name, typename Op>
Tensor& unary_ufunc_realhbbf16_to_floathbf16(
    const Op& fn,
    KernelRuntimeContext& ctx,
    const Tensor& in,
    Tensor& out) {
  ET_KERNEL_CHECK(ctx, tensor_is_floating_type(out), InvalidArgument, out);

  if (!check_and_resize_inputs(ctx, in, out)) {
    return out;
  }

  ET_SWITCH_REALHBBF16_TYPES(in.scalar_type(), ctx, op_name, CTYPE_IN, [&] {
    utils::apply_unitensor_elementwise_fn<CTYPE_IN, op_name>(
        [fn](const CTYPE_IN val_in) { return fn(val_in); },
        ctx,
        in,
        utils::SupportedTensorDtypes::REALHBBF16,
        out,
        utils::SupportedTensorDtypes::FLOATHBF16);
  });

  return out;
}

} // namespace internal
} // namespace native
} // namespace executor
} // namespace torch
