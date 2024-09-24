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

#include <executorch/runtime/kernel/kernel_includes.h>

namespace torch {
namespace executor {
namespace native {
namespace internal {

/**
 * Implements an op pattern for ops that take a single input tensor of any
 * realh dtye, no additional arguments, and outputs a tensor of the same size
 * and dtype. The function fn specifies the math operation which is applied to
 * the input tensor element-wise.
 */
Tensor& unary_ufunc_realh(
    double (*fn)(double),
    KernelRuntimeContext& ctx,
    const Tensor& in,
    Tensor& out);

/**
 * Implements an op pattern for ops that take a single input tensor of any
 * realhb dtye (real, half and boolean), no additional arguments, and outputs a
 * boolean tensor of the same size. The function fn specifies the math
 * operation which is applied to the input tensor element-wise.
 */
Tensor& unary_ufunc_realhb_to_bool(
    bool (*fn)(double),
    KernelRuntimeContext& ctx,
    const Tensor& in,
    Tensor& out);

/**
 * Implements an op pattern for ops that take a single input tensor of any
 * realhb dtye (real, half and boolean), no additional arguments, and outputs a
 * floating point tensor of the same size. The function fn specifies the math
 * operation which is applied to the input tensor element-wise.
 */
Tensor& unary_ufunc_realhb_to_floath(
    double (*fn)(double),
    KernelRuntimeContext& ctx,
    const Tensor& in,
    Tensor& out);

/**
 * Implements an op pattern for ops that take two broadcastable input tensors
 * of any realb dtye, no additional arguments, performs an element-wise binary
 * logical operation, and outputs a realb tensor. The function fn specifies the
 * binary logical operation which is applied to the input tensors element-wise.
 */
Tensor& binary_ufunc_realb_realb_to_realb_logical(
    bool (*fn)(bool, bool),
    KernelRuntimeContext& ctx,
    const Tensor& a,
    const Tensor& b,
    Tensor& out);

} // namespace internal
} // namespace native
} // namespace executor
} // namespace torch
