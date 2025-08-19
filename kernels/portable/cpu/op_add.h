/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/runtime/kernel/kernel_includes.h>

#pragma once

namespace torch {
namespace executor {
namespace native {
namespace utils {

Tensor& add_out(
    KernelRuntimeContext& ctx,
    const Tensor& a,
    const Tensor& b,
    const Scalar& alpha,
    Tensor& out);

Tensor& add_scalar_out(
    KernelRuntimeContext& ctx,
    const Tensor& a,
    const Scalar& b,
    const Scalar& alpha,
    Tensor& out);

/**
 * Computes the output shape for tensor addition with broadcasting.
 *
 * @param[in] a First input tensor
 * @param[in] b Second input tensor
 * @param[in] alpha Scalar multiplier for b (unused for shape computation)
 * @return Tuple containing the Error, output shape array, and number of
 * dimensions
 */
std::tuple<
    Error,
    std::array<executorch::aten::SizesType, kTensorDimensionLimit>,
    size_t>
add_out_shape(const Tensor& a, const Tensor& b, const Scalar& alpha);

/**
 * Computes the output shape for tensor-scalar addition.
 *
 * @param[in] a Input tensor
 * @param[in] b Scalar value (unused for shape computation)
 * @param[in] alpha Scalar multiplier for b (unused for shape computation)
 * @return Tuple containing the Error, output shape array, and number of
 * dimensions
 */
std::tuple<
    Error,
    std::array<executorch::aten::SizesType, kTensorDimensionLimit>,
    size_t>
add_scalar_out_shape(const Tensor& a, const Scalar& b, const Scalar& alpha);

} // namespace utils
} // namespace native
} // namespace executor
} // namespace torch
