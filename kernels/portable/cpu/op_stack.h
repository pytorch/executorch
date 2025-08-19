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

Tensor& stack_out(
    KernelRuntimeContext& ctx,
    executorch::aten::ArrayRef<Tensor> tensors,
    int64_t dim,
    Tensor& out);

/**
 * Computes the output shape for tensor stacking.
 *
 * @param[in] tensors Array of input tensors to stack
 * @param[in] dim Dimension along which to stack
 * @return Tuple containing the Error, output shape array, and number of
 * dimensions
 */
std::tuple<
    Error,
    std::array<executorch::aten::SizesType, kTensorDimensionLimit>,
    size_t>
stack_out_shape(executorch::aten::ArrayRef<Tensor> tensors, int64_t dim);

} // namespace utils
} // namespace native
} // namespace executor
} // namespace torch
