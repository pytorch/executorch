/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <executorch/runtime/kernel/kernel_includes.h>
#include <tuple>

namespace torch {
namespace executor {
namespace native {
namespace utils {

bool check_stack_args(
    executorch::aten::ArrayRef<Tensor> tensors,
    int64_t dim,
    Tensor& out);

void get_stack_out_target_size(
    executorch::aten::ArrayRef<Tensor> tensors,
    int64_t dim,
    executorch::aten::SizesType* out_sizes,
    size_t* out_ndim);

std::tuple<
    Error,
    std::array<executorch::aten::SizesType, kTensorDimensionLimit>,
    size_t>
stack_out_shape(executorch::aten::ArrayRef<Tensor> tensors, int64_t dim);

Tensor& stack_out_impl(
    KernelRuntimeContext& ctx,
    executorch::aten::ArrayRef<Tensor> tensors,
    int64_t dim,
    Tensor& out);

} // namespace utils
} // namespace native
} // namespace executor
} // namespace torch
