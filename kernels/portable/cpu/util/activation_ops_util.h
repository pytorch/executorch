/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <executorch/runtime/kernel/kernel_includes.h>

namespace torch {
namespace executor {

bool check_gelu_args(const Tensor& in, string_view approximate, Tensor& out);

bool check_glu_args(const Tensor& in, int64_t dim, Tensor& out);

bool check_log_softmax_args(
    const Tensor& in,
    int64_t dim,
    bool half_to_float,
    Tensor& out);

bool check_softmax_args(
    const Tensor& in,
    int64_t dim,
    bool half_to_float,
    Tensor& out);

Error resize_glu_out(const Tensor& in, int64_t dim, Tensor& out);

} // namespace executor
} // namespace torch
