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

bool check_narrow_copy_args(
    const Tensor& in,
    int64_t dim,
    int64_t start,
    int64_t length,
    Tensor& out);

void get_narrow_copy_out_target_size(
    const Tensor& in,
    int64_t dim,
    int64_t length,
    exec_aten::SizesType* out_sizes,
    size_t* out_ndim);

bool check_slice_copy_args(
    const Tensor& in,
    int64_t dim,
    int64_t step,
    Tensor& out);

void get_slice_copy_out_target_size(
    const Tensor& in,
    int64_t dim,
    int64_t num_values,
    exec_aten::SizesType* out_sizes,
    size_t* out_ndim);

bool check_slice_scatter_args(
    const Tensor& input,
    const Tensor& src,
    int64_t dim,
    int64_t num_values,
    int64_t step,
    Tensor output);

int64_t adjust_slice_indices(
    int64_t dim_length,
    int64_t* start,
    int64_t* end,
    int64_t step);

void compute_slice(
    const Tensor& in,
    int64_t dim,
    int64_t start,
    int64_t length,
    int64_t step,
    Tensor& out);

} // namespace executor
} // namespace torch
