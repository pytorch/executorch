/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <executorch/runtime/core/exec_aten/exec_aten.h>
#include <executorch/runtime/core/exec_aten/util/tensor_util.h>

namespace torch {
namespace executor {

bool check_index_select_args(
    const Tensor& in,
    int64_t dim,
    const Tensor& index,
    Tensor& out);

void get_index_select_out_target_size(
    const Tensor& in,
    int64_t dim,
    const Tensor& index,
    exec_aten::SizesType* out_sizes,
    size_t* out_ndim);

bool check_scatter_add_args(
    const Tensor& self,
    int64_t dim,
    const Tensor& index,
    const Tensor& src,
    Tensor& out);

bool check_nonzero_args(const Tensor& in, const Tensor& out);

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

bool check_select_scatter_args(
    const Tensor& in,
    const Tensor& src,
    int64_t dim,
    int64_t index,
    Tensor& output);

} // namespace executor
} // namespace torch
