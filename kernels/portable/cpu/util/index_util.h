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

} // namespace executor
} // namespace torch
