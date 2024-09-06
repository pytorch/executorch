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

bool check_addmm_args(
    const Tensor& in,
    const Tensor& mat1,
    const Tensor& mat2,
    const Scalar& beta,
    const Scalar& alpha,
    Tensor& out);

bool check_bmm_args(const Tensor& in, const Tensor& mat2, Tensor& out);

void get_bmm_out_target_size(
    const Tensor& mat1,
    const Tensor& mat2,
    Tensor::SizesType* out_sizes,
    size_t* out_ndim);

bool check_mm_args(const Tensor& in, const Tensor& mat2, Tensor& out);

void get_mm_out_target_size(
    const Tensor& mat1,
    const Tensor& mat2,
    Tensor::SizesType* out_sizes,
    size_t* out_ndim);

bool check_linear_args(const Tensor& in, const Tensor& mat2, Tensor& out);

void get_linear_out_target_size(
    const Tensor& mat1,
    const Tensor& mat2,
    Tensor::SizesType* out_sizes,
    size_t* out_ndim);

} // namespace executor
} // namespace torch
