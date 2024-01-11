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

bool check_as_strided_copy_args(
    const Tensor& in,
    ArrayRef<int64_t> size,
    ArrayRef<int64_t> stride,
    optional<int64_t> storage_offset,
    Tensor& out);

bool check_cat_args(
    exec_aten::ArrayRef<Tensor> tensors,
    int64_t dim,
    Tensor& out);

void get_cat_out_target_size(
    exec_aten::ArrayRef<Tensor> tensors,
    int64_t dim,
    Tensor::SizesType* out_sizes,
    size_t* out_ndim);

bool check_expand_copy_args(
    const Tensor& self,
    ArrayRef<int64_t> expand_sizes,
    bool implicit,
    Tensor& out);

bool check_permute_copy_args(const Tensor& in, IntArrayRef dims, Tensor& out);

bool check_unbind_copy_args(const Tensor& in, int64_t dim, TensorList out);

void get_permute_copy_out_target_size(
    const Tensor& in,
    IntArrayRef dims,
    Tensor::SizesType* out_sizes,
    size_t* out_ndim);

bool check_pixel_shuffle_args(
    const Tensor& in,
    int64_t upscale_factor,
    Tensor& out);

void get_pixel_shuffle_out_target_size(
    const Tensor& in,
    int64_t upscale_factor,
    Tensor::SizesType* out_sizes,
    size_t* out_ndim);

bool check_select_copy_out_args(
    const Tensor& in,
    int64_t dim,
    int64_t index,
    Tensor& out);

void get_select_copy_out_target_size(
    const Tensor& in,
    int64_t dim,
    Tensor::SizesType* out_sizes,
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
    Tensor::SizesType* out_sizes,
    size_t* out_ndim);

bool check_split_with_sizes_copy_args(
    const Tensor& in,
    exec_aten::ArrayRef<int64_t> split_sizes,
    int64_t dim,
    TensorList out);

void get_split_with_sizes_copy_out_target_size(
    const Tensor& in,
    int64_t split_size,
    int64_t dim,
    Tensor::SizesType* out_sizes,
    size_t* out_ndim);

bool check_squeeze_copy_dim_args(
    const Tensor in,
    int64_t dim,
    const Tensor out);

void get_squeeze_copy_dim_out_target_size(
    const Tensor in,
    int64_t dim,
    Tensor::SizesType* out_sizes,
    size_t* out_ndim);

bool check_squeeze_copy_dims_args(
    const Tensor in,
    const exec_aten::ArrayRef<int64_t> dims,
    const Tensor out);

void get_squeeze_copy_dims_out_target_size(
    const Tensor in,
    const exec_aten::ArrayRef<int64_t> dims,
    Tensor::SizesType* out_sizes,
    size_t* out_ndim);

bool check_stack_args(
    exec_aten::ArrayRef<Tensor> tensors,
    int64_t dim,
    Tensor& out);

void get_stack_out_target_size(
    exec_aten::ArrayRef<Tensor> tensors,
    int64_t dim,
    Tensor::SizesType* out_sizes,
    size_t* out_ndim);

bool check_tril_args(const Tensor& in, Tensor& out);

} // namespace executor
} // namespace torch
