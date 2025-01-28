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
namespace {

/**
 * Copy input_data to output_data according to the stride and shape recursively
 */
template <typename CTYPE>
void _as_strided_copy(
    CTYPE* input_data,
    CTYPE* output_data,
    Tensor& out,
    ArrayRef<int64_t> size,
    ArrayRef<int64_t> stride,
    int64_t dim) {
  // the last dimension, copy data
  if (dim == size.size() - 1) {
    for (size_t i = 0; i < size.at(dim); ++i) {
      output_data[i] = *input_data;
      input_data += stride.at(dim);
    }
    return;
  }
  size_t trailing_dims = getTrailingDims(out, dim);
  // recursively set data for the next dimension
  for (size_t i = 0; i < size.at(dim); ++i) {
    _as_strided_copy<CTYPE>(
        input_data, output_data, out, size, stride, dim + 1);
    input_data += stride.at(dim);
    output_data += trailing_dims;
  }
}

} // namespace

bool check_as_strided_copy_args(
    const Tensor& in,
    ArrayRef<int64_t> size,
    ArrayRef<int64_t> stride,
    optional<int64_t> storage_offset,
    Tensor& out);

template <typename CTYPE>
void as_strided_copy(
    const Tensor& in,
    ArrayRef<int64_t> size,
    ArrayRef<int64_t> stride,
    int64_t offset,
    Tensor& out) {
  CTYPE* in_data = in.mutable_data_ptr<CTYPE>() + offset;
  CTYPE* out_data = out.mutable_data_ptr<CTYPE>();

  if (size.empty()) {
    out_data[0] = in_data[0];
  } else {
    _as_strided_copy<CTYPE>(in_data, out_data, out, size, stride, 0);
  }
}

bool check_cat_args(
    exec_aten::ArrayRef<Tensor> tensors,
    int64_t dim,
    Tensor& out);

void get_cat_out_target_size(
    exec_aten::ArrayRef<Tensor> tensors,
    int64_t dim,
    exec_aten::SizesType* out_sizes,
    size_t* out_ndim);

bool check_expand_copy_args(
    const Tensor& self,
    ArrayRef<int64_t> expand_sizes,
    bool implicit,
    Tensor& out);

bool get_expand_copy_out_target_size(
    exec_aten::ArrayRef<exec_aten::SizesType> self_sizes,
    exec_aten::ArrayRef<int64_t> expand_sizes,
    exec_aten::SizesType* output_sizes,
    size_t* output_rank);

bool check_permute_copy_args(const Tensor& in, IntArrayRef dims, Tensor& out);

bool check_unbind_copy_args(const Tensor& in, int64_t dim, TensorList out);

void get_permute_copy_out_target_size(
    const Tensor& in,
    IntArrayRef dims,
    exec_aten::SizesType* out_sizes,
    size_t* out_ndim);

bool check_pixel_shuffle_args(
    const Tensor& in,
    int64_t upscale_factor,
    Tensor& out);

void get_pixel_shuffle_out_target_size(
    const Tensor& in,
    int64_t upscale_factor,
    exec_aten::SizesType* out_sizes,
    size_t* out_ndim);

bool check_pixel_unshuffle_args(
    const Tensor& in,
    int64_t upscale_factor,
    Tensor& out);

void get_pixel_unshuffle_out_target_size(
    const Tensor& in,
    int64_t upscale_factor,
    exec_aten::SizesType* out_sizes,
    size_t* out_ndim);

bool check_select_copy_out_args(
    const Tensor& in,
    int64_t dim,
    int64_t index,
    Tensor& out);

void get_select_copy_out_target_size(
    const Tensor& in,
    int64_t dim,
    exec_aten::SizesType* out_sizes,
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
    exec_aten::SizesType* out_sizes,
    size_t* out_ndim);

bool check_squeeze_copy_dim_args(
    const Tensor in,
    int64_t dim,
    const Tensor out);

void get_squeeze_copy_dim_out_target_size(
    const Tensor in,
    int64_t dim,
    exec_aten::SizesType* out_sizes,
    size_t* out_ndim);

bool check_squeeze_copy_dims_args(
    const Tensor in,
    const exec_aten::ArrayRef<int64_t> dims,
    const Tensor out);

void get_squeeze_copy_dims_out_target_size(
    const Tensor in,
    const exec_aten::ArrayRef<int64_t> dims,
    exec_aten::SizesType* out_sizes,
    size_t* out_ndim);

bool check_stack_args(
    exec_aten::ArrayRef<Tensor> tensors,
    int64_t dim,
    Tensor& out);

void get_stack_out_target_size(
    exec_aten::ArrayRef<Tensor> tensors,
    int64_t dim,
    exec_aten::SizesType* out_sizes,
    size_t* out_ndim);

bool check_tril_args(const Tensor& in, Tensor& out);

bool check_split_copy_args(
    const Tensor& input,
    int64_t split_size,
    int64_t dim,
    TensorList out);

bool check_to_copy_args(
    const Tensor& input,
    bool non_blocking,
    exec_aten::optional<exec_aten::MemoryFormat> memory_format,
    Tensor& out);

bool check__to_dim_order_copy_args(
    const Tensor& input,
    bool non_blocking,
    exec_aten::OptionalArrayRef<int64_t> dim_order,
    Tensor& out);

bool check_unsqueeze_copy_args(
    const Tensor input,
    int64_t dim,
    const Tensor out);

bool check_view_copy_args(
    const Tensor& self,
    exec_aten::ArrayRef<int64_t> size_int64_t,
    Tensor& out);

bool get_view_copy_target_size(
    const Tensor input,
    exec_aten::ArrayRef<int64_t> size_int64_t,
    int64_t dim,
    exec_aten::SizesType* out_sizes);

bool check_diagonal_copy_args(
    const Tensor& in,
    int64_t dim1,
    int64_t dim2,
    Tensor& out);

void get_diagonal_copy_out_target_size(
    const Tensor& in,
    int64_t offset,
    int64_t dim1,
    int64_t dim2,
    exec_aten::SizesType* out_sizes,
    size_t* out_ndim);

} // namespace executor
} // namespace torch
