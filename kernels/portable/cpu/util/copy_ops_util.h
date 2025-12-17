/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once
#include <c10/util/irange.h>

#include <executorch/kernels/portable/cpu/util/broadcast_util.h>
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
  const int64_t stride_dim = stride.at(dim);
  if (dim == static_cast<int64_t>(size.size()) - 1) {
    const size_t num_elements = size.at(dim);
    // use memcpy for contiguous memory
    if (stride_dim == 1) {
      memcpy(output_data, input_data, num_elements * sizeof(CTYPE));
    } else {
      for (const auto i : c10::irange(num_elements)) {
        output_data[i] = *input_data;
        input_data += stride_dim;
      }
    }
    return;
  }
  size_t trailing_dims = getTrailingDims(out, dim);
  // recursively set data for the next dimension
  for ([[maybe_unused]] const auto i : c10::irange(size.at(dim))) {
    _as_strided_copy<CTYPE>(
        input_data, output_data, out, size, stride, dim + 1);
    input_data += stride_dim;
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

/**
 * Copies and casts a tensor while preserving input dim_order.
 */
template <typename SELF_CTYPE, typename OUT_CTYPE>
void _to_dim_order_copy_impl(const Tensor& self, Tensor& out) {
  auto self_data = self.mutable_data_ptr<SELF_CTYPE>();
  auto out_data = out.mutable_data_ptr<OUT_CTYPE>();

  // Here we make a slightly off-label use of
  // BroadcastIndexesRange. It always assumes it doesn't have to care
  // about different dim_order between input and output, but we can
  // just force it to respect strides (and thus dim_order) for its
  // inputs using support_noncontiguous_input_tensors=true, and then pretend
  // the output is just another input.
  for (const auto [unused_index, self_data_index, out_data_index] :
       BroadcastIndexesRange<2, /*support_noncontiguous_input_tensors=*/true>(
           /*dummy output*/ self, self, out)) {
    (void)unused_index;
    out_data[out_data_index] =
        static_cast<OUT_CTYPE>(self_data[self_data_index]);
  }
}

bool check_cat_args(
    executorch::aten::ArrayRef<Tensor> tensors,
    int64_t dim,
    Tensor& out);

void get_cat_out_target_size(
    executorch::aten::ArrayRef<Tensor> tensors,
    int64_t dim,
    executorch::aten::SizesType* out_sizes,
    size_t* out_ndim);

bool check_expand_copy_args(
    const Tensor& self,
    ArrayRef<int64_t> expand_sizes,
    bool implicit,
    Tensor& out);

bool get_expand_copy_out_target_size(
    executorch::aten::ArrayRef<executorch::aten::SizesType> self_sizes,
    executorch::aten::ArrayRef<int64_t> expand_sizes,
    executorch::aten::SizesType* output_sizes,
    size_t* output_rank);

bool check_permute_copy_args(const Tensor& in, IntArrayRef dims, Tensor& out);

bool check_unbind_copy_args(const Tensor& in, int64_t dim, TensorList out);

void get_permute_copy_out_target_size(
    const Tensor& in,
    IntArrayRef dims,
    executorch::aten::SizesType* out_sizes,
    size_t* out_ndim);

bool check_pixel_shuffle_args(
    const Tensor& in,
    int64_t upscale_factor,
    Tensor& out);

bool get_pixel_shuffle_out_target_size(
    const Tensor& in,
    int64_t upscale_factor,
    executorch::aten::SizesType* out_sizes,
    size_t* out_ndim);

bool check_pixel_unshuffle_args(
    const Tensor& in,
    int64_t upscale_factor,
    Tensor& out);

void get_pixel_unshuffle_out_target_size(
    const Tensor& in,
    int64_t upscale_factor,
    executorch::aten::SizesType* out_sizes,
    size_t* out_ndim);

bool check_select_copy_out_args(
    const Tensor& in,
    int64_t dim,
    int64_t index,
    Tensor& out);

void get_select_copy_out_target_size(
    const Tensor& in,
    int64_t dim,
    executorch::aten::SizesType* out_sizes,
    size_t* out_ndim);

bool check_split_with_sizes_copy_args(
    const Tensor& in,
    executorch::aten::ArrayRef<int64_t> split_sizes,
    int64_t dim,
    TensorList out);

void get_split_with_sizes_copy_out_target_size(
    const Tensor& in,
    int64_t split_size,
    int64_t dim,
    executorch::aten::SizesType* out_sizes,
    size_t* out_ndim);

bool check_squeeze_copy_dim_args(
    const Tensor in,
    int64_t dim,
    const Tensor out);

void get_squeeze_copy_dim_out_target_size(
    const Tensor in,
    int64_t dim,
    executorch::aten::SizesType* out_sizes,
    size_t* out_ndim);

bool check_squeeze_copy_dims_args(
    const Tensor in,
    const executorch::aten::ArrayRef<int64_t> dims,
    const Tensor out);

void get_squeeze_copy_dims_out_target_size(
    const Tensor in,
    const executorch::aten::ArrayRef<int64_t> dims,
    executorch::aten::SizesType* out_sizes,
    size_t* out_ndim);

bool check_stack_args(
    executorch::aten::ArrayRef<Tensor> tensors,
    int64_t dim,
    Tensor& out);

void get_stack_out_target_size(
    executorch::aten::ArrayRef<Tensor> tensors,
    int64_t dim,
    executorch::aten::SizesType* out_sizes,
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
    std::optional<executorch::aten::MemoryFormat> memory_format,
    Tensor& out);

bool check__to_dim_order_copy_args(
    const Tensor& input,
    bool non_blocking,
    executorch::aten::OptionalArrayRef<int64_t> dim_order,
    Tensor& out);

bool check_unsqueeze_copy_args(
    const Tensor input,
    int64_t dim,
    const Tensor out);

bool check_view_copy_args(
    const Tensor& self,
    executorch::aten::ArrayRef<int64_t> size_int64_t,
    Tensor& out);

bool get_view_copy_target_size(
    const Tensor input,
    executorch::aten::ArrayRef<int64_t> size_int64_t,
    int64_t dim,
    executorch::aten::SizesType* out_sizes);

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
    executorch::aten::SizesType* out_sizes,
    size_t* out_ndim);

bool check_unfold_copy_args(
    const Tensor& self,
    int64_t dim,
    int64_t size,
    int64_t step);

void get_unfold_copy_out_target_size(
    const Tensor& self,
    int64_t dim,
    int64_t size,
    int64_t step,
    executorch::aten::SizesType* out_sizes,
    size_t* out_ndim);

void get_view_as_real_copy_out_target_size(
    const Tensor& self,
    executorch::aten::SizesType* out_sizes);

} // namespace executor
} // namespace torch
