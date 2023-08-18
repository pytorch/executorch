/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <cstring>

#include <executorch/kernels/portable/cpu/util/copy_ops_util.h>

namespace torch {
namespace executor {

using Tensor = exec_aten::Tensor;

bool check_cat_args(
    exec_aten::ArrayRef<Tensor> tensors,
    int64_t dim,
    Tensor& out) {
  // Ensure the input tensors list is non-empty
  ET_LOG_AND_RETURN_IF_FALSE(tensors.size() > 0);

  // Find the first non-empty tensor in the list to use as a reference
  size_t ref_i = 0;
  for (size_t i = 0; i < tensors.size(); ++i) {
    if (tensors[i].numel() > 0) {
      ref_i = i;
      break;
    }
  }

  // "All tensors must either have the same shape (except in the concatenating
  // dimension) or be empty."
  // https://pytorch.org/docs/stable/generated/torch.cat.html
  for (size_t i = 0; i < tensors.size(); ++i) {
    // All input dtypes must be castable to the output dtype.
    ET_LOG_AND_RETURN_IF_FALSE(
        canCast(tensors[i].scalar_type(), out.scalar_type()));

    // Empty tensors have no shape constraints.
    if (tensors[i].numel() == 0) {
      continue;
    }

    // All input tensors must have the same number of dimensions.
    ET_LOG_AND_RETURN_IF_FALSE(
        tensor_is_rank(tensors[ref_i], tensors[i].dim()));

    for (size_t d = 0; d < tensors[i].dim(); ++d) {
      if (d != dim) {
        ET_LOG_AND_RETURN_IF_FALSE(
            tensors_have_same_size_at_dims(tensors[i], d, tensors[ref_i], d));
      }
    }
  }

  // Ensure dim is in range.
  ET_LOG_AND_RETURN_IF_FALSE(
      tensors[ref_i].numel() == 0 || tensors[ref_i].dim() > dim);

  return true;
}

void get_cat_out_target_size(
    exec_aten::ArrayRef<Tensor> tensors,
    int64_t dim,
    Tensor::SizesType* out_sizes,
    size_t* out_ndim) {
  // Find the first non-1D-or-empty tensor in the list to use as a reference
  // because an 1D empty tensor is a wildcard and should be ignored when we
  // calculate out dim
  size_t ref_i = 0;
  size_t cat_dim_size = 0;
  for (size_t i = 0; i < tensors.size(); ++i) {
    if (tensors[i].numel() > 0) {
      cat_dim_size += tensors[i].size(dim);
    }
    if (tensors[i].dim() != 1 || tensors[i].numel() != 0) {
      ref_i = i;
    }
  }

  *out_ndim = tensors[ref_i].dim();

  for (size_t d = 0; d < *out_ndim; ++d) {
    if (d != dim) {
      out_sizes[d] = tensors[ref_i].size(d);
    } else {
      out_sizes[d] = cat_dim_size;
    }
  }
}

bool check_permute_copy_args(const Tensor& in, IntArrayRef dims, Tensor& out) {
  ET_LOG_AND_RETURN_IF_FALSE(tensor_is_rank(in, dims.size()));
  ET_LOG_AND_RETURN_IF_FALSE(tensors_have_same_dtype(in, out));

  // Make sure no dimensions are duplicated and all in the range [-in.dim(),
  // in.dim() - 1]. Use gaussian sum to check this.
  size_t expected_sum = (dims.size() * (dims.size() + 1)) / 2;
  size_t gauss_sum = 0;
  for (int i = 0; i < dims.size(); i++) {
    // Convert dimension to a non-negative number. dim_base is in the range
    // [0 .. in.dim() - 1].
    size_t dim = dims[i] > -1 ? dims[i] : in.dim() + dims[i];
    ET_LOG_AND_RETURN_IF_FALSE(tensor_has_dim(in, dim));
    gauss_sum += dim + 1;
  }

  ET_LOG_MSG_AND_RETURN_IF_FALSE(
      gauss_sum == expected_sum,
      "The dims passed to permute_copy must contain one of each dim!");

  return true;
}

void get_permute_copy_out_target_size(
    const Tensor& in,
    IntArrayRef dims,
    Tensor::SizesType* out_sizes,
    size_t* out_ndim) {
  *out_ndim = in.dim();

  for (size_t i = 0; i < in.dim(); ++i) {
    out_sizes[i] = in.size(dims[i] >= 0 ? dims[i] : dims[i] + in.dim());
  }
}

bool check_stack_args(
    exec_aten::ArrayRef<Tensor> tensors,
    int64_t dim,
    Tensor& out) {
  // Ensure the input tensors list is non-empty
  ET_LOG_AND_RETURN_IF_FALSE(tensors.size() > 0);

  // All input tensors need to be of the same size
  // https://pytorch.org/docs/stable/generated/torch.stack.html
  for (size_t i = 0; i < tensors.size(); i++) {
    // All input dtypes must be castable to the output dtype.
    ET_LOG_AND_RETURN_IF_FALSE(
        canCast(tensors[i].scalar_type(), out.scalar_type()));

    ET_LOG_AND_RETURN_IF_FALSE(tensor_is_rank(tensors[i], tensors[0].dim()));
    for (size_t d = 0; d < tensors[i].dim(); d++) {
      ET_LOG_AND_RETURN_IF_FALSE(
          tensors_have_same_size_at_dims(tensors[i], d, tensors[0], d));
    }
  }

  // The output tensor will have a dimension inserted, so dim should be between
  // 0 and ndim_of_inputs + 1
  ET_LOG_AND_RETURN_IF_FALSE(dim >= 0 && dim < tensors[0].dim() + 1);

  return true;
}

void get_stack_out_target_size(
    exec_aten::ArrayRef<Tensor> tensors,
    int64_t dim,
    Tensor::SizesType* out_sizes,
    size_t* out_ndim) {
  *out_ndim = tensors[0].dim() + 1;

  for (size_t d = 0; d < *out_ndim; ++d) {
    if (d < dim) {
      out_sizes[d] = tensors[0].size(d);
    } else if (d == dim) {
      out_sizes[d] = tensors.size();
    } else {
      out_sizes[d] = tensors[0].size(d - 1);
    }
  }
}

} // namespace executor
} // namespace torch
