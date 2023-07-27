// Copyright (c) Meta Platforms, Inc. and affiliates.

#include <cstring>

#include <executorch/runtime/core/exec_aten/exec_aten.h>
#include <executorch/runtime/core/exec_aten/util/scalar_type_util.h>
#include <executorch/runtime/core/exec_aten/util/tensor_util.h>
#include <executorch/runtime/platform/assert.h>

namespace torch {
namespace executor {

using Tensor = exec_aten::Tensor;

void check_cat_args(
    exec_aten::ArrayRef<Tensor> tensors,
    int64_t dim,
    Tensor& out) {
  // Ensure the input tensors list is non-empty
  ET_CHECK(tensors.size() > 0);

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
    ET_CHECK(canCast(tensors[i].scalar_type(), out.scalar_type()));

    // Empty tensors have no shape constraints.
    if (tensors[i].numel() == 0) {
      continue;
    }

    // All input tensors must have the same number of dimensions.
    ET_CHECK(tensors[i].dim() == tensors[ref_i].dim());

    for (size_t d = 0; d < tensors[i].dim(); ++d) {
      if (d != dim) {
        ET_CHECK(tensors[i].size(d) == tensors[ref_i].size(d));
      }
    }
  }

  // Ensure dim is in range.
  ET_CHECK(dim >= 0 && dim < tensors[ref_i].dim());
}

void get_cat_out_target_size(
    exec_aten::ArrayRef<Tensor> tensors,
    int64_t dim,
    Tensor::SizesType* out_sizes,
    size_t* out_ndim) {
  // Find the last non-empty tensor in the list to use as a reference
  size_t ref_i = 0;
  size_t cat_dim_size = 0;
  for (size_t i = 0; i < tensors.size(); ++i) {
    if (tensors[i].numel() > 0) {
      cat_dim_size += tensors[i].size(dim);
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

void check_permute_copy_args(const Tensor& in, IntArrayRef dims, Tensor& out) {
  ET_CHECK(in.dim() == dims.size());
  ET_CHECK_SAME_DTYPE2(in, out);

  // Make sure no dimensions are duplicated and all in the range [-in.dim(),
  // in.dim() - 1]. Use gaussian sum to check this.
  size_t expected_sum = (dims.size() * (dims.size() + 1)) / 2;
  size_t gauss_sum = 0;
  for (int i = 0; i < dims.size(); i++) {
    // Convert dimension to a non-negative number. dim_base is in the range
    // [0 .. in.dim() - 1].
    size_t dim = dims[i] > -1 ? dims[i] : in.dim() + dims[i];
    ET_CHECK(dim >= 0 && dim < in.dim());
    gauss_sum += dim + 1;
  }

  ET_CHECK_MSG(
      gauss_sum == expected_sum,
      "The dims passed to permute_copy must contain one of each dim!");
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

void check_stack_args(
    exec_aten::ArrayRef<Tensor> tensors,
    int64_t dim,
    Tensor& out) {
  // Ensure the input tensors list is non-empty
  ET_CHECK(tensors.size() > 0);

  // All input tensors need to be of the same size
  // https://pytorch.org/docs/stable/generated/torch.stack.html
  for (size_t i = 0; i < tensors.size(); i++) {
    // All input dtypes must be castable to the output dtype.
    ET_CHECK(canCast(tensors[i].scalar_type(), out.scalar_type()));

    ET_CHECK(tensors[i].dim() == tensors[0].dim());
    for (size_t d = 0; d < tensors[i].dim(); d++) {
      ET_CHECK(tensors[i].size(d) == tensors[0].size(d));
    }
  }

  // The output tensor will have a dimension inserted, so dim should be between
  // 0 and ndim_of_inputs + 1
  ET_CHECK(dim >= 0 && dim < tensors[0].dim() + 1);
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
