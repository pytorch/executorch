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

namespace {

size_t as_strided_copy_compute_storage_nbytes(
    IntArrayRef sizes,
    IntArrayRef strides,
    size_t itemsize_bytes) {
  // size of the underlying storage is 1 bigger than the offset
  // of the last element according to stride
  size_t size = 1;
  for (size_t i = 0; i < sizes.size(); ++i) {
    if (sizes[i] == 0) {
      return 0;
    }
    size += strides[i] * (sizes[i] - 1);
  }
  return size * itemsize_bytes;
}

} // namespace

bool check_as_strided_copy_args(
    const Tensor& in,
    ArrayRef<int64_t> size,
    ArrayRef<int64_t> stride,
    optional<int64_t> storage_offset,
    Tensor& out) {
  ET_LOG_AND_RETURN_IF_FALSE(tensors_have_same_dtype(in, out));
  ET_LOG_MSG_AND_RETURN_IF_FALSE(
      size.size() == stride.size(), "mismatch in length of strides and shape");
  for (const auto& val : stride) {
    ET_LOG_MSG_AND_RETURN_IF_FALSE(
        val >= 0,
        "as_strided: Negative strides are not supported at the moment");
  }

  int64_t offset = storage_offset.has_value() ? storage_offset.value() : 0;
  ET_LOG_MSG_AND_RETURN_IF_FALSE(offset >= 0, "Negative storage offset");

  // Check that the requested storage is within bounds of input storage
  size_t storage_size_bytes =
      as_strided_copy_compute_storage_nbytes(size, stride, in.element_size());
  size_t storage_offset_bytes = offset * in.element_size();
  if (storage_size_bytes == 0) {
    return true;
  }
  size_t new_storage_size_bytes = in.nbytes();
  ET_LOG_MSG_AND_RETURN_IF_FALSE(
      storage_size_bytes + storage_offset_bytes <= new_storage_size_bytes,
      "Requiring a storage size of %zd are out of bounds for storage of size %zd",
      storage_size_bytes + storage_offset_bytes,
      new_storage_size_bytes);
  return true;
}

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

bool check_expand_copy_args(
    const Tensor& input,
    ArrayRef<int64_t> expand_sizes,
    bool implicit,
    Tensor& out) {
  ET_LOG_MSG_AND_RETURN_IF_FALSE(
      implicit == false,
      "This operator is not implemented for when implicit == true.");

  ET_LOG_MSG_AND_RETURN_IF_FALSE(
      expand_sizes.size() >= input.sizes().size(),
      "The number of sizes provided (%zu) must at least be equal to the number of dimensions in the tensor (%zu)",
      expand_sizes.size(),
      input.sizes().size());

  ET_LOG_MSG_AND_RETURN_IF_FALSE(
      expand_sizes.size() <= kTensorDimensionLimit,
      "The number of expanded dims (%zu) exceeds the configured maximum (%zu). Increase this limit.",
      expand_sizes.size(),
      kTensorDimensionLimit);

  return true;
}

bool check_permute_copy_args(const Tensor& in, IntArrayRef dims, Tensor& out) {
  ET_LOG_AND_RETURN_IF_FALSE(tensor_is_rank(in, dims.size()));
  ET_LOG_AND_RETURN_IF_FALSE(tensors_have_same_dtype(in, out));

  // Make sure no dimensions are duplicated and all in the range [-in.dim(),
  // in.dim() - 1].
  bool dim_exist[kTensorDimensionLimit];
  memset(dim_exist, false, sizeof(dim_exist));

  for (int i = 0; i < dims.size(); i++) {
    ET_LOG_AND_RETURN_IF_FALSE(tensor_has_dim(in, dims[i]));
    // Convert dimension to a non-negative number in the range
    // [0 .. in.dim() - 1].
    size_t dim = dims[i] >= 0 ? dims[i] : in.dim() + dims[i];

    // Internal check, since we have already validated this
    ET_CHECK(dim < kTensorDimensionLimit && dim >= 0);

    // Check that the dimension hasn't been seen previously.
    ET_LOG_MSG_AND_RETURN_IF_FALSE(
        dim_exist[dim] == false, "duplicate dims are not allowed.");

    dim_exist[dim] = true;
  }

  return true;
}

bool check_unbind_copy_args(const Tensor& in, int64_t dim, TensorList out) {
  ET_LOG_MSG_AND_RETURN_IF_FALSE(
      in.dim() > 0, "in must have at least one dimension; saw %zd", in.dim());

  ET_LOG_MSG_AND_RETURN_IF_FALSE(
      dim >= 0 && dim < in.dim(),
      "dim %" PRId64 " out of range [0,%zd)",
      dim,
      in.dim());

  const ssize_t dim_size = in.size(dim);
  ET_LOG_MSG_AND_RETURN_IF_FALSE(
      dim_size == out.size(),
      "out tensorlist's length %zd must equal unbind dim %" PRId64
      " size = %zd.",
      out.size(),
      dim,
      dim_size);

  // Validate each output.
  for (size_t i = 0; i < out.size(); ++i) {
    // All output dtypes must be the same.
    ET_LOG_MSG_AND_RETURN_IF_FALSE(
        out[i].scalar_type() == out[0].scalar_type(),
        "out[%zu] dtype %" PRId8 " != out[0] dtype %" PRId8,
        i,
        static_cast<int8_t>(out[i].scalar_type()),
        static_cast<int8_t>(out[0].scalar_type()));

    // output tensor must have # of dims = in.dim() -1
    ET_LOG_MSG_AND_RETURN_IF_FALSE(
        out[i].dim() == (in.dim() - 1),
        "out[%zu] dim %zd != in dim %zd",
        i,
        out[i].dim(),
        in.dim() - 1);

    // Check the shape of the output.
    for (ssize_t d = 0, out_d = 0; d < in.dim(); ++d) {
      if (d != dim) {
        ET_LOG_MSG_AND_RETURN_IF_FALSE(
            out[i].size(out_d) == in.size(d),
            "out[%zu].size(%zd) %zd != in.size(%zd) %zd",
            i,
            d,
            out[i].size(out_d),
            d,
            in.size(d));
        out_d++;
      }
    }
  }

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

bool check_pixel_shuffle_args(
    const Tensor& in,
    int64_t upscale_factor,
    Tensor& out) {
  ET_LOG_AND_RETURN_IF_FALSE(tensors_have_same_dtype(in, out));
  ET_LOG_AND_RETURN_IF_FALSE(tensor_has_rank_greater_or_equal_to(in, 3));
  ET_LOG_AND_RETURN_IF_FALSE(tensor_has_rank_greater_or_equal_to(out, 3));
  ET_LOG_AND_RETURN_IF_FALSE(upscale_factor > 0);
  ET_LOG_AND_RETURN_IF_FALSE(
      in.size(in.dim() - 3) % (upscale_factor * upscale_factor) == 0);
  return true;
}

void get_pixel_shuffle_out_target_size(
    const Tensor& in,
    int64_t upscale_factor,
    Tensor::SizesType* out_sizes,
    size_t* out_ndim) {
  *out_ndim = in.dim();
  const Tensor::SizesType casted_upscale_factor = upscale_factor;

  size_t i = 0;
  for (; i < in.dim() - 3; ++i) {
    // Copy all leading dimensions in.
    out_sizes[i] = in.size(i);
  }
  // The last 3 dimensions are (channel, height, width). Divide by the upscale
  // factor squared and multiply the height and width by that factor.
  out_sizes[i] = in.size(i) / (casted_upscale_factor * casted_upscale_factor);
  i++;
  out_sizes[i] = in.size(i) * casted_upscale_factor;
  i++;
  out_sizes[i] = in.size(i) * casted_upscale_factor;
}

bool check_select_copy_out_args(
    const Tensor& in,
    int64_t dim,
    int64_t index,
    Tensor& out) {
  ET_LOG_AND_RETURN_IF_FALSE(tensor_has_rank_greater_or_equal_to(in, 1));
  ET_LOG_AND_RETURN_IF_FALSE(tensor_has_dim(in, dim));
  ET_LOG_AND_RETURN_IF_FALSE(tensor_dim_has_index(in, dim, index));
  ET_LOG_AND_RETURN_IF_FALSE(tensors_have_same_dtype(in, out));
  return true;
}

void get_select_copy_out_target_size(
    const Tensor& in,
    int64_t dim,
    Tensor::SizesType* out_sizes,
    size_t* out_ndim) {
  *out_ndim = in.dim() - 1;

  for (size_t d = 0; d < in.dim() - 1; ++d) {
    if (d < dim) {
      out_sizes[d] = in.size(d);
    } else {
      out_sizes[d] = in.size(d + 1);
    }
  }
}

bool check_slice_copy_args(
    const Tensor& in,
    int64_t dim,
    int64_t step,
    Tensor& out) {
  ET_LOG_AND_RETURN_IF_FALSE(tensors_have_same_dtype(in, out));
  ET_LOG_AND_RETURN_IF_FALSE(tensor_has_dim(in, dim));
  ET_LOG_MSG_AND_RETURN_IF_FALSE(
      step > 0, "slice step must be greater than zero");
  return true;
}

void get_slice_copy_out_target_size(
    const Tensor& in,
    int64_t dim,
    int64_t num_values,
    Tensor::SizesType* out_sizes,
    size_t* out_ndim) {
  *out_ndim = in.dim();

  for (size_t d = 0; d < in.dim(); ++d) {
    out_sizes[d] = in.size(d);
  }
  out_sizes[dim] = num_values;
}

bool check_split_with_sizes_copy_args(
    const Tensor& in,
    exec_aten::ArrayRef<int64_t> split_sizes,
    int64_t dim,
    TensorList out) {
  ET_LOG_AND_RETURN_IF_FALSE(tensor_has_rank_greater_or_equal_to(in, 1));
  ET_LOG_AND_RETURN_IF_FALSE(tensor_has_dim(in, dim));

  ET_LOG_MSG_AND_RETURN_IF_FALSE(
      split_sizes.size() == out.size(),
      "Number of split sizes must match the number of output tensors");

  int64_t sum = 0;
  for (int i = 0; i < split_sizes.size(); i++) {
    ET_LOG_MSG_AND_RETURN_IF_FALSE(
        split_sizes[i] >= 0, "All split sizes must be non negative.");
    sum += split_sizes[i];
  }

  const ssize_t dim_size = in.size(dim);
  ET_LOG_MSG_AND_RETURN_IF_FALSE(
      sum == dim_size,
      "Sum of split sizes does not match input size at given dim");

  return true;
}

void get_split_with_sizes_copy_out_target_size(
    const Tensor& in,
    int64_t split_size,
    int64_t dim,
    Tensor::SizesType* out_sizes,
    size_t* out_ndim) {
  *out_ndim = in.dim();

  for (size_t d = 0; d < in.dim(); ++d) {
    out_sizes[d] = in.size(d);
  }
  out_sizes[dim] = split_size;
}

bool check_squeeze_copy_dim_args(
    const Tensor in,
    int64_t dim,
    const Tensor out) {
  ET_LOG_AND_RETURN_IF_FALSE(tensors_have_same_dtype(in, out));
  ET_LOG_AND_RETURN_IF_FALSE(tensor_has_dim(in, dim));

  return true;
}

void get_squeeze_copy_dim_out_target_size(
    const Tensor in,
    int64_t dim,
    Tensor::SizesType* out_sizes,
    size_t* out_ndim) {
  // For 0 dim tensors, the output should also be 0 dim.
  if (in.dim() == 0) {
    *out_ndim = 0;
    return;
  }

  // Specified dim is only removed if the size at the given dim is 1.
  if (in.size(dim) == 1) {
    *out_ndim = in.dim() - 1;
  } else {
    *out_ndim = in.dim();
  }

  size_t out_d = 0;
  for (size_t in_d = 0; in_d < in.dim(); ++in_d) {
    if (in_d != dim || in.size(in_d) != 1) {
      out_sizes[out_d] = in.size(in_d);
      ++out_d;
    }
  }
}

bool check_squeeze_copy_dims_args(
    const Tensor in,
    const exec_aten::ArrayRef<int64_t> dims,
    const Tensor out) {
  ET_LOG_AND_RETURN_IF_FALSE(tensors_have_same_dtype(in, out));

  for (size_t i = 0; i < dims.size(); ++i) {
    const int64_t dim = dims[i] < 0 ? dims[i] + nonzero_dim(in) : dims[i];
    ET_LOG_AND_RETURN_IF_FALSE(tensor_has_dim(in, dim));

    // Check that a dim does not appear twice in dims
    for (size_t j = 0; j < dims.size(); ++j) {
      if (i != j) {
        const int64_t dim_temp =
            dims[j] < 0 ? dims[j] + nonzero_dim(in) : dims[j];
        ET_LOG_MSG_AND_RETURN_IF_FALSE(
            dim != dim_temp,
            "dim %" PRId64 " appears multiple times in dims!",
            dim);
      }
    }
  }

  return true;
}

void get_squeeze_copy_dims_out_target_size(
    const Tensor in,
    const exec_aten::ArrayRef<int64_t> dims,
    Tensor::SizesType* out_sizes,
    size_t* out_ndim) {
  // For 0 dim tensors, the output should also be 0 dim.
  if (in.dim() == 0) {
    *out_ndim = 0;
    return;
  }

  // A dim is only removed if the size at the given dim is 1.
  Tensor::SizesType dims_to_remove = 0;
  for (size_t i = 0; i < dims.size(); ++i) {
    int64_t dim = dims[i] < 0 ? dims[i] + nonzero_dim(in) : dims[i];
    if (in.size(dim) == 1) {
      ++dims_to_remove;
    }
  }
  *out_ndim = in.dim() - dims_to_remove;

  size_t out_d = 0;
  for (size_t in_d = 0; in_d < in.dim(); ++in_d) {
    bool in_d_in_dims = false;
    for (size_t i = 0; i < dims.size(); ++i) {
      int64_t dim = dims[i] < 0 ? dims[i] + nonzero_dim(in) : dims[i];
      if (in_d == dim) {
        in_d_in_dims = true;
        break;
      }
    }
    if (!in_d_in_dims || in.size(in_d) != 1) {
      out_sizes[out_d] = in.size(in_d);
      ++out_d;
    }
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

bool check_tril_args(const Tensor& in, Tensor& out) {
  ET_LOG_AND_RETURN_IF_FALSE(tensors_have_same_dtype(in, out));
  ET_LOG_AND_RETURN_IF_FALSE(tensor_has_rank_greater_or_equal_to(in, 2));
  return true;
}

} // namespace executor
} // namespace torch
