/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <c10/util/irange.h>
#include <cstring>

#include <executorch/kernels/portable/cpu/util/copy_ops_util.h>
#include <executorch/runtime/core/exec_aten/util/dim_order_util.h>
#include <executorch/runtime/core/exec_aten/util/tensor_util.h>

namespace torch {
namespace executor {

using Tensor = executorch::aten::Tensor;

namespace {

size_t as_strided_copy_compute_storage_nbytes(
    IntArrayRef sizes,
    IntArrayRef strides,
    size_t itemsize_bytes) {
  // size of the underlying storage is 1 bigger than the offset
  // of the last element according to stride
  size_t size = 1;
  for (const auto i : c10::irange(sizes.size())) {
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
    std::optional<int64_t> storage_offset,
    Tensor& out) {
  ET_LOG_AND_RETURN_IF_FALSE(tensors_have_same_dtype(in, out));
  ET_CHECK_OR_RETURN_FALSE(
      size.size() == stride.size(),
      "mismatch in length of strides and shape; size.size() = %zu, stride.size() = %zu",
      size.size(),
      stride.size());
  for (const auto& val : stride) {
    ET_CHECK_OR_RETURN_FALSE(
        val >= 0,
        "as_strided: Negative strides are not supported at the moment");
  }

  int64_t offset = storage_offset.has_value() ? storage_offset.value() : 0;
  ET_CHECK_OR_RETURN_FALSE(offset >= 0, "Negative storage offset");

  // Check that the requested storage is within bounds of input storage
  size_t storage_size_bytes =
      as_strided_copy_compute_storage_nbytes(size, stride, in.element_size());
  size_t storage_offset_bytes = offset * in.element_size();
  if (storage_size_bytes == 0) {
    return true;
  }
  size_t new_storage_size_bytes = in.nbytes();
  ET_CHECK_OR_RETURN_FALSE(
      storage_size_bytes + storage_offset_bytes <= new_storage_size_bytes,
      "Requiring a storage size of %zd are out of bounds for storage of size %zd",
      storage_size_bytes + storage_offset_bytes,
      new_storage_size_bytes);
  return true;
}

bool check_cat_args(
    executorch::aten::ArrayRef<Tensor> tensors,
    int64_t dim,
    Tensor& out) {
  // Ensure the input tensors list is non-empty
  ET_LOG_AND_RETURN_IF_FALSE(tensors.size() > 0);

  // Find the first non-empty tensor in the list to use as a reference
  size_t ref_i = 0;
  for (const auto i : c10::irange(tensors.size())) {
    if (tensors[i].numel() > 0) {
      ref_i = i;
      break;
    }
  }

  // "All tensors must either have the same shape (except in the concatenating
  // dimension) or be empty."
  // https://pytorch.org/docs/stable/generated/torch.cat.html
  for (const auto i : c10::irange(tensors.size())) {
    // All input dtypes must be castable to the output dtype.
    ET_LOG_AND_RETURN_IF_FALSE(
        canCast(tensors[i].scalar_type(), out.scalar_type()));

    ET_LOG_AND_RETURN_IF_FALSE(tensors_have_same_dim_order(tensors[i], out));

    // Empty tensors have no shape constraints.
    if (tensors[i].numel() == 0) {
      continue;
    }

    // All input tensors must have the same number of dimensions.
    ET_LOG_AND_RETURN_IF_FALSE(
        tensor_is_rank(tensors[ref_i], tensors[i].dim()));

    for (const auto d : c10::irange(tensors[i].dim())) {
      if (d != dim) {
        ET_LOG_AND_RETURN_IF_FALSE(
            tensors_have_same_size_at_dims(tensors[i], d, tensors[ref_i], d));
      }
    }
  }

  // Ensure dim is in range.
  ET_LOG_AND_RETURN_IF_FALSE(
      tensors[ref_i].numel() == 0 || tensors[ref_i].dim() > dim);
  ET_LOG_AND_RETURN_IF_FALSE(dim >= 0);

  return true;
}

void get_cat_out_target_size(
    executorch::aten::ArrayRef<Tensor> tensors,
    int64_t dim,
    executorch::aten::SizesType* out_sizes,
    size_t* out_ndim) {
  // Find the first non-1D-or-empty tensor in the list to use as a reference
  // because an 1D empty tensor is a wildcard and should be ignored when we
  // calculate out dim
  size_t ref_i = 0;
  size_t cat_dim_size = 0;
  for (const auto i : c10::irange(tensors.size())) {
    if (tensors[i].numel() > 0) {
      cat_dim_size += tensors[i].size(dim);
    }
    if (tensors[i].dim() != 1 || tensors[i].numel() != 0) {
      ref_i = i;
    }
  }

  *out_ndim = tensors[ref_i].dim();

  for (const auto d : c10::irange(*out_ndim)) {
    if (static_cast<int64_t>(d) != dim) {
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
  (void)out;

  ET_CHECK_OR_RETURN_FALSE(
      implicit == false,
      "This operator is not implemented for when implicit == true.");

  ET_CHECK_OR_RETURN_FALSE(
      expand_sizes.size() >= input.sizes().size(),
      "The number of sizes provided (%zu) must at least be equal to the number of dimensions in the tensor (%zu)",
      expand_sizes.size(),
      input.sizes().size());

  ET_CHECK_OR_RETURN_FALSE(
      expand_sizes.size() <= kTensorDimensionLimit,
      "The number of expanded dims (%zu) exceeds the configured maximum (%zu). Increase this limit.",
      expand_sizes.size(),
      kTensorDimensionLimit);

  ET_LOG_AND_RETURN_IF_FALSE(tensors_have_same_dtype(input, out));

  return true;
}

bool get_expand_copy_out_target_size(
    executorch::aten::ArrayRef<executorch::aten::SizesType> self_sizes,
    executorch::aten::ArrayRef<int64_t> expand_sizes,
    executorch::aten::SizesType* output_sizes,
    size_t* output_rank) {
  auto j{expand_sizes.size()};
  *output_rank = 0;

  for (size_t i{self_sizes.size()}; i > 0 && j > 0;) {
    --i;
    --j;

    output_sizes[j] = expand_sizes[j];

    if (expand_sizes[j] == -1) {
      // -1 can use for replacing any corresponding dimension
      output_sizes[j] = self_sizes[i];
    } else if (self_sizes[i] != 1) {
      ET_CHECK_OR_RETURN_FALSE(
          expand_sizes[j] == self_sizes[i],
          "The expanded size of the tensor (%zu) must match the existing size (%zu) at non-singleton dimension %zu.",
          (size_t)expand_sizes[j],
          (size_t)self_sizes[i],
          i);
    }
  }

  // The leading expand_sizes cannot be negative
  while (j > 0) {
    --j;
    output_sizes[j] = expand_sizes[j];
    ET_CHECK_OR_RETURN_FALSE(
        expand_sizes[j] >= 0,
        "The expanded size of the tensor (%zu) isn't allowed in a leading, non-existing dimension %zu",
        (size_t)expand_sizes[j],
        j);
  }

  *output_rank = expand_sizes.size();
  return true;
}

bool check_permute_copy_args(const Tensor& in, IntArrayRef dims, Tensor& out) {
  ET_LOG_AND_RETURN_IF_FALSE(tensor_is_rank(in, dims.size()));
  ET_LOG_AND_RETURN_IF_FALSE(tensors_have_same_dtype(in, out));

  // Make sure no dimensions are duplicated and all in the range [-in.dim(),
  // in.dim() - 1].
  bool dim_exist[kTensorDimensionLimit];
  memset(dim_exist, false, sizeof(dim_exist));

  for (const auto i : c10::irange(dims.size())) {
    ET_LOG_AND_RETURN_IF_FALSE(tensor_has_dim(in, dims[i]));
    // Convert dimension to a non-negative number in the range
    // [0 .. in.dim() - 1].
    size_t dim = dims[i] >= 0 ? dims[i] : in.dim() + dims[i];

    // Internal check, since we have already validated this
    ET_LOG_AND_RETURN_IF_FALSE(dim < kTensorDimensionLimit && dim >= 0);

    // Check that the dimension hasn't been seen previously.
    ET_CHECK_OR_RETURN_FALSE(
        dim_exist[dim] == false,
        "duplicate dims are not allowed; dim = %zu",
        dim);

    dim_exist[dim] = true;
  }

  return true;
}

bool check_unbind_copy_args(const Tensor& in, int64_t dim, TensorList out) {
  ET_CHECK_OR_RETURN_FALSE(
      in.dim() > 0, "in must have at least one dimension; saw %zd", in.dim());

  ET_LOG_AND_RETURN_IF_FALSE(dim_is_valid(dim, in.dim()));

  const ssize_t dim_size = in.size(dim);
  ET_CHECK_OR_RETURN_FALSE(
      dim_size == static_cast<ssize_t>(out.size()),
      "out tensorlist's length %zd must equal unbind dim %" PRId64
      " size = %zd.",
      out.size(),
      dim,
      dim_size);

  // Validate each output.
  for (const auto i : c10::irange(out.size())) {
    // All output dtypes must be the same.
    ET_CHECK_OR_RETURN_FALSE(
        out[i].scalar_type() == out[0].scalar_type(),
        "out[%zu] dtype %" PRId8 " != out[0] dtype %" PRId8,
        i,
        static_cast<int8_t>(out[i].scalar_type()),
        static_cast<int8_t>(out[0].scalar_type()));

    // output tensor must have # of dims = in.dim() -1
    ET_CHECK_OR_RETURN_FALSE(
        out[i].dim() == (in.dim() - 1),
        "out[%zu] dim %zd != in dim %zd",
        i,
        out[i].dim(),
        in.dim() - 1);

    // Check the shape of the output.
    ssize_t out_d = 0;
    for (const auto d : c10::irange(in.dim())) {
      if (d != dim) {
        ET_CHECK_OR_RETURN_FALSE(
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
    executorch::aten::SizesType* out_sizes,
    size_t* out_ndim) {
  *out_ndim = in.dim();

  for (const auto i : c10::irange(in.dim())) {
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

bool check_pixel_unshuffle_args(
    const Tensor& in,
    int64_t downscale_factor,
    Tensor& out) {
  ET_LOG_AND_RETURN_IF_FALSE(tensors_have_same_dtype(in, out));
  ET_LOG_AND_RETURN_IF_FALSE(tensor_has_rank_greater_or_equal_to(in, 3));
  ET_LOG_AND_RETURN_IF_FALSE(tensor_has_rank_greater_or_equal_to(out, 3));
  ET_LOG_AND_RETURN_IF_FALSE(downscale_factor > 0);
  ET_LOG_AND_RETURN_IF_FALSE(in.size(in.dim() - 1) % downscale_factor == 0);
  ET_LOG_AND_RETURN_IF_FALSE(in.size(in.dim() - 2) % downscale_factor == 0);
  return true;
}

void get_pixel_shuffle_out_target_size(
    const Tensor& in,
    int64_t upscale_factor,
    executorch::aten::SizesType* out_sizes,
    size_t* out_ndim) {
  *out_ndim = in.dim();
  const executorch::aten::SizesType casted_upscale_factor = upscale_factor;

  ssize_t i = 0;
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

void get_pixel_unshuffle_out_target_size(
    const Tensor& in,
    int64_t downscale_factor,
    executorch::aten::SizesType* out_sizes,
    size_t* out_ndim) {
  *out_ndim = in.dim();
  const executorch::aten::SizesType casted_factor = downscale_factor;

  ssize_t i = 0;
  for (; i < in.dim() - 3; ++i) {
    // Copy all leading dimensions in.
    out_sizes[i] = in.size(i);
  }
  // The last 3 dimensions are (channel, height, width). Multiply channel by
  // the downscale factor squared and divide the height and width by that
  // factor.
  out_sizes[i] = in.size(i) * (casted_factor * casted_factor);
  i++;
  out_sizes[i] = in.size(i) / casted_factor;
  i++;
  out_sizes[i] = in.size(i) / casted_factor;
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
    executorch::aten::SizesType* out_sizes,
    size_t* out_ndim) {
  *out_ndim = in.dim() - 1;

  for (const auto d : c10::irange(in.dim() - 1)) {
    if (d < dim) {
      out_sizes[d] = in.size(d);
    } else {
      out_sizes[d] = in.size(d + 1);
    }
  }
}

bool check_split_with_sizes_copy_args(
    const Tensor& in,
    executorch::aten::ArrayRef<int64_t> split_sizes,
    int64_t dim,
    TensorList out) {
  ET_LOG_AND_RETURN_IF_FALSE(tensor_has_rank_greater_or_equal_to(in, 1));
  ET_LOG_AND_RETURN_IF_FALSE(tensor_has_dim(in, dim));

  ET_CHECK_OR_RETURN_FALSE(
      split_sizes.size() == out.size(),
      "Number of split sizes must match the number of output tensors; split_sizes.size() = %zu, out.size() = %zu",
      split_sizes.size(),
      out.size());

  int64_t sum = 0;
  for (const auto i : c10::irange(split_sizes.size())) {
    ET_CHECK_OR_RETURN_FALSE(
        split_sizes[i] >= 0,
        "All split sizes must be non negative; split_sizes[%zu] = %" PRId64,
        i,
        split_sizes[i]);
    sum += split_sizes[i];
  }

  const ssize_t dim_size = in.size(dim);
  ET_CHECK_OR_RETURN_FALSE(
      sum == dim_size,
      "Sum of split sizes does not match input size at given dim; sum = %" PRId64
      ", dim_size = %zd",
      sum,
      dim_size);

  return true;
}

void get_split_with_sizes_copy_out_target_size(
    const Tensor& in,
    int64_t split_size,
    int64_t dim,
    executorch::aten::SizesType* out_sizes,
    size_t* out_ndim) {
  *out_ndim = in.dim();

  for (const auto d : c10::irange(in.dim())) {
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
    executorch::aten::SizesType* out_sizes,
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
  for (const auto in_d : c10::irange(in.dim())) {
    if (in_d != dim || in.size(in_d) != 1) {
      out_sizes[out_d] = in.size(in_d);
      ++out_d;
    }
  }
}

bool check_squeeze_copy_dims_args(
    const Tensor in,
    const executorch::aten::ArrayRef<int64_t> dims,
    const Tensor out) {
  ET_LOG_AND_RETURN_IF_FALSE(tensors_have_same_dtype(in, out));

  for (const auto i : c10::irange(dims.size())) {
    const int64_t dim = dims[i] < 0 ? dims[i] + nonzero_dim(in) : dims[i];
    ET_LOG_AND_RETURN_IF_FALSE(tensor_has_dim(in, dim));

    // Check that a dim does not appear twice in dims
    for (const auto j : c10::irange(dims.size())) {
      if (i != j) {
        const int64_t dim_temp =
            dims[j] < 0 ? dims[j] + nonzero_dim(in) : dims[j];
        ET_CHECK_OR_RETURN_FALSE(
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
    const executorch::aten::ArrayRef<int64_t> dims,
    executorch::aten::SizesType* out_sizes,
    size_t* out_ndim) {
  // For 0 dim tensors, the output should also be 0 dim.
  if (in.dim() == 0) {
    *out_ndim = 0;
    return;
  }

  // A dim is only removed if the size at the given dim is 1.
  executorch::aten::SizesType dims_to_remove = 0;
  for (const auto i : c10::irange(dims.size())) {
    int64_t dim = dims[i] < 0 ? dims[i] + nonzero_dim(in) : dims[i];
    if (in.size(dim) == 1) {
      ++dims_to_remove;
    }
  }
  *out_ndim = in.dim() - dims_to_remove;

  size_t out_d = 0;
  for (const auto in_d : c10::irange(in.dim())) {
    bool in_d_in_dims = false;
    for (const auto i : c10::irange(dims.size())) {
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
    executorch::aten::ArrayRef<Tensor> tensors,
    int64_t dim,
    Tensor& out) {
  // Ensure the input tensors list is non-empty
  ET_LOG_AND_RETURN_IF_FALSE(tensors.size() > 0);

  // All input tensors need to be of the same size
  // https://pytorch.org/docs/stable/generated/torch.stack.html
  for (const auto i : c10::irange(tensors.size())) {
    // All input dtypes must be castable to the output dtype.
    ET_LOG_AND_RETURN_IF_FALSE(
        canCast(tensors[i].scalar_type(), out.scalar_type()));

    ET_LOG_AND_RETURN_IF_FALSE(tensor_is_rank(tensors[i], tensors[0].dim()));
    for (const auto d : c10::irange(tensors[i].dim())) {
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
    executorch::aten::ArrayRef<Tensor> tensors,
    int64_t dim,
    executorch::aten::SizesType* out_sizes,
    size_t* out_ndim) {
  *out_ndim = tensors[0].dim() + 1;

  for (const auto d : c10::irange(*out_ndim)) {
    int64_t d_ = static_cast<int64_t>(d);
    if (d_ < dim) {
      out_sizes[d_] = tensors[0].size(d_);
    } else if (d_ == dim) {
      out_sizes[d_] = tensors.size();
    } else {
      out_sizes[d_] = tensors[0].size(d_ - 1);
    }
  }
}

bool check_tril_args(const Tensor& in, Tensor& out) {
  ET_LOG_AND_RETURN_IF_FALSE(tensors_have_same_dtype(in, out));
  ET_LOG_AND_RETURN_IF_FALSE(tensor_has_rank_greater_or_equal_to(in, 2));
  return true;
}

bool check_split_copy_args(
    const Tensor& input,
    int64_t split_size,
    int64_t dim,
    TensorList out) {
  ET_CHECK_OR_RETURN_FALSE(
      input.dim() > 0,
      "input must have at least one dimension; saw %zd",
      input.dim());
  ET_CHECK_OR_RETURN_FALSE(
      dim >= 0 && dim < input.dim(),
      "dim %" PRId64 " out of range [0,%zd)",
      dim,
      input.dim());

  const ssize_t dim_size = input.size(dim);
  ET_CHECK_OR_RETURN_FALSE(
      split_size >= 0,
      "split_size %" PRId64 " must be non-negative",
      split_size);
  ET_CHECK_OR_RETURN_FALSE(
      split_size > 0 || dim_size == 0,
      "split_size is zero but input.size(%" PRId64 ") %zd is non-zero",
      dim,
      dim_size);

  // Check the number of outputs.
  //
  // The specified dimension will be split into split_size-sized chunks, with
  // the final chunk possibly being smaller. So, the expected output length is
  // ceil(dim_size / split_size).
  //
  // E.g., splitting dim 0 of a [5,2] tensor with split_size 2 would produce
  // three tensors with size [2,2], [2,2], [1,2].
  int64_t remainder; // The size of the split dimension of the final out tensor.
  if (split_size >= dim_size) {
    // Note that this also handles the case where split_size == 0, avoiding a
    // division by zero in the other branch. When dim_size == 0 && split_size ==
    // 0, core PyTorch expects 1 output element.
    ET_CHECK_OR_RETURN_FALSE(
        out.size() == 1,
        "Unexpected out.size() %zu: should be 1 because split_size %" PRId64
        " >= input.size(%" PRId64 ") %zd",
        out.size(),
        split_size,
        dim,
        dim_size);
    remainder = dim_size;
  } else {
    int64_t expected_out_len = (dim_size + split_size - 1) / split_size;
    ET_CHECK_OR_RETURN_FALSE(
        static_cast<int64_t>(out.size()) == expected_out_len,
        "Unexpected out.size() %zu: ceil(input.size(%" PRId64
        ")=%zd"
        " / split_size=%" PRId64 ") is %" PRId64,
        out.size(),
        dim,
        dim_size,
        split_size,
        expected_out_len);
    remainder = dim_size % split_size;
    if (remainder == 0) {
      remainder = split_size;
    }
  }

  // Validate each output.
  for (const auto i : c10::irange(out.size())) {
    // All output dtypes must be the same.
    ET_CHECK_OR_RETURN_FALSE(
        out[i].scalar_type() == out[0].scalar_type(),
        "out[%zu] dtype %" PRId8 " != out[0] dtype %" PRId8,
        i,
        static_cast<int8_t>(out[i].scalar_type()),
        static_cast<int8_t>(out[0].scalar_type()));

    // All outputs must have the same number of dimensions as the input.
    ET_CHECK_OR_RETURN_FALSE(
        out[i].dim() == input.dim(),
        "out[%zu] dim %zd != input dim %zd",
        i,
        out[i].dim(),
        input.dim());

    // Check the shape of the output.
    for (const auto d : c10::irange(out[i].dim())) {
      if (d == dim) {
        // This is the split dimension, which may be different.
        if (i < out.size() - 1) {
          // All outputs except the final one: split dimension should be
          // split_size.
          ET_CHECK_OR_RETURN_FALSE(
              out[i].size(d) == split_size,
              "out[%zu].size(%zd) %zd != split_size %" PRId64,
              i,
              d,
              out[i].size(d),
              split_size);
        } else {
          // The final output: split dimension should be the remainder of
          // split_size.
          ET_CHECK_OR_RETURN_FALSE(
              out[i].size(d) == remainder,
              "out[%zu].size(%zd) %zd != remainder %" PRId64,
              i,
              d,
              out[i].size(d),
              remainder);
        }
      } else {
        // Non-split output dimensions must be the same as the input dimension.
        ET_LOG_AND_RETURN_IF_FALSE(
            tensors_have_same_size_at_dims(out[i], d, input, d));
      }
    }
  }

  return true;
}

bool check_to_copy_args(
    const Tensor& input,
    bool non_blocking,
    std::optional<executorch::aten::MemoryFormat> memory_format,
    Tensor& out) {
  (void)input;
  (void)out;

  // Right now we only support blocking data transfer
  ET_LOG_AND_RETURN_IF_FALSE(non_blocking == false);

  // Right now we only focus on contiguous memory, memory_format shall be
  // exec::aten::MemoryFormat::Contiguous or none.
  ET_LOG_AND_RETURN_IF_FALSE(
      !memory_format.has_value() ||
      memory_format.value() == MemoryFormat::Contiguous);

  return true;
}

bool check__to_dim_order_copy_args(
    const Tensor& input,
    bool non_blocking,
    executorch::aten::OptionalArrayRef<int64_t> dim_order,
    Tensor& out) {
  // Right now we only support blocking data transfer
  ET_LOG_AND_RETURN_IF_FALSE(non_blocking == false);

  if (dim_order.has_value()) {
    executorch::aten::ArrayRef<int64_t> dim_order_ref = dim_order.value();

    // dim order size shall equal to input dim
    ET_LOG_AND_RETURN_IF_FALSE(
        static_cast<ssize_t>(dim_order_ref.size()) == input.dim());

    ET_LOG_AND_RETURN_IF_FALSE(
        is_channels_last_dim_order(
            dim_order.value().data(), dim_order.value().size()) ||
        is_contiguous_dim_order(
            dim_order.value().data(), dim_order.value().size()));

    // Out tensor shall have same dim order as dim_order
    auto out_dim_order = out.dim_order();
    ET_LOG_AND_RETURN_IF_FALSE(out_dim_order.size() == dim_order_ref.size());
    for (const auto i : c10::irange(dim_order_ref.size())) {
      ET_LOG_AND_RETURN_IF_FALSE(out_dim_order[i] == dim_order_ref[i]);
    }
  } else { // dim_order is not set, preserve the dim order of input

    // Out tensor shall have same dim order as input dim_order
    auto out_dim_order = out.dim_order();
    auto input_dim_order = input.dim_order();
    ET_LOG_AND_RETURN_IF_FALSE(out_dim_order.size() == input_dim_order.size());
    for (const auto i : c10::irange(input_dim_order.size())) {
      ET_LOG_AND_RETURN_IF_FALSE(out_dim_order[i] == input_dim_order[i]);
    }
  }
  return true;
}

bool check_unsqueeze_copy_args(
    const Tensor input,
    int64_t dim,
    const Tensor out) {
  ET_LOG_AND_RETURN_IF_FALSE(dim >= 0);

  // The input and out shall share same dtype
  ET_LOG_AND_RETURN_IF_FALSE(tensors_have_same_dtype(input, out));

  ET_LOG_AND_RETURN_IF_FALSE(tensor_has_dim(out, dim));

  // The shape of input and out shall obey the relationship:
  // 1. input.dim() == out.dim()-1
  // 2. input.size(i) == out.size(i) for all i < dim
  // 3. input.size(i-1) == out.size(i) for all i >= dim
  // 4. out.size(dim) == 1
  ET_LOG_AND_RETURN_IF_FALSE(input.dim() == out.dim() - 1);

  for (auto const d : c10::irange(out.dim())) {
    auto dim_normalized = dim;
    if (dim_normalized < 0) {
      dim_normalized += out.dim();
    }

    if (d < dim_normalized) {
      ET_CHECK_OR_RETURN_FALSE(
          input.size(d) == out.size(d),
          "input.size(%zu) %zd != out.size(%zu) %zd | dim = %" PRId64,
          d,
          input.size(d),
          d,
          out.size(d),
          dim);
    } else if (d > dim_normalized) {
      ET_CHECK_OR_RETURN_FALSE(
          input.size(d - 1) == out.size(d),
          "input.size(%zu) %zd != out.size(%zu) %zd | dim = %" PRId64,
          d - 1,
          input.size(d),
          d,
          out.size(d),
          dim);
    } else { // d == dim
      ET_CHECK_OR_RETURN_FALSE(
          out.size(d) == 1,
          "out.size(%zu) %zd shall equal 1 | dim = %" PRId64,
          d,
          out.size(d),
          dim);
    }
  }

  return true;
}

bool check_view_copy_args(
    const Tensor& self,
    executorch::aten::ArrayRef<int64_t> size_int64_t,
    Tensor& out) {
  ET_LOG_AND_RETURN_IF_FALSE(size_int64_t.size() == out.sizes().size());

  // The input and out shall share same dtype and numel
  ET_CHECK_OR_RETURN_FALSE(
      self.numel() == out.numel(),
      "self.numel() %zd != out.numel() %zd",
      self.numel(),
      out.numel());
  ET_LOG_AND_RETURN_IF_FALSE(tensors_have_same_dtype(self, out));

  // The size of out should equal target size.
  bool size_inferred = false;
  for (auto const i : c10::irange(size_int64_t.size())) {
    // If this value is -1 it implies that this dimension is inferred.
    if (size_int64_t[i] == -1) {
      ET_CHECK_OR_RETURN_FALSE(
          !size_inferred, "Multiple dimensions cannot be inferred.");
      size_inferred = true;
    }
    ET_LOG_AND_RETURN_IF_FALSE(
        ((int64_t)out.sizes()[i] == size_int64_t[i]) ||
        (size_int64_t[i] == -1));
  }

  return true;
}

bool get_view_copy_target_size(
    const Tensor input,
    executorch::aten::ArrayRef<int64_t> size_int64_t,
    int64_t dim,
    executorch::aten::SizesType* out_sizes) {
  size_t out_numels_without_minus_1 = 1;
  int32_t minus_1_dim = -1;

  ET_LOG_AND_RETURN_IF_FALSE(static_cast<int64_t>(size_int64_t.size()) == dim);

  for (const auto i : c10::irange(dim)) {
    if (size_int64_t[i] != -1) {
      out_sizes[i] = static_cast<executorch::aten::SizesType>(size_int64_t[i]);
      out_numels_without_minus_1 = out_numels_without_minus_1 * size_int64_t[i];
    } else {
      // TODO(kimishpatel): Add test to hit this line
      ET_CHECK_OR_RETURN_FALSE(
          minus_1_dim == -1, "At most one view copy dim can be -1.");
      minus_1_dim = i;
    }
  }
  if (minus_1_dim >= 0) {
    out_sizes[minus_1_dim] = input.numel() / out_numels_without_minus_1;
  }

  return true;
}

bool check_diagonal_copy_args(
    const Tensor& in,
    int64_t dim1,
    int64_t dim2,
    Tensor& out) {
  ET_LOG_AND_RETURN_IF_FALSE(tensors_have_same_dtype(in, out));
  ET_LOG_AND_RETURN_IF_FALSE(tensor_has_rank_greater_or_equal_to(in, 2));
  ET_LOG_AND_RETURN_IF_FALSE(tensor_has_dim(in, dim1));
  ET_LOG_AND_RETURN_IF_FALSE(tensor_has_dim(in, dim2));
  if (dim1 < 0) {
    dim1 += nonzero_dim(in);
  }
  if (dim2 < 0) {
    dim2 += nonzero_dim(in);
  }
  ET_LOG_AND_RETURN_IF_FALSE(dim1 != dim2);
  return true;
}

void get_diagonal_copy_out_target_size(
    const Tensor& in,
    int64_t offset,
    int64_t dim1,
    int64_t dim2,
    executorch::aten::SizesType* out_sizes,
    size_t* out_ndim) {
  *out_ndim = in.dim() - 1;

  if (dim1 < 0) {
    dim1 += nonzero_dim(in);
  }
  if (dim2 < 0) {
    dim2 += nonzero_dim(in);
  }

  size_t diagonal_size = 0;
  if (offset >= 0) {
    if (in.size(dim2) <= offset) {
      diagonal_size = 0;
    } else {
      diagonal_size = std::min<size_t>(in.size(dim1), in.size(dim2) - offset);
    }
  } else {
    if (in.size(dim1) <= -offset) {
      diagonal_size = 0;
    } else {
      diagonal_size = std::min<size_t>(in.size(dim1) + offset, in.size(dim2));
    }
  }

  size_t shift = 0;
  for (const auto d : c10::irange(in.dim())) {
    if (d == dim1 || d == dim2) {
      shift++;
    } else {
      out_sizes[d - shift] = in.size(d);
    }
  }
  out_sizes[in.dim() - 2] = diagonal_size;
}

bool check_unfold_copy_args(
    const Tensor& self,
    int64_t dim,
    int64_t size,
    int64_t step) {
  if (dim < 0) {
    dim += nonzero_dim(self);
  }
  ET_LOG_AND_RETURN_IF_FALSE(tensor_has_dim(self, dim));
  ET_CHECK_OR_RETURN_FALSE(
      size >= 0, "size is %" PRId64 " but must be >= 0", size);
  ET_CHECK_OR_RETURN_FALSE(
      size <= self.size(dim),
      "maximum size for tensor at dimension %" PRId64
      " is %zd but size is %" PRId64,
      dim,
      self.size(dim),
      size);
  ET_CHECK_OR_RETURN_FALSE(
      step > 0, "step is %" PRId64 " but must be > 0", step);
  return true;
}

void get_unfold_copy_out_target_size(
    const Tensor& self,
    int64_t dim,
    int64_t size,
    int64_t step,
    executorch::aten::SizesType* out_sizes,
    size_t* out_ndim) {
  for (auto i : c10::irange(self.dim())) {
    out_sizes[i] = self.size(i);
  }
  // At `dim` dimension, we split the tensor into `size` chunks with `step`
  // stride.
  out_sizes[dim] = (self.size(dim) - size + step) / step;

  out_sizes[self.dim()] = size;
  *out_ndim = self.dim() + 1;
}

void get_view_as_real_copy_out_target_size(
    const Tensor& self,
    executorch::aten::SizesType* out_sizes) {
  for (auto i : c10::irange(self.dim())) {
    out_sizes[i] = self.size(i);
  }
  out_sizes[self.dim()] = 2;
}

} // namespace executor
} // namespace torch
