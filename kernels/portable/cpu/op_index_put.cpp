/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <c10/util/irange.h>
#include <cstring>

#include <executorch/kernels/portable/cpu/util/advanced_index_util.h>
#include <executorch/kernels/portable/cpu/util/broadcast_util.h>
#include <executorch/runtime/core/exec_aten/util/tensor_shape_to_c_string.h>
#include <executorch/runtime/kernel/kernel_includes.h>

namespace torch {
namespace executor {
namespace native {

using Tensor = executorch::aten::Tensor;
using TensorOptList = executorch::aten::ArrayRef<std::optional<Tensor>>;

Tensor& index_put_out(
    KernelRuntimeContext& ctx,
    const Tensor& in,
    TensorOptList indices,
    const Tensor& values,
    const bool accumulate,
    Tensor& out) {
  (void)ctx;

  ET_KERNEL_CHECK(
      ctx, check_index_args(in, indices, out), InvalidArgument, out);

  ET_KERNEL_CHECK(
      ctx, tensors_have_same_dtype(in, values), InvalidArgument, out);

  ET_KERNEL_CHECK(
      ctx, tensors_have_same_dim_order(in, out), InvalidArgument, out);

  ET_KERNEL_CHECK(ctx, tensor_is_default_dim_order(in), InvalidArgument, out);

  ScalarType in_type = in.scalar_type();
  size_t block_count = count_index_blocks(indices);

  // If indices list is empty or all indices are null, then the operation is
  // performed over then entire input tensor. So, this is equivalent to
  // out = values when accumulate is false. Otherwise, the operation is
  // out = in + values where accumulate is true.
  if (block_count == 0) {
    ET_KERNEL_CHECK(
        ctx, resize_tensor(out, in.sizes()) == Error::Ok, InvalidArgument, out);

    // Check that values tensors can be broadcasted to out
    ET_KERNEL_CHECK(
        ctx, tensor_is_broadcastable_to(values, out), InvalidArgument, out);

    ET_SWITCH_REALHBBF16_TYPES(in_type, ctx, "index_put.out", CTYPE, [&]() {
      apply_binary_elementwise_fn<CTYPE, CTYPE, CTYPE>(
          [accumulate](const CTYPE val_in, const CTYPE val) {
            return accumulate ? val_in + val : val;
          },
          in,
          values,
          out);
    });
    return out;
  }

  // The index output shape depends on whether all the non-null indices are
  // adjacent or not.
  bool adjacent = (block_count == 1);

  // Compute the expected index output shape.
  Tensor::SizesType x_sizes[kTensorDimensionLimit];
  size_t x_dim = 0;
  ET_KERNEL_CHECK(
      ctx,
      get_index_out_target_size(in, indices, adjacent, x_sizes, &x_dim),
      InvalidArgument,
      out);

  // Check that values tensors can be broadcasted to indexing result
  ET_KERNEL_CHECK(
      ctx,
      tensor_is_broadcastable_to(values.sizes(), {x_sizes, x_dim}),
      InvalidArgument,
      out);

  ET_KERNEL_CHECK(
      ctx, resize_tensor(out, in.sizes()) == Error::Ok, InvalidArgument, out);

  // No further action if the input is empty
  if (in.numel() == 0) {
    return out;
  }

  // To start, copy the input data into the out tensor
  memcpy(out.mutable_data_ptr<char>(), in.const_data_ptr<char>(), in.nbytes());

  // In what follows, `x = in[indices]`. This tensor is implicit, and it would
  // be much easier to be able to allocate memory, and then call index.Tensor
  // to compute `x`. But since we can't do that, we have to keep track of its
  // shape, number of dimensions, number of elements, and use it to translate
  // coordinates from `x` to `in`.

  // Compute the dim_map and ix_map needed for `x -> in` coordinate translation
  int32_t dim_map[kTensorDimensionLimit];
  int32_t ix_map[kTensorDimensionLimit];
  size_t start = 0;

  if (adjacent) {
    start = get_num_leading_null_indices(indices);
  }
  size_t bc_ndim = get_indices_broadcast_ndim(indices);
  compute_dim_map(in, indices, dim_map, block_count == 1);
  compute_index_map(in, indices, ix_map);

  // Compute the number of elements in the indexed space
  size_t x_numel = 1;
  for (const auto i : c10::irange(x_dim)) {
    x_numel *= x_sizes[i];
  }

  ET_SWITCH_REALHBBF16_TYPES(in_type, ctx, "index_put.out", CTYPE, [&]() {
    const CTYPE* const values_data = values.const_data_ptr<CTYPE>();
    CTYPE* const out_data = out.mutable_data_ptr<CTYPE>();

    for (const auto x_ix : c10::irange(x_numel)) {
      size_t in_ix = 0;

      size_t x_coord[kTensorDimensionLimit];
      delinearize_index(x_ix, {x_sizes, x_dim}, x_coord, kTensorDimensionLimit);

      size_t in_coord[kTensorDimensionLimit];

      ET_KERNEL_CHECK(
          ctx,
          get_in_coord(
              in, indices, start, bc_ndim, dim_map, ix_map, x_coord, in_coord),
          InvalidArgument, );

      in_ix = coordinateToIndex(in, in_coord);

      // Braodcast values
      size_t val_ix = linearize_access_indexes(x_coord, x_dim, values);
      if (accumulate) {
        out_data[in_ix] += values_data[val_ix];
      } else {
        out_data[in_ix] = values_data[val_ix];
      }
    }
  });

  return out;
}

namespace {

bool check_special_case_in_place_args(
    KernelRuntimeContext& ctx,
    Tensor& in,
    TensorOptList indices,
    const Tensor& values,
    const bool accumulate,
    size_t* dim) {
  ET_CHECK_OR_RETURN_FALSE(
      !accumulate,
      "Special case in-place index_put does not support accumulate");

  ET_CHECK_OR_RETURN_FALSE(
      static_cast<ssize_t>(indices.size()) <= in.dim(),
      "Indexing too many dimensions");

  bool found_index = false;
  for (const auto i : c10::irange(indices.size())) {
    if (indices[i].has_value()) {
      *dim = i;
      ET_CHECK_OR_RETURN_FALSE(
          !found_index,
          "Special case in-place index_put only supports a single non-null index tensor");
      found_index = true;
      const Tensor& index = indices[i].value();
      ScalarType ix_type = index.scalar_type();
      ET_CHECK_OR_RETURN_FALSE(
          ix_type == ScalarType::Long || ix_type == ScalarType::Int,
          "Special case in-place index_put only supports Long or Int index tensors; got %d",
          static_cast<int>(ix_type));
      ET_CHECK_OR_RETURN_FALSE(
          index.dim() == 1,
          "Special case in-place index_put only supports 1-dimensional index tensors; got %d",
          static_cast<int>(ix_type));
    }
  }

  ET_CHECK_OR_RETURN_FALSE(
      found_index,
      "Special case in-place index_put needs at least one non-null index tensor");

  const Tensor& index = indices[*dim].value();

  bool is_valid_index = true;
  ET_SWITCH_TWO_TYPES(
      Long, Int, index.scalar_type(), ctx, "index_put_", CTYPE, [&]() {
        const CTYPE* const index_arr = index.const_data_ptr<CTYPE>();
        for (const auto i : c10::irange(index.numel())) {
          if (index_arr[i] < 0 ||
              index_arr[i] >= static_cast<CTYPE>(in.size(*dim))) {
            ET_LOG(
                Error,
                "Index %" PRId64
                " out of range for tensor with size %zd"
                " at dimension %zu",
                static_cast<int64_t>(index_arr[i]),
                in.size(*dim),
                *dim);
            is_valid_index = false;
            break;
          }
        }
      });

  ET_CHECK_OR_RETURN_FALSE(
      is_valid_index,
      "Some index values are not within bounds of input tensor at indexed dim");

  ET_CHECK_OR_RETURN_FALSE(
      values.size(*dim) == index.size(0),
      "Special case in-place index_put requires values to match index length at the indexed dim; values.size(%zu) = %" ET_PRI_TENSOR_SIZE
      ", index_length = %zd",
      *dim,
      values.size(*dim),
      index.size(0));

  Tensor::SizesType expected_values_size[kTensorDimensionLimit] = {};
  size_t in_ndim = static_cast<size_t>(in.dim());
  for (const auto i : c10::irange(in_ndim)) {
    if (i != *dim) {
      expected_values_size[i] = static_cast<Tensor::SizesType>(in.size(i));
    }
  }
  expected_values_size[*dim] = static_cast<Tensor::SizesType>(index.size(0));

#if ET_LOG_ENABLED
  auto in_shape_str = executorch::runtime::tensor_shape_to_c_string(
      executorch::runtime::Span<const Tensor::SizesType>(
          in.sizes().data(), in.sizes().size()));
  auto values_shape_str = executorch::runtime::tensor_shape_to_c_string(
      executorch::runtime::Span<const Tensor::SizesType>(
          values.sizes().data(), values.sizes().size()));

  ET_CHECK_OR_RETURN_FALSE(
      tensor_has_expected_size(values, {expected_values_size, in_ndim}),
      "Special case in-place index_put requires values to match input shape except for indexed dim; got input shape %s and values shape %s",
      in_shape_str.data(),
      values_shape_str.data());
#else
  ET_CHECK_OR_RETURN_FALSE(
      tensor_has_expected_size(values, {expected_values_size, in_ndim}),
      "Special case in-place index_put requires values to match input shape except for indexed dim");
#endif // ET_LOG_ENABLED

  return true;
}

} // namespace

Tensor& index_put_(
    KernelRuntimeContext& ctx,
    Tensor& in,
    TensorOptList indices,
    const Tensor& values,
    const bool accumulate) {
  (void)ctx;

  ET_KERNEL_CHECK(
      ctx, tensors_have_same_dtype(in, values), InvalidArgument, in);

  ET_KERNEL_CHECK(
      ctx, tensors_have_same_dim_order(in, values), InvalidArgument, in);

  ET_KERNEL_CHECK(ctx, tensor_is_default_dim_order(in), InvalidArgument, in);

  size_t dim = 0;
  ET_KERNEL_CHECK(
      ctx,
      check_special_case_in_place_args(
          ctx, in, indices, values, accumulate, &dim),
      InvalidArgument,
      in);

  const Tensor& index = indices[dim].value();
  ScalarType index_type = index.scalar_type();

  if (in.dim() == 0) {
    memcpy(in.mutable_data_ptr(), values.const_data_ptr(), in.nbytes());
    return in;
  }

  size_t leading_dims = getLeadingDims(in, dim);
  size_t trailing_dims = getTrailingDims(in, dim);

  if (leading_dims == 0 || trailing_dims == 0) {
    return in;
  }

  size_t values_dim_length = values.size(dim);
  size_t in_dim_length = in.size(dim);

  size_t length_per_step = trailing_dims * in.element_size();

  const char* values_data = values.const_data_ptr<char>();
  char* in_data = in.mutable_data_ptr<char>();

  ET_SWITCH_TWO_TYPES(Long, Int, index_type, ctx, "index_put_", CTYPE, [&]() {
    const CTYPE* const index_arr = index.const_data_ptr<CTYPE>();
    for (const auto i : c10::irange(leading_dims)) {
      const char* src = values_data + i * values_dim_length * length_per_step;
      char* dest = in_data + i * in_dim_length * length_per_step;
      for (const auto j : c10::irange(values_dim_length)) {
        const char* copy_src = src + j * length_per_step;
        char* copy_dest = dest + index_arr[j] * length_per_step;
        memcpy(copy_dest, copy_src, length_per_step);
      }
    }
  });

  return in;
}

} // namespace native
} // namespace executor
} // namespace torch
