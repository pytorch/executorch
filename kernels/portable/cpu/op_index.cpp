/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 * Copyright 2025 Arm Limited and/or its affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <algorithm>
#include <cinttypes>
#include <cstdint>
#include <cstring>
#include <tuple>

#include <executorch/kernels/portable/cpu/util/advanced_index_util.h>
#include <executorch/kernels/portable/cpu/util/broadcast_util.h>
#include <executorch/runtime/kernel/kernel_includes.h>

namespace torch {
namespace executor {
namespace native {

using Tensor = executorch::aten::Tensor;
using TensorOptList = executorch::aten::ArrayRef<std::optional<Tensor>>;

namespace {

bool check_fast_path_conditions(
    ET_UNUSED const Tensor& in,
    TensorOptList indices,
    size_t* dim) {
  bool found_index = false;
  for (const auto i : c10::irange(indices.size())) {
    if (indices[i].has_value()) {
      *dim = i;
      // Fast path only supports a single non-null index tensor
      if (found_index) {
        return false;
      }
      found_index = true;
      const Tensor& index = indices[i].value();
      ScalarType ix_type = index.scalar_type();
      // Fast path only supports Long or Int index tensors
      if (ix_type != ScalarType::Long && ix_type != ScalarType::Int) {
        return false;
      }
      // Fast path only supports a 1-dimensional index tensor
      if (index.dim() != 1) {
        return false;
      }

      // Fast path only supports non-negative indices.
      if (ix_type == ScalarType::Int) {
        const int32_t* const data = index.const_data_ptr<int32_t>();
        if (std::any_of(data, data + index.numel(), [](const auto x) {
              return x < 0;
            })) {
          return false;
        }
      } else { // ScalarType::Long
        const int64_t* const data = index.const_data_ptr<int64_t>();
        if (std::any_of(data, data + index.numel(), [](const auto x) {
              return x < 0;
            })) {
          return false;
        }
      }
    }
  }

  // Fast path needs at least one non-null index tensor
  if (!found_index) {
    return false;
  }

  return true;
}

bool check_fast_path_args(
    KernelRuntimeContext& ctx,
    const Tensor& in,
    TensorOptList indices,
    size_t dim,
    Tensor& out) {
  ET_LOG_AND_RETURN_IF_FALSE(tensors_have_same_dtype(in, out));

  ET_CHECK_OR_RETURN_FALSE(
      static_cast<ssize_t>(indices.size()) <= in.dim(),
      "Indexing too many dimensions");

  const Tensor& index = indices[dim].value();

  bool is_valid_index = true;
  ET_SWITCH_TWO_TYPES(
      Long, Int, index.scalar_type(), ctx, "index.Tensor", CTYPE, [&]() {
        const CTYPE* const index_arr = index.const_data_ptr<CTYPE>();
        for (const auto i : c10::irange(index.numel())) {
          if (index_arr[i] < 0 ||
              index_arr[i] >= static_cast<CTYPE>(in.size(dim))) {
            ET_LOG(
                Error,
                "Index %" PRId64
                " out of range for tensor with size %zd"
                " at dimension %zu",
                static_cast<int64_t>(index_arr[i]),
                in.size(dim),
                dim);
            is_valid_index = false;
            break;
          }
        }
      });

  ET_CHECK_OR_RETURN_FALSE(
      is_valid_index,
      "Some index values are not within bounds of input tensor at indexed dim");

  return true;
}

void get_fast_path_index_out_target_size(
    const Tensor& in,
    TensorOptList indices,
    size_t dim,
    Tensor::SizesType* out_sizes,
    size_t* out_ndim) {
  *out_ndim = in.dim();

  for (const auto d : c10::irange(static_cast<size_t>(in.dim()))) {
    if (d != dim) {
      out_sizes[d] = static_cast<Tensor::SizesType>(in.size(d));
    } else {
      out_sizes[d] =
          static_cast<Tensor::SizesType>(indices[dim].value().numel());
    }
  }
}

Tensor& fast_path(
    KernelRuntimeContext& ctx,
    const Tensor& in,
    TensorOptList indices,
    size_t dim,
    Tensor& out) {
  ET_KERNEL_CHECK(
      ctx,
      check_fast_path_args(ctx, in, indices, dim, out),
      InvalidArgument,
      out);

  const Tensor& index = indices[dim].value();
  ScalarType index_type = index.scalar_type();

  // @lint-ignore CLANGTIDY facebook-hte-CArray
  Tensor::SizesType expected_size[kTensorDimensionLimit];
  size_t expected_ndim = 0;
  get_fast_path_index_out_target_size(
      in, indices, dim, expected_size, &expected_ndim);

  ET_KERNEL_CHECK(
      ctx,
      resize_tensor(out, {expected_size, expected_ndim}) == Error::Ok,
      InvalidArgument,
      out);

  if (out.dim() == 0) {
    memcpy(out.mutable_data_ptr(), in.const_data_ptr(), out.nbytes());
    return out;
  }

  size_t leading_dims = getLeadingDims(in, dim);
  size_t trailing_dims = getTrailingDims(in, dim);

  if (leading_dims == 0 || trailing_dims == 0) {
    return out;
  }

  size_t in_dim_length = in.size(dim);
  size_t out_dim_length = out.size(dim);

  size_t length_per_step = trailing_dims * in.element_size();

  const char* in_data = in.const_data_ptr<char>();
  char* out_data = out.mutable_data_ptr<char>();

  // @lint-ignore CLANGTIDY facebook-hte-CArray
  static constexpr const char op_name[] = "index.Tensor_out";

  ET_SWITCH_TWO_TYPES(Long, Int, index_type, ctx, op_name, CTYPE, [&]() {
    const CTYPE* const index_arr = index.const_data_ptr<CTYPE>();
    for (const auto i : c10::irange(leading_dims)) {
      const char* src = in_data + i * in_dim_length * length_per_step;
      char* dest = out_data + i * out_dim_length * length_per_step;
      for (const auto j : c10::irange(out_dim_length)) {
        const char* copy_src = src + index_arr[j] * length_per_step;
        char* copy_dest = dest + j * length_per_step;
        memcpy(copy_dest, copy_src, length_per_step);
      }
    }
  });

  return out;
}

} // namespace

Tensor& index_Tensor_out(
    KernelRuntimeContext& ctx,
    const Tensor& in,
    TensorOptList indices,
    Tensor& out) {
  ET_KERNEL_CHECK(
      ctx, tensors_have_same_dim_order(in, out), InvalidArgument, out);

  ET_KERNEL_CHECK(ctx, tensor_is_default_dim_order(in), InvalidArgument, out);

  size_t dim = 0;
  bool is_fast_path = check_fast_path_conditions(in, indices, &dim);
  if (is_fast_path) {
    return fast_path(ctx, in, indices, dim, out);
  }

  ET_KERNEL_CHECK(
      ctx, check_index_args(in, indices, out), InvalidArgument, out);

  ScalarType in_type = in.scalar_type();
  size_t block_count = count_index_blocks(indices);

  // If indices list is empty or all indices are null, just copy the input to
  // output and return early.
  if (block_count == 0) {
    ET_KERNEL_CHECK(
        ctx, resize_tensor(out, in.sizes()) == Error::Ok, InvalidArgument, out);
    ET_SWITCH_REALHBBF16_TYPES(in_type, ctx, "index.Tensor_out", CTYPE, [&]() {
      const CTYPE* const in_data = in.const_data_ptr<CTYPE>();
      CTYPE* const out_data = out.mutable_data_ptr<CTYPE>();
      memcpy(out_data, in_data, in.nbytes());
    });
    return out;
  }

  // The output shape depends on whether all the non-null indices are adjacent
  // or not.
  bool adjacent = (block_count == 1);

  Tensor::SizesType expected_size[kTensorDimensionLimit];
  size_t expected_ndim = 0;

  ET_KERNEL_CHECK(
      ctx,
      get_index_out_target_size(
          in, indices, adjacent, expected_size, &expected_ndim),
      InvalidArgument,
      out);

  ET_KERNEL_CHECK(
      ctx,
      resize_tensor(out, {expected_size, expected_ndim}) == Error::Ok,
      InvalidArgument,
      out);

  if (out.numel() == 0) {
    return out;
  }

  int32_t dim_map[kTensorDimensionLimit];
  int32_t ix_map[kTensorDimensionLimit];
  size_t start = 0;
  size_t xdim = 0;

  if (adjacent) {
    start = get_num_leading_null_indices(indices);
  }
  xdim = get_indices_broadcast_ndim(indices);
  compute_dim_map(in, indices, dim_map, block_count == 1);
  compute_index_map(in, indices, ix_map);

  ET_SWITCH_REALHBBF16_TYPES(in_type, ctx, "index.Tensor_out", CTYPE, [&]() {
    const CTYPE* const in_data = in.const_data_ptr<CTYPE>();
    CTYPE* const out_data = out.mutable_data_ptr<CTYPE>();

    for (auto out_ix = 0; out_ix < out.numel(); out_ix++) {
      size_t in_ix = 0;
      bool success = true;
      std::tie(in_ix, success) =
          get_in_ix(in, indices, out, out_ix, start, xdim, dim_map, ix_map);
      ET_KERNEL_CHECK(ctx, success, InvalidArgument, );
      out_data[out_ix] = in_data[in_ix];
    }
  });

  return out;
}

} // namespace native
} // namespace executor
} // namespace torch
