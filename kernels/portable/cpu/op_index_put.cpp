/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <cinttypes>
#include <cstdint>
#include <cstring>

#include <executorch/kernels/portable/cpu/util/index_util.h>
#include <executorch/runtime/kernel/kernel_includes.h>

namespace torch {
namespace executor {
namespace native {

using Tensor = exec_aten::Tensor;

namespace {

template <typename CTYPE>
void index_put_out_impl_mask(
    const Tensor& in,
    exec_aten::ArrayRef<exec_aten::optional<Tensor>> indices,
    const Tensor& values,
    const bool accum,
    Tensor& out) {
  // Data pointers
  const CTYPE* const in_data = in.const_data_ptr<CTYPE>();
  CTYPE* const out_data = out.mutable_data_ptr<CTYPE>();

  const CTYPE* val_data = values.const_data_ptr<CTYPE>();

  // To start, copy the in into the output
  memcpy(out_data, in_data, in.nbytes());

  const Tensor& mask = indices[0].value();
  const bool* const mask_ptr = mask.const_data_ptr<bool>();
  size_t count = 0;
  for (int i = 0; i < mask.numel(); ++i) {
    if (mask_ptr[i]) {
      if (accum) {
        out_data[i] += val_data[count];
      } else {
        out_data[i] = val_data[count];
      }
      if (values.numel() > 1) {
        count++;
      }
    }
  }
}

template <typename CTYPE>
void index_put_out_impl_list(
    const Tensor& in,
    exec_aten::ArrayRef<exec_aten::optional<Tensor>> indices,
    const Tensor& values,
    const bool accum,
    Tensor& out) {
  // Data pointers
  const CTYPE* const in_data = in.const_data_ptr<CTYPE>();
  CTYPE* const out_data = out.mutable_data_ptr<CTYPE>();

  const CTYPE* val = values.const_data_ptr<CTYPE>();

  // To start, copy the in into the output
  memcpy(out_data, in_data, in.nbytes());

  size_t num_idx_queries = get_indices_broadcast_len(indices);
  for (size_t idx = 0; idx < num_idx_queries; idx++) {
    const CTYPE* src = in_data;
    CTYPE* dst = out_data;

    // For each index query, align the src and dst pointers to the position
    // described by the query.
    size_t offset = get_index_query_pos_offset(idx, in, indices);
    src += offset;
    dst += offset;

    // Calculate the region of data to copy for this query.
    // For example, a 2x4x3x5 tensor indexing at [1, 1, :, :] should copy 15
    // elements.
    size_t copy_len = getTrailingDims(in, indices.size() - 1);

    // If values only contains 1 element, it needs to be broadcasted.
    if (values.numel() == 1) {
      CTYPE value = *val;

      for (size_t i = 0; i < copy_len; ++i) {
        if (accum) {
          dst[i] += value;
        } else {
          dst[i] = value;
        }
      }
    }
    // General case.
    else {
      if (accum) {
        for (size_t i = 0; i < copy_len; ++i) {
          dst[i] = src[i] + val[i];
        }
        val += copy_len;
      } else {
        size_t copy_size = copy_len * sizeof(CTYPE);
        memcpy(dst, val, copy_size);
        val += copy_len;
      }
    }
  }
}

} // namespace

Tensor& index_put_out(
    RuntimeContext& ctx,
    const Tensor& in,
    exec_aten::ArrayRef<exec_aten::optional<Tensor>> indices,
    const Tensor& values,
    const bool accumulate,
    Tensor& out) {
  ET_KERNEL_CHECK(
      ctx,
      check_index_put_args(in, indices, values, out),
      InvalidArgument,
      out);

  if (indices.empty() || in.numel() == 0) {
    ET_KERNEL_CHECK(
        ctx, resize_tensor(out, in.sizes()) == Error::Ok, InvalidArgument, out);
    memcpy(
        out.mutable_data_ptr<char>(), in.const_data_ptr<char>(), in.nbytes());
    return out;
  }

  ET_KERNEL_CHECK(
      ctx, resize_tensor(out, in.sizes()) == Error::Ok, InvalidArgument, out);

  ScalarType dtype = in.scalar_type();
  ET_SWITCH_REAL_TYPES_AND(Bool, dtype, ctx, "index_put", CTYPE, [&]() {
    if (is_index_mask(in, indices)) {
      index_put_out_impl_mask<CTYPE>(in, indices, values, accumulate, out);
    } else {
      index_put_out_impl_list<CTYPE>(in, indices, values, accumulate, out);
    }
  });

  return out;
}

} // namespace native
} // namespace executor
} // namespace torch
