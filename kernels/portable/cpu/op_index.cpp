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

#include <executorch/runtime/kernel/kernel_includes.h>

#include <executorch/kernels/portable/cpu/util/index_util.h>

namespace torch {
namespace executor {
namespace native {

using Tensor = exec_aten::Tensor;

namespace {

template <typename CTYPE_IN, typename CTYPE_OUT>
void index_out_impl_mask(
    const Tensor& in,
    exec_aten::ArrayRef<exec_aten::optional<Tensor>> indices,
    Tensor& out) {
  // Data pointers
  const CTYPE_IN* const in_data = in.const_data_ptr<CTYPE_IN>();
  CTYPE_OUT* const out_data = out.mutable_data_ptr<CTYPE_OUT>();

  const Tensor& mask = indices[0].value();
  const bool* const mask_ptr = mask.const_data_ptr<bool>();
  size_t count = 0;
  for (int i = 0; i < mask.numel(); ++i) {
    if (mask_ptr[i]) {
      out_data[count] = static_cast<CTYPE_OUT>(in_data[i]);
      count++;
    }
  }
}

template <typename CTYPE_IN, typename CTYPE_OUT>
void index_out_impl_list(
    const Tensor& in,
    exec_aten::ArrayRef<exec_aten::optional<Tensor>> indices,
    Tensor& out) {
  // Data pointers
  const CTYPE_IN* const in_data = in.const_data_ptr<CTYPE_IN>();
  CTYPE_OUT* dst = out.mutable_data_ptr<CTYPE_OUT>();

  size_t num_idx_queries = get_indices_broadcast_len(indices);
  for (size_t idx = 0; idx < num_idx_queries; idx++) {
    const CTYPE_IN* src = in_data;

    // For each index query, align the src and dst pointers to the position
    // described by the query.
    size_t offset = get_index_query_pos_offset(idx, in, indices);
    src += offset;

    // Calculate the region of data to copy for this query.
    // For example, a 2x4x3x5 tensor indexing at [1, 1, :, :] should copy 15
    // elements.
    size_t copy_len = getTrailingDims(in, indices.size() - 1);

    for (size_t i = 0; i < copy_len; ++i) {
      dst[i] = static_cast<CTYPE_OUT>(src[i]);
    }
    dst += copy_len;
  }
}

} // namespace

Tensor& index_Tensor_out(
    RuntimeContext& ctx,
    const Tensor& in,
    exec_aten::ArrayRef<exec_aten::optional<Tensor>> indices,
    Tensor& out) {
  ET_KERNEL_CHECK(
      ctx, check_index_args(in, indices, out), InvalidArgument, out);

  if (indices.empty()) {
    ET_KERNEL_CHECK(
        ctx, resize_tensor(out, in.sizes()) == Error::Ok, InvalidArgument, out);
    memcpy(
        out.mutable_data_ptr<char>(), in.const_data_ptr<char>(), in.nbytes());
    return out;
  }

  size_t expected_ndim = 0;
  Tensor::SizesType expected_size[kTensorDimensionLimit];
  get_index_out_target_size(in, indices, expected_size, &expected_ndim);
  ET_KERNEL_CHECK(
      ctx,
      resize_tensor(out, {expected_size, expected_ndim}) == Error::Ok,
      InvalidArgument,
      out);

  check_index_args(in, indices, out);

  if (in.numel() == 0) {
    return out;
  }

  ScalarType in_type = in.scalar_type();
  ScalarType out_type = out.scalar_type();
  ET_SWITCH_REAL_TYPES_AND(Bool, in_type, ctx, "index", CTYPE_IN, [&]() {
    ET_SWITCH_REAL_TYPES_AND(Bool, out_type, ctx, "index", CTYPE_OUT, [&]() {
      if (is_index_mask(in, indices)) {
        index_out_impl_mask<CTYPE_IN, CTYPE_OUT>(in, indices, out);
      } else {
        index_out_impl_list<CTYPE_IN, CTYPE_OUT>(in, indices, out);
      }
    });
  });

  return out;
}

} // namespace native
} // namespace executor
} // namespace torch
