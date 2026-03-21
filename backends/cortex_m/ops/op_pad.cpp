/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "cortex_m_ops_common.h"

namespace cortex_m {
namespace native {

using KernelRuntimeContext = torch::executor::KernelRuntimeContext;

namespace {

constexpr size_t kMaxSupportedDims = 4;

} // namespace

Tensor& pad_out(
    KernelRuntimeContext& context,
    const Tensor& input,
    const Int64ArrayRef pre_pad,
    const Int64ArrayRef post_pad,
    int64_t pad_value,
    Tensor& out) {
  if (input.scalar_type() != ScalarType::Char ||
      out.scalar_type() != ScalarType::Char) {
    ET_LOG(
        Error,
        "pad_out: only int8 tensors are supported (input=%d, out=%d)",
        static_cast<int>(input.scalar_type()),
        static_cast<int>(out.scalar_type()));
    context.fail(Error::InvalidArgument);
    return out;
  }

  const size_t rank = input.dim();
  if (rank == 0 || rank > kMaxSupportedDims) {
    ET_LOG(
        Error,
        "pad_out: expected tensor rank in [1, %zu], got %zu",
        kMaxSupportedDims,
        rank);
    context.fail(Error::InvalidArgument);
    return out;
  }

  // arm_pad_s8 processes data in {n, h, w, c} order where c is the
  // fastest-varying (innermost) dimension. Use dim_order to permute
  // logical sizes and padding into physical memory order so this holds
  // for both contiguous and channels_last tensors.
  const size_t offset = kMaxSupportedDims - rank;
  int32_t logical_dims[kMaxSupportedDims] = {1, 1, 1, 1};
  for (size_t i = 0; i < rank; ++i) {
    logical_dims[offset + i] = static_cast<int32_t>(input.size(i));
  }

  int32_t physical_dims[kMaxSupportedDims];
  int32_t physical_pre[kMaxSupportedDims];
  int32_t physical_post[kMaxSupportedDims];

  // Leading virtual dims (for rank < 4) are always identity-ordered.
  for (size_t i = 0; i < offset; ++i) {
    physical_dims[i] = 1;
    physical_pre[i] = static_cast<int32_t>(pre_pad[i]);
    physical_post[i] = static_cast<int32_t>(post_pad[i]);
  }

  // Permute the real dims according to dim_order.
  const auto dim_order = input.dim_order();
  for (size_t i = 0; i < rank; ++i) {
    const size_t logical_idx = offset + dim_order[i];
    physical_dims[offset + i] = logical_dims[logical_idx];
    physical_pre[offset + i] = static_cast<int32_t>(pre_pad[logical_idx]);
    physical_post[offset + i] = static_cast<int32_t>(post_pad[logical_idx]);
  }

  cmsis_nn_dims input_dims = {
      physical_dims[0], physical_dims[1], physical_dims[2], physical_dims[3]};
  cmsis_nn_dims cmsis_pre_pad = {
      physical_pre[0], physical_pre[1], physical_pre[2], physical_pre[3]};
  cmsis_nn_dims cmsis_post_pad = {
      physical_post[0], physical_post[1], physical_post[2], physical_post[3]};

  const int8_t* input_data = input.const_data_ptr<int8_t>();
  int8_t* output_data = out.mutable_data_ptr<int8_t>();

  const arm_cmsis_nn_status status = arm_pad_s8(
      input_data,
      output_data,
      static_cast<int8_t>(pad_value),
      &input_dims,
      &cmsis_pre_pad,
      &cmsis_post_pad);

  if (status != ARM_CMSIS_NN_SUCCESS) {
    ET_LOG(
        Error,
        "pad_out: arm_pad_s8 failed with status [%d]",
        static_cast<int>(status));
    context.fail(Error::Internal);
    return out;
  }

  return out;
}

} // namespace native
} // namespace cortex_m
