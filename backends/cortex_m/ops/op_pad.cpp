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

constexpr size_t kMaxSupportedDims = 4;

// cppcheck-suppress unusedFunction
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
        "cortex_m::pad: only int8 tensors are supported (input=%d, out=%d)",
        static_cast<int>(input.scalar_type()),
        static_cast<int>(out.scalar_type()));
    context.fail(Error::InvalidArgument);
    return out;
  }

  const size_t rank = input.dim();
  if (rank == 0 || rank > kMaxSupportedDims) {
    ET_LOG(
        Error,
        "cortex_m::pad: expected tensor rank in [1, %zu], got %zu",
        kMaxSupportedDims,
        rank);
    context.fail(Error::InvalidArgument);
    return out;
  }
  if (pre_pad.size() != kMaxSupportedDims ||
      post_pad.size() != kMaxSupportedDims) {
    ET_LOG(Error, "cortex_m::pad: pre_pad and post_pad must have length 4");
    context.fail(Error::InvalidArgument);
    return out;
  }

  // Read the sizes in physical memory order. The dim order says which logical
  // axis sits where, so this covers an NCHW-logical channels-last tensor and an
  // NHWC-logical contiguous one without asking which contract is in force.
  // Padding is already in physical order from the AOT pass.
  const size_t offset = kMaxSupportedDims - rank;
  const auto dim_order = input.dim_order();

  int32_t dims[kMaxSupportedDims] = {1, 1, 1, 1};
  for (size_t i = 0; i < rank; ++i) {
    dims[offset + i] = static_cast<int32_t>(input.size(dim_order[i]));
  }

  cmsis_nn_dims input_dims = {dims[0], dims[1], dims[2], dims[3]};
  cmsis_nn_dims cmsis_pre_pad = {
      static_cast<int32_t>(pre_pad[0]),
      static_cast<int32_t>(pre_pad[1]),
      static_cast<int32_t>(pre_pad[2]),
      static_cast<int32_t>(pre_pad[3])};
  cmsis_nn_dims cmsis_post_pad = {
      static_cast<int32_t>(post_pad[0]),
      static_cast<int32_t>(post_pad[1]),
      static_cast<int32_t>(post_pad[2]),
      static_cast<int32_t>(post_pad[3])};

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
        "cortex_m::pad: arm_pad_s8 failed with status [%d]",
        static_cast<int>(status));
    context.fail(Error::Internal);
    return out;
  }

  return out;
}

} // namespace native
} // namespace cortex_m
