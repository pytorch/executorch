/*
 * Copyright 2025-2026 Arm Limited and/or its affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "cortex_m_ops_common.h"

extern "C" {
#include "arm_nnfunctions.h"
}

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

  const size_t offset = kMaxSupportedDims - rank;

  cmsis_nn_dims input_dims = {1, 1, 1, 1};
  int32_t* d = &input_dims.n;
  for (size_t i = 0; i < rank; ++i) {
    d[offset + i] = static_cast<int32_t>(input.size(i));
  }

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
        "pad_out: arm_pad_s8 failed with status [%d]",
        static_cast<int>(status));
    context.fail(Error::Internal);
    return out;
  }

  return out;
}

} // namespace native
} // namespace cortex_m
