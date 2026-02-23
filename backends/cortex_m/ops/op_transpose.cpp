/*
 * Copyright 2025-2026 Arm Limited and/or its affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "cortex_m_ops_common.h"

#include <array>
#include <limits>
#include <vector>

// Include CMSIS-NN headers with C linkage
extern "C" {
#include "arm_nnfunctions.h"
}

namespace cortex_m {
namespace native {

using KernelRuntimeContext = torch::executor::KernelRuntimeContext;

namespace {

constexpr size_t kMaxSupportedDims = 4;

} // namespace

Tensor& transpose_out(
    KernelRuntimeContext& context,
    const Tensor& input,
    const Int64ArrayRef perm,
    Tensor& out) {
  if (input.scalar_type() != ScalarType::Char ||
      out.scalar_type() != ScalarType::Char) {
    ET_LOG(
        Error,
        "transpose_out: only int8 tensors are supported (input=%d, out=%d)",
        static_cast<int>(input.scalar_type()),
        static_cast<int>(out.scalar_type()));
    context.fail(Error::InvalidArgument);
    return out;
  }

  const size_t rank = input.dim();
  if (rank == 0 || rank > kMaxSupportedDims) {
    ET_LOG(
        Error,
        "transpose_out: expected tensor rank in [1, %zu], got %zu",
        kMaxSupportedDims,
        rank);
    context.fail(Error::InvalidArgument);
    return out;
  }

  if (perm.size() != static_cast<int64_t>(rank)) {
    ET_LOG(
        Error,
        "transpose_out: permutation length %zd does not match tensor rank %zu",
        perm.size(),
        rank);
    context.fail(Error::InvalidArgument);
    return out;
  }

  std::array<int32_t, kMaxSupportedDims> input_dims_arr{1, 1, 1, 1};
  std::array<int32_t, kMaxSupportedDims> output_dims_arr{1, 1, 1, 1};
  for (size_t i = 0; i < rank; ++i) {
    const auto in_size = input.size(i);
    const auto out_size = out.size(i);
    if (in_size > std::numeric_limits<int32_t>::max() ||
        out_size > std::numeric_limits<int32_t>::max()) {
      ET_LOG(
          Error,
          "transpose_out: dimension size exceeds int32_t range (input=%lld, output=%lld)",
          static_cast<long long>(in_size),
          static_cast<long long>(out_size));
      context.fail(Error::InvalidArgument);
      return out;
    }
    input_dims_arr[i] = static_cast<int32_t>(in_size);
    output_dims_arr[i] = static_cast<int32_t>(out_size);
  }

  cmsis_nn_dims input_dims = {
      input_dims_arr[0],
      input_dims_arr[1],
      input_dims_arr[2],
      input_dims_arr[3]};
  cmsis_nn_dims output_dims = {
      output_dims_arr[0],
      output_dims_arr[1],
      output_dims_arr[2],
      output_dims_arr[3]};

  std::array<uint32_t, kMaxSupportedDims> perm_buffer{0, 1, 2, 3};
  for (size_t i = 0; i < rank; ++i) {
    perm_buffer[i] = static_cast<uint32_t>(perm[i]);
  }

  const cmsis_nn_transpose_params transpose_params{
      static_cast<int32_t>(rank), perm_buffer.data()};

  const int8_t* input_data = input.const_data_ptr<int8_t>();
  int8_t* output_data = out.mutable_data_ptr<int8_t>();

  const arm_cmsis_nn_status status = arm_transpose_s8(
      input_data, output_data, &input_dims, &output_dims, &transpose_params);

  if (status != ARM_CMSIS_NN_SUCCESS) {
    ET_LOG(
        Error,
        "transpose_out: arm_transpose_s8 failed with status [%d]",
        static_cast<int>(status));
    context.fail(Error::Internal);
    return out;
  }

  return out;
}

} // namespace native
} // namespace cortex_m
