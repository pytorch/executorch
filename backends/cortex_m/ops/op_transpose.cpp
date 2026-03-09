/*
 * Copyright 2025-2026 Arm Limited and/or its affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "cortex_m_ops_common.h"

#include <array>
#include <limits>

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

  // Compute row-major strides for input and output
  std::array<int64_t, kMaxSupportedDims> input_strides{1, 1, 1, 1};
  for (int i = static_cast<int>(kMaxSupportedDims) - 2; i >= 0; --i) {
    input_strides[i] = input_strides[i + 1] * input_dims_arr[i + 1];
  }
  std::array<int64_t, kMaxSupportedDims> output_strides{1, 1, 1, 1};
  for (int i = static_cast<int>(kMaxSupportedDims) - 2; i >= 0; --i) {
    output_strides[i] = output_strides[i + 1] * output_dims_arr[i + 1];
  }

  std::array<uint32_t, kMaxSupportedDims> perm_buffer{0, 1, 2, 3};
  for (size_t i = 0; i < rank; ++i) {
    perm_buffer[i] = static_cast<uint32_t>(perm[i]);
  }

  const int8_t* input_data = input.const_data_ptr<int8_t>();
  int8_t* output_data = out.mutable_data_ptr<int8_t>();

  for (int32_t i0 = 0; i0 < output_dims_arr[0]; ++i0) {
    for (int32_t i1 = 0; i1 < output_dims_arr[1]; ++i1) {
      for (int32_t i2 = 0; i2 < output_dims_arr[2]; ++i2) {
        for (int32_t i3 = 0; i3 < output_dims_arr[3]; ++i3) {
          const std::array<int32_t, kMaxSupportedDims> out_idx{i0, i1, i2, i3};
          std::array<int32_t, kMaxSupportedDims> in_idx{0, 0, 0, 0};
          for (size_t k = 0; k < kMaxSupportedDims; ++k) {
            in_idx[perm_buffer[k]] = out_idx[k];
          }
          const int64_t in_offset = in_idx[0] * input_strides[0] +
                                     in_idx[1] * input_strides[1] +
                                     in_idx[2] * input_strides[2] +
                                     in_idx[3] * input_strides[3];
          const int64_t out_offset = i0 * output_strides[0] +
                                      i1 * output_strides[1] +
                                      i2 * output_strides[2] +
                                      i3 * output_strides[3];
          output_data[out_offset] = input_data[in_offset];
        }
      }
    }
  }

  return out;
}

} // namespace native
} // namespace cortex_m
