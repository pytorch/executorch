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

  const int32_t out_n = input_dims.n + cmsis_pre_pad.n + cmsis_post_pad.n;
  const int32_t out_h = input_dims.h + cmsis_pre_pad.h + cmsis_post_pad.h;
  const int32_t out_w = input_dims.w + cmsis_pre_pad.w + cmsis_post_pad.w;
  const int32_t out_c = input_dims.c + cmsis_pre_pad.c + cmsis_post_pad.c;

  const int8_t pad_byte = static_cast<int8_t>(pad_value);
  for (int32_t n = 0; n < out_n; ++n) {
    for (int32_t h = 0; h < out_h; ++h) {
      for (int32_t w = 0; w < out_w; ++w) {
        for (int32_t c = 0; c < out_c; ++c) {
          const int32_t out_idx = ((n * out_h + h) * out_w + w) * out_c + c;
          const int32_t in_n = n - cmsis_pre_pad.n;
          const int32_t in_h = h - cmsis_pre_pad.h;
          const int32_t in_w = w - cmsis_pre_pad.w;
          const int32_t in_c = c - cmsis_pre_pad.c;
          if (in_n >= 0 && in_n < input_dims.n && in_h >= 0 &&
              in_h < input_dims.h && in_w >= 0 && in_w < input_dims.w &&
              in_c >= 0 && in_c < input_dims.c) {
            const int32_t in_idx =
                ((in_n * input_dims.h + in_h) * input_dims.w + in_w) *
                    input_dims.c +
                in_c;
            output_data[out_idx] = input_data[in_idx];
          } else {
            output_data[out_idx] = pad_byte;
          }
        }
      }
    }
  }

  return out;
}

} // namespace native
} // namespace cortex_m
