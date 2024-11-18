/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/extension/kernel_util/make_boxed_from_unboxed_functor.h>
#include <executorch/extension/llm/custom_ops/op_tile_crop.h>
#include <executorch/runtime/kernel/kernel_includes.h>

namespace torch {
namespace executor {
namespace native {
namespace {

bool check_tile_crop_out_args(
    const Tensor& in,
    int64_t tile_size,
    Tensor& out) {
  ET_LOG_AND_RETURN_IF_FALSE(tensors_have_same_dtype(in, out));
  ET_LOG_AND_RETURN_IF_FALSE(tensor_is_rank(in, 3));
  ET_LOG_AND_RETURN_IF_FALSE(tensor_is_rank(out, 4));
  ET_LOG_AND_RETURN_IF_FALSE(tile_size > 0);
  ET_LOG_AND_RETURN_IF_FALSE(in.size(in.dim() - 1) % tile_size == 0);
  ET_LOG_AND_RETURN_IF_FALSE(in.size(in.dim() - 2) % tile_size == 0);
  return true;
}

void get_tile_crop_out_target_size(
    const Tensor& in,
    int64_t tile_size,
    exec_aten::SizesType* out_sizes,
    size_t* out_ndim) {
  *out_ndim = in.dim() + 1;

  out_sizes[0] = in.size(1) * in.size(2) / (tile_size * tile_size);
  out_sizes[1] = in.size(0);
  out_sizes[2] = tile_size;
  out_sizes[3] = tile_size;
}

template <typename CTYPE>
void tile_crop_impl(const Tensor& in, int64_t tile_size, Tensor& out) {
  const CTYPE* const in_data = in.const_data_ptr<CTYPE>();
  CTYPE* const out_data = out.mutable_data_ptr<CTYPE>();

  const auto channels = in.size(0);
  const auto height = in.size(1);
  const auto width = in.size(2);

  const auto HdivS = height / tile_size;
  const auto WdivS = width / tile_size;

  size_t out_ix = 0;
  for (size_t bH = 0; bH < HdivS; bH++) {
    for (size_t bW = 0; bW < WdivS; bW++) {
      for (size_t c = 0; c < channels; c++) {
        for (size_t h = 0; h < tile_size; h++) {
          for (size_t w = 0; w < tile_size; w++) {
            size_t in_h = bH * tile_size + h;
            size_t in_w = bW * tile_size + w;
            size_t in_ix = c * height * width + in_h * width + in_w;

            out_data[out_ix++] = in_data[in_ix];
          }
        }
      }
    }
  }
}

} // namespace

Tensor& tile_crop_out_impl(
    KernelRuntimeContext& ctx,
    const Tensor& input, // NOLINT
    const int64_t tile_size, // NOLINT
    Tensor& out) {
  ET_KERNEL_CHECK(
      ctx,
      check_tile_crop_out_args(input, tile_size, out),
      InvalidArgument,
      out);

  // @lint-ignore CLANGTIDY facebook-hte-CArray
  Tensor::SizesType expected_out_size[kTensorDimensionLimit];
  size_t expected_out_dim = 0;
  get_tile_crop_out_target_size(
      input, tile_size, expected_out_size, &expected_out_dim);

  ET_KERNEL_CHECK(
      ctx,
      resize_tensor(out, {expected_out_size, expected_out_dim}) == Error::Ok,
      InvalidArgument,
      out);

  constexpr auto name = "tile_crop.out";

  ET_SWITCH_ALL_TYPES(out.scalar_type(), ctx, name, CTYPE, [&]() {
    tile_crop_impl<CTYPE>(input, tile_size, out);
  });

  return out;
}

} // namespace native
} // namespace executor
} // namespace torch

EXECUTORCH_LIBRARY(
    preprocess,
    "tile_crop.out",
    torch::executor::native::tile_crop_out_impl);
