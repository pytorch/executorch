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

Tensor& tile_crop_out_impl(
    RuntimeContext& ctx,
    const Tensor& input, // NOLINT
    const int64_t tile_size, // NOLINT
    Tensor& out) {
  (void)ctx;
  return out;
}

} // namespace native
} // namespace executor
} // namespace torch

EXECUTORCH_LIBRARY(
    preprocess,
    "tile_crop.out",
    torch::executor::native::tile_crop_out_impl);
