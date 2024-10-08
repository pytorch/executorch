/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/extension/aten_util/make_aten_functor_from_et_functor.h>
#include <executorch/extension/kernel_util/make_boxed_from_unboxed_functor.h>
#include <executorch/extension/llm/custom_ops/op_tile_crop.h>

#include <torch/library.h>

namespace torch {
namespace executor {

namespace native {

Tensor&
tile_crop_out_no_context(const Tensor& input, int64_t tile_size, Tensor& out) {
  exec_aten::RuntimeContext context{};
  return tile_crop_out_impl(context, input, tile_size, out);
}

at::Tensor tile_crop_aten(const at::Tensor& input, int64_t tile_size) {
  // max_num_tiles = 4, num_channels = 3.
  auto output = at::empty({4, 3, tile_size, tile_size});

  WRAP_TO_ATEN(torch::executor::native::tile_crop_out_no_context, 2)
  (input, tile_size, output);
  return output;
}

} // namespace native
} // namespace executor
} // namespace torch

TORCH_LIBRARY(preprocess, m) {
  m.def("tile_crop(Tensor input, int tile_size) -> Tensor");
  m.def(
      "tile_crop.out(Tensor input, int tile_size, *, Tensor(a!) out) -> Tensor(a!)");
}

TORCH_LIBRARY_IMPL(preprocess, CompositeExplicitAutograd, m) {
  m.impl("tile_crop", torch::executor::native::tile_crop_aten);
  m.impl(
      "tile_crop.out",
      WRAP_TO_ATEN(torch::executor::native::tile_crop_out_no_context, 2));
}
