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
namespace {
template <typename EType, typename AType>
auto to_et_arg(AType&& value) {
  return executorch::extension::internal::type_convert<AType, EType>(
      std::forward<AType>(value));
}

at::Tensor& copy_et_result_to_out(Tensor& et_result, at::Tensor& out) {
  auto converted_result =
      executorch::extension::internal::type_convert<Tensor&, at::Tensor>(
          et_result)
          .call();
  at::native::resize_output(out, converted_result.sizes());
  out.copy_(converted_result);
  return out;
}
} // namespace

Tensor&
tile_crop_out_no_context(const Tensor& input, int64_t tile_size, Tensor& out);

at::Tensor&
tile_crop_out_aten(const at::Tensor& input, int64_t tile_size, at::Tensor& out);

Tensor&
tile_crop_out_no_context(const Tensor& input, int64_t tile_size, Tensor& out) {
  executorch::aten::RuntimeContext context{};
  return tile_crop_out_impl(context, input, tile_size, out);
}

at::Tensor tile_crop_aten(const at::Tensor& input, int64_t tile_size);

at::Tensor&
tile_crop_out_aten(const at::Tensor& input, int64_t tile_size, at::Tensor& out) {
  auto input_et = to_et_arg<Tensor>(input);
  auto out_et = to_et_arg<Tensor&>(out);
  auto& et_result =
      tile_crop_out_no_context(input_et.call(), tile_size, out_et.call());
  return copy_et_result_to_out(et_result, out);
}

at::Tensor tile_crop_aten(const at::Tensor& input, int64_t tile_size) {
  // max_num_tiles = 4, num_channels = 3.
  auto output = at::empty({4, 3, tile_size, tile_size});
  tile_crop_out_aten(input, tile_size, output);
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
      torch::executor::native::tile_crop_out_aten);
}
