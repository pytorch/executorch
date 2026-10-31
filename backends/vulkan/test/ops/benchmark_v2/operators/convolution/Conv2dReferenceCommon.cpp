// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <executorch/backends/vulkan/test/ops/benchmark_v2/operators/convolution/Conv2dReferenceCommon.h>

namespace vkcompute::shader_profiling::reference {

std::array<int64_t, 2> conv2d_pair_arg(const Json& spec, size_t index) {
  const auto values =
      spec.at("args").at(index).at("int_list").get<std::vector<int64_t>>();
  require(values.size() == 2, "Conv2d spatial parameters require two values");
  return {values[0], values[1]};
}

Conv2dGeometry conv2d_geometry(
    const Tensor& input,
    const Tensor& weight,
    const Tensor* bias,
    const Conv2dParameters& p) {
  require(
      input.sizes.size() == 4 && !weight.sizes.empty(),
      "Conv2d requires NCHW input");
  const int64_t channels = input.sizes[1];
  const int64_t out_channels = weight.sizes[0];
  require(
      p.groups > 0 && channels % p.groups == 0 && out_channels % p.groups == 0,
      "Conv2d channels must be divisible by groups");
  for (size_t dim = 0; dim < 2; ++dim) {
    require(
        p.kernel[dim] > 0 && p.stride[dim] > 0 && p.dilation[dim] > 0 &&
            p.padding[dim] >= 0,
        "Conv2d requires positive kernel/stride/dilation and nonnegative padding");
  }
  require(
      !bias || bias->values.size() == size_t(out_channels),
      "Conv2d bias size must equal output channels");
  const int64_t hn =
      input.sizes[2] + 2 * p.padding[0] - p.dilation[0] * (p.kernel[0] - 1) - 1;
  const int64_t wn =
      input.sizes[3] + 2 * p.padding[1] - p.dilation[1] * (p.kernel[1] - 1) - 1;
  require(hn >= 0 && wn >= 0, "Conv2d kernel exceeds padded input");
  return {
      input.sizes[0],
      channels,
      input.sizes[2],
      input.sizes[3],
      out_channels,
      channels / p.groups,
      out_channels / p.groups,
      channels / p.groups * p.kernel[0] * p.kernel[1],
      hn / p.stride[0] + 1,
      wn / p.stride[1] + 1};
}

} // namespace vkcompute::shader_profiling::reference
