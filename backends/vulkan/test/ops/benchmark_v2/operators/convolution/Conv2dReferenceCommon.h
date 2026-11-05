// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include <executorch/backends/vulkan/test/ops/benchmark_v2/framework/Reference.h>

#include <array>

namespace vkcompute::shader_profiling::reference {

struct Conv2dParameters {
  std::array<int64_t, 2> kernel;
  std::array<int64_t, 2> stride;
  std::array<int64_t, 2> padding;
  std::array<int64_t, 2> dilation;
  int64_t groups;
};

struct Conv2dGeometry {
  int64_t batches;
  int64_t channels;
  int64_t height;
  int64_t width;
  int64_t out_channels;
  int64_t in_channels_per_group;
  int64_t out_channels_per_group;
  int64_t row_size;
  int64_t out_height;
  int64_t out_width;
};

std::array<int64_t, 2> conv2d_pair_arg(const Json& spec, size_t index);
Conv2dGeometry conv2d_geometry(
    const Tensor& input,
    const Tensor& weight,
    const Tensor* bias,
    const Conv2dParameters& p);

template <
    typename Accumulator,
    typename WeightIndex,
    typename Accumulate,
    typename Finish>
Tensor convolve(
    const Tensor& input,
    const Tensor& weight,
    const Conv2dParameters& p,
    const Conv2dGeometry& g,
    size_t threads,
    WeightIndex weight_index,
    Accumulate accumulate,
    Finish finish) {
  Tensor output{
      {g.batches, g.out_channels, g.out_height, g.out_width},
      std::vector<double>(
          size_t(g.batches * g.out_channels * g.out_height * g.out_width))};
  const size_t rows = size_t(g.batches * g.out_channels * g.out_height);
  parallel_for(rows, threads, [&](size_t row) {
    const int64_t oy = row % g.out_height;
    const int64_t oc = (row / g.out_height) % g.out_channels;
    const int64_t batch = row / (g.out_height * g.out_channels);
    const int64_t input_channel_start =
        (oc / g.out_channels_per_group) * g.in_channels_per_group;
    for (int64_t ox = 0; ox < g.out_width; ++ox) {
      Accumulator sum = 0;
      for (int64_t ci = 0; ci < g.in_channels_per_group; ++ci) {
        const int64_t input_channel_offset =
            (batch * g.channels + input_channel_start + ci) * g.height *
            g.width;
        for (int64_t ky = 0; ky < p.kernel[0]; ++ky) {
          const int64_t iy =
              oy * p.stride[0] - p.padding[0] + ky * p.dilation[0];
          if (iy < 0 || iy >= g.height)
            continue;
          for (int64_t kx = 0; kx < p.kernel[1]; ++kx) {
            const int64_t ix =
                ox * p.stride[1] - p.padding[1] + kx * p.dilation[1];
            if (ix < 0 || ix >= g.width)
              continue;
            sum = accumulate(
                sum,
                input.values[input_channel_offset + iy * g.width + ix],
                weight.values[weight_index(oc, ci, ky, kx)]);
          }
        }
      }
      output.values[row * g.out_width + ox] = finish(sum, oc);
    }
  });
  return output;
}

} // namespace vkcompute::shader_profiling::reference
