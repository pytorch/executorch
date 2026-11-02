// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <executorch/backends/vulkan/test/ops/benchmark_v2/framework/ReferenceRegistry.h>
#include <executorch/backends/vulkan/test/ops/benchmark_v2/operators/convolution/Conv2dReference.h>
#include <executorch/backends/vulkan/test/ops/benchmark_v2/operators/convolution/Conv2dReferenceCommon.h>
#include <executorch/backends/vulkan/test/ops/benchmark_v2/operators/convolution/conv1d/Conv1dReference.h>

#include <cmath>

namespace vkcompute::shader_profiling::reference {

Tensor
conv2d_reference(const Json& spec, const TensorMap& tensors, size_t threads) {
  if (tensor_arg(spec, 0, tensors).sizes.size() == 3) {
    return conv1d_reference(spec, tensors, threads);
  }
  require(
      !spec.at("args").at(6).at("bool").get<bool>(),
      "Conv2d reference does not support transposed convolution");
  const Tensor& input = tensor_arg(spec, 0, tensors);
  const Tensor& weight = tensor_arg(spec, 1, tensors);
  require(weight.sizes.size() == 4, "Conv2d reference requires rank-4 weight");
  const Conv2dParameters p{
      {weight.sizes[2], weight.sizes[3]},
      conv2d_pair_arg(spec, 3),
      conv2d_pair_arg(spec, 4),
      conv2d_pair_arg(spec, 5),
      spec.at("args").at(8).at("int")};
  const Tensor* bias =
      spec.at("args").at(2).is_null() ? nullptr : &tensor_arg(spec, 2, tensors);
  const auto g = conv2d_geometry(input, weight, bias, p);
  require(
      weight.sizes ==
          std::vector<int64_t>(
              {g.out_channels,
               g.in_channels_per_group,
               p.kernel[0],
               p.kernel[1]}),
      "Conv2d weight shape must be [Cout, Cin/groups, Kh, Kw]");
  return convolve<float>(
      input,
      weight,
      p,
      g,
      threads,
      [&](int64_t oc, int64_t ci, int64_t ky, int64_t kx) {
        return ((oc * g.in_channels_per_group + ci) * p.kernel[0] + ky) *
            p.kernel[1] +
            kx;
      },
      [](float sum, double value, double weight_value) {
        return std::fma(float(value), float(weight_value), sum);
      },
      [&](float sum, int64_t oc) {
        return sum + (bias ? float(bias->values[oc]) : 0.0f);
      });
}

REGISTER_REFERENCE_FUNCTIONS {
  REGISTER_REFERENCE_FN(conv2d, conv2d_reference);
  REGISTER_OPERATOR_REFERENCE(aten.convolution.default, conv2d);
}

} // namespace vkcompute::shader_profiling::reference
