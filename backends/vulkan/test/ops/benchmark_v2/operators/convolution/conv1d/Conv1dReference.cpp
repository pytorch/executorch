// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <executorch/backends/vulkan/test/ops/benchmark_v2/operators/convolution/conv1d/Conv1dReference.h>
#include <algorithm>
#include <cmath>
namespace vkcompute::shader_profiling::reference {
Tensor
conv1d_reference(const Json& spec, const TensorMap& tensors, size_t threads) {
  const Tensor& input = tensor_arg(spec, 0, tensors);
  const Tensor& weight = tensor_arg(spec, 1, tensors);
  require(
      input.sizes.size() == 3 && weight.sizes.size() == 3,
      "Conv1d requires rank3 input and weight");
  const int64_t n = input.sizes[0], ci = input.sizes[1], l = input.sizes[2];
  const int64_t co = weight.sizes[0], cg = weight.sizes[1], k = weight.sizes[2];
  const int64_t groups = spec.at("args").at(8).at("int");
  const int64_t stride = spec.at("args").at(3).at("int_list").at(0);
  const int64_t padding = spec.at("args").at(4).at("int_list").at(0);
  const int64_t dilation = spec.at("args").at(5).at("int_list").at(0);
  require(
      groups > 0 && ci == cg * groups && co % groups == 0 && stride > 0 &&
          dilation > 0,
      "Invalid Conv1d dimensions or groups");
  require(
      !spec.at("args").at(6).at("bool").get<bool>(),
      "Transposed Conv1d reference is unsupported");
  const int64_t lo = (l + 2 * padding - dilation * (k - 1) - 1) / stride + 1;
  require(lo > 0, "Conv1d output length must be positive");
  const Tensor* bias =
      spec.at("args").at(2).is_null() ? nullptr : &tensor_arg(spec, 2, tensors);
  Tensor out{{n, co, lo}, std::vector<double>(n * co * lo)};
  parallel_for(n * co, threads, [&](size_t slice) {
    const int64_t batch = slice / co, oc = slice % co,
                  group = oc / (co / groups);
    for (int64_t x = 0; x < lo; ++x) {
      float sum = 0;
      for (int64_t c = 0; c < cg; ++c)
        for (int64_t j = 0; j < k; ++j) {
          const int64_t ix = x * stride - padding + j * dilation;
          if (ix >= 0 && ix < l)
            sum = std::fma(
                float(input.values[(batch * ci + group * cg + c) * l + ix]),
                float(weight.values[(oc * cg + c) * k + j]),
                sum);
        }
      out.values[slice * lo + x] =
          sum + (bias ? float(bias->values[oc]) : 0.0f);
    }
  });
  return out;
}
} // namespace vkcompute::shader_profiling::reference
