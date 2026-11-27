// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <executorch/backends/vulkan/test/ops/benchmark_v2/framework/ReferenceRegistry.h>
#include <executorch/backends/vulkan/test/ops/benchmark_v2/operators/embedding/qw/q4gsw/EmbeddingReference.h>
namespace vkcompute::shader_profiling::reference {
Tensor embedding_q4gsw_reference(
    const Json& spec,
    const TensorMap& tensors,
    size_t threads) {
  const Tensor& weight = tensor_arg(spec, 0, tensors);
  const Tensor& scales = tensor_arg(spec, 1, tensors);
  const Tensor& indices = tensor_arg(spec, 3, tensors);
  const int64_t group = spec.at("args").at(2).at("int");
  const bool linear = spec.at("args").at(4).at("bool");
  require(
      weight.sizes.size() == 2 && group > 0,
      "Embedding needs rank2 weight and positive group size");
  const size_t dim = weight.sizes[1] * 2, vocab = weight.sizes[0],
               groups = dim / group;
  require(
      dim % group == 0 && scales.values.size() == vocab * groups,
      "Embedding scale dimensions differ");
  Tensor out{indices.sizes, std::vector<double>(indices.values.size() * dim)};
  out.sizes.push_back(dim);
  parallel_for(indices.values.size(), threads, [&](size_t i) {
    const int64_t row = indices.values[i];
    require(row >= 0 && row < int64_t(vocab), "Embedding index out of bounds");
    for (size_t d = 0; d < dim; ++d) {
      const uint8_t byte = weight.values[row * (dim / 2) + d / 2];
      const size_t shift = linear ? 4 * (d % 2) : 4 * (1 - d % 2);
      out.values[i * dim + d] = float(int((byte >> shift) & 15) - 8) *
          float(scales.values[row * groups + d / group]);
    }
  });
  return out;
}
REGISTER_REFERENCE_FUNCTIONS {
  REGISTER_REFERENCE_FN(embedding_q4gsw, embedding_q4gsw_reference);
  REGISTER_OPERATOR_REFERENCE(et_vk.embedding_q4gsw.default, embedding_q4gsw);
}
} // namespace vkcompute::shader_profiling::reference
