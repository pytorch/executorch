// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <executorch/backends/vulkan/test/ops/benchmark_v2/framework/ReferenceRegistry.h>
#include <executorch/backends/vulkan/test/ops/benchmark_v2/operators/attention/sdpa/SdpaReference.h>
#include <algorithm>
#include <cmath>
#include <limits>
namespace vkcompute::shader_profiling::reference {
Tensor
sdpa_reference(const Json& spec, const TensorMap& tensors, size_t threads) {
  const Tensor& q = tensor_arg(spec, 0, tensors);
  const bool fused = spec.at("operator") == "sdpa_with_kv_cache.default";
  const Tensor& k = tensor_arg(spec, fused ? 3 : 1, tensors);
  const Tensor& v = tensor_arg(spec, fused ? 4 : 2, tensors);
  const int64_t pos = spec.at("args").at(fused ? 5 : 3).at("int");
  require(
      q.sizes.size() == 4 && k.sizes.size() == 4 && v.sizes == k.sizes &&
          q.sizes[0] == 1,
      "SDPA expects batch1 rank4 tensors");
  const size_t s = q.sizes[1], h = q.sizes[2], d = q.sizes[3], hk = k.sizes[2],
               capacity = k.sizes[1];
  require(
      h % hk == 0 && k.sizes[3] == int64_t(d) && pos >= 0 &&
          size_t(pos) + s <= capacity,
      "Invalid SDPA head dimensions or cache position");
  const Json& scale_arg = spec.at("args").at(fused ? 10 : 7);
  const float scale = scale_arg.is_null() ? 1 / std::sqrt(float(d))
                                          : scale_arg.at("float").get<float>();
  const Tensor* pk = fused ? &tensor_arg(spec, 1, tensors) : nullptr;
  const Tensor* pv = fused ? &tensor_arg(spec, 2, tensors) : nullptr;
  Tensor out{q.sizes, std::vector<double>(q.values.size())};
  parallel_for(s * h, threads, [&](size_t job) {
    const size_t token = job / h, head = job % h, kvhead = head / (h / hk),
                 context = pos + token + 1;
    std::vector<float> probs(context);
    float max = -std::numeric_limits<float>::infinity();
    for (size_t c = 0; c < context; ++c) {
      float dot = 0;
      const Tensor& cache = pk && c >= size_t(pos) ? *pk : k;
      const size_t ci = pk && c >= size_t(pos) ? c - pos : c;
      for (size_t j = 0; j < d; ++j)
        dot = std::fma(
            float(q.values[job * d + j]),
            float(cache.values[(ci * hk + kvhead) * d + j]),
            dot);
      probs[c] = dot * scale;
      max = std::max(max, probs[c]);
    }
    float denom = 0;
    for (float& p : probs) {
      p = std::exp(p - max);
      denom += p;
    }
    for (size_t j = 0; j < d; ++j) {
      float value = 0;
      for (size_t c = 0; c < context; ++c) {
        const Tensor& cache = pv && c >= size_t(pos) ? *pv : v;
        const size_t ci = pv && c >= size_t(pos) ? c - pos : c;
        value +=
            probs[c] / denom * float(cache.values[(ci * hk + kvhead) * d + j]);
      }
      out.values[job * d + j] = value;
    }
  });
  return out;
}
static ReferenceOutputs sdpa_reference_outputs(
    const Json& spec,
    const TensorMap& tensors,
    size_t threads) {
  Tensor out = sdpa_reference(spec, tensors, threads);
  ReferenceOutputs result{
      {spec.at("args").back().at("tensor").get<std::string>(),
       std::move(out.values)}};
  if (spec.at("operator") == "sdpa_with_kv_cache.default") {
    const size_t pos = spec.at("args").at(5).at("int");
    for (const auto& pair : {std::pair<size_t, size_t>{3, 1}, {4, 2}}) {
      const Tensor& cache = tensor_arg(spec, pair.first, tensors);
      const Tensor& projected = tensor_arg(spec, pair.second, tensors);
      auto values = cache.values;
      const size_t row_size = cache.sizes[2] * cache.sizes[3];
      require(
          pos * row_size + projected.values.size() <= values.size(),
          "SDPA projected cache update exceeds capacity");
      std::copy(
          projected.values.begin(),
          projected.values.end(),
          values.begin() + pos * row_size);
      result[spec.at("args").at(pair.first).at("tensor").get<std::string>()] =
          std::move(values);
    }
  }
  return result;
}
REGISTER_REFERENCE_FUNCTIONS {
  REGISTER_REFERENCE_OUTPUTS_FN(sdpa, sdpa_reference_outputs);
  REGISTER_OPERATOR_REFERENCE(llama.custom_sdpa.default, sdpa);
  REGISTER_OPERATOR_REFERENCE(sdpa_with_kv_cache.default, sdpa);
}
} // namespace vkcompute::shader_profiling::reference
