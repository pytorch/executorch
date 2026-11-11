// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

// What a caller needs to build the off-graph KV cache a method's
// kvcache::update_and_attend calls use, read from the method and its package.

#include <cstdint>
#include <limits>
#include <map>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#include <executorch/backends/native/runtime/Method.h>
#include <executorch/backends/native/runtime/deserialize/Package.h>
#include <executorch/extension/llm/cache/cache.h>
#include <executorch/extension/llm/runner/constants.h>

namespace ptn {

// The cache geometry a method's update_and_attend calls need: one layer per
// cache id, shaped like its K, with the sliding window (or none) the package's
// metadata gives it. nullopt if the method makes no such calls. Throws
// std::runtime_error when the calls or the metadata disagree, or the metadata
// has no windows.
inline std::optional<executorch::extension::llm::cache::CacheGeometry>
offgraph_cache_geometry(const Method& method, const Package& package) {
  namespace cache = executorch::extension::llm::cache;
  namespace llm = executorch::extension::llm;
  constexpr std::string_view kUpdateAndAttend =
      "torch.ops.kvcache.update_and_attend.default";
  std::map<int64_t, cache::LayerGeometry> layers;
  for (const Node& node : method.graph.nodes) {
    if (node.target != kUpdateAndAttend) {
      continue;
    }
    const auto arg = [&](std::string_view name,
                         size_t positional) -> const Argument& {
      for (const NamedArgument& input : node.inputs) {
        if (input.name == name) {
          return input.arg;
        }
      }
      if (positional >= node.inputs.size()) {
        throw std::runtime_error(
            "update_and_attend is missing argument " + std::string(name));
      }
      return node.inputs[positional].arg;
    };
    const Argument& layer_arg = arg("layer_id", 4);
    const Argument& k_arg = arg("k", 1);
    if (layer_arg.kind() != ArgKind::Int || valid(layer_arg.as_int().id) ||
        k_arg.kind() != ArgKind::Tensor) {
      throw std::runtime_error(
          "update_and_attend needs a constant layer_id and a K tensor");
    }
    const std::vector<int64_t>& k_sizes =
        method.graph.value(k_arg.as_tensor().id).tensor_meta().sizes;
    if (k_sizes.size() != 4) {
      throw std::runtime_error("update_and_attend K must be [B, H, T, D]");
    }
    const cache::LayerGeometry layer{
        .n_kv_heads = static_cast<int>(k_sizes[1]),
        .head_dim = static_cast<int>(k_sizes[3])};
    const auto [it, inserted] =
        layers.try_emplace(layer_arg.as_int().value, layer);
    if (!inserted &&
        (it->second.n_kv_heads != layer.n_kv_heads ||
         it->second.head_dim != layer.head_dim)) {
      throw std::runtime_error(
          "update_and_attend calls disagree on cache layer " +
          std::to_string(it->first) + "'s shape");
    }
  }
  if (layers.empty()) {
    return std::nullopt;
  }
  cache::CacheGeometry geometry;
  for (const auto& [id, layer] : layers) {
    if (id != static_cast<int64_t>(geometry.layers.size())) {
      throw std::runtime_error("update_and_attend cache ids must be 0..n-1");
    }
    geometry.layers.push_back(layer);
  }

  const auto per_layer = [&](const char* name) {
    std::optional<std::vector<int64_t>> values = package.metadata_ints(name);
    if (values && values->size() != geometry.layers.size()) {
      throw std::runtime_error(
          std::string("package metadata ") + name + " has " +
          std::to_string(values->size()) + " entries for " +
          std::to_string(geometry.layers.size()) + " cache layers");
    }
    return values;
  };
  const auto kv_heads = per_layer(llm::kKVHeads);
  const auto head_dims = per_layer(llm::kHeadDims);
  const auto windows = per_layer(llm::kWindows);
  // Without them a sliding layer would silently attend its whole history.
  if (!windows) {
    throw std::runtime_error(
        std::string("package metadata has no ") + llm::kWindows +
        "; export the package with export_hf_llm.py --offgraph_cache");
  }
  for (size_t i = 0; i < geometry.layers.size(); ++i) {
    cache::LayerGeometry& layer = geometry.layers[i];
    if ((kv_heads && (*kv_heads)[i] != layer.n_kv_heads) ||
        (head_dims && (*head_dims)[i] != layer.head_dim)) {
      throw std::runtime_error(
          "package metadata disagrees with the graph on cache layer " +
          std::to_string(i) + "'s shape");
    }
    const int64_t window = (*windows)[i];
    if (window < 0 || window > std::numeric_limits<int>::max()) {
      throw std::runtime_error(
          "package metadata gives cache layer " + std::to_string(i) +
          " an invalid window");
    }
    if (window > 0) {
      layer.policy = {cache::LayerPolicy::Kind::Ring, static_cast<int>(window)};
    }
  }
  return geometry;
}

// The most tokens one call of `method` takes, from its [1, T] token input's
// serialized upper bound: the max_write that sizes a ring layer.
inline int max_step_tokens(const Method& method) {
  const std::vector<int64_t>& bound =
      method.graph.value(method.graph.input_ids.at(0)).tensor_meta().sizes;
  if (bound.empty() || bound.back() <= 0 ||
      bound.back() > std::numeric_limits<int>::max()) {
    throw std::runtime_error("token input must have shape [1, sequence]");
  }
  return static_cast<int>(bound.back());
}

} // namespace ptn
