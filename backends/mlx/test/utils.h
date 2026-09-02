/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <executorch/extension/llm/cache/cache.h>
#include <mlx/mlx.h>

#include <optional>

namespace executorch {
namespace backends {
namespace mlx {

// Max absolute difference within tolerance. Computed in float32: item<float>()
// reads sizeof(float) bytes, so calling it on an fp16 scalar misreads the
// buffer.
inline bool
allclose(const ::mlx::core::array& a, const ::mlx::core::array& b, float atol) {
  using namespace ::mlx::core;
  array m = max(abs(subtract(astype(a, float32), astype(b, float32))));
  eval(m);
  return m.item<float>() <= atol;
}

// A single unwindowed layer, so a step's slots are one run over the capacity.
inline ::executorch::extension::llm::cache::CacheConfig flat_config(
    int capacity,
    int n_layers,
    int n_kv_heads,
    int head_dim,
    int kv_dtype,
    std::optional<int> initial_capacity = std::nullopt) {
  namespace cache = ::executorch::extension::llm::cache;
  cache::CacheConfig cfg;
  cfg.capacity = capacity;
  cfg.n_layers = n_layers;
  cfg.layers = {cache::LayerConfig{
      cache::LayerPolicy{cache::LayerPolicy::Kind::Flat, 0},
      n_kv_heads,
      head_dim}};
  cfg.kv_dtype = kv_dtype;
  if (initial_capacity) {
    cfg.initial_capacity = *initial_capacity;
  }
  return cfg;
}

// One ring layer of `window` cells; max_write bounds a multi-token step.
inline ::executorch::extension::llm::cache::CacheConfig ring_config(
    int capacity,
    int window,
    int max_write,
    int n_kv_heads,
    int head_dim,
    int kv_dtype) {
  namespace cache = ::executorch::extension::llm::cache;
  cache::CacheConfig cfg =
      flat_config(capacity, /*n_layers=*/1, n_kv_heads, head_dim, kv_dtype);
  cfg.layers[0].policy =
      cache::LayerPolicy{cache::LayerPolicy::Kind::Ring, window};
  cfg.max_write = max_write;
  return cfg;
}

} // namespace mlx
} // namespace backends
} // namespace executorch
