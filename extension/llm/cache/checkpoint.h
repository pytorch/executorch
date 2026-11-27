/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <algorithm>
#include <cstdint>
#include <limits>
#include <string>
#include <variant>
#include <vector>

#include <executorch/extension/llm/cache/cache.h>

namespace executorch::extension::llm::cache {

struct ET_EXPERIMENTAL CheckpointError {
  std::string message;
};
template <typename T>
using CheckpointResult = std::variant<T, CheckpointError>;

// Owned, contiguous CPU bytes in logical [token, head, channel] order.
// No allocator, device address, padding, or uncommitted rows cross this
// boundary.
struct ET_EXPERIMENTAL LayerCheckpoint {
  int begin = 0;
  int heads = 0;
  int head_dim = 0;
  int window = 0;
  std::vector<uint8_t> key, value;
};

struct ET_EXPERIMENTAL SequenceCheckpoint {
  int position = 0;
  int dtype = 0; // ExecuTorch ScalarType; bytes are little endian.
  std::optional<int> dflash_block_size;
  std::vector<LayerCheckpoint> layers;
};

inline size_t checkpoint_itemsize(int dtype) {
  // Half, Float, BFloat16. Other representations need their own format.
  return dtype == 5 || dtype == 15 ? 2 : dtype == 6 ? 4 : 0;
}

inline bool valid_checkpoint(
    const SequenceCheckpoint& state,
    const CacheGeometry& geometry,
    const CacheConfig& cfg,
    int upto) {
  const auto itemsize = checkpoint_itemsize(state.dtype);
  if (!itemsize || state.dtype != cfg.kv_dtype || upto < 0 ||
      upto > state.position || upto > cfg.capacity || state.position < 0 ||
      state.dflash_block_size != geometry.dflash_block_size ||
      state.layers.size() != geometry.layers.size()) {
    return false;
  }
  for (size_t i = 0; i < state.layers.size(); ++i) {
    const auto& layer = state.layers[i];
    const auto& expected = geometry.layers[i];
    if (layer.heads != expected.n_kv_heads ||
        layer.head_dim != expected.head_dim ||
        layer.window != expected.policy.window || layer.begin < 0 ||
        layer.begin > state.position ||
        (layer.window == 0 && layer.begin != 0)) {
      return false;
    }
    const int required = layer.window
        ? std::max(0, upto - layer.window + (state.dflash_block_size ? 0 : 1))
        : 0;
    if (upto && (layer.begin > required || layer.begin > upto)) {
      return false;
    }
    size_t bytes = itemsize;
    for (const auto n :
         {state.position - layer.begin, layer.heads, layer.head_dim}) {
      if (n < 0 || (n && bytes > std::numeric_limits<size_t>::max() / n)) {
        return false;
      }
      bytes *= n;
    }
    if (layer.key.size() != bytes || layer.value.size() != bytes) {
      return false;
    }
  }
  return true;
}

// Optional cache face. Calls run between forwards on the executor thread.
// Export synchronizes device writes. Import owns all copied bytes before it
// returns and publishes a new sequence only after every layer is restored.
class ET_EXPERIMENTAL CheckpointControl {
 public:
  static constexpr const char* kFaceName = "et.cache.CheckpointControl";
  virtual ~CheckpointControl() = default;
  virtual CheckpointResult<SequenceCheckpoint> export_to_cpu(
      int32_t id,
      int upto) = 0;
  virtual CheckpointResult<int32_t> import_from_cpu(
      const SequenceCheckpoint& state,
      int upto) = 0;
};

} // namespace executorch::extension::llm::cache
