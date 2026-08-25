/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <cstdint>
#include <optional>
#include <vector>

#include "MLXExecutor.h" // Tensor, StreamOrDevice

namespace executorch {
namespace backends {
namespace mlx {

// The K/V window to attend over + how to mask it; the cache owns the semantic
// and hands over whatever MLX SDPA needs to apply it. `kind` mirrors MLX's mask
// forms: no mask, its fused "causal", or an explicit tensor for anything MLX
// cannot express -- a sliding window, and later tree/speculative patterns.
struct AttendSpec {
  Tensor K;
  Tensor V;
  enum class Mask { None, Causal, Explicit } kind;
  std::optional<Tensor> mask; // Explicit only
};

// Tensor-typed op face of the off-graph KV cache, kept separate from the
// neutral CacheBase (which is tensor-free) so a cache can expose both without a
// diamond. ExecutionState holds one; nothing assigns it yet -- the registry
// that owns the cache and hands this pointer to the executor lands in a
// follow-up, until which exec_update_and_attend is unreachable.
class MLXCache {
 public:
  virtual ~MLXCache() = default;

  // Write this step's K/V for `layer` at `positions`, one host int per query
  // token, and return the window plus the mask kind. k/v are BHSD.
  virtual AttendSpec update_and_fetch(
      int layer,
      const std::vector<int32_t>& positions,
      const Tensor& k,
      const Tensor& v,
      StreamOrDevice s) = 0;
};

} // namespace mlx
} // namespace backends
} // namespace executorch
