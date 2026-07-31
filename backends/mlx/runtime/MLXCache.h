/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <optional>

#include "MLXExecutor.h" // Tensor, StreamOrDevice

namespace executorch {
namespace backends {
namespace mlx {

// The K/V window to attend over + how to mask it; the cache owns the semantic.
// `kind` mirrors MLX SDPA's mask forms (no mask / "causal" / explicit tensor).
// Will be added: a `window` on Causal (sliding-window -- the
// mask_mod axis) and a `softcap`/bias field (ALiBi, softcap -- the score_mod
// axis).
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

  // Write this step's K/V for `layer` at `position` (the run's logical start);
  // return the window + mask kind. k/v are BHSD. `position` is a host int --
  // the caller reads it off the graph so the cache stays pure graph + integer
  // bookkeeping. The cache owns the mask: a multi-token chain is Causal, a
  // single decode token is None.
  virtual AttendSpec update_and_fetch(
      int layer,
      int position,
      const Tensor& k,
      const Tensor& v,
      StreamOrDevice s) = 0;
};

} // namespace mlx
} // namespace backends
} // namespace executorch
