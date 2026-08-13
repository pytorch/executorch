/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <executorch/runtime/core/error.h>
#include <executorch/runtime/core/result.h>

namespace executorch {
namespace backends {
namespace xnnpack {

// The K/V window to attend over, and how to attend it. The caller shapes the
// pool as [1, H, slots, D] at the call site.
//
// The pool is wider than the live history: `slots` is what is allocated,
// `valid_len` what is live. A consumer that takes its strides from the shape
// and bounds its key extent at `valid_len` reads no further.
struct AttendSpec {
  // The cache decides how its window is attended; a windowed or tree layer
  // needs more than a flag can carry.
  enum class Mask {
    None, // one query over the whole window
    Causal, // multi-token step, right-aligned at the window's tail
    Explicit, // per-query visibility; see `mask`
  };

  // The pool grows by doubling, which reallocates, so both pointers are valid
  // only until this layer's next update_and_fetch.
  const void* k; // this layer's K pool, dense [1, H, slots, D]
  const void* v; // this layer's V pool, same shape
  int slots; // pool's seq extent -- shapes the tensor
  int valid_len; // live history [0, valid_len) -- bounds the read
  Mask kind;
  // Explicit only, else nullptr. Additive [n_tok, valid_len]: 0 attends, -inf
  // suppresses; float for every storage dtype. Owned by the cache, valid until
  // this layer's next update_and_fetch.
  const float* mask;
};

// Byte-facing side of the off-graph KV cache, kept separate from the neutral
// tensor-free face so one cache can expose both without a diamond.
class XnnCache {
 public:
  virtual ~XnnCache() = default;

  // Append this step's `n_tok` K/V for `layer` at logical `position`, then
  // return the window to attend over. k and v hold the step's new tokens,
  // contiguous [1, H, n_tok, D] in the cache's storage dtype, so each head
  // copies in one run; H and D come from the layer's config.
  virtual runtime::Result<AttendSpec> update_and_fetch(
      int layer,
      int position,
      const void* k,
      const void* v,
      int n_tok) = 0;
};

} // namespace xnnpack
} // namespace backends
} // namespace executorch
