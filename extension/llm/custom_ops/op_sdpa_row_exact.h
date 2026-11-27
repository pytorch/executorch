/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

namespace torch {
namespace executor {
namespace native {

/**
 * Turns on row-exact mode for `custom_sdpa` and `custom_quantized_sdpa` on the
 * calling thread while the scope is alive. In row-exact mode a causal call
 * without an attention mask computes each query row as a one-row call at that
 * row's position, so every row is bit for bit what a one-row call over the
 * same memory returns (on x86-64 Linux the float path goes through MKL, whose
 * results also depend on buffer alignment). By default all rows of a call are
 * computed together, which is faster for long prefills, but a row's result can
 * then depend on how many rows share the call. Scopes nest; the previous mode
 * is restored on destruction.
 */
class SdpaRowExactScope {
 public:
  explicit SdpaRowExactScope(bool enabled = true);
  ~SdpaRowExactScope();
  SdpaRowExactScope(const SdpaRowExactScope&) = delete;
  SdpaRowExactScope& operator=(const SdpaRowExactScope&) = delete;

 private:
  bool previous_;
};

/// Whether row-exact mode is on for the calling thread.
bool sdpa_row_exact_enabled();

} // namespace native
} // namespace executor
} // namespace torch
