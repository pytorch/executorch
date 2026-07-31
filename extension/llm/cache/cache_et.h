/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

// ExecuTorch adapter for the neutral cache core. The core (cache.h /
// sequence_cache.h) is ET-independent and reports failures as
// bool/std::optional so it is usable outside an ET runner. These thin inline
// adapters map those results to ExecuTorch Error/Result (logging on failure)
// for ET consumers -- the runner and the delegate byte layer. (The registry is
// delegate-specific and already returns Result directly, so it needs no
// adapter.)

#include <optional>

#include <executorch/extension/llm/cache/cache.h>
#include <executorch/runtime/core/error.h>
#include <executorch/runtime/core/result.h>

namespace executorch {
namespace extension {
namespace llm {
namespace cache {
namespace et {

using ::executorch::runtime::Error;
using ::executorch::runtime::Result;

// Plan a layer's step, or OutOfResources if it would exceed capacity (or the
// layer is out of range).
inline Result<SeqStepPlan>
plan(const SequencePlanner& planner, int layer, int position, int T) {
  std::optional<SeqStepPlan> p = planner.plan(layer, position, T);
  ET_CHECK_OR_RETURN_ERROR(
      p.has_value(),
      OutOfResources,
      "cache: plan(layer=%d, position=%d, T=%d) exceeds capacity or bad layer",
      layer,
      position,
      T);
  return *p;
}

// Truncate the history, or InvalidArgument if new_len would grow it (or is
// older than an evicting layer retains).
inline Error rewind(SequenceControl& control, int new_len) {
  ET_CHECK_OR_RETURN_ERROR(
      control.rewind(new_len),
      InvalidArgument,
      "rewind: cannot grow to %d",
      new_len);
  return Error::Ok;
}

} // namespace et
} // namespace cache
} // namespace llm
} // namespace extension
} // namespace executorch
