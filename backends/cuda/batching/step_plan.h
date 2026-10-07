/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <algorithm>
#include <vector>

#include <executorch/runtime/platform/assert.h>

namespace executorch::backends::cuda::batching {

// Which exported method runs a slice of a batch.
enum class StepMethod {
  // Static, one token: the method a CUDA graph is captured for.
  Decode,
  // Dynamic over [min_prefill_tokens, max_step_tokens].
  Prefill,
};

struct StepSlice {
  int offset;
  int length;
  StepMethod method;
};

// Cuts a batch of `total` packed tokens into forwards the exported methods can
// run, in order, so a slice attends every cell its predecessors wrote.
//
// Slices take up to `max_step_tokens` each. A one-token slice runs Decode. A
// slice shorter than `min_prefill_tokens` -- the lower bound the prefill
// method was exported with -- runs as that many Decode forwards; anything
// else runs Prefill. `max_step_tokens` must be positive, or no slice would
// advance through the batch.
inline std::vector<StepSlice>
plan_slices(int total, int max_step_tokens, int min_prefill_tokens) {
  ET_CHECK_MSG(
      max_step_tokens > 0,
      "plan_slices: max_step_tokens must be positive, got %d",
      max_step_tokens);
  std::vector<StepSlice> slices;
  for (int offset = 0; offset < total;) {
    const int length = std::min(max_step_tokens, total - offset);
    if (length >= min_prefill_tokens && length > 1) {
      slices.push_back({offset, length, StepMethod::Prefill});
    } else {
      for (int i = 0; i < length; ++i) {
        slices.push_back({offset + i, 1, StepMethod::Decode});
      }
    }
    offset += length;
  }
  return slices;
}

} // namespace executorch::backends::cuda::batching
