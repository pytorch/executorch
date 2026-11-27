/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <executorch/extension/llm/batching/types.h>
#include <executorch/extension/llm/cache/checkpoint.h>

namespace executorch::extension::llm::batching {

using cache::CheckpointError;
using cache::CheckpointResult;

// Complete committed continuation state. Predictions, sampling state, and
// request policy are intentionally excluded, as they are for Session::clone.
struct ET_EXPERIMENTAL Checkpoint {
  std::string format;
  Position position = 0;
  std::vector<cache::SequenceCheckpoint> caches;
  std::vector<uint8_t> hidden;
  int hidden_row_bytes = 0;
};

} // namespace executorch::extension::llm::batching
