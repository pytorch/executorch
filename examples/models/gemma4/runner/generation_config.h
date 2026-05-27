/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <cstdint>
#include <vector>

namespace executorch::examples::gemma4 {

/**
 * Configuration for text generation.
 *
 * Controls sampling behavior, sequence limits, and stop conditions.
 */
struct GenerationConfig {
  /// Maximum number of new tokens to generate.
  int32_t max_new_tokens = 100;

  /// Sampling temperature. 0 = greedy (argmax), >0 = random sampling.
  float temperature = 0.0f;

  /// Top-p (nucleus) sampling threshold. Only used when temperature > 0.
  float topp = 0.9f;

  /// Stop token IDs. Generation stops when any of these tokens is produced.
  /// Default: EOS (1) and EndOfTurn (106).
  std::vector<int64_t> stop_tokens = {1, 106};
};

} // namespace executorch::examples::gemma4
