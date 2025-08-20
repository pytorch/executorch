/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

// A simple audio struct.

#pragma once
#include <executorch/runtime/platform/compiler.h>
#include <cstdint>
#include <vector>

namespace executorch {
namespace extension {
namespace llm {

struct ET_EXPERIMENTAL Audio {
  // Raw audio data (typically float32 values)
  std::vector<float> data;
  int32_t sample_rate;
  int32_t channels;
  int32_t num_samples;
};

} // namespace llm
} // namespace extension
} // namespace executorch

namespace torch {
namespace executor {
// TODO(T197294990): Remove these deprecated aliases once all users have moved
// to the new `::executorch` namespaces.
using ::executorch::extension::llm::Audio;
} // namespace executor
} // namespace torch