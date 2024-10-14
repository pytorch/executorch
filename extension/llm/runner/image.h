/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

// A simple image struct.

#pragma once
#include <cstdint>
// patternlint-disable-next-line executorch-cpp-nostdinc
#include <executorch/runtime/platform/compiler.h>
#include <vector>

namespace executorch {
namespace extension {
namespace llm {

struct ET_EXPERIMENTAL Image {
  // Assuming NCHW format
  std::vector<uint8_t> data;
  int32_t width;
  int32_t height;
  int32_t channels;
};

} // namespace llm
} // namespace extension
} // namespace executorch

namespace torch {
namespace executor {
// TODO(T197294990): Remove these deprecated aliases once all users have moved
// to the new `::executorch` namespaces.
using ::executorch::extension::llm::Image;
} // namespace executor
} // namespace torch
