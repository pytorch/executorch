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
#include <vector>

namespace torch::executor {

struct Image {
  // Assuming NCHW format
  std::vector<uint8_t> data;
  int32_t width;
  int32_t height;
  int32_t channels;
};

} // namespace torch::executor
