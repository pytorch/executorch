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

class BasicSampler {
 public:
  int64_t sample(const std::vector<float>& logits) {
    // Find the token with the highest log probability.
    int64_t max_index =
        std::max_element(logits.begin(), logits.end()) - logits.begin();
    return max_index;
  }
};
