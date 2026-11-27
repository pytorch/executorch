/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <algorithm>
#include <cstddef>
#include <tuple>
#include <vector>

#include <executorch/examples/models/llama/runner/static_attention_io_manager.h>

#include <gtest/gtest.h>

namespace example {
namespace {

size_t count_visible(const float* mask, size_t row, size_t row_size) {
  return std::count(mask + row * row_size, mask + (row + 1) * row_size, 0.0f);
}

void check_mask_invariant(size_t input_len, size_t window, size_t prompt_len) {
  constexpr float kZeroVal = 0.0f;
  constexpr float kMaskVal = -1.0f;
  const size_t global_cache_len = prompt_len + input_len;
  StaticAttentionMask<float> local_mask(
      window,
      input_len,
      1,
      kZeroVal,
      kMaskVal,
      StaticAttentionUpdateStyle::SMART_MASK,
      true);
  StaticAttentionMask<float> global_mask(
      global_cache_len, input_len, 1, kZeroVal, kMaskVal);
  local_mask.set_causal_mask();
  global_mask.set_causal_mask();

  for (size_t pos = 0; pos < prompt_len; pos += input_len) {
    const size_t update_len = std::min(input_len, prompt_len - pos);
    local_mask.set_sliding_window_mask(pos);
    global_mask.set_sliding_window_mask(pos);
    for (size_t row = 0; row < update_len; row++) {
      EXPECT_EQ(
          count_visible(local_mask.get(), row, window + input_len),
          std::min(pos + row + 1, window))
          << "input_len=" << input_len << ", window=" << window
          << ", prompt_len=" << prompt_len << ", pos=" << pos
          << ", row=" << row;
      EXPECT_EQ(
          count_visible(global_mask.get(), row, global_cache_len + input_len),
          pos + row + 1)
          << "input_len=" << input_len << ", prompt_len=" << prompt_len
          << ", pos=" << pos << ", row=" << row;
    }
    local_mask.unmask(update_len);
    global_mask.unmask(update_len);
  }
}

TEST(StaticAttentionMaskTest, SlidingWindowInvariant) {
  const std::vector<std::tuple<size_t, size_t, size_t>> cases = {
      {8, 4, 24},
      {6, 12, 24},
      {64, 256, 1242},
      {1024, 256, 1242},
      {1, 256, 600},
  };
  for (const auto& [input_len, window, prompt_len] : cases) {
    check_mask_invariant(input_len, window, prompt_len);
  }
}

TEST(StaticAttentionMaskTest, SlidingWindowMaskRotatesWithCacheRing) {
  constexpr size_t kWindow = 4;
  constexpr size_t kInputLen = 4;
  StaticAttentionMask<float> mask(
      kWindow,
      kInputLen,
      1,
      0.0f,
      -1.0f,
      StaticAttentionUpdateStyle::SMART_MASK,
      true);
  mask.set_causal_mask();
  mask.set_sliding_window_mask(9);

  const std::vector<std::vector<float>> expected_cache = {
      {0.0f, -1.0f, 0.0f, 0.0f},
      {0.0f, -1.0f, -1.0f, 0.0f},
      {0.0f, -1.0f, -1.0f, -1.0f},
      {-1.0f, -1.0f, -1.0f, -1.0f},
  };
  for (size_t row = 0; row < kInputLen; row++) {
    for (size_t col = 0; col < kWindow; col++) {
      EXPECT_EQ(
          mask.get()[row * (kWindow + kInputLen) + col],
          expected_cache[row][col]);
    }
  }
}

} // namespace
} // namespace example
