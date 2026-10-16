// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <executorch/backends/native/examples/hf_llm/runtime/LlmRunnerUtils.h>

#include <algorithm>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <vector>

#include <gtest/gtest.h>

namespace ptn {
namespace {

// cppcheck-suppress-begin syntaxError
TEST(LlmRunnerUtilsTest, GreedyArgmaxSelectsLastLogitRow) {
  EXPECT_EQ(
      greedy_argmax_last_row({100.0f, 0.0f, 0.0f, 1.0f, 3.0f, 2.0f}, 3), 1);
  EXPECT_EQ(greedy_argmax_last_row({3.0f, 3.0f}, 2), 0);
}

TEST(LlmRunnerUtilsTest, GreedyArgmaxMatchesMaxElementOnEdgeValues) {
  const float nan = std::numeric_limits<float>::quiet_NaN();
  const float inf = std::numeric_limits<float>::infinity();
  // 33 floats span two full 16-lane blocks and the scalar tail.
  const std::vector<float> filled(33, -1.0f);
  const auto with = [](std::vector<float> row, size_t index, float value) {
    row[index] = value;
    return row;
  };
  const std::vector<std::vector<float>> rows = {
      {nan, 5.0f, 7.0f},
      {1.0f, nan, 7.0f, nan},
      {-0.0f, 0.0f, -1.0f},
      {-inf, -inf},
      {inf, 1.0f, inf},
      with(with(filled, 0, nan), 20, 10.0f),
      with(with(filled, 5, 10.0f), 21, nan),
      with(with(filled, 3, -0.0f), 18, 0.0f),
      with(filled, 32, inf),
      std::vector<float>(33, -inf),
  };
  for (const std::vector<float>& row : rows) {
    EXPECT_EQ(
        greedy_argmax_last_row(row, row.size()),
        static_cast<size_t>(
            std::max_element(row.begin(), row.end()) - row.begin()));
  }
}

TEST(LlmRunnerUtilsTest, GreedyArgmaxFindsFirstMaximumAcrossLanes) {
  std::vector<float> row(37, -1.0f);
  row[19] = 2.0f;
  row[33] = 2.0f;
  EXPECT_EQ(greedy_argmax_last_row(row, row.size()), 19);
  row[36] = 3.0f;
  EXPECT_EQ(greedy_argmax_last_row(row, row.size()), 36);
}

TEST(LlmRunnerUtilsTest, GreedyArgmaxRejectsIncompleteLogitRows) {
  EXPECT_THROW(greedy_argmax_last_row({1.0f, 2.0f}, 3), std::invalid_argument);
}

TEST(LlmRunnerUtilsTest, ParsesAndDeduplicatesTokenIds) {
  EXPECT_EQ(parse_token_ids("128001,128009,128001").size(), 2);
  EXPECT_TRUE(parse_token_ids("128001,128009").contains(128009));
}

TEST(LlmRunnerUtilsTest, RejectsInvalidTokenIds) {
  EXPECT_THROW(parse_token_ids(""), std::invalid_argument);
  EXPECT_THROW(parse_token_ids("128001,nope"), std::invalid_argument);
  EXPECT_THROW(parse_token_ids("-1"), std::invalid_argument);
  EXPECT_THROW(parse_token_ids("128001,+2"), std::invalid_argument);
  EXPECT_THROW(parse_token_ids("128001,"), std::invalid_argument);
  EXPECT_THROW(parse_token_ids(",128001"), std::invalid_argument);
}

TEST(LlmRunnerUtilsTest, SelectsUniqueUserOutput) {
  EXPECT_EQ(resolve_logits_output(3, {false, true, false}), 0);
  EXPECT_EQ(resolve_logits_output(1, {}), 0);
}

TEST(LlmRunnerUtilsTest, RejectsAmbiguousOutputMetadata) {
  EXPECT_THROW(resolve_logits_output(2, {}), std::invalid_argument);
  EXPECT_THROW(resolve_logits_output(2, {false, false}), std::invalid_argument);
  EXPECT_THROW(resolve_logits_output(2, {true, true}), std::invalid_argument);
  EXPECT_THROW(resolve_logits_output(2, {true}), std::invalid_argument);
}

constexpr const char* kUpdateCache = "torch.ops.llama.update_cache.default";
constexpr const char* kIndexPut = "torch.ops.aten.index_put_.default";

TEST(LlmRunnerUtilsTest, KvCacheSequenceLengthReadsEachCacheLayout) {
  EXPECT_EQ(
      kv_cache_sequence_length(kUpdateCache, "cache", {1, 64, 8, 16}), 64);
  EXPECT_EQ(kv_cache_sequence_length(kIndexPut, "self", {1, 8, 32, 16}), 32);
}

TEST(LlmRunnerUtilsTest, KvCacheSequenceLengthIgnoresNonCacheWrites) {
  EXPECT_EQ(kv_cache_sequence_length(kUpdateCache, "value", {1, 64, 8, 16}), 0);
  EXPECT_EQ(kv_cache_sequence_length(kIndexPut, "values", {1, 8, 32, 16}), 0);
  EXPECT_EQ(kv_cache_sequence_length(kIndexPut, "self", {8, 32, 16}), 0);
  EXPECT_EQ(kv_cache_sequence_length(kIndexPut, "self", {1, 8, 0, 16}), 0);
  EXPECT_EQ(kv_cache_sequence_length(kIndexPut, "self", {1, 8, -1, 16}), 0);
  EXPECT_EQ(
      kv_cache_sequence_length(
          "torch.ops.aten.copy_.default", "self", {1, 8, 32, 16}),
      0);
}

TEST(LlmRunnerUtilsTest, GenerationLimitFitsOneTokenPastTheCache) {
  EXPECT_EQ(generation_limit(10, 32, 2048, 64, 128), 32);
  EXPECT_EQ(generation_limit(60, 32, 2048, 64, 128), 5);
  EXPECT_EQ(generation_limit(64, 32, 2048, 64, 128), 1);
  EXPECT_THROW(generation_limit(65, 32, 2048, 64, 128), std::runtime_error);
}

TEST(LlmRunnerUtilsTest, GenerationLimitHonorsMaxSeqLength) {
  EXPECT_EQ(generation_limit(10, 32, 16, 64, 128), 6);
  EXPECT_THROW(generation_limit(16, 32, 16, 64, 128), std::runtime_error);
}

TEST(LlmRunnerUtilsTest, GenerationLimitFallsBackToTokenInputBound) {
  EXPECT_EQ(generation_limit(120, 32, 2048, 0, 128), 9);
  EXPECT_THROW(generation_limit(129, 32, 2048, 0, 128), std::runtime_error);
}

TEST(LlmRunnerUtilsTest, TopLogitIndicesRankNanFirst) {
  const float nan = std::numeric_limits<float>::quiet_NaN();
  std::vector<float> row = {10.0f, 1.0f, nan, 3.0f, 4.0f};
  row.resize(105);
  std::iota(row.begin() + 5, row.end(), 20.0f);
  EXPECT_EQ(
      top_logit_indices(row.data(), row.size(), 5),
      (std::vector<size_t>{2, 104, 103, 102, 101}));
  EXPECT_EQ(top_logit_indices(row.data(), 2, 5), (std::vector<size_t>{0, 1}));
}
// cppcheck-suppress-end syntaxError

} // namespace
} // namespace ptn
