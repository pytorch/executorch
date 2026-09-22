/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "MLXBatchedSequenceCache.h"
#include "MLXSequenceCache.h"
#include "utils.h"

#include <mlx/mlx.h>

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <numeric>
#include <vector>

using namespace ::executorch::backends::mlx;
namespace cache = ::executorch::extension::llm::cache;
using ::mlx::core::array;

namespace {

// Real MLX attention and pools; requires Apple Silicon, no model or tokenizer.
class MLXSequenceCloneTest : public ::testing::Test {
 protected:
  static constexpr int H = 2;
  static constexpr int D = 8;
  static constexpr int kMaxWrite = 3;
  ::mlx::core::StreamOrDevice s = {};

  CacheArgs config() {
    auto args =
        flat_config(256, 2, H, D, static_cast<int>(ScalarType::Half), 2);
    args.geometry.layers[1].policy = {cache::LayerPolicy::Kind::Ring, 4};
    args.config.max_write = kMaxWrite;
    return args;
  }

  std::vector<uint64_t> tokens(size_t count) {
    std::vector<uint64_t> result(count);
    std::iota(result.begin(), result.end(), uint64_t{1});
    return result;
  }

  array forward(
      MLXCache& c,
      cache::BatchControl* control,
      int32_t seq,
      const std::vector<uint64_t>& prompt,
      size_t begin = 0,
      int chunk_size = kMaxWrite) {
    using namespace ::mlx::core;
    std::vector<array> outputs;
    while (begin < prompt.size()) {
      const size_t count =
          std::min(static_cast<size_t>(chunk_size), prompt.size() - begin);
      std::vector<int32_t> positions(count);
      std::iota(
          positions.begin(), positions.end(), static_cast<int32_t>(begin));
      if (control) {
        EXPECT_TRUE(control->declare_step(std::vector<int32_t>(count, seq)));
      }
      std::vector<float> values;
      values.reserve(static_cast<size_t>(H * D) * count);
      for (int h = 0; h < H; ++h) {
        for (size_t i = begin; i < begin + count; ++i) {
          for (int d = 0; d < D; ++d) {
            values.push_back(std::sin(
                static_cast<float>(prompt[i]) * 0.37f +
                static_cast<float>(h) * 0.41f + static_cast<float>(d) * 0.19f));
          }
        }
      }
      array input = astype(
          array(
              values.data(), Shape{1, H, static_cast<int>(count), D}, float32),
          float16,
          s);
      const float scale = 1.0f / std::sqrt(static_cast<float>(D));
      // The SWA layer's K/V depend on full attention, including old tokens
      // outside its window. Reusing a matching tail alone is insufficient.
      array hidden =
          add(input, c.attend(0, positions, input, input, input, scale, s), s);
      array output = c.attend(1, positions, hidden, hidden, hidden, scale, s);
      eval(output);
      outputs.push_back(output);
      begin += count;
    }
    return concatenate(outputs, 2, s);
  }

  void expect_cold_match(
      MLXBatchedSequenceCache& c,
      int32_t sequence,
      size_t position,
      const std::vector<uint64_t>& prompt) {
    using namespace ::mlx::core;
    ASSERT_LT(position, prompt.size());
    ASSERT_EQ(c.pos(sequence), static_cast<int>(position));
    array got = forward(c, &c, sequence, prompt, position);
    auto cold = make_cache<MLXSequenceCache>(config());
    array expected = forward(cold, nullptr, 0, prompt, 0, 1);
    expected = slice(
        expected,
        Shape{0, 0, static_cast<int>(position), 0},
        Shape{1, H, static_cast<int>(prompt.size()), D},
        s);
    EXPECT_TRUE(allclose(got, expected, 1e-2f));
    EXPECT_EQ(c.pos(sequence), static_cast<int>(prompt.size()));
  }
};

TEST_F(MLXSequenceCloneTest, SnapshotSurvivesSourceAndForkWrites) {
  auto c = make_cache<MLXBatchedSequenceCache>(config());
  c.bind_controller_stream(::mlx::core::to_stream(s));
  const auto source = c.seq_new();
  ASSERT_TRUE(source);
  const auto prefix = tokens(21);
  forward(c, &c, *source, prefix);
  const auto snapshot = c.seq_clone(*source, 21);
  ASSERT_TRUE(snapshot);
  ASSERT_EQ(c.pos(*snapshot), 21);

  // Overwrite every source ring slot after the snapshot, then release it.
  forward(c, &c, *source, tokens(41), prefix.size());
  ASSERT_TRUE(c.seq_rm(*source));

  auto request = prefix;
  request.insert(request.end(), {50, 51, 52, 53, 54, 55, 56});
  const auto first = c.seq_clone(*snapshot, 21);
  ASSERT_TRUE(first);
  expect_cold_match(c, *first, 21, request);
  ASSERT_TRUE(c.seq_rm(*first));

  // The previous fork also wrapped; an equal prompt still replays its last
  // token against the untouched snapshot to obtain fresh logits.
  const auto equal = c.seq_clone(*snapshot, 20);
  ASSERT_TRUE(equal);
  expect_cold_match(c, *equal, 20, prefix);
  ASSERT_TRUE(c.seq_rm(*equal));

  request.back() = 90;
  const auto independent = c.seq_clone(*snapshot, 21);
  ASSERT_TRUE(independent);
  ASSERT_TRUE(c.seq_rm(*snapshot));
  expect_cold_match(c, *independent, 21, request);
  EXPECT_TRUE(c.seq_rm(*independent));
}

TEST_F(MLXSequenceCloneTest, RetainedBoundaryAndOlderSnapshot) {
  auto c = make_cache<MLXBatchedSequenceCache>(config());
  c.bind_controller_stream(::mlx::core::to_stream(s));
  const auto source = c.seq_new();
  ASSERT_TRUE(source);
  forward(c, &c, *source, tokens(21));
  const auto snapshot = c.seq_clone(*source, 21);
  ASSERT_TRUE(snapshot);
  ASSERT_EQ(c.pos(*snapshot), 21);
  ASSERT_TRUE(c.seq_rm(*source));

  // Ring size is 4 + 3 - 1 = 6. At position 21 it retains the keys needed
  // to resume at 18, but resuming at 17 would read overwritten slots.
  auto boundary = tokens(18);
  boundary.insert(boundary.end(), {80, 81, 82, 83});
  const auto hit = c.seq_clone(*snapshot, 18);
  ASSERT_TRUE(hit);
  expect_cold_match(c, *hit, 18, boundary);
  ASSERT_TRUE(c.seq_rm(*hit));

  const auto shortened = c.seq_clone(*snapshot, 18);
  ASSERT_TRUE(shortened);
  expect_cold_match(c, *shortened, 18, tokens(19));
  ASSERT_TRUE(c.seq_rm(*shortened));

  auto missing = tokens(17);
  missing.insert(missing.end(), {80, 81, 82});
  EXPECT_FALSE(c.seq_clone(*snapshot, 17));
  EXPECT_EQ(c.pos(*snapshot), 21);

  const auto older = c.seq_new();
  ASSERT_TRUE(older);
  forward(c, &c, *older, tokens(17));
  const auto older_snapshot = c.seq_clone(*older, 17);
  ASSERT_TRUE(older_snapshot);
  ASSERT_EQ(c.pos(*older_snapshot), 17);
  ASSERT_TRUE(c.seq_rm(*older));
  EXPECT_FALSE(c.seq_clone(*snapshot, 17));
  const auto retained = c.seq_clone(*older_snapshot, 17);
  ASSERT_TRUE(retained);
  expect_cold_match(c, *retained, 17, missing);
  EXPECT_TRUE(c.seq_rm(*retained));
  EXPECT_TRUE(c.seq_rm(*older_snapshot));
  EXPECT_TRUE(c.seq_rm(*snapshot));
}

} // namespace
