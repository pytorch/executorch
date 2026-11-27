/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <gtest/gtest.h>
#include <numeric>
#include "MLXBatchedCache.h"

namespace {
namespace mx = ::mlx::core;
namespace cache = executorch::extension::llm::cache;
using executorch::backends::mlx::MLXBatchedCache;

cache::CacheGeometry geometry() {
  return {
      {{{cache::LayerPolicy::Kind::Flat, 0}, 2, 8},
       {{cache::LayerPolicy::Kind::Ring, 4}, 2, 8}},
      std::nullopt};
}
cache::CacheConfig config() {
  return {256, 5, 4, 3, std::nullopt};
}

mx::array step(MLXBatchedCache& c, int id, int start, int count) {
  std::vector<float> values(2 * count * 8);
  for (size_t i = 0; i < values.size(); ++i)
    values[i] = std::sin(static_cast<float>(start + i) * 0.17f);
  mx::array input(values.begin(), {1, 2, count, 8}, mx::float16);
  std::vector<int32_t> positions(count);
  std::iota(positions.begin(), positions.end(), start);
  EXPECT_TRUE(c.declare_step(std::vector<int32_t>(count, id)));
  auto output = input;
  for (int layer = 0; layer < 2; ++layer)
    output = c.attend(layer, positions, input, input, input, 0.35f, {});
  output.eval();
  return output;
}

TEST(MLXCheckpoint, WrappedRingRestoresContinuationAndRejectsMissingHistory) {
  for (int end : {18, 20}) {
    SCOPED_TRACE(end);
    MLXBatchedCache source(geometry(), config());
    auto id = source.seq_new().value();
    for (int p = 0; p < end; p += 3)
      step(source, id, p, std::min(3, end - p));
    auto exported = source.export_to_cpu(id, end);
    ASSERT_TRUE(std::holds_alternative<cache::SequenceCheckpoint>(exported));
    auto checkpoint = std::get<cache::SequenceCheckpoint>(std::move(exported));
    EXPECT_EQ(checkpoint.layers[0].begin, 0);
    EXPECT_EQ(checkpoint.layers[1].begin, end - 6);
    for (int position : {end, end - 1, end - 3}) {
      MLXBatchedCache restored(geometry(), config());
      auto result = restored.import_from_cpu(checkpoint, position);
      ASSERT_TRUE(std::holds_alternative<int32_t>(result));
      auto reference = source.seq_clone(id, position);
      ASSERT_TRUE(reference);
      auto expected = step(source, *reference, position, 1);
      auto actual = step(restored, std::get<int32_t>(result), position, 1);
      EXPECT_TRUE(mx::allclose(expected, actual, 0, 0).item<bool>());
      source.seq_rm(*reference);
    }
    MLXBatchedCache rejected(geometry(), config());
    EXPECT_TRUE(std::holds_alternative<cache::CheckpointError>(
        rejected.import_from_cpu(checkpoint, end - 4)));
    EXPECT_EQ(rejected.seq_new().value(), 0);
  }
}

TEST(MLXCheckpoint, ReexportPreservesRetentionAndOwnsBytes) {
  MLXBatchedCache source(geometry(), config());
  auto id = source.seq_new().value();
  for (int p = 0; p < 18; p += 3)
    step(source, id, p, 3);
  auto state =
      std::get<cache::SequenceCheckpoint>(source.export_to_cpu(id, 17));
  const auto original = state.layers;
  step(source, id, 18, 3);
  EXPECT_EQ(state.layers[1].key, original[1].key);
  MLXBatchedCache restored(geometry(), config());
  auto target = std::get<int32_t>(restored.import_from_cpu(state, 17));
  for (auto& layer : state.layers) {
    layer.key.assign(layer.key.size(), 0);
    layer.value.assign(layer.value.size(), 0);
  }
  auto again =
      std::get<cache::SequenceCheckpoint>(restored.export_to_cpu(target, 17));
  EXPECT_EQ(again.layers[1].begin, 12);
  ASSERT_EQ(again.layers.size(), original.size());
  for (size_t i = 0; i < original.size(); ++i) {
    EXPECT_EQ(again.layers[i].key, original[i].key);
    EXPECT_EQ(again.layers[i].value, original[i].value);
  }
  EXPECT_FALSE(restored.rewind(target, 14));
}

TEST(MLXCheckpoint, DFlashRequiresTheFullWindowBeforeTheNextBlock) {
  auto g = geometry();
  g.dflash_block_size = 2;
  MLXBatchedCache source(g, config());
  const auto id = source.seq_new().value();
  auto forward = [](MLXBatchedCache& c, int seq, int start, int count) {
    const auto kv = mx::reshape(
        mx::arange(2 * (count + 2) * 8, mx::float32), {1, 2, count + 2, 8});
    const auto query = mx::ones({1, 2, 2, 8}, mx::float32);
    std::vector<int32_t> positions(count + 2);
    std::iota(positions.begin(), positions.end(), start);
    EXPECT_TRUE(c.declare_step(std::vector<int32_t>(count + 2, seq)));
    auto output = query;
    for (int layer = 0; layer < 2; ++layer)
      output = c.attend(layer, positions, query, kv, kv, 0.35f, {});
    output.eval();
    return output;
  };
  for (int p = 0; p < 20; p += 3)
    forward(source, id, p, std::min(3, 20 - p));
  const auto checkpoint =
      std::get<cache::SequenceCheckpoint>(source.export_to_cpu(id, 20));
  const int boundary = checkpoint.layers[1].begin + 4;
  executorch::backends::mlx::MLXSequenceCache direct(g, config());
  EXPECT_FALSE(direct.import_from_cpu(checkpoint.layers, boundary - 1));
  MLXBatchedCache restored(g, config());
  EXPECT_TRUE(std::holds_alternative<cache::CheckpointError>(
      restored.import_from_cpu(checkpoint, boundary - 1)));
  const auto result = restored.import_from_cpu(checkpoint, boundary);
  ASSERT_TRUE(std::holds_alternative<int32_t>(result));
  const auto reference = source.seq_clone(id, boundary);
  ASSERT_TRUE(reference);
  const auto expected = forward(source, *reference, boundary, 1);
  const auto actual = forward(restored, std::get<int32_t>(result), boundary, 1);
  EXPECT_TRUE(mx::allclose(expected, actual, 0, 0).item<bool>());
}

TEST(MLXCheckpoint, InvalidGeometryAndPartialForwardLeaveNoSession) {
  MLXBatchedCache source(geometry(), config());
  auto id = source.seq_new().value();
  step(source, id, 0, 3);
  auto state = std::get<cache::SequenceCheckpoint>(source.export_to_cpu(id, 3));
  MLXBatchedCache restored(geometry(), config());
  state.layers[1].key.pop_back();
  EXPECT_TRUE(std::holds_alternative<cache::CheckpointError>(
      restored.import_from_cpu(state, 3)));
  EXPECT_EQ(restored.seq_new().value(), 0);
  EXPECT_TRUE(source.declare_step({id}));
  auto value = mx::ones({1, 2, 1, 8}, mx::float16);
  source.attend(0, {3}, value, value, value, 0.35f, {}).eval();
  EXPECT_TRUE(std::holds_alternative<cache::CheckpointError>(
      source.export_to_cpu(id, 3)));
}
} // namespace
