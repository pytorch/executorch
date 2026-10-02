/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/extension/llm/batching/util/session_table.h>

#include <executorch/extension/llm/cache/cell_cache.h>
#include <executorch/extension/tensor/tensor.h>
#include <executorch/runtime/platform/runtime.h>

#include <gtest/gtest.h>

#include <memory>
#include <vector>

namespace batching = ::executorch::extension::llm::batching;
namespace cache = ::executorch::extension::llm::cache;
namespace util = ::executorch::extension::llm::batching::util;
using ::executorch::extension::make_tensor_ptr;
using ::executorch::runtime::Error;

namespace {

constexpr int kVocab = 8;

// The neutral cell cache is a complete BatchControl without any bytes, so the
// table runs against the real sequence semantics on the CPU.
class SessionTableTest : public ::testing::Test {
 protected:
  void SetUp() override {
    ::executorch::runtime::runtime_init();
    cache::CacheGeometry geometry;
    geometry.layers = {{{cache::LayerPolicy::Kind::Flat, 0}, 1, 1}};
    cache::CacheConfig cfg{/*capacity=*/64, /*kv_dtype=*/6};
    cells_ = std::make_unique<cache::CellCache>(geometry, cfg);
    table_ = std::make_unique<util::SessionTable>(
        *cells_, /*max_sessions=*/3, /*max_session_tokens=*/16, kVocab);
  }

  // Declares and places a step, as a forward would.
  void run(const util::PackedStep& step) {
    ASSERT_TRUE(cells_->declare_step(step.seq_ids));
    std::vector<int32_t> positions(step.positions.begin(), step.positions.end());
    ASSERT_NE(
        cells_->place_step(0, positions.data(), static_cast<int>(positions.size())),
        nullptr);
  }

  static batching::Input input(
      batching::SessionId sid,
      std::vector<batching::Token> tokens,
      batching::Position position,
      bool produce_output = true,
      std::size_t offset = 0,
      std::size_t size = 0) {
    auto shared = std::make_shared<std::vector<batching::Token>>(std::move(tokens));
    batching::Input in;
    in.sid = sid;
    in.tokens = shared;
    in.offset = offset;
    in.size = size == 0 ? shared->size() - offset : size;
    in.position = position;
    in.produce_output = produce_output;
    return in;
  }

  std::unique_ptr<cache::CellCache> cells_;
  std::unique_ptr<util::SessionTable> table_;
};

} // namespace

TEST_F(SessionTableTest, PacksInputsInOrderWithTheirRows) {
  const auto a = *table_->open();
  const auto b = *table_->open();
  batching::BatchInput batch;
  batch.inputs = {input(a, {1, 2, 3}, 0), input(b, {7}, 0, false)};
  const auto step = table_->pack(batch);
  ASSERT_EQ(step.error(), Error::Ok);
  EXPECT_EQ(step->tokens, std::vector<std::int64_t>({1, 2, 3, 7}));
  EXPECT_EQ(step->positions, std::vector<std::int64_t>({0, 1, 2, 0}));
  EXPECT_EQ(step->seq_ids[0], step->seq_ids[2]);
  EXPECT_NE(step->seq_ids[0], step->seq_ids[3]);
  EXPECT_EQ(step->logit_indices, std::vector<int>({2, -1}));
}

TEST_F(SessionTableTest, ConsecutiveChunksOfOnePromptAbut) {
  const auto a = *table_->open();
  batching::BatchInput batch;
  const std::vector<batching::Token> prompt{1, 2, 3, 4, 5};
  batch.inputs = {
      input(a, prompt, 0, false, 0, 2), input(a, prompt, 0, true, 2, 3)};
  const auto step = table_->pack(batch);
  ASSERT_EQ(step.error(), Error::Ok);
  EXPECT_EQ(step->positions, std::vector<std::int64_t>({0, 1, 2, 3, 4}));
  EXPECT_EQ(step->logit_indices, std::vector<int>({-1, 4}));
}

TEST_F(SessionTableTest, RewindsOnlyAfterEveryInputChecks) {
  const auto a = *table_->open();
  const auto b = *table_->open();
  batching::BatchInput first;
  first.inputs = {input(a, {1, 2, 3, 4}, 0), input(b, {5, 6}, 0)};
  const auto step = table_->pack(first);
  ASSERT_EQ(step.error(), Error::Ok);
  run(*step);

  // a reopens at 2; b skips ahead, which refuses the batch before a rewinds.
  batching::BatchInput bad;
  bad.inputs = {input(a, {9}, 2), input(b, {9}, 5)};
  EXPECT_EQ(table_->pack(bad).error(), Error::InvalidArgument);
  EXPECT_EQ(cells_->pos(step->seq_ids[0]), 4);

  batching::BatchInput good;
  good.inputs = {input(a, {9}, 2)};
  const auto rewound = table_->pack(good);
  ASSERT_EQ(rewound.error(), Error::Ok);
  EXPECT_EQ(cells_->pos(rewound->seq_ids[0]), 2);
  EXPECT_EQ(rewound->positions, std::vector<std::int64_t>({2}));
}

TEST_F(SessionTableTest, RefusesOverlapsRestartsAndOverruns) {
  const auto a = *table_->open();
  batching::BatchInput first;
  first.inputs = {input(a, {1, 2, 3}, 0)};
  const auto step = table_->pack(first);
  ASSERT_EQ(step.error(), Error::Ok);
  run(*step);

  batching::BatchInput overlap;
  overlap.inputs = {input(a, {4}, 3), input(a, {5}, 2)};
  EXPECT_EQ(table_->pack(overlap).error(), Error::InvalidArgument);
  batching::BatchInput restart;
  restart.inputs = {input(a, {4}, 0)};
  EXPECT_EQ(table_->pack(restart).error(), Error::InvalidArgument);
  batching::BatchInput overrun;
  overrun.inputs = {input(a, std::vector<batching::Token>(14, 1), 3)};
  EXPECT_EQ(table_->pack(overrun).error(), Error::OutOfResources);
  batching::BatchInput unknown;
  unknown.inputs = {input(a + 100, {4}, 0)};
  EXPECT_EQ(table_->pack(unknown).error(), Error::InvalidArgument);
}

TEST_F(SessionTableTest, SessionsAreBoundedAndIdsNeverReused) {
  const auto a = *table_->open();
  const auto b = *table_->open();
  const auto c = *table_->open();
  EXPECT_FALSE(table_->open().has_value());
  table_->close(b);
  const auto d = *table_->open();
  EXPECT_NE(d, b);
  EXPECT_EQ(table_->size(), 3u);
  (void)a;
  (void)c;
}

TEST_F(SessionTableTest, CloneSharesTheSourcePrefix) {
  const auto a = *table_->open();
  batching::BatchInput batch;
  batch.inputs = {input(a, {1, 2, 3, 4}, 0)};
  const auto step = table_->pack(batch);
  ASSERT_EQ(step.error(), Error::Ok);
  run(*step);

  const auto c = table_->clone(a, 3);
  ASSERT_TRUE(c.has_value());
  EXPECT_FALSE(table_->clone(a, 5).has_value()); // past the source
  batching::BatchInput next;
  next.inputs = {input(*c, {8}, 3)};
  const auto cloned = table_->pack(next);
  ASSERT_EQ(cloned.error(), Error::Ok);
  EXPECT_EQ(cloned->positions, std::vector<std::int64_t>({3}));
}

TEST_F(SessionTableTest, SamplesEachSessionsRowWithItsOwnPolicy) {
  const auto a = *table_->open();
  const auto b = *table_->open();
  batching::SamplingParams greedy;
  greedy.temperature = 0.0f;
  table_->set_sampling(a, greedy, 0);
  std::vector<float> values(2 * kVocab, 0.0f);
  values[3] = 5.0f; // row 0 peaks at 3
  values[kVocab + 6] = 5.0f; // row 1 peaks at 6
  auto logits = make_tensor_ptr({2, kVocab}, values);
  EXPECT_EQ(table_->sample(a, *logits, 1), 6u);
  EXPECT_EQ(table_->sample(a, *logits, 0), 3u);
  // No policy yet, and no such row.
  EXPECT_FALSE(table_->sample(b, *logits, 0).has_value());
  EXPECT_FALSE(table_->sample(a, *logits, 2).has_value());
}

TEST(SelectRowsTest, KeepsOnlyTheSlicesWantedRows) {
  util::PackedStep step;
  step.logit_indices = {2, -1, 5, 9};
  auto rows = util::select_rows(step, 0, 6);
  EXPECT_EQ(rows.selector, std::vector<std::int64_t>({2, 5}));
  EXPECT_EQ(rows.inputs, std::vector<std::size_t>({0, 2}));
  rows = util::select_rows(step, 6, 4);
  EXPECT_EQ(rows.selector, std::vector<std::int64_t>({3}));
  EXPECT_EQ(rows.inputs, std::vector<std::size_t>({3}));
  // Nothing wanted: the last token still runs, and nothing reads it.
  rows = util::select_rows(step, 10, 3);
  EXPECT_EQ(rows.selector, std::vector<std::int64_t>({2}));
  EXPECT_TRUE(rows.inputs.empty());
}
