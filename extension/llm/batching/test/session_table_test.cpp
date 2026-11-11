/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/extension/llm/batching/util/sequence_batch_control.h>
#include <executorch/extension/llm/batching/util/session_table.h>

#include <executorch/extension/llm/cache/cell_cache.h>
#include <executorch/extension/llm/cache/sequence_cache.h>
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
    std::vector<int32_t> positions(
        step.positions.begin(), step.positions.end());
    ASSERT_NE(
        cells_->place_step(
            0, positions.data(), static_cast<int>(positions.size())),
        nullptr);
  }

  static batching::Input input(
      batching::SessionId sid,
      std::vector<batching::Token> tokens,
      batching::Position position,
      bool produce_output = true,
      std::size_t offset = 0,
      std::size_t size = 0) {
    auto shared =
        std::make_shared<std::vector<batching::Token>>(std::move(tokens));
    batching::Input in;
    in.sid = sid;
    in.payload = batching::TokenInputPtr(shared);
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

TEST_F(SessionTableTest, DeviceRowsCarryThePolicyAndTheHostsCoins) {
  const auto a = *table_->open();
  const auto b = *table_->open();
  const auto c = *table_->open();
  EXPECT_FALSE(table_->device_sampling(a).has_value());
  batching::SamplingParams stochastic;
  stochastic.temperature = 0.7f;
  stochastic.top_p = 0.9f;
  stochastic.top_k = 5;
  table_->set_sampling(a, stochastic, 42);
  table_->set_sampling(b, stochastic, 42);
  batching::SamplingParams greedy;
  table_->set_sampling(c, greedy, 42);
  EXPECT_FALSE(table_->greedy(a));
  EXPECT_TRUE(table_->greedy(c));

  // Sampler's xorshift from the same seed: the host draws these coins.
  std::uint64_t state = 42;
  auto coin = [&state]() {
    state ^= state >> 12;
    state ^= state << 25;
    state ^= state >> 27;
    const auto bits =
        static_cast<std::uint32_t>((state * 0x2545F4914F6CDD1Dull) >> 32);
    return static_cast<float>(bits >> 8) / 16777216.0f;
  };
  for (int step = 0; step < 4; ++step) {
    const auto row = table_->device_sampling(a);
    ASSERT_TRUE(row.has_value());
    EXPECT_FLOAT_EQ(row->temperature, 0.7f);
    EXPECT_FLOAT_EQ(row->top_p, 0.9f);
    EXPECT_FLOAT_EQ(row->top_k, 5.0f);
    EXPECT_EQ(row->coin, coin()) << step;
    // One seed, one sequence of coins, whatever else runs.
    EXPECT_EQ(table_->device_sampling(b)->coin, row->coin) << step;
    // A greedy row draws none.
    EXPECT_EQ(table_->device_sampling(c)->coin, 0.0f);
    EXPECT_EQ(table_->device_sampling(c)->temperature, 0.0f);
  }
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

// One session over a single-sequence cache, through the same table.
TEST(SequenceBatchControlTest, ServesOneSessionAtATime) {
  ::executorch::runtime::runtime_init();
  cache::CacheGeometry geometry;
  geometry.layers = {{{cache::LayerPolicy::Kind::Flat, 0}, 1, 1}};
  cache::SequenceCache sequence(geometry, cache::CacheConfig{16, 6});
  util::SequenceBatchControl control(sequence);
  util::SessionTable table(control, /*max_sessions=*/2, 16, kVocab);
  // Commits a step as the cache's own byte layer would.
  const auto commit = [&](const util::PackedStep& step) {
    ASSERT_TRUE(control.declare_step(step.seq_ids));
    const auto plan = sequence.plan(
        0,
        static_cast<int>(step.positions.front()),
        static_cast<int>(step.positions.size()));
    ASSERT_TRUE(plan.has_value());
    sequence.commit(*plan);
  };
  const auto input = [](batching::SessionId sid,
                        std::vector<batching::Token> tokens,
                        batching::Position position) {
    batching::Input in;
    in.sid = sid;
    in.size = tokens.size();
    in.offset = 0;
    in.payload = batching::TokenInputPtr(
        std::make_shared<std::vector<batching::Token>>(std::move(tokens)));
    in.position = position;
    in.produce_output = true;
    return in;
  };

  const auto session = table.open();
  ASSERT_TRUE(session.has_value());
  EXPECT_FALSE(table.open().has_value());
  EXPECT_FALSE(table.clone(*session, 1).has_value());

  auto step = table.pack({{input(*session, {1, 2, 3}, 0)}});
  ASSERT_TRUE(step.ok());
  EXPECT_EQ(step->seq_ids, (std::vector<int32_t>{0, 0, 0}));
  EXPECT_EQ(step->positions, (std::vector<int64_t>{0, 1, 2}));
  commit(*step);
  EXPECT_EQ(control.pos(0), 3);

  // A step starting inside the committed history rewinds to its start.
  const auto rewound = table.pack({{input(*session, {7}, 1)}});
  ASSERT_TRUE(rewound.ok());
  EXPECT_EQ(control.pos(0), 1);
  commit(*rewound);
  EXPECT_EQ(control.pos(0), 2);
  EXPECT_FALSE(table.pack({{input(*session, {8}, 4)}}).ok());

  table.close(*session);
  EXPECT_EQ(sequence.length(), 0);
  const auto reopened = table.open();
  ASSERT_TRUE(reopened.has_value());
  EXPECT_NE(*reopened, *session);
  EXPECT_EQ(control.pos(0), 0);
}
