/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/extension/llm/cache/cache.h>
#include <executorch/extension/llm/cache/cache_et.h>
#include <executorch/extension/llm/cache/cache_registry.h>
#include <executorch/extension/llm/cache/cell_cache.h>
#include <executorch/extension/llm/cache/sequence_cache.h>

#include <memory>
#include <string>
#include <vector>

#include <executorch/runtime/core/error.h>
#include <executorch/runtime/platform/runtime.h>
#include <gtest/gtest.h>

using executorch::extension::llm::cache::BatchControl;
using executorch::extension::llm::cache::CacheBase;
using executorch::extension::llm::cache::CacheBuilderRegistry;
using executorch::extension::llm::cache::CacheConfig;
using executorch::extension::llm::cache::CacheRegistry;
using executorch::extension::llm::cache::CacheSession;
using executorch::extension::llm::cache::CellCache;
using executorch::extension::llm::cache::CellPlanner;
using executorch::extension::llm::cache::CellStepPlan;
using executorch::extension::llm::cache::LayerConfig;
using executorch::extension::llm::cache::LayerPolicy;
using executorch::extension::llm::cache::make_unique_key;
using executorch::extension::llm::cache::MaskKind;
using executorch::extension::llm::cache::SequenceCache;
using executorch::runtime::Error;
namespace et = executorch::extension::llm::cache::et;

namespace {
LayerConfig flat_layer() {
  return LayerConfig{LayerPolicy{LayerPolicy::Kind::Flat, 0}, 2, 8};
}
LayerConfig ring_layer(int window) {
  return LayerConfig{LayerPolicy{LayerPolicy::Kind::Ring, window}, 2, 8};
}
} // namespace

// Initializes the ExecuTorch PAL so the ET adapter's error paths (which ET_LOG)
// can run.
class CacheTest : public ::testing::Test {
 protected:
  void SetUp() override {
    executorch::runtime::runtime_init();
  }
};

// ---- Flat policy -----------------------------------------------------------

TEST_F(CacheTest, FlatPlanAppendsAndReadsAllHistory) {
  SequenceCache cache(CacheConfig{8, 1, {flat_layer()}});

  auto p0 = cache.plan(/*layer=*/0, /*position=*/0, /*T=*/4); // prefill
  ASSERT_TRUE(p0.has_value());
  EXPECT_EQ(p0->n_write, 1);
  EXPECT_EQ(p0->write[0].start, 0);
  EXPECT_EQ(p0->write[0].len, 4);
  EXPECT_EQ(p0->n_read, 1);
  EXPECT_EQ(p0->read[0].start, 0);
  EXPECT_EQ(p0->read[0].len, 4);
  EXPECT_EQ(p0->read_base_pos, 0);
  cache.commit(*p0); // accept the prefill step

  auto p1 = cache.plan(0, 4, 1); // decode
  ASSERT_TRUE(p1.has_value());
  EXPECT_EQ(p1->write[0].start, 4);
  EXPECT_EQ(p1->read[0].len, 5);
}

// ---- Ring policy: wrap, eviction, read base position -----------------------

TEST_F(CacheTest, RingPlanWrapsAndEvicts) {
  // window 4, max_write unset -> defaults to window -> ring of 2*4 - 1 = 7.
  SequenceCache cache(CacheConfig{100, 1, {ring_layer(4)}});

  // Prefill chunk of 4 at position 0: one write run, reads [0,4), no wrap.
  auto p0 = cache.plan(0, 0, 4);
  ASSERT_TRUE(p0.has_value());
  EXPECT_EQ(p0->n_write, 1);
  EXPECT_EQ(p0->write[0].start, 0);
  EXPECT_EQ(p0->write[0].len, 4);
  EXPECT_EQ(p0->n_read, 1);
  EXPECT_EQ(p0->read[0].start, 0);
  EXPECT_EQ(p0->read[0].len, 4);
  EXPECT_EQ(p0->read_base_pos, 0);
  cache.commit(*p0);

  // Decode at position 7: write wraps to slot 0 (7 % 7); the read window
  // [4,8) wraps the ring -> phys [4,6] then [0,0]. Positions 0-3 evicted.
  auto p1 = cache.plan(0, 7, 1);
  ASSERT_TRUE(p1.has_value());
  EXPECT_EQ(p1->n_write, 1);
  EXPECT_EQ(p1->write[0].start, 0);
  EXPECT_EQ(p1->write[0].len, 1);
  EXPECT_EQ(p1->n_read, 2);
  EXPECT_EQ(p1->read[0].start, 4);
  EXPECT_EQ(p1->read[0].len, 3);
  EXPECT_EQ(p1->read[1].start, 0);
  EXPECT_EQ(p1->read[1].len, 1);
  EXPECT_EQ(p1->read_base_pos, 4); // oldest retained is logical position 4

  // A step larger than max_write would overrun the ring -> rejected.
  EXPECT_FALSE(cache.plan(0, 0, 5).has_value());
}

// ---- Mixed flat/ring: one shared length across layers ----------------------

TEST_F(CacheTest, MixedFlatRingShareOneLength) {
  SequenceCache cache(CacheConfig{100, 2, {flat_layer(), ring_layer(4)}});

  // Same step drives both layers (T <= window); commit once.
  auto p = cache.plan(0, 0, 3);
  ASSERT_TRUE(p.has_value());
  ASSERT_TRUE(cache.plan(1, 0, 3).has_value());
  cache.commit(*p);

  // Length advanced once (not per layer): length == 3.
  EXPECT_TRUE(cache.can_extend(97));
  EXPECT_FALSE(cache.can_extend(98));
}

// ---- Admission / rewind ----------------------------------------------------

TEST_F(CacheTest, CanExtendBoundedByCapacity) {
  SequenceCache cache(CacheConfig{2, 1, {flat_layer()}});
  EXPECT_TRUE(cache.can_extend(2));
  auto p = cache.plan(0, 0, 2);
  ASSERT_TRUE(p.has_value());
  cache.commit(*p);
  EXPECT_FALSE(cache.can_extend(1));
  EXPECT_FALSE(cache.plan(0, 2, 1).has_value()); // exceeds capacity
}

TEST_F(CacheTest, FlatRewindsFreelyToZero) {
  SequenceCache cache(CacheConfig{8, 1, {flat_layer()}});
  auto p = cache.plan(0, 0, 5);
  ASSERT_TRUE(p.has_value());
  cache.commit(*p);
  EXPECT_TRUE(cache.rewind(0)); // flat retains all history
  EXPECT_FALSE(cache.rewind(1)); // cannot grow
  cache.clear();
  EXPECT_TRUE(cache.can_extend(8));
}

TEST_F(CacheTest, RewindBoundedByRingWindow) {
  SequenceCache cache(CacheConfig{100, 2, {flat_layer(), ring_layer(4)}});
  // Advance length to 10 in chunks of 2 (<= window).
  for (int pos = 0; pos < 10; pos += 2) {
    auto p = cache.plan(0, pos, 2);
    ASSERT_TRUE(p.has_value());
    ASSERT_TRUE(cache.plan(1, pos, 2).has_value());
    cache.commit(*p);
  }
  // ring(4) has evicted everything older than 10 - 4 = 6.
  EXPECT_FALSE(cache.rewind(5)); // older than the ring retains
  EXPECT_TRUE(cache.rewind(6)); // exactly the floor
  EXPECT_FALSE(cache.rewind(11)); // cannot grow
}

// ---- Faces / registry / session --------------------------------------------

TEST_F(CacheTest, FaceRecoveryReturnsSameObject) {
  SequenceCache cache(CacheConfig{4, 1, {flat_layer()}});
  CacheBase* base = &cache;
  ASSERT_NE(base->as_control(), nullptr);
  ASSERT_NE(base->as_planner(), nullptr);
  EXPECT_TRUE(base->as_control()->can_extend(4));
  auto plan = base->as_planner()->plan(0, 0, 1);
  ASSERT_TRUE(plan.has_value());
  EXPECT_EQ(plan->read[0].len, 1);
}

TEST_F(CacheTest, RegistryInstallGetErase) {
  auto& reg = CacheRegistry::global();
  const std::string key = make_unique_key();
  EXPECT_EQ(reg.get(key), nullptr);

  std::shared_ptr<CacheBase> cache =
      std::make_shared<SequenceCache>(CacheConfig{16, 1, {flat_layer()}});
  reg.install(key, cache);
  EXPECT_EQ(reg.get(key), cache);
  EXPECT_TRUE(reg.get(key)->as_control()->can_extend(16));

  reg.erase(key);
  EXPECT_EQ(reg.get(key), nullptr);
}

TEST_F(CacheTest, UniqueKeysDoNotCollide) {
  EXPECT_NE(make_unique_key(), make_unique_key());
}

TEST_F(CacheTest, BuilderBuildsRegisteredKindElseError) {
  auto& reg = CacheBuilderRegistry::global();
  reg.register_builder("TestBackend", "seq", [](const CacheConfig& cfg) {
    return std::static_pointer_cast<CacheBase>(
        std::make_shared<SequenceCache>(cfg));
  });

  CacheConfig cfg{32, 1, {flat_layer()}};
  auto cache = reg.build("TestBackend", "seq", cfg);
  ASSERT_TRUE(cache.ok());
  EXPECT_EQ(cache.get()->as_control()->capacity(), 32);

  EXPECT_EQ(reg.build("TestBackend", "missing", cfg).error(), Error::NotFound);
}

TEST_F(CacheTest, SessionInstallsOnCtorErasesOnDtor) {
  const std::string key = make_unique_key();
  {
    CacheSession session(
        key,
        std::make_shared<SequenceCache>(CacheConfig{4, 1, {flat_layer()}}));
    EXPECT_NE(CacheRegistry::global().get(key), nullptr);
    EXPECT_TRUE(session.control()->can_extend(4));
  }
  EXPECT_EQ(CacheRegistry::global().get(key), nullptr);
}

// ---- ET adapter (maps core bool/optional to Error/Result) ------------------

TEST_F(CacheTest, EtAdapterMapsResultsAndCodes) {
  SequenceCache cache(CacheConfig{2, 1, {flat_layer()}});
  auto ok = et::plan(cache, /*layer=*/0, /*position=*/0, /*T=*/2);
  ASSERT_TRUE(ok.ok());
  EXPECT_EQ(ok->read[0].len, 2);
  cache.commit(ok.get()); // accept the step so rewind has history to truncate
  EXPECT_EQ(
      et::plan(cache, 0, 2, 1).error(), Error::OutOfResources); // over capacity
  EXPECT_FALSE(et::plan(cache, 5, 0, 1).ok()); // bad layer

  EXPECT_EQ(et::rewind(cache, 9), Error::InvalidArgument); // cannot grow
  EXPECT_EQ(et::rewind(cache, 1), Error::Ok);
}

// ---- Cell layout -----------------------------------------------------------

namespace {
// A cell cache and the two faces a caller holds: the runner drives the verbs,
// the backend asks for the step's plan.
struct Cells {
  explicit Cells(
      int capacity,
      std::vector<LayerConfig> layers = {flat_layer(), flat_layer()})
      : cache(CacheConfig{capacity, static_cast<int>(layers.size()), layers}),
        ctl(cache.as_batch_control()),
        planner(cache.as_cell_planner()) {}

  const CellStepPlan* step(
      std::vector<int32_t> seq_ids,
      std::vector<int32_t> positions) {
    if (!ctl->begin_step(seq_ids.data(), static_cast<int>(seq_ids.size()))) {
      return nullptr;
    }
    return planner->plan(
        0, positions.data(), static_cast<int>(positions.size()));
  }

  CellCache cache;
  BatchControl* ctl;
  CellPlanner* planner;
};

// The mask row for query `i`, as a string of '.' and '1' -- readable.
std::string row(const CellStepPlan& plan, int i) {
  std::string out(plan.read_len, '.');
  for (int j = 0; j < plan.read_len; ++j) {
    out[j] = plan.mask_bits[i * plan.read_len + j] ? '1' : '.';
  }
  return out;
}
} // namespace

TEST_F(CacheTest, CellSingleSequenceIsFused) {
  Cells c(16);
  const auto* prefill =
      c.step({0, 0, 0, 0}, {0, 1, 2, 3}); // seq 0 places 4 tokens at 0..3
  ASSERT_NE(prefill, nullptr);
  EXPECT_EQ(prefill->kind, MaskKind::Causal); // one run at the tail it owns
  EXPECT_EQ(prefill->write_start, 0);
  EXPECT_EQ(prefill->read_len, 4);
  EXPECT_TRUE(prefill->mask_bits.empty()); // fused kinds carry no mask

  const auto* decode = c.step({0}, {4}); // seq 0 places one more at 4
  ASSERT_NE(decode, nullptr);
  EXPECT_EQ(decode->kind, MaskKind::None); // one query over its own window
  EXPECT_EQ(decode->write_start, 4);
}

TEST_F(CacheTest, CellSecondSequenceForcesAnExplicitMask) {
  Cells c(16);
  c.step({0, 0, 0, 0}, {0, 1, 2, 3}); // seq 0 places 4 tokens at 0..3

  const auto* plan =
      c.step({1, 1}, {0, 1}); // seq 1 places 2 at positions seq 0 also holds
  ASSERT_NE(plan, nullptr);
  EXPECT_EQ(plan->kind, MaskKind::Explicit);
  EXPECT_EQ(plan->read_len, 6);
  EXPECT_EQ(plan->cells, (std::vector<int32_t>{4, 5}));
  // sequence 1 sees none of sequence 0's cells, though they share positions
  EXPECT_EQ(row(*plan, 0), "....1.");
  EXPECT_EQ(row(*plan, 1), "....11");
}

TEST_F(CacheTest, CellPlanIsSharedByEveryLayerOfTheStep) {
  Cells c(16);
  const int32_t seqs[2] = {0, 1};
  const int32_t pos[2] = {0, 0};
  ASSERT_TRUE(c.ctl->begin_step(seqs, 2));

  const auto* first = c.planner->plan(0, pos, 2); // layer 0 places the cells
  ASSERT_NE(first, nullptr);
  EXPECT_EQ(c.planner->plan(1, pos, 2), first); // later layers reuse them
  EXPECT_EQ(c.planner->plan(0, pos, 2), nullptr); // asking twice is a new step
  EXPECT_EQ(c.cache.free_cells(), 14); // placed once, not once per layer
}

TEST_F(CacheTest, CellForkSharesCellsAndEvictionRefcounts) {
  Cells c(16);
  c.step({0, 0, 0, 0}, {0, 1, 2, 3}); // seq 0 places 4 tokens at 0..3

  ASSERT_EQ(c.cache.free_cells(), 12); // the four it placed, out of sixteen

  c.ctl->seq_cp(0, 1, std::nullopt);
  EXPECT_EQ(c.cache.free_cells(), 12); // no cell, no byte copied
  EXPECT_EQ(c.ctl->seq_len(1), 4);
  EXPECT_EQ(c.ctl->next_pos(1), 4);

  c.ctl->seq_rm(0, 0, std::nullopt);
  EXPECT_EQ(c.ctl->seq_len(0), 0);
  EXPECT_EQ(c.ctl->seq_len(1), 4); // the fork still owns them
  EXPECT_EQ(c.cache.free_cells(), 12); // so nothing is reclaimed yet

  c.ctl->seq_rm(1, 0, std::nullopt);
  EXPECT_EQ(c.cache.free_cells(), 16);
  EXPECT_EQ(c.cache.used_end(), 0); // the extent comes back too
}

TEST_F(CacheTest, CellRangedRemovalFreesOnlyThatWindow) {
  Cells c(16);
  c.step({0, 0, 0, 0, 0}, {0, 1, 2, 3, 4}); // seq 0 places 5 tokens at 0..4

  c.ctl->seq_rm(0, 0, 2); // sliding window: drop the oldest two
  EXPECT_EQ(c.ctl->seq_len(0), 3);
  EXPECT_EQ(c.cache.free_cells(), 13);
  EXPECT_EQ(c.ctl->next_pos(0), 5); // a count is not a position

  c.ctl->seq_rm(0, 4, std::nullopt); // backtrack: drop position 4 onwards
  EXPECT_EQ(c.ctl->seq_len(0), 2);
  EXPECT_EQ(c.ctl->next_pos(0), 4);
}

TEST_F(CacheTest, CellRefillingHolesGivesUpTheFusedPath) {
  Cells c(16);
  c.step({0, 0, 0, 0}, {0, 1, 2, 3}); // seq 0 places 4 tokens at 0..3
  c.ctl->seq_rm(0, 0, 2); // free the oldest cells, leaving holes at 0 and 1

  // The new tokens take those holes, so the sequence owns the whole window
  // again -- but its cells no longer ascend with its positions.
  const auto* plan = c.step({0, 0}, {4, 5}); // seq 0 places two more at 4..5
  ASSERT_NE(plan, nullptr);
  EXPECT_EQ(plan->cells, (std::vector<int32_t>{0, 1}));
  EXPECT_EQ(plan->kind, MaskKind::Explicit);
}

TEST_F(CacheTest, CellRejectsAPositionASequenceStillHolds) {
  Cells c(16);
  c.step({0, 0, 0, 0}, {0, 1, 2, 3}); // seq 0 places 4 tokens at 0..3
  c.ctl->seq_rm(0, 3, std::nullopt); // free the newest cell

  // A step only extends its sequences. Writing 0 again would leave seq 0 with
  // two cells for one token and a window that is no longer a causal prefix.
  EXPECT_EQ(c.step({0}, {0}), nullptr);
  EXPECT_EQ(c.step({0, 0}, {5, 4}), nullptr); // nor may its own tokens descend
  EXPECT_NE(c.step({0}, {3}), nullptr); // the position it just freed is fine
}

TEST_F(CacheTest, CellLayersSharingAWindowShareAPlan) {
  // gemma-style: one flat layer beside two windowed ones. The placement is
  // shared, the plan is per policy -- so the flat layer still fuses while the
  // windowed pair gets one banded plan between them.
  Cells c(16, {flat_layer(), ring_layer(2), ring_layer(2)});
  const int32_t seqs[4] = {0, 0, 0, 0};
  const int32_t pos[4] = {0, 1, 2, 3};
  ASSERT_TRUE(c.ctl->begin_step(seqs, 4));

  const auto* flat = c.planner->plan(0, pos, 4);
  const auto* windowed = c.planner->plan(1, pos, 4);
  ASSERT_NE(flat, nullptr);
  ASSERT_NE(windowed, nullptr);
  EXPECT_EQ(flat->kind, MaskKind::Causal);
  EXPECT_TRUE(flat->mask_bits.empty());
  EXPECT_EQ(windowed->kind, MaskKind::Explicit);
  EXPECT_EQ(row(*windowed, 2), ".11."); // query 2 drops position 0

  EXPECT_EQ(c.planner->plan(2, pos, 4), windowed); // same window, same plan
  EXPECT_EQ(c.cache.free_cells(), 12); // placed once for all three layers
}

TEST_F(CacheTest, CellStepProtocolIsEnforced) {
  Cells c(4);
  const int32_t seqs[2] = {0, 0};
  const int32_t pos[2] = {0, 1};

  EXPECT_FALSE(c.ctl->begin_step(seqs, 0)); // no tokens
  const int32_t many[5] = {0, 0, 0, 0, 0};
  EXPECT_FALSE(c.ctl->begin_step(many, 5)); // 5 tokens, 4 cells
  const int32_t bad[1] = {CellCache::kMaxSeqs};
  EXPECT_FALSE(c.ctl->begin_step(bad, 1)); // seq id past the last owner bit

  EXPECT_EQ(c.planner->plan(0, pos, 2), nullptr); // no declaration
  ASSERT_TRUE(c.ctl->begin_step(seqs, 2));
  EXPECT_EQ(c.planner->plan(0, pos, 1), nullptr); // token count disagrees
  ASSERT_NE(c.planner->plan(0, pos, 2), nullptr); // 2 tokens, as declared

  const int32_t next[2] = {2, 3};
  EXPECT_EQ(
      c.planner->plan(0, next, 2), nullptr); // a second step, no begin_step
}
