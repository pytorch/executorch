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
#include <numeric>
#include <string>
#include <vector>

#include <executorch/runtime/core/error.h>
#include <executorch/runtime/platform/runtime.h>
#include <gtest/gtest.h>

using executorch::extension::llm::cache::BatchControl;
using executorch::extension::llm::cache::Cache;
using executorch::extension::llm::cache::CacheFactory;
using executorch::extension::llm::cache::CacheConfig;
using executorch::extension::llm::cache::CacheRegistry;
using executorch::extension::llm::cache::CacheLease;
using executorch::extension::llm::cache::CellCache;
using executorch::extension::llm::cache::CellStep;
using executorch::extension::llm::cache::CellStepper;
using executorch::extension::llm::cache::SequenceControl;
using executorch::extension::llm::cache::SequencePlanner;
namespace kind = executorch::extension::llm::cache::kind;
using executorch::extension::llm::cache::LayerConfig;
using executorch::extension::llm::cache::LayerPolicy;
using executorch::extension::llm::cache::new_cache_key;
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
  Cache* base = &cache;
  ASSERT_NE(base->as<SequenceControl>(), nullptr);
  ASSERT_NE(base->as<SequencePlanner>(), nullptr);
  EXPECT_TRUE(base->as<SequenceControl>()->can_extend(4));
  auto plan = base->as<SequencePlanner>()->plan(0, 0, 1);
  ASSERT_TRUE(plan.has_value());
  EXPECT_EQ(plan->read[0].len, 1);
}

TEST_F(CacheTest, RegistryInstallGetErase) {
  auto& reg = CacheRegistry::global();
  const std::string key = new_cache_key();
  EXPECT_EQ(reg.get(key), nullptr);

  std::shared_ptr<Cache> cache =
      std::make_shared<SequenceCache>(CacheConfig{16, 1, {flat_layer()}});
  reg.install(key, cache);
  EXPECT_EQ(reg.get(key), cache);
  EXPECT_TRUE(reg.get(key)->as<SequenceControl>()->can_extend(16));

  reg.erase(key);
  EXPECT_EQ(reg.get(key), nullptr);
}

TEST_F(CacheTest, UniqueKeysDoNotCollide) {
  EXPECT_NE(new_cache_key(), new_cache_key());
}

TEST_F(CacheTest, BuilderBuildsRegisteredKindElseError) {
  auto& reg = CacheFactory::global();
  reg.register_builder(
      "TestBackend", kind::kSingle, [](const CacheConfig& cfg) {
    return std::static_pointer_cast<Cache>(
        std::make_shared<SequenceCache>(cfg));
  });

  CacheConfig cfg{32, 1, {flat_layer()}};
  auto cache = reg.build("TestBackend", kind::kSingle, cfg);
  ASSERT_TRUE(cache.ok());
  EXPECT_EQ(cache.get()->as<SequenceControl>()->capacity(), 32);

  EXPECT_EQ(reg.build("TestBackend", "missing", cfg).error(), Error::NotFound);

  // A layers list that is neither size 1 nor n_layers would be indexed past
  // the end, so build refuses it before the cache is constructed.
  EXPECT_EQ(
      reg.build("TestBackend", kind::kSingle, CacheConfig{32, 3, {}}).error(),
      Error::InvalidArgument);
  EXPECT_EQ(
      reg.build(
             "TestBackend",
             kind::kSingle,
             CacheConfig{32, 3, {flat_layer(), flat_layer()}})
          .error(),
      Error::InvalidArgument);
}

TEST_F(CacheTest, SessionInstallsOnCtorErasesOnDtor) {
  std::string key;
  {
    CacheLease lease(
        std::make_shared<SequenceCache>(CacheConfig{4, 1, {flat_layer()}}));
    key = lease.key();
    EXPECT_NE(CacheRegistry::global().get(key), nullptr);
    EXPECT_TRUE(lease.cache()->as<SequenceControl>()->can_extend(4));
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
// One sequence's tokens in a step: `length` of them from `start_pos` onward.
struct SeqTokens {
  int32_t seq_id;
  int32_t start_pos;
  int length;
};

// The per-token arrays a step is made of.
struct StepArgs {
  std::vector<int32_t> seq_ids;
  std::vector<int32_t> positions;
};

// Lay a step's sequences on one token axis, building the two arrays together
// so they cannot fall out of alignment.
StepArgs flatten_step(std::initializer_list<SeqTokens> sequences) {
  StepArgs args;
  for (const SeqTokens& t : sequences) {
    for (int i = 0; i < t.length; ++i) {
      args.seq_ids.push_back(t.seq_id);
      args.positions.push_back(t.start_pos + i);
    }
  }
  return args;
}

// A cell cache and the two faces a caller holds: the runner drives the verbs,
// the backend places each step.
struct Cells {
  explicit Cells(
      int capacity,
      std::vector<LayerConfig> layers = {flat_layer(), flat_layer()})
      : cache(CacheConfig{capacity, static_cast<int>(layers.size()), layers}),
        ctl(cache.as<BatchControl>()),
        stepper(cache.as<CellStepper>()) {}

  // Ids come from the cache, so a test names sequences by allocating them.
  // Unlike the face's, this one unwraps and fails the test if none is free.
  int32_t seq_new() {
    const auto id = ctl->seq_new();
    EXPECT_TRUE(id) << "no free sequence id";
    return id ? *id : -1;
  }

  // One layer of an already-declared step.
  const CellStep* place(int layer, std::vector<int32_t> positions) {
    return stepper->place_step(
        layer, positions.data(), static_cast<int>(positions.size()));
  }

  // A whole single-layer step: declare it, then place it. A nullptr from here
  // is the placement's refusal -- admission failing is a test failure.
  const CellStep* step(
      std::vector<int32_t> seq_ids,
      std::vector<int32_t> positions) {
    EXPECT_TRUE(ctl->declare_step(seq_ids)) << "the step was not admitted";
    return place(/*layer=*/0, positions);
  }

  const CellStep* step(StepArgs args) {
    return step(std::move(args.seq_ids), std::move(args.positions));
  }

  CellCache cache;
  BatchControl* ctl;
  CellStepper* stepper;
};

// The mask row for query `i`, as a string of '.' and '1'.
std::string row(const CellStep& step, int i) {
  EXPECT_EQ(
      step.mask_bits.size(),
      static_cast<size_t>(step.length) * static_cast<size_t>(step.read_len));
  if (step.mask_bits.empty()) {
    return {};
  }
  std::string out(step.read_len, '.');
  for (int j = 0; j < step.read_len; ++j) {
    out[j] = step.mask_bits[i * step.read_len + j] ? '1' : '.';
  }
  return out;
}
} // namespace

TEST_F(CacheTest, CellSingleSequenceAttendsItsOwnPrefix) {
  Cells c(16);
  const int32_t s0 = c.seq_new();
  // seq 0 places 4 tokens at 0..3
  const auto* prefill = c.step(flatten_step({{s0, 0, 4}}));
  ASSERT_NE(prefill, nullptr);
  EXPECT_EQ(prefill->read_len, 4);
  EXPECT_EQ(prefill->cells, (std::vector<int32_t>{0, 1, 2, 3}));
  EXPECT_EQ(row(*prefill, 0), "1...");
  EXPECT_EQ(row(*prefill, 3), "1111");

  const auto* decode = c.step(flatten_step({{s0, 4, 1}})); // one more at 4
  ASSERT_NE(decode, nullptr);
  EXPECT_EQ(decode->cells, (std::vector<int32_t>{4}));
  EXPECT_EQ(row(*decode, 0), "11111"); // the whole history it owns
}

TEST_F(CacheTest, CellSecondSequenceForcesAnExplicitMask) {
  Cells c(16);
  const int32_t s0 = c.seq_new();
  const int32_t s1 = c.seq_new();
  c.step(flatten_step({{s0, 0, 4}})); // seq 0 places 4 tokens at 0..3

  const auto* step =
      // seq 1 places 2 at positions seq 0 also holds
      c.step(flatten_step({{s1, 0, 2}}));
  ASSERT_NE(step, nullptr);
  EXPECT_EQ(step->read_len, 6);
  EXPECT_EQ(step->cells, (std::vector<int32_t>{4, 5}));
  // sequence 1 sees none of sequence 0's cells, though they share positions
  EXPECT_EQ(row(*step, 0), "....1.");
  EXPECT_EQ(row(*step, 1), "....11");
}

TEST_F(CacheTest, CellBatchedPrefillKeepsSequencesApart) {
  Cells c(16);
  const int32_t s0 = c.seq_new();
  const int32_t s1 = c.seq_new();

  // One step, two prefills of different lengths on a single token axis.
  const auto* step = c.step(flatten_step({{s0, 0, 3}, {s1, 0, 2}}));
  ASSERT_NE(step, nullptr);
  EXPECT_EQ(step->cells, (std::vector<int32_t>{0, 1, 2, 3, 4}));
  EXPECT_EQ(step->read_len, 5);

  EXPECT_EQ(row(*step, 0), "1...."); // seq 0 causally over its own three
  EXPECT_EQ(row(*step, 1), "11...");
  EXPECT_EQ(row(*step, 2), "111..");
  EXPECT_EQ(row(*step, 3), "...1."); // seq 1 sees none of seq 0's, though
  EXPECT_EQ(row(*step, 4), "...11"); // they hold the same positions
}

TEST_F(CacheTest, CellDecodeAndPrefillShareOneStep) {
  Cells c(16);
  const int32_t s0 = c.seq_new();
  c.step(flatten_step({{s0, 0, 2}})); // an existing conversation at 0..1

  // Continuous batching: s0 decodes while s1 arrives and prefills.
  const int32_t s1 = c.seq_new();
  const auto* step = c.step(flatten_step({{s0, 2, 1}, {s1, 0, 2}}));
  ASSERT_NE(step, nullptr);
  EXPECT_EQ(step->cells, (std::vector<int32_t>{2, 3, 4}));

  EXPECT_EQ(row(*step, 0), "111.."); // s0's decode sees its own history
  EXPECT_EQ(row(*step, 1), "...1."); // s1 starts from nothing
  EXPECT_EQ(row(*step, 2), "...11");
  EXPECT_EQ(c.ctl->next_pos(s0), 3);
  EXPECT_EQ(c.ctl->next_pos(s1), 2);
}

TEST_F(CacheTest, CellExtendsIsCheckedPerSequence) {
  Cells c(16);
  const int32_t s0 = c.seq_new();
  const int32_t s1 = c.seq_new();
  c.step(flatten_step({{s0, 0, 1}, {s1, 0, 1}})); // both at position 0

  // One sequence advancing does not license another to repeat: the check is
  // against what each sequence itself already holds.
  EXPECT_EQ(c.step({s0, s1}, {1, 0}), nullptr);
  EXPECT_NE(c.step(flatten_step({{s0, 1, 1}, {s1, 1, 1}})), nullptr);
}

TEST_F(CacheTest, CellPlacementIsSharedByEveryLayerOfTheStep) {
  Cells c(16);
  const int32_t s0 = c.seq_new();
  const int32_t s1 = c.seq_new();
  const auto args = flatten_step({{s0, 0, 1}, {s1, 0, 1}});
  ASSERT_TRUE(c.ctl->declare_step(args.seq_ids));

  const auto* first = c.place(0, args.positions); // layer 0 places the cells
  ASSERT_NE(first, nullptr);
  EXPECT_EQ(c.place(1, args.positions), first); // later layers reuse them
  EXPECT_EQ(c.place(0, args.positions), nullptr); // asking twice is a new step
  EXPECT_EQ(c.cache.free_cells(), 14); // placed once, not once per layer
}

TEST_F(CacheTest, CellForkSharesCellsAndEvictionRefcounts) {
  Cells c(16);
  const int32_t s0 = c.seq_new();
  c.step(flatten_step({{s0, 0, 4}})); // seq 0 places 4 tokens at 0..3

  ASSERT_EQ(c.cache.free_cells(), 12); // the four it placed, out of sixteen

  const auto s1 = c.ctl->seq_clone(s0, std::nullopt);
  ASSERT_TRUE(s1);
  EXPECT_EQ(c.cache.free_cells(), 12); // no cell, no byte copied
  EXPECT_EQ(c.ctl->seq_len(*s1), 4);
  EXPECT_EQ(c.ctl->next_pos(*s1), 4);

  c.ctl->seq_rm(s0, 0, std::nullopt);
  EXPECT_EQ(c.ctl->seq_len(s0), 0);
  EXPECT_EQ(c.ctl->seq_len(*s1), 4); // the fork still owns them
  EXPECT_EQ(c.cache.free_cells(), 12); // so nothing is reclaimed yet

  c.ctl->seq_rm(*s1, 0, std::nullopt);
  EXPECT_EQ(c.cache.free_cells(), 16);
  EXPECT_EQ(c.cache.used_end(), 0); // the extent comes back too
}

TEST_F(CacheTest, CellSeqNewHandsOutIdsUntilTheyAreReleased) {
  Cells c(16);
  const auto a = c.ctl->seq_new();
  const auto b = c.ctl->seq_new();
  ASSERT_TRUE(a && b);
  EXPECT_NE(*a, *b);

  // Reserved before it holds anything, so the next call cannot hand it out.
  EXPECT_EQ(c.ctl->seq_len(*a), 0);
  EXPECT_NE(*c.ctl->seq_new(), *a);

  // Removing everything a sequence holds returns its id, whether or not it
  // ever held a slot.
  ASSERT_TRUE(c.ctl->seq_rm(*a, 0, std::nullopt));
  EXPECT_EQ(*c.ctl->seq_new(), *a);

  // An id nobody was handed does not start a sequence.
  EXPECT_FALSE(c.ctl->declare_step({40}));

  while (c.ctl->seq_new()) {
  }
  EXPECT_FALSE(c.ctl->seq_new()); // every id is now in use

  c.ctl->clear();
  EXPECT_EQ(*c.ctl->seq_new(), 0);
}

TEST_F(CacheTest, CellForkCanShareAPrefixOnly) {
  Cells c(16);
  const int32_t s0 = c.seq_new();
  c.step(flatten_step({{s0, 0, 4}})); // positions 0..3

  const auto s1 = c.ctl->seq_clone(s0, /*upto=*/2); // positions 0..1 only
  ASSERT_TRUE(s1);
  EXPECT_EQ(c.ctl->seq_len(*s1), 2);
  EXPECT_EQ(c.ctl->next_pos(*s1), 2); // the fork resumes where the prefix ends
  EXPECT_EQ(c.ctl->seq_len(s0), 4); // the source keeps all of its own
  EXPECT_EQ(c.cache.free_cells(), 12); // still no cell copied

  // Removing the source's shared range frees nothing: the fork still owns it.
  ASSERT_TRUE(c.ctl->seq_rm(s0, 0, 2));
  EXPECT_EQ(c.cache.free_cells(), 12);
  EXPECT_EQ(c.ctl->seq_len(*s1), 2);
}

TEST_F(CacheTest, CellWindowAndSequenceBothNarrowTheMask) {
  // The two mask bounds together: a query sees only its own sequence, and only
  // inside its window.
  Cells c(16, {ring_layer(2), ring_layer(2)});
  const int32_t s0 = c.seq_new();
  const int32_t s1 = c.seq_new();
  c.step(flatten_step({{s0, 0, 3}})); // seq 0 at 0..2
  const auto* step = c.step(flatten_step({{s1, 0, 1}, {s0, 3, 1}}));
  ASSERT_NE(step, nullptr);
  EXPECT_EQ(step->read_len, 5);

  // Seq 1 holds only its own new cell, and its window reaches no further.
  EXPECT_EQ(row(*step, 0), "...1.");
  // Seq 0 at position 3 sees positions 2 and 3 -- cells 2 and 4 -- but not its
  // own cells 0 and 1, which the window excludes.
  EXPECT_EQ(row(*step, 1), "..1.1");
}

TEST_F(CacheTest, CellVerbBetweenLayersInvalidatesTheStep) {
  Cells c(16);
  const int32_t s0 = c.seq_new();
  c.step(flatten_step({{s0, 0, 2}}));

  ASSERT_TRUE(c.ctl->declare_step({s0}));
  ASSERT_NE(c.place(0, {2}), nullptr);
  // A verb rebuilds the table under the step: the placement no longer stands
  // and the remaining layers are refused.
  ASSERT_TRUE(c.ctl->seq_rm(s0, 0, 1));
  EXPECT_EQ(c.place(1, {2}), nullptr);
}

TEST_F(CacheTest, CellRangedRemovalFreesOnlyThatWindow) {
  Cells c(16);
  const int32_t s0 = c.seq_new();
  c.step(flatten_step({{s0, 0, 5}})); // seq 0 places 5 tokens at 0..4

  c.ctl->seq_rm(s0, 0, 2); // sliding window: drop the oldest two
  EXPECT_EQ(c.ctl->seq_len(s0), 3);
  EXPECT_EQ(c.cache.free_cells(), 13);
  EXPECT_EQ(c.ctl->next_pos(s0), 5); // a count is not a position

  c.ctl->seq_rm(s0, 4, std::nullopt); // backtrack: drop position 4 onwards
  EXPECT_EQ(c.ctl->seq_len(s0), 2);
  EXPECT_EQ(c.ctl->next_pos(s0), 4);
}

TEST_F(CacheTest, CellRefillingHolesKeepsTheMaskOnPositions) {
  Cells c(16);
  const int32_t s0 = c.seq_new();
  c.step(flatten_step({{s0, 0, 4}})); // seq 0 places 4 tokens at 0..3
  c.ctl->seq_rm(s0, 0, 2); // free the oldest cells, leaving holes at 0 and 1

  // The new tokens take those holes, so the sequence's cells no longer ascend
  // with its positions -- the mask keys off pos/owners, never the index.
  // seq 0 places two more at 4..5
  const auto* step = c.step(flatten_step({{s0, 4, 2}}));
  ASSERT_NE(step, nullptr);
  EXPECT_EQ(step->cells, (std::vector<int32_t>{0, 1}));
  // cells 0,1 hold positions 4,5; cells 2,3 hold 2,3
  EXPECT_EQ(row(*step, 0), "1.11"); // query at 4 sees 2, 3 and itself
  EXPECT_EQ(row(*step, 1), "1111"); // query at 5 sees all of them
}

TEST_F(CacheTest, CellRejectsAPositionASequenceStillHolds) {
  Cells c(16);
  const int32_t s0 = c.seq_new();
  c.step(flatten_step({{s0, 0, 4}})); // seq 0 places 4 tokens at 0..3
  c.ctl->seq_rm(s0, 3, std::nullopt); // free the newest cell

  // A step only extends its sequences. Writing 0 again would leave seq 0 with
  // two cells for one token and a window that is no longer a causal prefix.
  EXPECT_EQ(c.step({s0}, {0}), nullptr);
  // nor may its own tokens descend
  EXPECT_EQ(c.step({s0, s0}, {5, 4}), nullptr);
  EXPECT_NE(c.step({s0}, {3}), nullptr); // the position it just freed is fine

  // A refusal places nothing, so the declaration stands and the same step can
  // be placed again with corrected positions.
  ASSERT_TRUE(c.ctl->declare_step({s0}));
  EXPECT_EQ(c.place(0, {0}), nullptr);
  EXPECT_NE(c.place(0, {4}), nullptr);
}

TEST_F(CacheTest, CellWindowBoundsEachQueryByPosition) {
  {
    Cells c(16, {ring_layer(4)});
    // span 3 < window 4: the last query still sees position 0.
    EXPECT_EQ(row(*c.step(flatten_step({{c.seq_new(), 0, 4}})), 3), "1111");
  }
  {
    Cells c(16, {ring_layer(3)});
    // span 3 == window 3: position 0 is one too old for the query at 3.
    EXPECT_EQ(row(*c.step(flatten_step({{c.seq_new(), 0, 4}})), 3), ".111");
  }
  {
    Cells c(16, {ring_layer(2)});
    // Two cells, five positions apart: the span is what counts.
    const int32_t s = c.seq_new();
    c.step({s}, {0});
    EXPECT_EQ(row(*c.step({s}, {5}), 0), ".1");
  }
  {
    Cells c(16, {ring_layer(8)});
    // Sparse but inside the window, so both cells stay visible.
    const int32_t s = c.seq_new();
    c.step({s}, {0});
    EXPECT_EQ(row(*c.step({s}, {5}), 0), "11");
  }
}

TEST_F(CacheTest, CellLayersSharingAWindowShareAStep) {
  // gemma-style: one flat layer beside two windowed ones. The placement is
  // shared, the step is per policy -- so the flat layer keeps the whole prefix
  // while the windowed pair gets one banded step between them.
  Cells c(16, {flat_layer(), ring_layer(2), ring_layer(2)});
  const int32_t s0 = c.seq_new();
  const auto args = flatten_step({{s0, 0, 4}});
  ASSERT_TRUE(c.ctl->declare_step(args.seq_ids));

  const auto* flat = c.place(0, args.positions);
  const auto* windowed = c.place(1, args.positions);
  ASSERT_NE(flat, nullptr);
  ASSERT_NE(windowed, nullptr);
  EXPECT_EQ(row(*flat, 2), "111."); // the flat layer keeps position 0
  EXPECT_EQ(row(*windowed, 2), ".11."); // the windowed one drops it

  EXPECT_EQ(c.place(2, args.positions), windowed); // same window, same step
  EXPECT_EQ(c.cache.free_cells(), 12); // placed once for all three layers
}

TEST_F(CacheTest, CellStepProtocolIsEnforced) {
  Cells c(4);
  const int32_t s0 = c.seq_new();
  const std::vector<int32_t> seqs{s0, s0};
  EXPECT_FALSE(c.ctl->declare_step({})); // no tokens
  EXPECT_FALSE(c.ctl->declare_step({s0, s0, s0, s0, s0})); // 5 tokens, 4 cells
  EXPECT_FALSE(
      c.ctl->declare_step({CellCache::kMaxSeqs})); // seq id past the last bit
  // The verbs report a bad sequence rather than doing nothing quietly.
  EXPECT_FALSE(c.ctl->seq_clone(CellCache::kMaxSeqs, std::nullopt));
  EXPECT_FALSE(c.ctl->seq_clone(s0 + 30, std::nullopt)); // src holds nothing
  EXPECT_FALSE(c.ctl->seq_rm(-1, 0, std::nullopt));

  EXPECT_EQ(c.place(0, {0, 1}), nullptr); // no declaration
  ASSERT_TRUE(c.ctl->declare_step(seqs));
  EXPECT_EQ(c.place(0, {0}), nullptr); // token count disagrees
  ASSERT_NE(c.place(0, {0, 1}), nullptr); // 2 tokens, as declared

  EXPECT_EQ(c.place(0, {2, 3}), nullptr); // a second step, no declare_step

  EXPECT_EQ(c.place(-1, {0, 1}), nullptr); // layers outside the model
  EXPECT_EQ(c.place(2, {0, 1}), nullptr);
}

TEST_F(CacheTest, CellClearReturnsEveryCell) {
  Cells c(4);
  const int32_t s0 = c.seq_new();
  EXPECT_EQ(c.cache.capacity(), 4);
  EXPECT_TRUE(c.ctl->can_extend(4));

  c.step(flatten_step({{s0, 0, 3}}));
  EXPECT_FALSE(c.ctl->can_extend(2)); // one cell left
  EXPECT_EQ(c.ctl->seq_len(s0), 3);

  c.ctl->clear();
  EXPECT_TRUE(c.ctl->can_extend(4));
  EXPECT_EQ(c.cache.free_cells(), 4);
  EXPECT_EQ(c.cache.used_end(), 0);
  EXPECT_EQ(c.ctl->seq_len(s0), 0);
  EXPECT_EQ(c.ctl->next_pos(s0), 0); // the sequence is gone
  EXPECT_EQ(c.place(0, {0}), nullptr); // and the step went with it
}
