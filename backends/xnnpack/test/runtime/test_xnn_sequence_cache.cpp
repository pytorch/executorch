/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <gtest/gtest.h>

#include <executorch/backends/xnnpack/runtime/XnnSequenceCache.h>
#include <executorch/runtime/platform/runtime.h>

#include <random>
#include <vector>

using executorch::aten::ScalarType;
using executorch::backends::xnnpack::AttendSpec;
using executorch::backends::xnnpack::XnnSequenceCache;
using executorch::runtime::Error;

namespace cache = executorch::extension::llm::cache;

namespace {

constexpr int kHeads = 3;
constexpr int kDim = 4;

cache::CacheConfig
config(int capacity, int n_layers = 1, int initial_capacity = 4) {
  cache::CacheConfig cfg;
  cfg.capacity = capacity;
  cfg.n_layers = n_layers;
  cfg.layers = {cache::LayerConfig{{}, kHeads, kDim}};
  cfg.kv_dtype = static_cast<int>(ScalarType::Float);
  cfg.initial_capacity = initial_capacity;
  return cfg;
}

// A dense [1, heads, n_tok, dim] step. The values only have to be distinct from
// every other step's for a misplaced copy to show up, so they are random; the
// seed is explicit so a failure reproduces.
std::vector<float>
random_step(int n_tok, uint32_t seed, int heads = kHeads, int dim = kDim) {
  std::mt19937 rng(seed);
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
  std::vector<float> out(static_cast<size_t>(heads) * n_tok * dim);
  for (float& x : out) {
    x = dist(rng);
  }
  return out;
}

// Read slot `slot` of head `h`, exactly as a consumer reading the pool as a
// [1, kHeads, slots, kDim] tensor would.
float pool_at(const void* pool, int slots, int h, int slot, int d) {
  return static_cast<const float*>(pool)[(h * slots + slot) * kDim + d];
}

// The first `n_check` tokens of `step` sit at pool slots [start,
// start+n_check). n_check defaults to all of them; a shorter count checks the
// prefix that survived a rewind.
void expect_step_at(
    const void* pool,
    int slots,
    int start,
    const std::vector<float>& step,
    int n_check = -1) {
  const int n_tok = static_cast<int>(step.size()) / (kHeads * kDim);
  if (n_check < 0) {
    n_check = n_tok;
  }
  for (int h = 0; h < kHeads; ++h) {
    for (int t = 0; t < n_check; ++t) {
      for (int d = 0; d < kDim; ++d) {
        EXPECT_FLOAT_EQ(
            pool_at(pool, slots, h, start + t, d),
            step[(h * n_tok + t) * kDim + d])
            << "head " << h << " slot " << (start + t) << " dim " << d;
      }
    }
  }
}

} // namespace

class XnnSequenceCacheTest : public ::testing::Test {
 protected:
  void SetUp() override {
    executorch::runtime::runtime_init();
  }
};

TEST_F(XnnSequenceCacheTest, RejectsInvalidConfig) {
  cache::CacheConfig cfg = config(/*capacity=*/0);
  EXPECT_EQ(XnnSequenceCache::create(cfg).error(), Error::InvalidArgument);
}

// `layers` is indexed directly under the size-1 broadcast rule, so a list that
// is neither 1 nor n_layers would read past the end.
TEST_F(XnnSequenceCacheTest, RejectsPartialLayerList) {
  cache::CacheConfig cfg = config(/*capacity=*/32, /*n_layers=*/4);
  cfg.layers.push_back(cfg.layers.front()); // size 2, neither 1 nor 4
  EXPECT_EQ(XnnSequenceCache::create(cfg).error(), Error::InvalidArgument);
}

TEST_F(XnnSequenceCacheTest, RejectsNegativeInitialCapacity) {
  cache::CacheConfig cfg = config(/*capacity=*/32, 1, /*initial=*/-1);
  EXPECT_EQ(XnnSequenceCache::create(cfg).error(), Error::InvalidArgument);
}

// An empty pool has no buffer to copy from when it first grows.
TEST_F(XnnSequenceCacheTest, ZeroInitialCapacityGrowsOnFirstWrite) {
  auto created =
      XnnSequenceCache::create(config(/*capacity=*/32, 1, /*initial=*/0));
  ASSERT_EQ(created.error(), Error::Ok);
  auto& kv = *created.get();

  const std::vector<float> k = random_step(3, /*seed=*/1);
  auto result = kv.update_and_fetch(0, 0, k.data(), k.data(), 3);
  ASSERT_EQ(result.error(), Error::Ok);
  const AttendSpec spec = result.get();
  EXPECT_EQ(spec.valid_len, 3);
  EXPECT_GE(spec.slots, 3);
  expect_step_at(spec.k, spec.slots, 0, k);
}

TEST_F(XnnSequenceCacheTest, RejectsRingLayer) {
  cache::CacheConfig cfg = config(/*capacity=*/16);
  cfg.layers = {cache::LayerConfig{
      {cache::LayerPolicy::Kind::Ring, /*window=*/8}, kHeads, kDim}};
  EXPECT_EQ(XnnSequenceCache::create(cfg).error(), Error::NotSupported);
}

TEST_F(XnnSequenceCacheTest, RejectsUnsupportedDtype) {
  cache::CacheConfig cfg = config(/*capacity=*/16);
  cfg.kv_dtype = static_cast<int>(ScalarType::Int);
  EXPECT_EQ(XnnSequenceCache::create(cfg).error(), Error::NotSupported);
}

TEST_F(XnnSequenceCacheTest, AcceptsHalfAndBFloat16) {
  for (ScalarType dtype : {ScalarType::Half, ScalarType::BFloat16}) {
    cache::CacheConfig cfg = config(/*capacity=*/16);
    cfg.kv_dtype = static_cast<int>(dtype);
    EXPECT_EQ(XnnSequenceCache::create(cfg).error(), Error::Ok);
  }
}

// THE parity test. Drive the cache through a realistic step sequence --
// prefill, chunked prefill, decodes, crossing several pool doublings -- while
// an independent naive model just appends each step to a flat per-head history.
// After every step the pool, read as [1, H, slots, D] over [0, valid_len), must
// equal that history. This is what a stride, growth or re-layout bug breaks.
TEST_F(XnnSequenceCacheTest, ParityWithNaiveAppendedHistory) {
  auto created =
      XnnSequenceCache::create(config(/*capacity=*/256, 1, /*initial=*/2));
  ASSERT_EQ(created.error(), Error::Ok);
  auto& kv = *created.get();

  // Naive reference: history[h] is head h's cells, appended in order.
  std::vector<std::vector<float>> history(kHeads);

  const int steps[] = {5, 1, 1, 3, 1, 1, 1, 1, 8, 1, 1, 17, 1};
  int position = 0;
  uint32_t seed = 1;

  for (int n_tok : steps) {
    const std::vector<float> k = random_step(n_tok, seed++);
    const std::vector<float> v = random_step(n_tok, seed++);
    for (int h = 0; h < kHeads; ++h) {
      for (int t = 0; t < n_tok; ++t) {
        const float* src = &k[(h * n_tok + t) * kDim];
        history[h].insert(history[h].end(), src, src + kDim);
      }
    }

    auto result = kv.update_and_fetch(0, position, k.data(), v.data(), n_tok);
    ASSERT_EQ(result.error(), Error::Ok) << "position " << position;
    const AttendSpec spec = result.get();

    position += n_tok;
    ASSERT_EQ(spec.valid_len, position);
    ASSERT_GE(spec.slots, spec.valid_len);

    for (int h = 0; h < kHeads; ++h) {
      for (int s = 0; s < spec.valid_len; ++s) {
        for (int d = 0; d < kDim; ++d) {
          ASSERT_FLOAT_EQ(
              pool_at(spec.k, spec.slots, h, s, d), history[h][s * kDim + d])
              << "after position " << position << ": head " << h << " slot "
              << s << " dim " << d;
        }
      }
    }
  }
}

TEST_F(XnnSequenceCacheTest, PrefillThenDecode) {
  auto created = XnnSequenceCache::create(config(/*capacity=*/64));
  ASSERT_EQ(created.error(), Error::Ok);
  auto& kv = *created.get();

  const std::vector<float> k0 = random_step(/*n_tok=*/3, /*seed=*/1);
  const std::vector<float> v0 = random_step(/*n_tok=*/3, /*seed=*/2);
  auto prefill_result =
      kv.update_and_fetch(/*layer=*/0, /*position=*/0, k0.data(), v0.data(), 3);
  ASSERT_EQ(prefill_result.error(), Error::Ok);
  const AttendSpec prefill_spec = prefill_result.get();
  EXPECT_EQ(prefill_spec.valid_len, 3);
  EXPECT_EQ(prefill_spec.kind, AttendSpec::Mask::Causal);
  expect_step_at(prefill_spec.k, prefill_spec.slots, 0, k0);
  expect_step_at(prefill_spec.v, prefill_spec.slots, 0, v0);

  const std::vector<float> k1 = random_step(/*n_tok=*/1, /*seed=*/3);
  const std::vector<float> v1 = random_step(/*n_tok=*/1, /*seed=*/4);
  auto decode_result =
      kv.update_and_fetch(/*layer=*/0, /*position=*/3, k1.data(), v1.data(), 1);
  ASSERT_EQ(decode_result.error(), Error::Ok);
  const AttendSpec decode_spec = decode_result.get();
  EXPECT_EQ(decode_spec.valid_len, 4);
  EXPECT_EQ(decode_spec.kind, AttendSpec::Mask::None);
  // The decode token appended after the prefill, which is still intact.
  expect_step_at(decode_spec.k, decode_spec.slots, 3, k1);
  expect_step_at(decode_spec.k, decode_spec.slots, 0, k0);
  expect_step_at(decode_spec.v, decode_spec.slots, 3, v1);
  expect_step_at(decode_spec.v, decode_spec.slots, 0, v0);
}

// The cache decides how its window is attended. A flat layer only ever needs
// the two flag-expressible kinds; Explicit arrives with the windowed and tree
// patterns that need it.
TEST_F(XnnSequenceCacheTest, FlatLayerDeclaresItsOwnMasking) {
  auto created = XnnSequenceCache::create(config(/*capacity=*/64));
  ASSERT_EQ(created.error(), Error::Ok);
  auto& kv = *created.get();

  int position = 0;
  for (int n_tok : {4, 1, 1, 3, 1}) {
    const std::vector<float> k = random_step(n_tok, /*seed=*/1);
    auto result = kv.update_and_fetch(0, position, k.data(), k.data(), n_tok);
    ASSERT_EQ(result.error(), Error::Ok);
    const AttendSpec spec = result.get();
    EXPECT_EQ(
        spec.kind,
        n_tok == 1 ? AttendSpec::Mask::None : AttendSpec::Mask::Causal)
        << "n_tok " << n_tok;
    EXPECT_EQ(spec.mask, nullptr) << "flat never returns Explicit";
    position += n_tok;
  }
}

// Growth widens the row stride, so every head's history moves to a new offset.
TEST_F(XnnSequenceCacheTest, GrowthPreservesHistoryAcrossHeads) {
  auto created =
      XnnSequenceCache::create(config(/*capacity=*/64, 1, /*initial=*/2));
  ASSERT_EQ(created.error(), Error::Ok);
  auto& kv = *created.get();

  const std::vector<float> k0 = random_step(2, /*seed=*/1);
  auto first_result = kv.update_and_fetch(0, 0, k0.data(), k0.data(), 2);
  ASSERT_EQ(first_result.error(), Error::Ok);
  ASSERT_EQ(first_result.get().slots, 2);

  // Six more tokens forces two doublings (2 -> 4 -> 8).
  const std::vector<float> k1 = random_step(6, /*seed=*/2);
  auto grown_result = kv.update_and_fetch(0, 2, k1.data(), k1.data(), 6);
  ASSERT_EQ(grown_result.error(), Error::Ok);
  const AttendSpec grown = grown_result.get();
  EXPECT_EQ(grown.slots, 8);
  EXPECT_EQ(grown.valid_len, 8);

  expect_step_at(grown.k, grown.slots, 0, k0);
  expect_step_at(grown.k, grown.slots, 2, k1);
}

// Growth reallocates, so a caller must re-read the pointers each step instead
// of caching them. The old and new buffers are alive together during the copy,
// so the address is guaranteed to change.
TEST_F(XnnSequenceCacheTest, GrowthMovesThePoolPointer) {
  auto created =
      XnnSequenceCache::create(config(/*capacity=*/64, 1, /*initial=*/2));
  ASSERT_EQ(created.error(), Error::Ok);
  auto& kv = *created.get();

  const std::vector<float> k = random_step(1, /*seed=*/1);
  auto first_result = kv.update_and_fetch(0, 0, k.data(), k.data(), 1);
  ASSERT_EQ(first_result.error(), Error::Ok);
  const AttendSpec first = first_result.get();
  const void* before = first.k;
  const int slots_before = first.slots;

  int position = 1;
  const void* after = before;
  int slots_after = slots_before;
  while (slots_after == slots_before) { // step until the pool doubles
    auto result = kv.update_and_fetch(0, position++, k.data(), k.data(), 1);
    ASSERT_EQ(result.error(), Error::Ok);
    after = result.get().k;
    slots_after = result.get().slots;
  }
  EXPECT_GT(slots_after, slots_before);
  EXPECT_NE(after, before);
}

TEST_F(XnnSequenceCacheTest, GrowthClampsToCapacity) {
  auto created =
      XnnSequenceCache::create(config(/*capacity=*/12, 1, /*initial=*/2));
  ASSERT_EQ(created.error(), Error::Ok);
  auto& kv = *created.get();

  const std::vector<float> k = random_step(12, /*seed=*/1);
  auto result = kv.update_and_fetch(0, 0, k.data(), k.data(), 12);
  ASSERT_EQ(result.error(), Error::Ok);
  const AttendSpec spec = result.get();
  EXPECT_EQ(spec.slots, 12);
  EXPECT_EQ(spec.valid_len, 12);
  expect_step_at(spec.k, spec.slots, 0, k);
}

// A per-layer `layers` list sizes each layer's pools from its own entry. Every
// other test passes a single entry, which takes the broadcast path instead.
TEST_F(XnnSequenceCacheTest, PerLayerHeadsAndDims) {
  cache::CacheConfig cfg = config(/*capacity=*/16, /*n_layers=*/2);
  cfg.layers = {
      cache::LayerConfig{{}, /*n_kv_heads=*/2, /*head_dim=*/4},
      cache::LayerConfig{{}, /*n_kv_heads=*/5, /*head_dim=*/8}};
  auto created = XnnSequenceCache::create(cfg);
  ASSERT_EQ(created.error(), Error::Ok);
  auto& kv = *created.get();

  constexpr int kTokens = 3;
  for (int layer = 0; layer < 2; ++layer) {
    const int heads = cfg.layers[layer].n_kv_heads;
    const int dim = cfg.layers[layer].head_dim;

    const std::vector<float> k =
        random_step(kTokens, /*seed=*/layer + 1, heads, dim);

    auto result = kv.update_and_fetch(layer, 0, k.data(), k.data(), kTokens);
    ASSERT_EQ(result.error(), Error::Ok) << "layer " << layer;
    const AttendSpec spec = result.get();
    ASSERT_EQ(spec.valid_len, kTokens);

    const auto* pool = static_cast<const float*>(spec.k);
    for (int h = 0; h < heads; ++h) {
      for (int t = 0; t < kTokens; ++t) {
        for (int d = 0; d < dim; ++d) {
          EXPECT_FLOAT_EQ(
              pool[(h * spec.slots + t) * dim + d],
              k[(h * kTokens + t) * dim + d])
              << "layer " << layer << " head " << h << " slot " << t << " dim "
              << d;
        }
      }
    }
  }
}

// One controller and one logical length, but each layer owns its own bytes.
TEST_F(XnnSequenceCacheTest, LayersAreIndependent) {
  auto created = XnnSequenceCache::create(config(/*capacity=*/64, 3));
  ASSERT_EQ(created.error(), Error::Ok);
  auto& kv = *created.get();

  const std::vector<float> ka = random_step(2, /*seed=*/1);
  const std::vector<float> kb = random_step(2, /*seed=*/2);
  auto la_result = kv.update_and_fetch(0, 0, ka.data(), ka.data(), 2);
  auto lb_result = kv.update_and_fetch(2, 0, kb.data(), kb.data(), 2);
  ASSERT_EQ(la_result.error(), Error::Ok);
  ASSERT_EQ(lb_result.error(), Error::Ok);
  const AttendSpec la = la_result.get();
  const AttendSpec lb = lb_result.get();

  EXPECT_NE(la.k, lb.k);
  expect_step_at(la.k, la.slots, 0, ka);
  expect_step_at(lb.k, lb.slots, 0, kb);

  // commit() is idempotent across the layers of one step, so the shared logical
  // length advanced once.
  EXPECT_EQ(la.valid_len, 2);
  EXPECT_EQ(lb.valid_len, 2);
}

TEST_F(XnnSequenceCacheTest, KeyAndValueAreSeparate) {
  auto created = XnnSequenceCache::create(config(/*capacity=*/64));
  ASSERT_EQ(created.error(), Error::Ok);
  auto& kv = *created.get();

  const std::vector<float> k = random_step(4, /*seed=*/1);
  const std::vector<float> v = random_step(4, /*seed=*/2);
  auto result = kv.update_and_fetch(0, 0, k.data(), v.data(), 4);
  ASSERT_EQ(result.error(), Error::Ok);
  const AttendSpec spec = result.get();
  EXPECT_NE(spec.k, spec.v);
  expect_step_at(spec.k, spec.slots, 0, k);
  expect_step_at(spec.v, spec.slots, 0, v);
}

TEST_F(XnnSequenceCacheTest, RejectsStepPastCapacity) {
  auto created = XnnSequenceCache::create(config(/*capacity=*/4));
  ASSERT_EQ(created.error(), Error::Ok);
  auto& kv = *created.get();

  const std::vector<float> k = random_step(6, /*seed=*/1);
  EXPECT_EQ(
      kv.update_and_fetch(0, 0, k.data(), k.data(), 6).error(),
      Error::InvalidArgument);
}

TEST_F(XnnSequenceCacheTest, RejectsBadLayer) {
  auto created = XnnSequenceCache::create(config(/*capacity=*/16, 2));
  ASSERT_EQ(created.error(), Error::Ok);
  auto& kv = *created.get();

  const std::vector<float> k = random_step(1, /*seed=*/1);
  EXPECT_EQ(
      kv.update_and_fetch(2, 0, k.data(), k.data(), 1).error(),
      Error::InvalidArgument);
  EXPECT_EQ(
      kv.update_and_fetch(-1, 0, k.data(), k.data(), 1).error(),
      Error::InvalidArgument);
}

// Rewind truncates without moving bytes; the next step overwrites in place and
// the read window shrinks to match.
TEST_F(XnnSequenceCacheTest, RewindThenOverwrite) {
  auto created = XnnSequenceCache::create(config(/*capacity=*/64));
  ASSERT_EQ(created.error(), Error::Ok);
  auto& kv = *created.get();

  const std::vector<float> k0 = random_step(4, /*seed=*/1);
  ASSERT_EQ(
      kv.update_and_fetch(0, 0, k0.data(), k0.data(), 4).error(), Error::Ok);
  ASSERT_TRUE(kv.as_control()->rewind(2));

  const std::vector<float> k1 = random_step(1, /*seed=*/2);
  auto result = kv.update_and_fetch(0, 2, k1.data(), k1.data(), 1);
  ASSERT_EQ(result.error(), Error::Ok);
  const AttendSpec spec = result.get();
  EXPECT_EQ(spec.valid_len, 3);
  expect_step_at(spec.k, spec.slots, 0, k0, /*n_check=*/2);
  expect_step_at(spec.k, spec.slots, 2, k1);
}

TEST_F(XnnSequenceCacheTest, ClearResetsLength) {
  auto created = XnnSequenceCache::create(config(/*capacity=*/64));
  ASSERT_EQ(created.error(), Error::Ok);
  auto& kv = *created.get();

  const std::vector<float> k0 = random_step(6, /*seed=*/1);
  auto before_result = kv.update_and_fetch(0, 0, k0.data(), k0.data(), 6);
  ASSERT_EQ(before_result.error(), Error::Ok);
  const int slots_before = before_result.get().slots;

  kv.as_control()->clear();

  const std::vector<float> k1 = random_step(2, /*seed=*/2);
  auto after_result = kv.update_and_fetch(0, 0, k1.data(), k1.data(), 2);
  ASSERT_EQ(after_result.error(), Error::Ok);
  const AttendSpec after = after_result.get();
  EXPECT_EQ(after.valid_len, 2);
  EXPECT_EQ(after.slots, slots_before);
  expect_step_at(after.k, after.slots, 0, k1);
}
