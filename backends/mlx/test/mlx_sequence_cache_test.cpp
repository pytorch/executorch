/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

// Op-level test for the off-graph KV cache (MLXSequenceCache / Pool).
//
// Drives MLXSequenceCache::update_and_fetch directly (no interpreter / .pte)
// and checks the AttendSpec it returns against the K/V history the cache should
// have assembled -- verifying the plan/write/read window and the mask kind
// across prefill (Causal) and decode (None), plus the capacity-reject and
// storage-dtype paths. The window is compared directly rather than through
// SDPA: attending both sides would only test equality through a lossy kernel.
//
// Must run on Apple Silicon: MLX needs the Metal backend.

#include "MLXSequenceCache.h"
#include "utils.h" // allclose, flat_config, ring_config

#include <mlx/mlx.h>

#include <gtest/gtest.h>

#include <numeric>
#include <optional>
#include <vector>

using namespace ::executorch::backends::mlx;
namespace cache = ::executorch::extension::llm::cache;
using ::mlx::core::array;

namespace {

// A step's positions: the contiguous run of k.shape(2) tokens starting at
// `start`, which is what every single-sequence step is.
AttendSpec step(
    MLXSequenceCache& c,
    int layer,
    int start,
    const array& k,
    const array& v,
    ::mlx::core::StreamOrDevice s) {
  const int T = static_cast<int>(k.shape(2));
  std::vector<int32_t> positions(static_cast<size_t>(T));
  std::iota(positions.begin(), positions.end(), start);
  return c.update_and_fetch(layer, positions, k, v, s);
}

class MLXSequenceCacheTest : public ::testing::Test {
 protected:
  const int H = 2;
  const int D = 8;
  ::mlx::core::StreamOrDevice s = {};

  array randn(int T, ::mlx::core::Dtype dt) {
    return ::mlx::core::random::normal(::mlx::core::Shape{1, H, T, D}, dt);
  }
};

// Prefill: T=4 at position 0 -> Causal (lower-right aligned).
TEST_F(MLXSequenceCacheTest, PrefillIsCausal) {
  using namespace ::mlx::core;
  auto c = make_cache<MLXSequenceCache>(flat_config(
      /*capacity=*/32,
      /*n_layers=*/1,
      H,
      D,
      static_cast<int>(ScalarType::Half)));
  const int T0 = 4;
  array k0 = randn(T0, float16);
  array v0 = randn(T0, float16);

  AttendSpec spec0 = step(c, 0, /*position=*/0, k0, v0, s);
  EXPECT_EQ(spec0.kind, AttendSpec::Mask::Causal);
  EXPECT_TRUE(allclose(spec0.K, k0, 0.0f));
  EXPECT_TRUE(allclose(spec0.V, v0, 0.0f));
}

// Decode: after a T=4 prefill, a single token at position 4 -> None, and the
// window is the full assembled history (prefill ++ the new token).
TEST_F(MLXSequenceCacheTest, DecodeReadsFullHistory) {
  using namespace ::mlx::core;
  auto c = make_cache<MLXSequenceCache>(flat_config(
      /*capacity=*/32,
      /*n_layers=*/1,
      H,
      D,
      static_cast<int>(ScalarType::Half)));
  const int T0 = 4;
  array k0 = randn(T0, float16);
  array v0 = randn(T0, float16);
  step(c, 0, /*position=*/0, k0, v0, s); // prefill

  array k1 = randn(1, float16);
  array v1 = randn(1, float16);
  AttendSpec spec1 = step(c, 0, /*position=*/T0, k1, v1, s);
  EXPECT_EQ(spec1.kind, AttendSpec::Mask::None);
  EXPECT_TRUE(
      allclose(spec1.K, concatenate(std::vector<array>{k0, k1}, 2, s), 0.0f));
  EXPECT_TRUE(
      allclose(spec1.V, concatenate(std::vector<array>{v0, v1}, 2, s), 0.0f));
}

// A step past capacity is rejected (plan returns nullopt).
TEST_F(MLXSequenceCacheTest, StepPastCapacityThrows) {
  using namespace ::mlx::core;
  auto c = make_cache<MLXSequenceCache>(flat_config(
      /*capacity=*/32,
      /*n_layers=*/1,
      H,
      D,
      static_cast<int>(ScalarType::Half)));
  array kx = randn(1, float16);
  EXPECT_ANY_THROW(step(c, 0, /*position=*/32, kx, kx, s));
}

// The step carries one position per query token, and this layout can only hold
// a contiguous run of one sequence. Anything else names cells it cannot
// address, so it is refused rather than stored at the wrong positions.
TEST_F(MLXSequenceCacheTest, NonContiguousOrMiscountedPositionsThrow) {
  using namespace ::mlx::core;
  auto c = make_cache<MLXSequenceCache>(flat_config(
      /*capacity=*/32,
      /*n_layers=*/1,
      H,
      D,
      static_cast<int>(ScalarType::Half)));
  array k = randn(3, float16);

  const std::vector<int32_t> gap{0, 1, 3};
  EXPECT_ANY_THROW(c.update_and_fetch(0, gap, k, k, s));

  const std::vector<int32_t> two_seqs{0, 0, 1};
  EXPECT_ANY_THROW(c.update_and_fetch(0, two_seqs, k, k, s));

  const std::vector<int32_t> short_run{0, 1};
  EXPECT_ANY_THROW(c.update_and_fetch(0, short_run, k, k, s));

  EXPECT_NO_THROW(step(c, 0, /*position=*/0, k, k, s));
}

// Storage dtype != compute: fp32 input, fp16 storage. The cache casts on write,
// so the read-back K/V are exactly the fp16 of the input.
TEST_F(MLXSequenceCacheTest, StorageDtypeDiffersCastsOnWrite) {
  using namespace ::mlx::core;
  auto c16 = make_cache<MLXSequenceCache>(flat_config(
      /*capacity=*/32,
      /*n_layers=*/1,
      H,
      D,
      static_cast<int>(ScalarType::Half)));
  const int T0 = 4;
  array k2 = randn(T0, float32);
  array v2 = randn(T0, float32);
  AttendSpec spec2 = step(c16, 0, /*position=*/0, k2, v2, s);
  EXPECT_EQ(spec2.K.dtype(), float16);
  EXPECT_EQ(spec2.V.dtype(), float16);
  EXPECT_TRUE(allclose(spec2.K, astype(k2, float16, s), 0.0f));
  EXPECT_TRUE(allclose(spec2.V, astype(v2, float16, s), 0.0f));
}

// A run is placed and fetched at its own physical start. Flat runs always start
// at 0, so this is driven on Pool directly -- a ring layer's read starts
// mid-pool, and dropping the start would silently return the wrong cells.
TEST_F(MLXSequenceCacheTest, PoolHonorsRunStart) {
  using namespace ::mlx::core;
  Pool p(/*initial_slots=*/8, /*max_slots=*/8, H, D, float16);
  array x = randn(3, float16);
  p.write(/*start=*/2, /*len=*/3, x, s);

  EXPECT_TRUE(allclose(p.read(2, 3, s), x, 0.0f));
  // The cells before the run are untouched, so reading from 0 is not the same
  // window -- the regression this guards against.
  EXPECT_FALSE(allclose(p.read(0, 3, s), x, 0.0f));
}

// A partial per-layer list is rejected instead of indexing past the end.
TEST_F(MLXSequenceCacheTest, EmptyGeometryThrows) {
  CacheArgs args = flat_config(
      /*capacity=*/32,
      /*n_layers=*/4,
      H,
      D,
      static_cast<int>(ScalarType::Half));
  args.geometry.layers.clear();
  EXPECT_ANY_THROW(MLXSequenceCache(args.geometry, args.config));
}

// A step past the allocated slots grows the pool instead of failing, and the
// result is the same window a fully-allocated pool would have returned.
TEST_F(MLXSequenceCacheTest, GrowsPastInitialCapacity) {
  using namespace ::mlx::core;
  auto c = make_cache<MLXSequenceCache>(flat_config(
      /*capacity=*/32,
      /*n_layers=*/1,
      H,
      D,
      static_cast<int>(ScalarType::Half),
      /*initial_capacity=*/2));

  const int T0 = 5; // > initial_capacity
  array k0 = randn(T0, float16);
  array v0 = randn(T0, float16);
  AttendSpec spec0 = step(c, 0, /*position=*/0, k0, v0, s);
  EXPECT_EQ(spec0.K.shape(2), T0);
  EXPECT_TRUE(allclose(spec0.K, k0, 0.0f));
  EXPECT_TRUE(allclose(spec0.V, v0, 0.0f));
}

// Growth preserves cells already written: a decode crossing the allocated
// boundary must still read back the full history.
TEST_F(MLXSequenceCacheTest, GrowthPreservesExistingCells) {
  using namespace ::mlx::core;
  auto c = make_cache<MLXSequenceCache>(flat_config(
      /*capacity=*/32,
      /*n_layers=*/1,
      H,
      D,
      static_cast<int>(ScalarType::Half),
      /*initial_capacity=*/2));

  array k0 = randn(2, float16); // exactly fills the initial allocation
  array v0 = randn(2, float16);
  step(c, 0, /*position=*/0, k0, v0, s);

  array k1 = randn(1, float16); // crosses the boundary -> grows
  array v1 = randn(1, float16);
  AttendSpec spec1 = step(c, 0, /*position=*/2, k1, v1, s);
  EXPECT_EQ(spec1.K.shape(2), 3);
  EXPECT_TRUE(
      allclose(spec1.K, concatenate(std::vector<array>{k0, k1}, 2, s), 0.0f));
  EXPECT_TRUE(
      allclose(spec1.V, concatenate(std::vector<array>{v0, v1}, 2, s), 0.0f));
}

// Growth doubles until the run fits, and never allocates past max_slots --
// including when the last doubling would overshoot it.
TEST_F(MLXSequenceCacheTest, PoolDoublesAndClampsToMaxSlots) {
  using namespace ::mlx::core;
  Pool p(/*initial_slots=*/2, /*max_slots=*/32, H, D, float16);
  EXPECT_EQ(p.slots(), 0) << "a pool holds no slots until its first write";
  p.write(0, 5, randn(5, float16), s); // 2 -> 4 -> 8
  EXPECT_EQ(p.slots(), 8);

  // 16 -> 32 overshoots a cap of 20, so it clamps.
  Pool q(/*initial_slots=*/16, /*max_slots=*/20, H, D, float16);
  q.write(0, 17, randn(17, float16), s);
  EXPECT_EQ(q.slots(), 20);

  // initial_slots above the cap is clamped when the pool allocates.
  Pool r(/*initial_slots=*/512, /*max_slots=*/4, H, D, float16);
  r.write(0, 1, randn(1, float16), s);
  EXPECT_EQ(r.slots(), 4);
}

TEST_F(MLXSequenceCacheTest, PoolClonePrefixCompactsAGrownPool) {
  using namespace ::mlx::core;
  Pool source(/*initial_slots=*/4, /*max_slots=*/64, H, D, float16);
  array values = randn(40, float16);
  source.write(0, 40, values, s);
  ASSERT_EQ(source.slots(), 64); // 4 doubled until it fit 40

  Pool clone = source.clone_prefix(8, s);
  EXPECT_EQ(clone.slots(), 8);
  EXPECT_TRUE(allclose(
      clone.read(0, 8, s),
      slice(values, Shape{0, 0, 0, 0}, Shape{1, H, 8, D}, s),
      0.0f));
}

// A pool no larger than a fresh one is shared: the copy would buy nothing and
// the fork would double straight back out of it.
TEST_F(MLXSequenceCacheTest, PoolClonePrefixSharesWhenItWouldNotShrink) {
  using namespace ::mlx::core;
  Pool source(/*initial_slots=*/16, /*max_slots=*/32, H, D, float16);
  source.write(0, 8, randn(8, float16), s);
  ASSERT_EQ(source.slots(), 16);

  EXPECT_EQ(source.clone_prefix(3, s).slots(), 16);
}

// gemma4 alternates flat and ring layers. A clone compacts the flat pools and
// leaves the ring ones, and the ring is what bounds how far back it may fork.
TEST_F(MLXSequenceCacheTest, CloneOfAMixedModelIsBoundedByItsRingLayer) {
  using namespace ::mlx::core;
  const int kHalf = static_cast<int>(ScalarType::Half);
  CacheArgs cfg = flat_config(
      /*capacity=*/64, /*n_layers=*/2, H, D, kHalf, /*initial_capacity=*/4);
  cfg.geometry.layers[1].policy =
      cache::LayerPolicy{cache::LayerPolicy::Kind::Ring, /*window=*/4};
  cfg.config.max_write = 4;

  auto c = make_cache<MLXSequenceCache>(cfg);
  auto oracle = make_cache<MLXSequenceCache>(cfg);
  std::vector<array> ks, vs;
  for (int32_t p = 0; p < 10; ++p) {
    ks.push_back(randn(1, float16));
    vs.push_back(randn(1, float16));
    c.update_and_fetch(0, {p}, ks[p], vs[p], s);
    c.update_and_fetch(1, {p}, ks[p], vs[p], s);
    if (p < 6) {
      oracle.update_and_fetch(0, {p}, ks[p], vs[p], s);
      oracle.update_and_fetch(1, {p}, ks[p], vs[p], s);
    }
  }

  // The flat layer holds every position, but the ring retains only down to
  // written - max_write, and the most restrictive layer decides.
  EXPECT_FALSE(c.can_rewind(5));
  ASSERT_TRUE(c.can_rewind(6));
  MLXSequenceCache fork(c, 6, ::mlx::core::to_stream(s));
  EXPECT_EQ(fork.length(), 6);

  // Both layers keep decoding correctly: the flat one out of its compacted
  // pool, the ring one over slots that still hold the donor's later positions
  // until it overwrites them.
  for (int32_t p = 6; p < 10; ++p) {
    array k = randn(1, float16), v = randn(1, float16);
    for (int layer = 0; layer < 2; ++layer) {
      AttendSpec got = fork.update_and_fetch(layer, {p}, k, v, s);
      AttendSpec want = oracle.update_and_fetch(layer, {p}, k, v, s);
      EXPECT_TRUE(allclose(got.K, want.K, 0.0f))
          << "layer " << layer << " @" << p;
      EXPECT_TRUE(allclose(got.V, want.V, 0.0f))
          << "layer " << layer << " @" << p;
    }
  }
}

// A compacted fork keeps growing: its pool doubles from the smaller size the
// clone gave it, and the history it copied survives that.
TEST_F(MLXSequenceCacheTest, CompactedForkGrowsPastItsClonedSize) {
  using namespace ::mlx::core;
  auto c = make_cache<MLXSequenceCache>(flat_config(
      /*capacity=*/64,
      /*n_layers=*/1,
      H,
      D,
      static_cast<int>(ScalarType::Half),
      /*initial_capacity=*/4));
  auto oracle = make_cache<MLXSequenceCache>(flat_config(
      /*capacity=*/64,
      /*n_layers=*/1,
      H,
      D,
      static_cast<int>(ScalarType::Half),
      /*initial_capacity=*/4));
  std::vector<array> ks, vs;
  for (int32_t p = 0; p < 40; ++p) {
    ks.push_back(randn(1, float16));
    vs.push_back(randn(1, float16));
    c.update_and_fetch(0, {p}, ks[p], vs[p], s);
    if (p < 6) {
      oracle.update_and_fetch(0, {p}, ks[p], vs[p], s);
    }
  }

  ASSERT_TRUE(c.can_rewind(6));
  MLXSequenceCache fork(c, 6, ::mlx::core::to_stream(s));
  // Well past the 6 slots the clone compacted to, so the pool doubles again.
  for (int32_t p = 6; p < 30; ++p) {
    array k = randn(1, float16), v = randn(1, float16);
    AttendSpec got = fork.update_and_fetch(0, {p}, k, v, s);
    AttendSpec want = oracle.update_and_fetch(0, {p}, k, v, s);
    EXPECT_EQ(got.K.shape(2), p + 1);
    EXPECT_TRUE(allclose(got.K, want.K, 0.0f)) << "position " << p;
  }
}

// The fork constructor takes a position the source can rewind to. Anything
// else is refused loudly rather than producing a fork reading slots it does
// not own.
TEST_F(MLXSequenceCacheTest, ForkRejectsPositionsItCannotRewindTo) {
  using namespace ::mlx::core;
  auto c = make_cache<MLXSequenceCache>(ring_config(
      /*capacity=*/64,
      /*window=*/4,
      /*max_write=*/1,
      H,
      D,
      static_cast<int>(ScalarType::Half)));
  for (int32_t p = 0; p < 10; ++p) {
    array k = randn(1, float16);
    c.update_and_fetch(0, {p}, k, k, s);
  }

  const ::mlx::core::Stream stream = ::mlx::core::to_stream(s);
  EXPECT_ANY_THROW(MLXSequenceCache(c, 8, stream)); // below the ring floor
  EXPECT_ANY_THROW(MLXSequenceCache(c, 11, stream)); // never reached
  EXPECT_ANY_THROW(MLXSequenceCache(c, 0, stream)); // nothing to fork
  EXPECT_NO_THROW(MLXSequenceCache(c, 9, stream)); // exactly the floor
}

// A pool that starts empty is allowed, and grows on the first write.
TEST_F(MLXSequenceCacheTest, ZeroInitialCapacityGrowsOnFirstWrite) {
  using namespace ::mlx::core;
  auto c = make_cache<MLXSequenceCache>(flat_config(
      /*capacity=*/32,
      /*n_layers=*/1,
      H,
      D,
      static_cast<int>(ScalarType::Half),
      /*initial_capacity=*/0));
  array k0 = randn(3, float16);
  array v0 = randn(3, float16);
  AttendSpec spec0 = step(c, 0, /*position=*/0, k0, v0, s);
  EXPECT_TRUE(allclose(spec0.K, k0, 0.0f));
}

// The sliding-window mask is a band on (key - query): causal above, window
// below. The span is right-aligned, so the last query's own key is the last
// key, and each query attends `window` keys ending at its own.
TEST_F(MLXSequenceCacheTest, WindowCausalMaskIsABand) {
  using namespace ::mlx::core;
  // T=3 queries over S=5 keys, window 3. Query i owns key i + (S - T), and
  // attends the `window` keys ending there -- a band, one row per query.
  // clang-format off
  std::vector<int> want = {
      1, 1, 1, 0, 0,
      0, 1, 1, 1, 0,
      0, 0, 1, 1, 1};
  // A window covering the whole span leaves plain causal: no lower bound bites.
  std::vector<int> causal = {
      1, 1, 1, 0, 0,
      1, 1, 1, 1, 0,
      1, 1, 1, 1, 1};
  // clang-format on
  array m = astype(window_causal_mask(3, 5, 3, s), int32, s);
  EXPECT_TRUE(allclose(m, array(want.data(), Shape{1, 1, 3, 5}, int32), 0.0f));

  array full = astype(window_causal_mask(3, 5, 5, s), int32, s);
  EXPECT_TRUE(
      allclose(full, array(causal.data(), Shape{1, 1, 3, 5}, int32), 0.0f));
}

// A ring layer decodes past its window: the read span stops at `window` cells
// and holds the newest tokens, evicting the oldest. No mask -- a single query
// may attend its whole span.
TEST_F(MLXSequenceCacheTest, RingDecodeEvictsOldestAndNeedsNoMask) {
  using namespace ::mlx::core;
  const int W = 4;
  auto c = make_cache<MLXSequenceCache>(ring_config(
      /*capacity=*/64,
      /*window=*/W,
      /*max_write=*/1,
      H,
      D,
      static_cast<int>(ScalarType::Half)));

  // Feed 6 single tokens; the last 4 must survive, oldest -> newest.
  std::vector<array> toks;
  for (int i = 0; i < 6; ++i) {
    toks.push_back(randn(1, float16));
  }
  AttendSpec spec{toks[0], toks[0], AttendSpec::Mask::None, {}};
  for (int i = 0; i < 6; ++i) {
    spec = step(c, 0, /*position=*/i, toks[i], toks[i], s);
  }
  EXPECT_EQ(spec.kind, AttendSpec::Mask::None);
  EXPECT_EQ(spec.K.shape(2), W);
  EXPECT_TRUE(allclose(
      spec.K,
      concatenate(std::vector<array>{toks[2], toks[3], toks[4], toks[5]}, 2, s),
      0.0f));
}

// A ring layer stays Causal while its window still covers the span, so MLX
// applies its fused mask and no tensor is built. A flat layer always does.
TEST_F(MLXSequenceCacheTest, WindowWiderThanSpanStaysCausal) {
  using namespace ::mlx::core;
  auto ring = make_cache<MLXSequenceCache>(ring_config(
      /*capacity=*/64,
      /*window=*/4,
      /*max_write=*/2,
      H,
      D,
      static_cast<int>(ScalarType::Half)));
  array k = randn(2, float16); // span 2 < window 4
  AttendSpec rspec = step(ring, 0, /*position=*/0, k, k, s);
  EXPECT_EQ(rspec.kind, AttendSpec::Mask::Causal);
  EXPECT_FALSE(rspec.mask.has_value());

  auto flat = make_cache<MLXSequenceCache>(flat_config(
      /*capacity=*/64,
      /*n_layers=*/1,
      H,
      D,
      static_cast<int>(ScalarType::Half)));
  AttendSpec fspec = step(flat, 0, /*position=*/0, k, k, s);
  EXPECT_EQ(fspec.kind, AttendSpec::Mask::Causal);
  EXPECT_FALSE(fspec.mask.has_value());
}

// A step whose runs wrap the ring is scattered and gathered in logical order.
TEST_F(MLXSequenceCacheTest, RingStepWrapsAndRejoinsInOrder) {
  using namespace ::mlx::core;
  const int W = 4;
  const int MW = 2;
  auto c = make_cache<MLXSequenceCache>(ring_config(
      /*capacity=*/64,
      /*window=*/W,
      /*max_write=*/MW,
      H,
      D,
      static_cast<int>(ScalarType::Half)));

  // ring_size = W + MW - 1 = 5. Fill 4 tokens, then a 2-token step at
  // position 4 writes slots 4 and 0 -- a wrap.
  std::vector<array> toks;
  for (int i = 0; i < 4; ++i) {
    toks.push_back(randn(1, float16));
    step(c, 0, /*position=*/i, toks.back(), toks.back(), s);
  }
  array pair = randn(2, float16);
  AttendSpec spec = step(c, 0, /*position=*/4, pair, pair, s);

  // The span is the union of the two queries' windows -- position 4 attends
  // 1..4 and position 5 attends 2..5 -- so it is window + T - 1 = 5 cells, not
  // 4. Since the window is now narrower than the span, the cache hands back a
  // band to narrow each query back to its own 4.
  EXPECT_EQ(spec.K.shape(2), W + 1);
  EXPECT_EQ(spec.kind, AttendSpec::Mask::Explicit);
  ASSERT_TRUE(spec.mask.has_value());
  // Span indices 0..4 are positions 1..5. Query 0 is position 4 and attends
  // 1..4; query 1 is position 5 and attends 2..5 -- four keys each.
  // clang-format off
  std::vector<int> band = {
      1, 1, 1, 1, 0,
      0, 1, 1, 1, 1};
  // clang-format on
  EXPECT_TRUE(allclose(
      astype(*spec.mask, int32, s),
      array(band.data(), Shape{1, 1, 2, W + 1}, int32),
      0.0f));
  EXPECT_TRUE(allclose(
      spec.K,
      concatenate(std::vector<array>{toks[1], toks[2], toks[3], pair}, 2, s),
      0.0f));
}

// A negative initial_capacity is rejected rather than reaching MLX as a
// negative dimension.
TEST_F(MLXSequenceCacheTest, NegativeInitialCapacityThrows) {
  CacheArgs args = flat_config(
      /*capacity=*/32,
      /*n_layers=*/1,
      H,
      D,
      static_cast<int>(ScalarType::Half),
      /*initial_capacity=*/-1);
  EXPECT_ANY_THROW(MLXSequenceCache(args.geometry, args.config));
}

} // namespace
