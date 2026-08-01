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

#include <mlx/mlx.h>

#include <gtest/gtest.h>

#include <optional>
#include <vector>

using namespace ::executorch::backends::mlx;
namespace cache = ::executorch::extension::llm::cache;
using ::mlx::core::array;

namespace {

// Max absolute difference within tolerance. Computed in float32: item<float>()
// reads sizeof(float) bytes, so calling it on an fp16 scalar misreads the
// buffer.
bool allclose(const array& a, const array& b, float atol) {
  using namespace ::mlx::core;
  array m = max(abs(subtract(astype(a, float32), astype(b, float32))));
  eval(m);
  return m.item<float>() <= atol;
}

cache::CacheConfig flat_config(
    int capacity,
    int n_layers,
    int n_kv_heads,
    int head_dim,
    int kv_dtype,
    std::optional<int> initial_capacity = std::nullopt) {
  cache::CacheConfig cfg;
  cfg.capacity = capacity;
  cfg.n_layers = n_layers;
  cfg.layers = {cache::LayerConfig{
      cache::LayerPolicy{cache::LayerPolicy::Kind::Flat, 0},
      n_kv_heads,
      head_dim}};
  cfg.kv_dtype = kv_dtype;
  if (initial_capacity) {
    cfg.initial_capacity = *initial_capacity;
  }
  return cfg;
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
  MLXSequenceCache c(flat_config(
      /*capacity=*/32,
      /*n_layers=*/1,
      H,
      D,
      static_cast<int>(ScalarType::Half)));
  const int T0 = 4;
  array k0 = randn(T0, float16);
  array v0 = randn(T0, float16);

  AttendSpec spec0 = c.update_and_fetch(0, /*position=*/0, k0, v0, s);
  EXPECT_EQ(spec0.kind, AttendSpec::Mask::Causal);
  EXPECT_TRUE(allclose(spec0.K, k0, 0.0f));
  EXPECT_TRUE(allclose(spec0.V, v0, 0.0f));
}

// Decode: after a T=4 prefill, a single token at position 4 -> None, and the
// window is the full assembled history (prefill ++ the new token).
TEST_F(MLXSequenceCacheTest, DecodeReadsFullHistory) {
  using namespace ::mlx::core;
  MLXSequenceCache c(flat_config(
      /*capacity=*/32,
      /*n_layers=*/1,
      H,
      D,
      static_cast<int>(ScalarType::Half)));
  const int T0 = 4;
  array k0 = randn(T0, float16);
  array v0 = randn(T0, float16);
  c.update_and_fetch(0, /*position=*/0, k0, v0, s); // prefill

  array k1 = randn(1, float16);
  array v1 = randn(1, float16);
  AttendSpec spec1 = c.update_and_fetch(0, /*position=*/T0, k1, v1, s);
  EXPECT_EQ(spec1.kind, AttendSpec::Mask::None);
  EXPECT_TRUE(
      allclose(spec1.K, concatenate(std::vector<array>{k0, k1}, 2, s), 0.0f));
  EXPECT_TRUE(
      allclose(spec1.V, concatenate(std::vector<array>{v0, v1}, 2, s), 0.0f));
}

// A step past capacity is rejected (plan returns nullopt).
TEST_F(MLXSequenceCacheTest, StepPastCapacityThrows) {
  using namespace ::mlx::core;
  MLXSequenceCache c(flat_config(
      /*capacity=*/32,
      /*n_layers=*/1,
      H,
      D,
      static_cast<int>(ScalarType::Half)));
  array kx = randn(1, float16);
  EXPECT_ANY_THROW(c.update_and_fetch(0, /*position=*/32, kx, kx, s));
}

// Storage dtype != compute: fp32 input, fp16 storage. The cache casts on write,
// so the read-back K/V are exactly the fp16 of the input.
TEST_F(MLXSequenceCacheTest, StorageDtypeDiffersCastsOnWrite) {
  using namespace ::mlx::core;
  MLXSequenceCache c16(flat_config(
      /*capacity=*/32,
      /*n_layers=*/1,
      H,
      D,
      static_cast<int>(ScalarType::Half)));
  const int T0 = 4;
  array k2 = randn(T0, float32);
  array v2 = randn(T0, float32);
  AttendSpec spec2 = c16.update_and_fetch(0, /*position=*/0, k2, v2, s);
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
  p.write(cache::Run{/*start=*/2, /*len=*/3}, x, s);

  EXPECT_TRUE(allclose(p.read(cache::Run{2, 3}, s), x, 0.0f));
  // The cells before the run are untouched, so reading from 0 is not the same
  // window -- the regression this guards against.
  EXPECT_FALSE(allclose(p.read(cache::Run{0, 3}, s), x, 0.0f));
}

// A partial per-layer list is rejected instead of indexing past the end.
TEST_F(MLXSequenceCacheTest, PartialLayerListThrows) {
  cache::CacheConfig cfg = flat_config(
      /*capacity=*/32,
      /*n_layers=*/4,
      H,
      D,
      static_cast<int>(ScalarType::Half));
  cfg.layers.push_back(cfg.layers.front()); // size 2, neither 1 nor n_layers
  EXPECT_ANY_THROW(MLXSequenceCache{cfg});
}

// A step past the allocated slots grows the pool instead of failing, and the
// result is the same window a fully-allocated pool would have returned.
TEST_F(MLXSequenceCacheTest, GrowsPastInitialCapacity) {
  using namespace ::mlx::core;
  MLXSequenceCache c(flat_config(
      /*capacity=*/32,
      /*n_layers=*/1,
      H,
      D,
      static_cast<int>(ScalarType::Half),
      /*initial_capacity=*/2));

  const int T0 = 5; // > initial_capacity
  array k0 = randn(T0, float16);
  array v0 = randn(T0, float16);
  AttendSpec spec0 = c.update_and_fetch(0, /*position=*/0, k0, v0, s);
  EXPECT_EQ(spec0.K.shape(2), T0);
  EXPECT_TRUE(allclose(spec0.K, k0, 0.0f));
  EXPECT_TRUE(allclose(spec0.V, v0, 0.0f));
}

// Growth preserves cells already written: a decode crossing the allocated
// boundary must still read back the full history.
TEST_F(MLXSequenceCacheTest, GrowthPreservesExistingCells) {
  using namespace ::mlx::core;
  MLXSequenceCache c(flat_config(
      /*capacity=*/32,
      /*n_layers=*/1,
      H,
      D,
      static_cast<int>(ScalarType::Half),
      /*initial_capacity=*/2));

  array k0 = randn(2, float16); // exactly fills the initial allocation
  array v0 = randn(2, float16);
  c.update_and_fetch(0, /*position=*/0, k0, v0, s);

  array k1 = randn(1, float16); // crosses the boundary -> grows
  array v1 = randn(1, float16);
  AttendSpec spec1 = c.update_and_fetch(0, /*position=*/2, k1, v1, s);
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
  EXPECT_EQ(p.slots(), 2);
  p.write(cache::Run{0, 5}, randn(5, float16), s); // 2 -> 4 -> 8
  EXPECT_EQ(p.slots(), 8);

  // 16 -> 32 overshoots a cap of 20, so it clamps.
  Pool q(/*initial_slots=*/16, /*max_slots=*/20, H, D, float16);
  q.write(cache::Run{0, 17}, randn(17, float16), s);
  EXPECT_EQ(q.slots(), 20);

  // initial_slots above the cap is clamped at construction.
  Pool r(/*initial_slots=*/512, /*max_slots=*/4, H, D, float16);
  EXPECT_EQ(r.slots(), 4);
}

// A pool that starts empty is allowed, and grows on the first write.
TEST_F(MLXSequenceCacheTest, ZeroInitialCapacityGrowsOnFirstWrite) {
  using namespace ::mlx::core;
  MLXSequenceCache c(flat_config(
      /*capacity=*/32,
      /*n_layers=*/1,
      H,
      D,
      static_cast<int>(ScalarType::Half),
      /*initial_capacity=*/0));
  array k0 = randn(3, float16);
  array v0 = randn(3, float16);
  AttendSpec spec0 = c.update_and_fetch(0, /*position=*/0, k0, v0, s);
  EXPECT_TRUE(allclose(spec0.K, k0, 0.0f));
}

// A negative initial_capacity is rejected rather than reaching MLX as a
// negative dimension.
TEST_F(MLXSequenceCacheTest, NegativeInitialCapacityThrows) {
  cache::CacheConfig cfg = flat_config(
      /*capacity=*/32,
      /*n_layers=*/1,
      H,
      D,
      static_cast<int>(ScalarType::Half),
      /*initial_capacity=*/-1);
  EXPECT_ANY_THROW(MLXSequenceCache{cfg});
}

} // namespace
