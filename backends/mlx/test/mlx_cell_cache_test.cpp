/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

// Op-level test for the cell layout's MLX byte layer (MLXCellCache / Pool).
//
// Drives MLXCellCache::update_and_fetch directly (no interpreter / .pte) and
// checks the AttendSpec against the cells the layout should have claimed: that
// a token lands in its own cell, that the window is the occupied prefix, and
// that the mask isolates sequences, hides freed cells and honours a layer's
// window. The scatter path is covered by refilling a hole below a live cell,
// which no contiguous write could express.
//
// Must run on Apple Silicon: MLX needs the Metal backend.

#include "MLXCellCache.h"
#include "backend_options.h" // kMLXBackendId
#include "utils.h" // allclose, flat_config

#include <executorch/extension/llm/cache/cache_registry.h>
#include <mlx/mlx.h>

#include <gtest/gtest.h>

#include <optional>
#include <vector>

using namespace ::executorch::backends::mlx;
namespace cache = ::executorch::extension::llm::cache;
using ::mlx::core::array;

namespace {

// Cell j of a [1, H, S, D] window.
array cell_of(const array& window, int j) {
  using namespace ::mlx::core;
  const int H = static_cast<int>(window.shape(1));
  const int D = static_cast<int>(window.shape(3));
  return slice(window, Shape{0, 0, j, 0}, Shape{1, H, j + 1, D});
}

class MLXCellCacheTest : public ::testing::Test {
 protected:
  const int H = 2;
  const int D = 8;
  const int kHalf = static_cast<int>(ScalarType::Half);
  ::mlx::core::StreamOrDevice s = {};

  array randn(int T, ::mlx::core::Dtype dt = ::mlx::core::float16) {
    return ::mlx::core::random::normal(::mlx::core::Shape{1, H, T, D}, dt);
  }

  // Declare `seq_ids` and serve layer 0 with K/V of matching length.
  AttendSpec step(
      MLXCellCache& c,
      const std::vector<int32_t>& seq_ids,
      const std::vector<int32_t>& positions,
      const array& k,
      const array& v) {
    EXPECT_TRUE(c.declare_step(seq_ids));
    return c.update_and_fetch(0, positions, k, v, s);
  }

  // The mask as ints, so a row can be compared against an expected pattern.
  array bits(const AttendSpec& spec) {
    return ::mlx::core::astype(*spec.mask, ::mlx::core::int32);
  }
};

// Prefill of one sequence: tokens take cells 0..3, the window is exactly them,
// and the mask is lower-triangular.
TEST_F(MLXCellCacheTest, PrefillClaimsCellsInOrder) {
  using namespace ::mlx::core;
  MLXCellCache c(flat_config(/*capacity=*/32, /*n_layers=*/1, H, D, kHalf));
  const int32_t a = *c.seq_new();
  array k = randn(4), v = randn(4);

  AttendSpec spec = step(c, {a, a, a, a}, {0, 1, 2, 3}, k, v);

  EXPECT_EQ(spec.kind, AttendSpec::Mask::Explicit);
  EXPECT_EQ(spec.K.shape(2), 4);
  EXPECT_TRUE(allclose(spec.K, k, 0.0f));
  EXPECT_TRUE(allclose(spec.V, v, 0.0f));

  const std::vector<int32_t> causal = {
      1, 0, 0, 0, 1, 1, 0, 0, 1, 1, 1, 0, 1, 1, 1, 1};
  EXPECT_TRUE(allclose(
      bits(spec), array(causal.data(), Shape{1, 1, 4, 4}, int32), 0.0f));
}

// Decode appends one cell and reads the whole prefix back.
TEST_F(MLXCellCacheTest, DecodeExtendsTheWindow) {
  using namespace ::mlx::core;
  MLXCellCache c(flat_config(32, 1, H, D, kHalf));
  const int32_t a = *c.seq_new();
  array k0 = randn(2), v0 = randn(2);
  step(c, {a, a}, {0, 1}, k0, v0);

  array k1 = randn(1), v1 = randn(1);
  AttendSpec spec = step(c, {a}, {2}, k1, v1);

  EXPECT_EQ(spec.K.shape(2), 3);
  EXPECT_TRUE(allclose(cell_of(spec.K, 2), k1, 0.0f));
  EXPECT_TRUE(allclose(cell_of(spec.V, 2), v1, 0.0f));
  const std::vector<int32_t> all = {1, 1, 1};
  EXPECT_TRUE(
      allclose(bits(spec), array(all.data(), Shape{1, 1, 1, 3}, int32), 0.0f));
}

// Two sequences share the pool: the window spans both, the mask attends only
// the querying sequence's cells.
TEST_F(MLXCellCacheTest, MaskIsolatesSequences) {
  using namespace ::mlx::core;
  MLXCellCache c(flat_config(32, 1, H, D, kHalf));
  const int32_t a = *c.seq_new();
  const int32_t b = *c.seq_new();
  EXPECT_NE(a, b);

  step(c, {a, a}, {0, 1}, randn(2), randn(2)); // cells 0, 1
  step(c, {b, b}, {0, 1}, randn(2), randn(2)); // cells 2, 3

  AttendSpec spec = step(c, {b}, {2}, randn(1), randn(1)); // cell 4
  EXPECT_EQ(spec.K.shape(2), 5);
  const std::vector<int32_t> only_b = {0, 0, 1, 1, 1};
  EXPECT_TRUE(allclose(
      bits(spec), array(only_b.data(), Shape{1, 1, 1, 5}, int32), 0.0f));
}

// A removed sequence frees its cells; the next token refills the lowest one,
// landing below a live cell. The scatter must place it there and the mask must
// skip the cell still free.
TEST_F(MLXCellCacheTest, FreedCellsRefillBelowLiveOnes) {
  using namespace ::mlx::core;
  MLXCellCache c(flat_config(32, 1, H, D, kHalf));
  const int32_t a = *c.seq_new();
  const int32_t b = *c.seq_new();

  step(c, {a, a, a}, {0, 1, 2}, randn(3), randn(3)); // cells 0, 1, 2
  array kb = randn(1), vb = randn(1);
  step(c, {b}, {0}, kb, vb); // cell 3

  EXPECT_TRUE(c.seq_rm(a, 0, 2)); // frees cells 0 and 1
  EXPECT_EQ(c.used_end(), 4);

  array kb1 = randn(1), vb1 = randn(1);
  AttendSpec spec = step(c, {b}, {1}, kb1, vb1); // refills cell 0

  EXPECT_EQ(spec.K.shape(2), 4);
  EXPECT_TRUE(allclose(cell_of(spec.K, 0), kb1, 0.0f));
  EXPECT_TRUE(allclose(cell_of(spec.V, 0), vb1, 0.0f));
  EXPECT_TRUE(allclose(cell_of(spec.K, 3), kb, 0.0f));

  const std::vector<int32_t> b_only = {1, 0, 0, 1};
  EXPECT_TRUE(allclose(
      bits(spec), array(b_only.data(), Shape{1, 1, 1, 4}, int32), 0.0f));
}

// A windowed layer hides cells older than its window; a flat layer keeps them.
TEST_F(MLXCellCacheTest, WindowHidesOlderCells) {
  using namespace ::mlx::core;
  cache::CacheConfig cfg = flat_config(32, /*n_layers=*/2, H, D, kHalf);
  cfg.layers = {
      cache::LayerConfig{
          cache::LayerPolicy{cache::LayerPolicy::Kind::Flat, 0}, H, D},
      cache::LayerConfig{
          cache::LayerPolicy{cache::LayerPolicy::Kind::Ring, 2}, H, D}};
  MLXCellCache c(cfg);
  const int32_t a = *c.seq_new();

  EXPECT_TRUE(c.declare_step({a, a, a, a}));
  const std::vector<int32_t> pos = {0, 1, 2, 3};
  array k = randn(4), v = randn(4);
  AttendSpec flat = c.update_and_fetch(0, pos, k, v, s);
  AttendSpec ring = c.update_and_fetch(1, pos, k, v, s);

  // The last query attends every cell on the flat layer, its own window on the
  // ring one.
  const std::vector<int32_t> flat_last = {1, 1, 1, 1};
  const std::vector<int32_t> ring_last = {0, 0, 1, 1};
  EXPECT_TRUE(allclose(
      slice(bits(flat), Shape{0, 0, 3, 0}, Shape{1, 1, 4, 4}),
      array(flat_last.data(), Shape{1, 1, 1, 4}, int32),
      0.0f));
  EXPECT_TRUE(allclose(
      slice(bits(ring), Shape{0, 0, 3, 0}, Shape{1, 1, 4, 4}),
      array(ring_last.data(), Shape{1, 1, 1, 4}, int32),
      0.0f));
}

// The window is shared: both pools grow past the initial allocation and the
// cells written before growth survive it.
TEST_F(MLXCellCacheTest, GrowthPreservesExistingCells) {
  using namespace ::mlx::core;
  MLXCellCache c(flat_config(32, 1, H, D, kHalf, /*initial_capacity=*/2));
  const int32_t a = *c.seq_new();
  array k0 = randn(2), v0 = randn(2);
  step(c, {a, a}, {0, 1}, k0, v0);

  array k1 = randn(3), v1 = randn(3);
  AttendSpec spec = step(c, {a, a, a}, {2, 3, 4}, k1, v1);

  EXPECT_EQ(spec.K.shape(2), 5);
  EXPECT_TRUE(
      allclose(slice(spec.K, Shape{0, 0, 0, 0}, Shape{1, H, 2, D}), k0, 0.0f));
  EXPECT_TRUE(
      allclose(slice(spec.K, Shape{0, 0, 2, 0}, Shape{1, H, 5, D}), k1, 0.0f));
}

// K/V are cast to the configured storage dtype on the way in.
TEST_F(MLXCellCacheTest, StorageDtypeDiffersCastsOnWrite) {
  using namespace ::mlx::core;
  MLXCellCache c(
      flat_config(32, 1, H, D, static_cast<int>(ScalarType::BFloat16)));
  const int32_t a = *c.seq_new();
  array k = randn(2, float32), v = randn(2, float32);

  AttendSpec spec = step(c, {a, a}, {0, 1}, k, v);

  EXPECT_EQ(spec.K.dtype(), bfloat16);
  EXPECT_TRUE(allclose(spec.K, k, 1e-2f));
}

// The step verbs are a contract: no declaration, a miscounted call, a repeated
// layer and a position a sequence already holds are all refused.
TEST_F(MLXCellCacheTest, IllFormedStepsThrow) {
  using namespace ::mlx::core;
  MLXCellCache c(flat_config(32, 1, H, D, kHalf));
  const int32_t a = *c.seq_new();
  array k = randn(2), v = randn(2);

  EXPECT_ANY_THROW(c.update_and_fetch(0, {0, 1}, k, v, s)); // no declare_step

  EXPECT_TRUE(c.declare_step({a, a}));
  EXPECT_ANY_THROW(c.update_and_fetch(0, {0}, k, v, s)); // positions != tokens
  EXPECT_ANY_THROW(c.update_and_fetch(1, {0, 1}, k, v, s)); // no such layer

  c.update_and_fetch(0, {0, 1}, k, v, s);
  EXPECT_ANY_THROW(
      c.update_and_fetch(0, {0, 1}, k, v, s)); // layer served twice

  EXPECT_TRUE(c.declare_step({a}));
  array k1 = randn(1), v1 = randn(1);
  EXPECT_ANY_THROW(c.update_and_fetch(0, {1}, k1, v1, s)); // position not newer
}

// A step wider than the free cells is refused, and refusing claims nothing.
TEST_F(MLXCellCacheTest, StepPastCapacityIsRefused) {
  MLXCellCache c(flat_config(/*capacity=*/2, 1, H, D, kHalf));
  const int32_t a = *c.seq_new();
  step(c, {a, a}, {0, 1}, randn(2), randn(2));

  EXPECT_FALSE(c.declare_step({a}));
  EXPECT_EQ(c.free_cells(), 0);
}

TEST_F(MLXCellCacheTest, InvalidConfigThrows) {
  cache::CacheConfig cfg = flat_config(32, /*n_layers=*/2, H, D, kHalf);
  cfg.layers.resize(1);
  cfg.layers.push_back(cache::LayerConfig{{}, H, D});
  cfg.capacity = 0;
  EXPECT_ANY_THROW(MLXCellCache{cfg});
}

// A runner reaches a layout by (backend_id, kind), so the builder registration
// is as much a part of the layout as the class.
TEST_F(MLXCellCacheTest, RegistryBuildsCellLayout) {
  auto built = cache::CacheFactory::global().build(
      kMLXBackendId, "cell", flat_config(32, 1, H, D, kHalf));
  ASSERT_TRUE(built.ok());
  const std::shared_ptr<cache::Cache>& c = *built;
  EXPECT_NE(c->as<cache::BatchControl>(), nullptr);
  EXPECT_NE(c->as<MLXCache>(), nullptr) << "the backend face comes back too";
  // A cell layout is multi-sequence, so it offers no single-sequence face.
  EXPECT_EQ(c->as<cache::SequenceControl>(), nullptr);
}

} // namespace
