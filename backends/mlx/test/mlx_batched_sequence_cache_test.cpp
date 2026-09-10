/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

// Op-level test for the batched-sequence layout's MLX byte layer
// (MLXBatchedSequenceCache / Pool).
//
// Drives attend() directly (no interpreter / .pte). The subject is isolation:
// a step carries several sequences on one token axis, and each must come back
// with exactly what it would have computed alone. So the oracle throughout is
// an MLXSequenceCache fed the same tokens -- if a span ever saw a neighbour's
// cells, or read the wrong pool, the two would part.
//
// Must run on Apple Silicon: MLX needs the Metal backend.

#include "MLXBatchedSequenceCache.h"
#include "MLXSequenceCache.h"
#include "backend_options.h" // kMLXBackendId
#include "utils.h" // allclose, flat_config, ring_config

#include <executorch/extension/llm/cache/cache_registry.h>
#include <mlx/mlx.h>

#include <gtest/gtest.h>

#include <vector>

using namespace ::executorch::backends::mlx;
namespace cache = ::executorch::extension::llm::cache;
using ::mlx::core::array;

namespace {

class MLXBatchedSequenceCacheTest : public ::testing::Test {
 protected:
  const int H = 2;
  const int D = 8;
  const int kHalf = static_cast<int>(ScalarType::Half);
  const float kScale = 1.0f / 2.828427f; // 1/sqrt(D)
  ::mlx::core::StreamOrDevice s = {};

  array randn(int T, ::mlx::core::Dtype dt = ::mlx::core::float16) {
    return ::mlx::core::random::normal(::mlx::core::Shape{1, H, T, D}, dt);
  }

  // Declare `seq_ids` and attend layer 0 with q/k/v of matching length.
  array step(
      MLXBatchedSequenceCache& c,
      const std::vector<int32_t>& seq_ids,
      const std::vector<int32_t>& positions,
      const array& q,
      const array& k,
      const array& v) {
    EXPECT_TRUE(c.declare_step(seq_ids));
    return c.attend(0, positions, q, k, v, kScale, s);
  }

  // The same tokens through a cache that holds one sequence: what this span
  // would have computed had it run alone.
  array alone(
      MLXSequenceCache& c,
      const std::vector<int32_t>& positions,
      const array& q,
      const array& k,
      const array& v) {
    return c.attend(0, positions, q, k, v, kScale, s);
  }

  // Tokens [off, off+len) of a [1, H, T, D] step.
  array span(const array& t, int off, int len) {
    using namespace ::mlx::core;
    return slice(
        t,
        Shape{0, 0, off, 0},
        Shape{t.shape(0), t.shape(1), off + len, t.shape(3)});
  }
};

// Two sequences in one step: each comes back with what it would have computed
// alone, so neither saw the other's cells.
TEST_F(MLXBatchedSequenceCacheTest, SpansMatchSeparateRuns) {
  MLXBatchedSequenceCache c(
      flat_config(/*capacity=*/32, /*n_layers=*/1, H, D, kHalf));
  const int32_t a = *c.seq_new();
  const int32_t b = *c.seq_new();

  array q = randn(5), k = randn(5), v = randn(5);
  array out = step(c, {a, a, a, b, b}, {0, 1, 2, 0, 1}, q, k, v);
  EXPECT_EQ(out.shape(2), 5);

  MLXSequenceCache solo_a(flat_config(32, 1, H, D, kHalf));
  MLXSequenceCache solo_b(flat_config(32, 1, H, D, kHalf));
  EXPECT_TRUE(allclose(
      span(out, 0, 3),
      alone(solo_a, {0, 1, 2}, span(q, 0, 3), span(k, 0, 3), span(v, 0, 3)),
      1e-2f));
  EXPECT_TRUE(allclose(
      span(out, 3, 2),
      alone(solo_b, {0, 1}, span(q, 3, 2), span(k, 3, 2), span(v, 3, 2)),
      1e-2f));
}

// Decode: each span continues its own history, not the step before it.
TEST_F(MLXBatchedSequenceCacheTest, DecodeContinuesEachPrivateHistory) {
  MLXBatchedSequenceCache c(flat_config(32, 1, H, D, kHalf));
  const int32_t a = *c.seq_new();
  const int32_t b = *c.seq_new();
  MLXSequenceCache solo_a(flat_config(32, 1, H, D, kHalf));
  MLXSequenceCache solo_b(flat_config(32, 1, H, D, kHalf));

  array q0 = randn(3), k0 = randn(3), v0 = randn(3);
  step(c, {a, a, b}, {0, 1, 0}, q0, k0, v0);
  alone(solo_a, {0, 1}, span(q0, 0, 2), span(k0, 0, 2), span(v0, 0, 2));
  alone(solo_b, {0}, span(q0, 2, 1), span(k0, 2, 1), span(v0, 2, 1));

  array q1 = randn(2), k1 = randn(2), v1 = randn(2);
  array out = step(c, {a, b}, {2, 1}, q1, k1, v1);

  EXPECT_TRUE(allclose(
      span(out, 0, 1),
      alone(solo_a, {2}, span(q1, 0, 1), span(k1, 0, 1), span(v1, 0, 1)),
      1e-2f));
  EXPECT_TRUE(allclose(
      span(out, 1, 1),
      alone(solo_b, {1}, span(q1, 1, 1), span(k1, 1, 1), span(v1, 1, 1)),
      1e-2f));
}

// A sequence named twice in one step: its second span continues its first,
// and the outputs come back in the order the step declared them.
TEST_F(MLXBatchedSequenceCacheTest, SequenceSpannedTwicePreservesOrder) {
  MLXBatchedSequenceCache c(flat_config(32, 1, H, D, kHalf));
  const int32_t a = *c.seq_new();
  const int32_t b = *c.seq_new();

  array q = randn(3), k = randn(3), v = randn(3);
  array out = step(c, {a, b, a}, {0, 0, 1}, q, k, v);

  MLXSequenceCache solo_a(flat_config(32, 1, H, D, kHalf));
  array a0 = alone(solo_a, {0}, span(q, 0, 1), span(k, 0, 1), span(v, 0, 1));
  array a1 = alone(solo_a, {1}, span(q, 2, 1), span(k, 2, 1), span(v, 2, 1));
  EXPECT_TRUE(allclose(span(out, 0, 1), a0, 1e-2f));
  EXPECT_TRUE(allclose(span(out, 2, 1), a1, 1e-2f));

  MLXSequenceCache solo_b(flat_config(32, 1, H, D, kHalf));
  EXPECT_TRUE(allclose(
      span(out, 1, 1),
      alone(solo_b, {0}, span(q, 1, 1), span(k, 1, 1), span(v, 1, 1)),
      1e-2f));
}

// A windowed layer bounds each span to its own window, and only the span's own
// history is in it -- a neighbour's cells are in another pool entirely.
TEST_F(MLXBatchedSequenceCacheTest, WindowBoundsEachSpanWithinItsSequence) {
  const int window = 2;
  MLXBatchedSequenceCache c(
      ring_config(/*capacity=*/32, window, /*max_write=*/4, H, D, kHalf));
  const int32_t a = *c.seq_new();
  const int32_t b = *c.seq_new();
  MLXSequenceCache solo_a(ring_config(32, window, 4, H, D, kHalf));

  array q0 = randn(4), k0 = randn(4), v0 = randn(4);
  step(c, {a, a, a, b}, {0, 1, 2, 0}, q0, k0, v0);
  alone(solo_a, {0, 1, 2}, span(q0, 0, 3), span(k0, 0, 3), span(v0, 0, 3));

  array q1 = randn(2), k1 = randn(2), v1 = randn(2);
  array out = step(c, {a, b}, {3, 1}, q1, k1, v1);
  EXPECT_TRUE(allclose(
      span(out, 0, 1),
      alone(solo_a, {3}, span(q1, 0, 1), span(k1, 0, 1), span(v1, 0, 1)),
      1e-2f));
}

// K/V are cast to the configured storage dtype on the way in, per sequence.
TEST_F(MLXBatchedSequenceCacheTest, StorageDtypeDiffersCastsOnWrite) {
  using namespace ::mlx::core;
  MLXBatchedSequenceCache c(
      flat_config(32, 1, H, D, static_cast<int>(ScalarType::BFloat16)));
  const int32_t a = *c.seq_new();
  const int32_t b = *c.seq_new();

  array q = randn(2, float32), k = randn(2, float32), v = randn(2, float32);
  array out = step(c, {a, b}, {0, 0}, q, k, v);
  EXPECT_EQ(out.shape(2), 2);

  MLXSequenceCache solo(
      flat_config(32, 1, H, D, static_cast<int>(ScalarType::BFloat16)));
  EXPECT_TRUE(allclose(
      span(out, 0, 1),
      alone(solo, {0}, span(q, 0, 1), span(k, 0, 1), span(v, 0, 1)),
      1e-2f));
}

// A sequence removed frees its pools; the id comes back and starts empty.
TEST_F(MLXBatchedSequenceCacheTest, RemovedSequenceStartsOverOnReuse) {
  MLXBatchedSequenceCache c(flat_config(32, 1, H, D, kHalf));
  const int32_t a = *c.seq_new();
  step(c, {a, a, a}, {0, 1, 2}, randn(3), randn(3), randn(3));
  EXPECT_EQ(c.seq_len(a), 3);

  EXPECT_TRUE(c.seq_rm(a));
  EXPECT_EQ(c.seq_len(a), 0);
  EXPECT_EQ(*c.seq_new(), a); // the id frees for reuse

  array q = randn(1), k = randn(1), v = randn(1);
  array out = step(c, {a}, {0}, q, k, v); // and starts at position 0 again
  MLXSequenceCache solo(flat_config(32, 1, H, D, kHalf));
  EXPECT_TRUE(allclose(out, alone(solo, {0}, q, k, v), 1e-2f));
}

// The step verbs are a contract: no declaration, a miscounted call, a repeated
// layer and a position a sequence does not continue are all refused.
TEST_F(MLXBatchedSequenceCacheTest, IllFormedStepsThrow) {
  MLXBatchedSequenceCache c(flat_config(32, /*n_layers=*/2, H, D, kHalf));
  const int32_t a = *c.seq_new();
  array q = randn(2), k = randn(2), v = randn(2);

  EXPECT_ANY_THROW(c.attend(0, {0, 1}, q, k, v, kScale, s)); // no declare_step

  EXPECT_TRUE(c.declare_step({a, a}));
  EXPECT_ANY_THROW(c.attend(0, {0}, q, k, v, kScale, s)); // positions != tokens
  EXPECT_ANY_THROW(c.attend(2, {0, 1}, q, k, v, kScale, s)); // no such layer
  EXPECT_ANY_THROW(c.attend(0, {1, 2}, q, k, v, kScale, s)); // a holds nothing

  c.attend(0, {0, 1}, q, k, v, kScale, s);
  // A KV-shared layer re-serves its donor's id with the same tokens, so the
  // repeat is idempotent rather than an error.
  EXPECT_NO_THROW(c.attend(0, {0, 1}, q, k, v, kScale, s));
  c.attend(1, {0, 1}, q, k, v, kScale, s);
  EXPECT_EQ(c.seq_len(a), 2); // and nothing advanced twice
}

// A fork holds the donor's prefix and then diverges: the copy is its own, so
// what either writes afterwards is invisible to the other.
TEST_F(MLXBatchedSequenceCacheTest, ForkCopiesThePrefixThenDiverges) {
  MLXBatchedSequenceCache c(flat_config(32, 1, H, D, kHalf));
  const int32_t a = *c.seq_new();
  array q0 = randn(2), k0 = randn(2), v0 = randn(2);
  step(c, {a, a}, {0, 1}, q0, k0, v0);

  const int32_t b = *c.seq_clone(a, std::nullopt);
  EXPECT_EQ(c.seq_len(b), 2);

  // Both continue from position 2 with the *same* token, so both must produce
  // what a lone sequence carrying the whole history would.
  array q1 = randn(1), k1 = randn(1), v1 = randn(1);
  auto twice = [](const array& t) {
    return ::mlx::core::concatenate({t, t}, 2);
  };
  array out = step(c, {a, b}, {2, 2}, twice(q1), twice(k1), twice(v1));

  MLXSequenceCache solo(flat_config(32, 1, H, D, kHalf));
  alone(solo, {0, 1}, q0, k0, v0);
  array want = alone(solo, {2}, q1, k1, v1);
  EXPECT_TRUE(allclose(span(out, 0, 1), want, 1e-2f)); // the source
  EXPECT_TRUE(allclose(span(out, 1, 1), want, 1e-2f)); // and its fork
}

// A runner reaches a layout by (backend_id, kind), so the builder registration
// is as much a part of the layout as the class.
TEST_F(MLXBatchedSequenceCacheTest, RegistryBuildsBatchedSequenceLayout) {
  auto built = cache::CacheFactory::global().build(
      kMLXBackendId,
      cache::kind::kBatchedSequence,
      flat_config(32, 1, H, D, kHalf));
  ASSERT_TRUE(built.ok());
  const std::shared_ptr<cache::Cache>& c = *built;
  EXPECT_NE(c->as<cache::BatchControl>(), nullptr);
  EXPECT_NE(c->as<MLXCache>(), nullptr);
  EXPECT_EQ(c->as<cache::SequenceControl>(), nullptr);
}

} // namespace
