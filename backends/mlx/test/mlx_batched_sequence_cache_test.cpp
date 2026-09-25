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
  auto c = make_cache<MLXBatchedSequenceCache>(
      flat_config(/*capacity=*/32, /*n_layers=*/1, H, D, kHalf));
  const int32_t a = *c.seq_new();
  const int32_t b = *c.seq_new();

  array q = randn(5), k = randn(5), v = randn(5);
  array out = step(c, {a, a, a, b, b}, {0, 1, 2, 0, 1}, q, k, v);
  EXPECT_EQ(out.shape(2), 5);

  auto solo_a = make_cache<MLXSequenceCache>(flat_config(32, 1, H, D, kHalf));
  auto solo_b = make_cache<MLXSequenceCache>(flat_config(32, 1, H, D, kHalf));
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
  auto c = make_cache<MLXBatchedSequenceCache>(flat_config(32, 1, H, D, kHalf));
  const int32_t a = *c.seq_new();
  const int32_t b = *c.seq_new();
  auto solo_a = make_cache<MLXSequenceCache>(flat_config(32, 1, H, D, kHalf));
  auto solo_b = make_cache<MLXSequenceCache>(flat_config(32, 1, H, D, kHalf));

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
  auto c = make_cache<MLXBatchedSequenceCache>(flat_config(32, 1, H, D, kHalf));
  const int32_t a = *c.seq_new();
  const int32_t b = *c.seq_new();

  array q = randn(3), k = randn(3), v = randn(3);
  array out = step(c, {a, b, a}, {0, 0, 1}, q, k, v);

  auto solo_a = make_cache<MLXSequenceCache>(flat_config(32, 1, H, D, kHalf));
  array a0 = alone(solo_a, {0}, span(q, 0, 1), span(k, 0, 1), span(v, 0, 1));
  array a1 = alone(solo_a, {1}, span(q, 2, 1), span(k, 2, 1), span(v, 2, 1));
  EXPECT_TRUE(allclose(span(out, 0, 1), a0, 1e-2f));
  EXPECT_TRUE(allclose(span(out, 2, 1), a1, 1e-2f));

  auto solo_b = make_cache<MLXSequenceCache>(flat_config(32, 1, H, D, kHalf));
  EXPECT_TRUE(allclose(
      span(out, 1, 1),
      alone(solo_b, {0}, span(q, 1, 1), span(k, 1, 1), span(v, 1, 1)),
      1e-2f));
}

// A windowed layer bounds each span to its own window, and only the span's own
// history is in it -- a neighbour's cells are in another pool entirely.
TEST_F(MLXBatchedSequenceCacheTest, WindowBoundsEachSpanWithinItsSequence) {
  const int window = 2;
  auto c = make_cache<MLXBatchedSequenceCache>(
      ring_config(/*capacity=*/32, window, /*max_write=*/4, H, D, kHalf));
  const int32_t a = *c.seq_new();
  const int32_t b = *c.seq_new();
  auto solo_a =
      make_cache<MLXSequenceCache>(ring_config(32, window, 4, H, D, kHalf));

  array q0 = randn(4), k0 = randn(4), v0 = randn(4);
  step(c, {a, a, a, b}, {0, 1, 2, 0}, q0, k0, v0);
  alone(solo_a, {0, 1, 2}, span(q0, 0, 3), span(k0, 0, 3), span(v0, 0, 3));

  array q1 = randn(2), k1 = randn(2), v1 = randn(2);
  array out = step(c, {a, b}, {3, 1}, q1, k1, v1);
  EXPECT_TRUE(allclose(
      span(out, 0, 1),
      alone(solo_a, {3}, span(q1, 0, 1), span(k1, 0, 1), span(v1, 0, 1)),
      1e-2f));

  // Past the ring's window + max_write - 1 slots, so a's write wraps and its
  // read comes back as two runs. Its neighbour keeps decoding through it.
  for (int32_t p = 4; p < 9; ++p) {
    array q = randn(2), k = randn(2), v = randn(2);
    array o = step(c, {a, b}, {p, p - 2}, q, k, v);
    EXPECT_TRUE(allclose(
        span(o, 0, 1),
        alone(solo_a, {p}, span(q, 0, 1), span(k, 0, 1), span(v, 0, 1)),
        1e-2f))
        << "position " << p;
  }
}

// K/V are cast to the configured storage dtype on the way in, per sequence.
TEST_F(MLXBatchedSequenceCacheTest, StorageDtypeDiffersCastsOnWrite) {
  using namespace ::mlx::core;
  auto c = make_cache<MLXBatchedSequenceCache>(
      flat_config(32, 1, H, D, static_cast<int>(ScalarType::BFloat16)));
  const int32_t a = *c.seq_new();
  const int32_t b = *c.seq_new();

  array q = randn(2, float32), k = randn(2, float32), v = randn(2, float32);
  array out = step(c, {a, b}, {0, 0}, q, k, v);
  EXPECT_EQ(out.shape(2), 2);

  auto solo = make_cache<MLXSequenceCache>(
      flat_config(32, 1, H, D, static_cast<int>(ScalarType::BFloat16)));
  EXPECT_TRUE(allclose(
      span(out, 0, 1),
      alone(solo, {0}, span(q, 0, 1), span(k, 0, 1), span(v, 0, 1)),
      1e-2f));
}

// A sequence removed frees its pools; the id comes back and starts empty.
TEST_F(MLXBatchedSequenceCacheTest, RemovedSequenceStartsOverOnReuse) {
  auto c = make_cache<MLXBatchedSequenceCache>(flat_config(32, 1, H, D, kHalf));
  const int32_t a = *c.seq_new();
  step(c, {a, a, a}, {0, 1, 2}, randn(3), randn(3), randn(3));
  EXPECT_EQ(c.pos(a), 3);

  EXPECT_TRUE(c.seq_rm(a));
  EXPECT_EQ(c.pos(a), 0);
  EXPECT_EQ(*c.seq_new(), a); // the id frees for reuse

  array q = randn(1), k = randn(1), v = randn(1);
  array out = step(c, {a}, {0}, q, k, v); // and starts at position 0 again
  auto solo = make_cache<MLXSequenceCache>(flat_config(32, 1, H, D, kHalf));
  EXPECT_TRUE(allclose(out, alone(solo, {0}, q, k, v), 1e-2f));
}

// The step verbs are a contract: no declaration, a miscounted call, a repeated
// layer and a position a sequence does not continue are all refused.
TEST_F(MLXBatchedSequenceCacheTest, IllFormedStepsThrow) {
  auto c = make_cache<MLXBatchedSequenceCache>(
      flat_config(32, /*n_layers=*/2, H, D, kHalf));
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
  EXPECT_EQ(c.pos(a), 2); // and nothing advanced twice
}

// A fork holds the donor's prefix and then diverges: the copy is its own, so
// what either writes afterwards is invisible to the other.
TEST_F(MLXBatchedSequenceCacheTest, ForkCopiesThePrefixThenDiverges) {
  auto c = make_cache<MLXBatchedSequenceCache>(flat_config(32, 1, H, D, kHalf));
  c.bind_controller_stream(::mlx::core::to_stream(s));
  const int32_t a = *c.seq_new();
  array q0 = randn(2), k0 = randn(2), v0 = randn(2);
  step(c, {a, a}, {0, 1}, q0, k0, v0);

  const int32_t b = *c.seq_clone(a, std::nullopt);
  EXPECT_EQ(c.pos(b), 2);

  // Both continue from position 2 with the *same* token, so both must produce
  // what a lone sequence carrying the whole history would.
  array q1 = randn(1), k1 = randn(1), v1 = randn(1);
  auto twice = [](const array& t) {
    return ::mlx::core::concatenate({t, t}, 2);
  };
  array out = step(c, {a, b}, {2, 2}, twice(q1), twice(k1), twice(v1));

  auto solo = make_cache<MLXSequenceCache>(flat_config(32, 1, H, D, kHalf));
  alone(solo, {0, 1}, q0, k0, v0);
  array want = alone(solo, {2}, q1, k1, v1);
  EXPECT_TRUE(allclose(span(out, 0, 1), want, 1e-2f)); // the source
  EXPECT_TRUE(allclose(span(out, 1, 1), want, 1e-2f)); // and its fork
}

TEST_F(MLXBatchedSequenceCacheTest, PartialFlatForkCopiesOnlyPrefix) {
  auto c = make_cache<MLXBatchedSequenceCache>(flat_config(32, 1, H, D, kHalf));
  c.bind_controller_stream(::mlx::core::to_stream(s));
  const int32_t source = *c.seq_new();
  array q0 = randn(8), k0 = randn(8), v0 = randn(8);
  step(
      c, std::vector<int32_t>(8, source), {0, 1, 2, 3, 4, 5, 6, 7}, q0, k0, v0);

  const auto fork = c.seq_clone(source, /*upto=*/3);
  ASSERT_TRUE(fork);
  EXPECT_EQ(c.pos(*fork), 3);
  EXPECT_TRUE(c.seq_rm(source));

  auto oracle = make_cache<MLXSequenceCache>(flat_config(32, 1, H, D, kHalf));
  alone(oracle, {0, 1, 2}, span(q0, 0, 3), span(k0, 0, 3), span(v0, 0, 3));
  array q1 = randn(1), k1 = randn(1), v1 = randn(1);
  array got = step(c, {*fork}, {3}, q1, k1, v1);
  EXPECT_TRUE(allclose(got, alone(oracle, {3}, q1, k1, v1), 1e-2f));
}

// A compacted fork owns its pool: what it writes must not reach the source,
// nor the reverse. Feeding the two different tokens is what tells a real copy
// from a shared buffer -- the same token would agree either way.
TEST_F(MLXBatchedSequenceCacheTest, ForkAndSourceDivergeOnDifferentTokens) {
  // A small initial capacity so the pool grows past it and the fork compacts.
  auto cfg = [&] { return flat_config(32, 1, H, D, kHalf, 2); };
  auto c = make_cache<MLXBatchedSequenceCache>(cfg());
  c.bind_controller_stream(::mlx::core::to_stream(s));
  const int32_t a = *c.seq_new();

  std::vector<array> qs, ks, vs;
  for (int32_t p = 0; p < 8; ++p) {
    qs.push_back(randn(1));
    ks.push_back(randn(1));
    vs.push_back(randn(1));
    step(c, {a}, {p}, qs[p], ks[p], vs[p]);
  }
  const int32_t b = *c.seq_clone(a, /*upto=*/3);
  ASSERT_EQ(c.pos(b), 3);

  auto oa = make_cache<MLXSequenceCache>(cfg());
  auto ob = make_cache<MLXSequenceCache>(cfg());
  for (int32_t p = 0; p < 8; ++p) {
    alone(oa, {p}, qs[p], ks[p], vs[p]);
    if (p < 3) {
      alone(ob, {p}, qs[p], ks[p], vs[p]);
    }
  }

  auto pair = [](const array& x, const array& y) {
    return ::mlx::core::concatenate({x, y}, 2);
  };
  // The source continues at 8, the fork at 3, with different tokens each.
  for (int i = 0; i < 3; ++i) {
    array qa = randn(1), ka = randn(1), va = randn(1);
    array qb = randn(1), kb = randn(1), vb = randn(1);
    array out = step(
        c, {a, b}, {8 + i, 3 + i}, pair(qa, qb), pair(ka, kb), pair(va, vb));
    EXPECT_TRUE(
        allclose(span(out, 0, 1), alone(oa, {8 + i}, qa, ka, va), 1e-2f))
        << "source, step " << i;
    EXPECT_TRUE(
        allclose(span(out, 1, 1), alone(ob, {3 + i}, qb, kb, vb), 1e-2f))
        << "fork, step " << i;
  }
}

// A runner reaches a layout by (backend_id, kind), so the builder registration
// is as much a part of the layout as the class.
// A refusal is not a verb: the declaration it refused stays standing.
TEST_F(
    MLXBatchedSequenceCacheTest,
    RejectedDeclarationLeavesTheLastOneStanding) {
  auto c = make_cache<MLXBatchedSequenceCache>(flat_config(32, 1, H, D, kHalf));
  const int32_t a = *c.seq_new();

  EXPECT_TRUE(c.declare_step({a}));
  EXPECT_FALSE(c.declare_step({a + 5})); // never handed out
  EXPECT_FALSE(c.declare_step({})); // a step carries a token

  array q = randn(1), k = randn(1), v = randn(1);
  EXPECT_NO_THROW(c.attend(0, {0}, q, k, v, kScale, s));
  EXPECT_EQ(c.pos(a), 1);
}

// One capacity behind the private histories: what either holds is what the
// other cannot, and a fork needs room for the prefix it copies.
TEST_F(MLXBatchedSequenceCacheTest, SequencesShareOneCapacity) {
  auto c = make_cache<MLXBatchedSequenceCache>(
      flat_config(/*capacity=*/4, 1, H, D, kHalf));
  c.bind_controller_stream(::mlx::core::to_stream(s));
  const int32_t a = *c.seq_new();
  const int32_t b = *c.seq_new();

  step(c, {a, a, b}, {0, 1, 0}, randn(3), randn(3), randn(3));
  EXPECT_EQ(c.pos(a), 2);
  EXPECT_EQ(c.pos(b), 1);

  EXPECT_TRUE(c.declare_step({a})); // the fourth cell
  array q = randn(1), k = randn(1), v = randn(1);
  c.attend(0, {2}, q, k, v, kScale, s);

  EXPECT_FALSE(c.declare_step({a})); // all four held, whoever holds them
  EXPECT_FALSE(c.declare_step({b}));
  EXPECT_FALSE(c.seq_clone(a, std::nullopt)); // no room for the copy

  EXPECT_TRUE(c.seq_rm(b)); // b's cell comes back
  EXPECT_TRUE(c.declare_step({a}));
}

TEST_F(MLXBatchedSequenceCacheTest, RegistryBuildsBatchedSequenceLayout) {
  CacheArgs args = flat_config(32, 1, H, D, kHalf);
  auto built = cache::CacheFactory::global().build(
      kMLXBackendId, cache::kind::kBatchedSequence, args.geometry, args.config);
  ASSERT_TRUE(built.ok());
  const std::shared_ptr<cache::Cache>& c = *built;
  EXPECT_NE(c->as<cache::BatchControl>(), nullptr);
  EXPECT_NE(c->as<MLXCache>(), nullptr);
  EXPECT_EQ(c->as<cache::SequenceControl>(), nullptr);
}

// A partial fork of a wrapped ring. The donor's slots for the fork's window
// may already hold its later positions, so copying the pools whole is not
// enough on its own.
TEST_F(MLXBatchedSequenceCacheTest, PartialForkOfAWrappedRing) {
  const int window = 4, max_write = 1; // ring of window + max_write - 1 = 4
  auto c = make_cache<MLXBatchedSequenceCache>(
      ring_config(32, window, max_write, H, D, kHalf));
  c.bind_controller_stream(::mlx::core::to_stream(s));
  auto oracle = make_cache<MLXSequenceCache>(
      ring_config(32, window, max_write, H, D, kHalf));
  const int32_t a = *c.seq_new();

  // Decode the donor to 10, feeding the oracle only the first 6.
  std::vector<array> qs, ks, vs;
  for (int32_t p = 0; p < 10; ++p) {
    qs.push_back(randn(1));
    ks.push_back(randn(1));
    vs.push_back(randn(1));
    step(c, {a}, {p}, qs[p], ks[p], vs[p]);
    if (p < 6) {
      alone(oracle, {p}, qs[p], ks[p], vs[p]);
    }
  }

  // Position 6 is more than max_write behind, so the slots its window needs
  // now hold 7, 8 and 9. Nothing can recover them, so the fork is refused.
  EXPECT_FALSE(c.seq_clone(a, /*upto=*/6));

  // One step back is still in the ring, and reads what the donor read.
  const auto fork = c.seq_clone(a, /*upto=*/9);
  ASSERT_TRUE(fork);
  EXPECT_EQ(c.pos(*fork), 9);

  for (int32_t p = 6; p < 9; ++p) {
    alone(oracle, {p}, qs[p], ks[p], vs[p]);
  }
  array q = randn(1), k = randn(1), v = randn(1);
  array got = step(c, {*fork}, {9}, q, k, v);
  EXPECT_TRUE(allclose(got, alone(oracle, {9}, q, k, v), 1e-2f));
}

// A fork copies the same slots, so the second one's target is as unreachable
// as it was from the donor.
TEST_F(MLXBatchedSequenceCacheTest, ForkOfAForkKeepsTheDonorsFloor) {
  const int window = 4, max_write = 1; // ring of window + max_write - 1 = 4
  auto c = make_cache<MLXBatchedSequenceCache>(
      ring_config(32, window, max_write, H, D, kHalf));
  c.bind_controller_stream(::mlx::core::to_stream(s));
  const int32_t a = *c.seq_new();
  for (int32_t p = 0; p < 10; ++p) {
    array q = randn(1), k = randn(1), v = randn(1);
    step(c, {a}, {p}, q, k, v);
  }

  const auto b = c.seq_clone(a, /*upto=*/9);
  ASSERT_TRUE(b);
  EXPECT_FALSE(c.seq_clone(*b, /*upto=*/8));
}

} // namespace
