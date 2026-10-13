/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

// Op-level tests for the registered batched cache kinds.
//
// Drives attend() directly (no interpreter / .pte). Each packed sequence must
// produce what it would have computed alone. Ordinary-attention tests use an
// MLXSequenceCache fed the same tokens as the oracle.
//
// Must run on Apple Silicon: MLX needs the Metal backend.

#include "MLXCache.h"
#include "MLXSequenceCache.h"
#include "backend_options.h" // kMLXBackendId
#include "utils.h" // allclose, flat_config, ring_config

#include <executorch/extension/llm/cache/cache_registry.h>
#include <mlx/mlx.h>

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>

using namespace ::executorch::backends::mlx;
namespace cache = ::executorch::extension::llm::cache;
using ::mlx::core::array;

namespace {

class MLXBatchedCacheTest : public ::testing::TestWithParam<std::string> {
 protected:
  const int H = 2;
  const int D = 8;
  const int kHalf = static_cast<int>(ScalarType::Half);
  const float kScale = 1.0f / 2.828427f; // 1/sqrt(D)
  ::mlx::core::StreamOrDevice s = {};
  cache::BatchControl* batch_ = nullptr;
  MLXCache* attention_ = nullptr;

  void create_cache(const CacheArgs& args) {
    auto built = cache::CacheFactory::global().build(
        kMLXBackendId, GetParam(), args.geometry, args.config);
    ASSERT_TRUE(built.ok());
    owner_ = *built;
    batch_ = owner_->as<cache::BatchControl>();
    attention_ = owner_->as<MLXCache>();
    ASSERT_NE(batch_, nullptr);
    ASSERT_NE(attention_, nullptr);
    EXPECT_EQ(owner_->as<cache::SequenceControl>(), nullptr);
    attention_->bind_controller_stream(::mlx::core::to_stream(s));
  }

  array randn(int T, ::mlx::core::Dtype dt = ::mlx::core::float16) {
    return ::mlx::core::random::normal(::mlx::core::Shape{1, H, T, D}, dt);
  }

  // Declare `seq_ids` and attend layer 0 with q/k/v of matching length.
  array step(
      const std::vector<int32_t>& seq_ids,
      const std::vector<int32_t>& positions,
      const array& q,
      const array& k,
      const array& v) {
    EXPECT_TRUE(batch_->declare_step(seq_ids));
    return attention_->attend(0, positions, q, k, v, kScale, s);
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

  static constexpr int kCloneMaxWrite = 3;

  CacheArgs clone_config() {
    auto args = flat_config(256, 2, H, D, kHalf, 2);
    args.geometry.layers[1].policy = {cache::LayerPolicy::Kind::Ring, 4};
    args.config.max_write = kCloneMaxWrite;
    return args;
  }

  std::vector<uint64_t> clone_tokens(size_t count) {
    std::vector<uint64_t> result(count);
    std::iota(result.begin(), result.end(), uint64_t{1});
    return result;
  }

  array clone_forward(
      MLXCache& c,
      cache::BatchControl* control,
      int32_t seq,
      const std::vector<uint64_t>& prompt,
      size_t begin = 0,
      int chunk_size = kCloneMaxWrite) {
    using namespace ::mlx::core;
    std::vector<array> outputs;
    while (begin < prompt.size()) {
      const size_t count =
          std::min(static_cast<size_t>(chunk_size), prompt.size() - begin);
      std::vector<int32_t> positions(count);
      std::iota(
          positions.begin(), positions.end(), static_cast<int32_t>(begin));
      if (control) {
        EXPECT_TRUE(control->declare_step(std::vector<int32_t>(count, seq)));
      }
      std::vector<float> values;
      values.reserve(static_cast<size_t>(H * D) * count);
      for (int h = 0; h < H; ++h) {
        for (size_t i = begin; i < begin + count; ++i) {
          for (int d = 0; d < D; ++d) {
            values.push_back(std::sin(
                static_cast<float>(prompt[i]) * 0.37f +
                static_cast<float>(h) * 0.41f + static_cast<float>(d) * 0.19f));
          }
        }
      }
      array input = astype(
          array(
              values.data(), Shape{1, H, static_cast<int>(count), D}, float32),
          float16,
          s);
      const float scale = 1.0f / std::sqrt(static_cast<float>(D));
      // The SWA layer's K/V depend on full attention, including old tokens
      // outside its window. Reusing a matching tail alone is insufficient.
      array hidden =
          add(input, c.attend(0, positions, input, input, input, scale, s), s);
      array output = c.attend(1, positions, hidden, hidden, hidden, scale, s);
      eval(output);
      outputs.push_back(output);
      begin += count;
    }
    return concatenate(outputs, 2, s);
  }

  void expect_cold_clone_match(
      int32_t sequence,
      size_t position,
      const std::vector<uint64_t>& prompt) {
    using namespace ::mlx::core;
    ASSERT_LT(position, prompt.size());
    ASSERT_EQ(batch_->pos(sequence), static_cast<int>(position));
    array got = clone_forward(*attention_, batch_, sequence, prompt, position);
    auto cold = make_cache<MLXSequenceCache>(clone_config());
    array expected = clone_forward(cold, nullptr, 0, prompt, 0, 1);
    expected = slice(
        expected,
        Shape{0, 0, static_cast<int>(position), 0},
        Shape{1, H, static_cast<int>(prompt.size()), D},
        s);
    EXPECT_TRUE(allclose(got, expected, 1e-2f));
    EXPECT_EQ(batch_->pos(sequence), static_cast<int>(prompt.size()));
  }

 private:
  std::shared_ptr<cache::Cache> owner_;
};

// Two sequences in one step: each comes back with what it would have computed
// alone, so neither saw the other's cells.
TEST_P(MLXBatchedCacheTest, SpansMatchSeparateRuns) {
  ASSERT_NO_FATAL_FAILURE(
      create_cache(flat_config(/*capacity=*/32, /*n_layers=*/1, H, D, kHalf)));
  const int32_t a = *batch_->seq_new();
  const int32_t b = *batch_->seq_new();

  array q = randn(5), k = randn(5), v = randn(5);
  array out = step({a, a, a, b, b}, {0, 1, 2, 0, 1}, q, k, v);
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
TEST_P(MLXBatchedCacheTest, DecodeContinuesEachPrivateHistory) {
  ASSERT_NO_FATAL_FAILURE(create_cache(flat_config(32, 1, H, D, kHalf)));
  const int32_t a = *batch_->seq_new();
  const int32_t b = *batch_->seq_new();
  auto solo_a = make_cache<MLXSequenceCache>(flat_config(32, 1, H, D, kHalf));
  auto solo_b = make_cache<MLXSequenceCache>(flat_config(32, 1, H, D, kHalf));

  array q0 = randn(3), k0 = randn(3), v0 = randn(3);
  step({a, a, b}, {0, 1, 0}, q0, k0, v0);
  alone(solo_a, {0, 1}, span(q0, 0, 2), span(k0, 0, 2), span(v0, 0, 2));
  alone(solo_b, {0}, span(q0, 2, 1), span(k0, 2, 1), span(v0, 2, 1));

  array q1 = randn(2), k1 = randn(2), v1 = randn(2);
  array out = step({a, b}, {2, 1}, q1, k1, v1);

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
TEST_P(MLXBatchedCacheTest, SequenceSpannedTwicePreservesOrder) {
  ASSERT_NO_FATAL_FAILURE(create_cache(flat_config(32, 1, H, D, kHalf)));
  const int32_t a = *batch_->seq_new();
  const int32_t b = *batch_->seq_new();

  array q = randn(3), k = randn(3), v = randn(3);
  array out = step({a, b, a}, {0, 0, 1}, q, k, v);

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
TEST_P(MLXBatchedCacheTest, WindowBoundsEachSpanWithinItsSequence) {
  const int window = 2;
  ASSERT_NO_FATAL_FAILURE(create_cache(
      ring_config(/*capacity=*/32, window, /*max_write=*/4, H, D, kHalf)));
  const int32_t a = *batch_->seq_new();
  const int32_t b = *batch_->seq_new();
  auto solo_a =
      make_cache<MLXSequenceCache>(ring_config(32, window, 4, H, D, kHalf));

  array q0 = randn(4), k0 = randn(4), v0 = randn(4);
  step({a, a, a, b}, {0, 1, 2, 0}, q0, k0, v0);
  alone(solo_a, {0, 1, 2}, span(q0, 0, 3), span(k0, 0, 3), span(v0, 0, 3));

  array q1 = randn(2), k1 = randn(2), v1 = randn(2);
  array out = step({a, b}, {3, 1}, q1, k1, v1);
  EXPECT_TRUE(allclose(
      span(out, 0, 1),
      alone(solo_a, {3}, span(q1, 0, 1), span(k1, 0, 1), span(v1, 0, 1)),
      1e-2f));

  // Past the ring's window + max_write - 1 slots, so a's write wraps and its
  // read comes back as two runs. Its neighbour keeps decoding through it.
  for (int32_t p = 4; p < 9; ++p) {
    array q = randn(2), k = randn(2), v = randn(2);
    array o = step({a, b}, {p, p - 2}, q, k, v);
    EXPECT_TRUE(allclose(
        span(o, 0, 1),
        alone(solo_a, {p}, span(q, 0, 1), span(k, 0, 1), span(v, 0, 1)),
        1e-2f))
        << "position " << p;
  }
}

// K/V are cast to the configured storage dtype on the way in, per sequence.
TEST_P(MLXBatchedCacheTest, StorageDtypeDiffersCastsOnWrite) {
  using namespace ::mlx::core;
  ASSERT_NO_FATAL_FAILURE(create_cache(
      flat_config(32, 1, H, D, static_cast<int>(ScalarType::BFloat16))));
  const int32_t a = *batch_->seq_new();
  const int32_t b = *batch_->seq_new();

  array q = randn(2, float32), k = randn(2, float32), v = randn(2, float32);
  array out = step({a, b}, {0, 0}, q, k, v);
  EXPECT_EQ(out.shape(2), 2);

  auto solo = make_cache<MLXSequenceCache>(
      flat_config(32, 1, H, D, static_cast<int>(ScalarType::BFloat16)));
  EXPECT_TRUE(allclose(
      span(out, 0, 1),
      alone(solo, {0}, span(q, 0, 1), span(k, 0, 1), span(v, 0, 1)),
      1e-2f));
}

// A sequence removed frees its pools; the id comes back and starts empty.
TEST_P(MLXBatchedCacheTest, RemovedSequenceStartsOverOnReuse) {
  ASSERT_NO_FATAL_FAILURE(create_cache(flat_config(32, 1, H, D, kHalf)));
  const int32_t a = *batch_->seq_new();
  step({a, a, a}, {0, 1, 2}, randn(3), randn(3), randn(3));
  EXPECT_EQ(batch_->pos(a), 3);

  EXPECT_TRUE(batch_->seq_rm(a));
  EXPECT_EQ(batch_->pos(a), 0);
  EXPECT_EQ(*batch_->seq_new(), a); // the id frees for reuse

  array q = randn(1), k = randn(1), v = randn(1);
  array out = step({a}, {0}, q, k, v); // and starts at position 0 again
  auto solo = make_cache<MLXSequenceCache>(flat_config(32, 1, H, D, kHalf));
  EXPECT_TRUE(allclose(out, alone(solo, {0}, q, k, v), 1e-2f));
}

// Undeclared, miscounted and out-of-order steps are refused.
TEST_P(MLXBatchedCacheTest, IllFormedStepsThrow) {
  ASSERT_NO_FATAL_FAILURE(
      create_cache(flat_config(32, /*n_layers=*/2, H, D, kHalf)));
  const int32_t a = *batch_->seq_new();
  array q = randn(2), k = randn(2), v = randn(2);

  EXPECT_ANY_THROW(attention_->attend(0, {0, 1}, q, k, v, kScale, s));

  EXPECT_TRUE(batch_->declare_step({a, a}));
  EXPECT_ANY_THROW(attention_->attend(0, {0}, q, k, v, kScale, s));
  EXPECT_ANY_THROW(attention_->attend(2, {0, 1}, q, k, v, kScale, s));
  EXPECT_ANY_THROW(attention_->attend(0, {1, 2}, q, k, v, kScale, s));

  attention_->attend(0, {0, 1}, q, k, v, kScale, s);
  // A KV-shared layer re-serves its donor's id with the same tokens, so the
  // repeat is idempotent rather than an error.
  EXPECT_NO_THROW(attention_->attend(0, {0, 1}, q, k, v, kScale, s));
  attention_->attend(1, {0, 1}, q, k, v, kScale, s);
  EXPECT_EQ(batch_->pos(a), 2); // and nothing advanced twice
}

// A fork holds the donor's prefix and then diverges: the copy is its own, so
// what either writes afterwards is invisible to the other.
TEST_P(MLXBatchedCacheTest, ForkCopiesThePrefixThenDiverges) {
  ASSERT_NO_FATAL_FAILURE(create_cache(flat_config(32, 1, H, D, kHalf)));
  const int32_t a = *batch_->seq_new();
  array q0 = randn(2), k0 = randn(2), v0 = randn(2);
  step({a, a}, {0, 1}, q0, k0, v0);

  const auto fork = batch_->seq_clone(a, std::nullopt);
  ASSERT_TRUE(fork);
  const int32_t b = *fork;
  EXPECT_EQ(batch_->pos(b), 2);

  // Both continue from position 2 with the *same* token, so both must produce
  // what a lone sequence carrying the whole history would.
  array q1 = randn(1), k1 = randn(1), v1 = randn(1);
  auto twice = [](const array& t) {
    return ::mlx::core::concatenate({t, t}, 2);
  };
  array out = step({a, b}, {2, 2}, twice(q1), twice(k1), twice(v1));

  auto solo = make_cache<MLXSequenceCache>(flat_config(32, 1, H, D, kHalf));
  alone(solo, {0, 1}, q0, k0, v0);
  array want = alone(solo, {2}, q1, k1, v1);
  EXPECT_TRUE(allclose(span(out, 0, 1), want, 1e-2f)); // the source
  EXPECT_TRUE(allclose(span(out, 1, 1), want, 1e-2f)); // and its fork
}

TEST_P(MLXBatchedCacheTest, PartialFlatForkCopiesOnlyPrefix) {
  ASSERT_NO_FATAL_FAILURE(create_cache(flat_config(32, 1, H, D, kHalf)));
  const int32_t source = *batch_->seq_new();
  array q0 = randn(8), k0 = randn(8), v0 = randn(8);
  step(std::vector<int32_t>(8, source), {0, 1, 2, 3, 4, 5, 6, 7}, q0, k0, v0);

  const auto fork = batch_->seq_clone(source, /*upto=*/3);
  ASSERT_TRUE(fork);
  EXPECT_EQ(batch_->pos(*fork), 3);
  EXPECT_TRUE(batch_->seq_rm(source));

  auto oracle = make_cache<MLXSequenceCache>(flat_config(32, 1, H, D, kHalf));
  alone(oracle, {0, 1, 2}, span(q0, 0, 3), span(k0, 0, 3), span(v0, 0, 3));
  array q1 = randn(1), k1 = randn(1), v1 = randn(1);
  array got = step({*fork}, {3}, q1, k1, v1);
  EXPECT_TRUE(allclose(got, alone(oracle, {3}, q1, k1, v1), 1e-2f));
}

// A fork's writes must not reach the source, nor the reverse. Feeding the two
// different tokens distinguishes isolation from a shared writable buffer.
TEST_P(MLXBatchedCacheTest, ForkAndSourceDivergeOnDifferentTokens) {
  // A small initial capacity so the pool grows past it before the fork.
  auto cfg = [&] { return flat_config(32, 1, H, D, kHalf, 2); };
  ASSERT_NO_FATAL_FAILURE(create_cache(cfg()));
  const int32_t a = *batch_->seq_new();

  std::vector<array> qs, ks, vs;
  for (int32_t p = 0; p < 8; ++p) {
    qs.push_back(randn(1));
    ks.push_back(randn(1));
    vs.push_back(randn(1));
    step({a}, {p}, qs[p], ks[p], vs[p]);
  }
  const auto fork = batch_->seq_clone(a, /*upto=*/3);
  ASSERT_TRUE(fork);
  const int32_t b = *fork;
  ASSERT_EQ(batch_->pos(b), 3);

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
    array out =
        step({a, b}, {8 + i, 3 + i}, pair(qa, qb), pair(ka, kb), pair(va, vb));
    EXPECT_TRUE(
        allclose(span(out, 0, 1), alone(oa, {8 + i}, qa, ka, va), 1e-2f))
        << "source, step " << i;
    EXPECT_TRUE(
        allclose(span(out, 1, 1), alone(ob, {3 + i}, qb, kb, vb), 1e-2f))
        << "fork, step " << i;
  }
}

// A refusal is not a verb: the declaration it refused stays standing.
TEST_P(MLXBatchedCacheTest, RejectedDeclarationLeavesTheLastOneStanding) {
  ASSERT_NO_FATAL_FAILURE(create_cache(flat_config(32, 1, H, D, kHalf)));
  const int32_t a = *batch_->seq_new();

  EXPECT_TRUE(batch_->declare_step({a}));
  EXPECT_FALSE(batch_->declare_step({a + 5})); // never handed out
  EXPECT_FALSE(batch_->declare_step({})); // a step carries a token

  array q = randn(1), k = randn(1), v = randn(1);
  EXPECT_NO_THROW(attention_->attend(0, {0}, q, k, v, kScale, s));
  EXPECT_EQ(batch_->pos(a), 1);
}

// All kinds share capacity. Cell forks share their prefix without using
// more cells; private-history forks consume their retained logical length.
TEST_P(MLXBatchedCacheTest, CapacityAndCloneAdmissionFollowCacheKind) {
  ASSERT_NO_FATAL_FAILURE(
      create_cache(flat_config(/*capacity=*/4, 1, H, D, kHalf)));
  const int32_t a = *batch_->seq_new();
  const int32_t b = *batch_->seq_new();

  step({a, a, b}, {0, 1, 0}, randn(3), randn(3), randn(3));
  EXPECT_EQ(batch_->pos(a), 2);
  EXPECT_EQ(batch_->pos(b), 1);

  EXPECT_TRUE(batch_->declare_step({a})); // the fourth cell
  array q = randn(1), k = randn(1), v = randn(1);
  attention_->attend(0, {2}, q, k, v, kScale, s);

  EXPECT_FALSE(batch_->declare_step({a}));
  EXPECT_FALSE(batch_->declare_step({b}));
  const auto fork = batch_->seq_clone(a, std::nullopt);
  EXPECT_EQ(fork.has_value(), GetParam() == cache::kind::kBatchedCell);
  if (fork) {
    EXPECT_EQ(batch_->pos(*fork), 3);
    EXPECT_FALSE(batch_->declare_step({*fork}));
    EXPECT_TRUE(batch_->seq_rm(*fork));
  }

  EXPECT_TRUE(batch_->seq_rm(b));
  EXPECT_TRUE(batch_->declare_step({a}));
  attention_->attend(0, {3}, q, k, v, kScale, s);
  EXPECT_EQ(batch_->pos(a), 4);
  EXPECT_FALSE(batch_->declare_step({a})); // a reached the capacity limit
}

TEST_P(MLXBatchedCacheTest, SharedCapacityDeclarationReplacement) {
  ASSERT_NO_FATAL_FAILURE(create_cache(flat_config(4, 1, H, D, kHalf)));
  const int32_t a = *batch_->seq_new();
  const int32_t b = *batch_->seq_new();
  step({a, b, a}, {0, 0, 1}, randn(3), randn(3), randn(3));

  ASSERT_TRUE(batch_->declare_step({a}));
  ASSERT_TRUE(batch_->declare_step({b})); // replaces, rather than adds to, a
  EXPECT_FALSE(batch_->declare_step({a, b}));
  if (GetParam() != cache::kind::kBatchedCell) {
    EXPECT_FALSE(batch_->seq_clone(a, std::nullopt));
  }
  // Rejected declarations and clones leave b's admitted step intact.
  EXPECT_NO_THROW(
      attention_->attend(0, {1}, randn(1), randn(1), randn(1), kScale, s));
  EXPECT_EQ(batch_->pos(a), 2);
  EXPECT_EQ(batch_->pos(b), 2);
  EXPECT_FALSE(batch_->declare_step({a}));
  ASSERT_TRUE(batch_->rewind(b, 1));
  EXPECT_TRUE(batch_->declare_step({a}));
}

TEST_P(MLXBatchedCacheTest, SharedCapacityChargesRetainedClonePrefixes) {
  if (GetParam() == cache::kind::kBatchedCell) {
    GTEST_SKIP()
        << "Cell clones share cells instead of charging logical prefixes";
  }
  ASSERT_NO_FATAL_FAILURE(create_cache(flat_config(6, 1, H, D, kHalf)));
  const int32_t a = *batch_->seq_new();
  step({a, a, a}, {0, 1, 2}, randn(3), randn(3), randn(3));

  ASSERT_TRUE(batch_->declare_step({a}));
  const auto full = batch_->seq_clone(a, std::nullopt);
  ASSERT_TRUE(full);
  EXPECT_EQ(batch_->pos(*full), 3);
  // A successful clone invalidates the now-unaffordable declaration.
  EXPECT_ANY_THROW(
      attention_->attend(0, {3}, randn(1), randn(1), randn(1), kScale, s));
  EXPECT_FALSE(batch_->seq_clone(a, 1));
  EXPECT_FALSE(batch_->declare_step({a}));

  ASSERT_TRUE(batch_->rewind(*full, 1));
  EXPECT_FALSE(batch_->seq_clone(a, std::nullopt));
  const auto partial = batch_->seq_clone(a, 2);
  ASSERT_TRUE(partial);
  EXPECT_EQ(batch_->pos(*partial), 2);
  EXPECT_FALSE(batch_->declare_step({a}));
  ASSERT_TRUE(batch_->seq_rm(*full));
  EXPECT_TRUE(batch_->declare_step({a}));
  ASSERT_TRUE(batch_->seq_rm(*partial));
  const auto replacement = batch_->seq_clone(a, std::nullopt);
  ASSERT_TRUE(replacement);
  EXPECT_FALSE(batch_->declare_step({a}));
}

TEST_P(MLXBatchedCacheTest, SharedCapacityIncludesPartialForwards) {
  if (GetParam() != cache::kind::kBatched) {
    GTEST_SKIP() << "V2 commits positions only after all layers have advanced";
  }
  ASSERT_NO_FATAL_FAILURE(create_cache(flat_config(4, 2, H, D, kHalf)));
  const int32_t a = *batch_->seq_new();
  const int32_t b = *batch_->seq_new();
  array q = randn(2), k = randn(2), v = randn(2);
  ASSERT_TRUE(batch_->declare_step({a, a}));
  for (int layer = 0; layer < 2; ++layer) {
    attention_->attend(layer, {0, 1}, q, k, v, kScale, s);
  }
  ASSERT_EQ(batch_->pos(a), 2);
  ASSERT_TRUE(batch_->declare_step({b, b}));
  attention_->attend(0, {0, 1}, q, k, v, kScale, s);
  EXPECT_EQ(batch_->pos(b), 0);
  EXPECT_ANY_THROW(attention_->attend(2, {0, 1}, q, k, v, kScale, s));
  EXPECT_FALSE(batch_->declare_step({a}));
  EXPECT_FALSE(batch_->seq_clone(a, 1));
  EXPECT_FALSE(batch_->rewind(b, 1));

  // The remaining layer consumes the existing reservation, not new capacity.
  attention_->attend(1, {0, 1}, q, k, v, kScale, s);
  EXPECT_EQ(batch_->pos(b), 2);
  ASSERT_TRUE(batch_->rewind(b, 1));
  EXPECT_TRUE(batch_->declare_step({a}));
  EXPECT_FALSE(batch_->declare_step({a, a}));

  ASSERT_TRUE(batch_->rewind(b, 0));
  ASSERT_TRUE(batch_->declare_step({b, b}));
  attention_->attend(0, {0, 1}, q, k, v, kScale, s);
  EXPECT_FALSE(batch_->declare_step({a}));
  ASSERT_TRUE(batch_->rewind(b, 0)); // releases an unfinished reservation
  const auto fork = batch_->seq_clone(a, std::nullopt);
  ASSERT_TRUE(fork);
  EXPECT_FALSE(batch_->declare_step({a}));
  ASSERT_TRUE(batch_->seq_rm(*fork));
  EXPECT_TRUE(batch_->declare_step({a, a}));
  batch_->clear();
  const int32_t fresh = *batch_->seq_new();
  EXPECT_TRUE(batch_->declare_step(std::vector<int32_t>(4, fresh)));
}

// Every test uses the registry; also check the exposed capacity and admission.
TEST_P(MLXBatchedCacheTest, RegistryBuildsSelectedLayout) {
  ASSERT_NO_FATAL_FAILURE(create_cache(flat_config(32, 1, H, D, kHalf)));
  EXPECT_EQ(batch_->capacity(), 32);
  const auto a = batch_->seq_new();
  const auto b = batch_->seq_new();
  ASSERT_TRUE(a);
  ASSERT_TRUE(b);
  std::vector<int32_t> ids(32, *a);
  ids.push_back(*b);
  EXPECT_FALSE(batch_->declare_step(ids));
  ids.erase(ids.begin());
  EXPECT_TRUE(batch_->declare_step(ids));
}

// Exercise the default registration through its public faces. The parameter
// is -1 for ordinary attention, or the DFlash block size (including zero).
class MLXBatchedCacheRegistryTest : public ::testing::TestWithParam<int> {};

TEST_P(MLXBatchedCacheRegistryTest, GeometrySelectsPolicies) {
  using namespace ::mlx::core;
  constexpr int D = 8;
  const int block = GetParam();
  auto args = flat_config(8, 2, 1, D, static_cast<int>(ScalarType::Half));
  args.geometry.layers[1].policy = {cache::LayerPolicy::Kind::Ring, 2};
  args.config.max_write = 4;
  if (block >= 0) {
    args.geometry.dflash_block_size = block;
  }
  auto built = cache::CacheFactory::global().build(
      kMLXBackendId, cache::kind::kBatched, args.geometry, args.config);
  ASSERT_TRUE(built.ok());
  auto* batch = (*built)->as<cache::BatchControl>();
  auto* attention = (*built)->as<MLXCache>();
  ASSERT_NE(batch, nullptr);
  ASSERT_NE(attention, nullptr);
  EXPECT_EQ((*built)->as<cache::SequenceControl>(), nullptr);
  EXPECT_EQ(batch->capacity(), 8);
  const auto a = batch->seq_new();
  const auto b = batch->seq_new();
  ASSERT_TRUE(a);
  ASSERT_TRUE(b);
  const int n = 4 + std::max(block, 0);
  const int queries = block < 0 ? n : std::max(block, 1);
  std::vector<int32_t> ids, positions;
  for (int32_t id : {*a, *b}) {
    for (int32_t p = 0; p < n; ++p) {
      ids.push_back(id);
      positions.push_back(p);
    }
  }
  // Both owners fill the shared budget; temporary blocks do not consume it.
  ASSERT_TRUE(batch->declare_step(ids));
  const auto q = zeros({1, 1, 2 * queries, D}, float32);
  const auto k = zeros({1, 1, 2 * n, D}, float32);
  const auto values = broadcast_to(
      reshape(arange(1, n + 1, float32), {1, 1, n, 1}), {1, 1, n, D});
  const auto v = concatenate({values, add(values, array(100.0f))}, 2);
  for (int layer = 0; layer < 2; ++layer) {
    std::vector<float> means;
    if (block >= 0) {
      // Uniform, noncausal attention over history + block; layer 1 retains
      // only context values 3 and 4, but still attends to the entire block.
      means.assign(queries, (static_cast<float>(n) + (layer ? 3 : 1)) / 2);
    } else {
      means = layer ? std::vector<float>{1, 1.5f, 2.5f, 3.5f}
                    : std::vector<float>{1, 1.5f, 2, 2.5f};
    }
    const auto want = broadcast_to(
        array(means.data(), {1, 1, queries, 1}), {1, 1, queries, D});
    const auto out = attention->attend(layer, positions, q, k, v, 1.0f, {});
    EXPECT_EQ(out.shape(), (Shape{1, 1, 2 * queries, D}));
    EXPECT_TRUE(
        allclose(out, concatenate({want, add(want, array(100.0f))}, 2), 1e-4f));
  }
  EXPECT_EQ(batch->pos(*a), 4);
  EXPECT_EQ(batch->pos(*b), 4);
  EXPECT_FALSE(batch->seq_clone(*a, 1));
  EXPECT_FALSE(
      batch->declare_step(std::vector<int32_t>(std::max(block, 0) + 1, *a)));
  if (block > 0) {
    // Repeated temporary-only forwards must fit without advancing either owner.
    const auto temporary = zeros({1, 1, block, D}, float32);
    std::vector<int32_t> temporary_positions;
    for (int p = 4; p < 4 + block; ++p) {
      temporary_positions.push_back(p);
    }
    for (int step = 0; step < 2; ++step) {
      ASSERT_TRUE(batch->declare_step(std::vector<int32_t>(block, *a)));
      for (int layer = 0; layer < 2; ++layer) {
        eval(attention->attend(
            layer,
            temporary_positions,
            temporary,
            temporary,
            temporary,
            1.0f,
            {}));
      }
      EXPECT_EQ(batch->pos(*a), 4);
      EXPECT_EQ(batch->pos(*b), 4);
    }
  }
}

INSTANTIATE_TEST_SUITE_P(
    BlockSizes,
    MLXBatchedCacheRegistryTest,
    ::testing::Values(-1, 0, 1, 2));

class MLXDFlashBatchedCacheTest
    : public ::testing::TestWithParam<std::tuple<std::string, int, int>> {
 protected:
  struct Row {
    int32_t id;
    std::vector<float> history;
  };

  int block() const {
    return std::get<1>(GetParam());
  }
  int window() const {
    return std::get<2>(GetParam());
  }

  bool create_cache() {
    auto args = flat_config(64, 1, 1, 4, static_cast<int>(ScalarType::Half));
    args.geometry.dflash_block_size = block();
    args.config.max_write = 2;
    if (window()) {
      args.geometry.layers[0].policy = {
          cache::LayerPolicy::Kind::Ring, window()};
    }
    const auto& kind = std::get<0>(GetParam());
    if (kind != cache::kind::kBatched) {
      try {
        cache::CacheFactory::global().build(
            kMLXBackendId, kind, args.geometry, args.config);
        ADD_FAILURE() << kind << " accepted DFlash geometry";
      } catch (const std::runtime_error& error) {
        EXPECT_STREQ(
            error.what(),
            kind == cache::kind::kBatchedCell
                ? "MLXCellCache: DFlash is unsupported"
                : "MLXBatchedSequenceCache: DFlash is unsupported");
      }
      return false;
    }
    auto built = cache::CacheFactory::global().build(
        kMLXBackendId, kind, args.geometry, args.config);
    EXPECT_TRUE(built.ok());
    if (!built.ok()) {
      return false;
    }
    owner_ = *built;
    batch_ = owner_->as<cache::BatchControl>();
    attention_ = owner_->as<MLXCache>();
    EXPECT_NE(batch_, nullptr);
    EXPECT_NE(attention_, nullptr);
    return batch_ && attention_;
  }

  Row new_row() {
    const auto id = batch_->seq_new();
    EXPECT_TRUE(id);
    return {id.value_or(-1), {}};
  }

  void forward(std::vector<Row*> rows, int context = 2) {
    using namespace ::mlx::core;
    const int n = context + block();
    std::vector<std::vector<float>> incoming;
    std::vector<int> starts;
    std::vector<float> means;
    for (auto* row : rows) {
      starts.push_back(static_cast<int>(row->history.size()));
      std::vector<float> values;
      for (int p = 0; p < context; ++p) {
        values.push_back(
            100 * (row->id + 1) + 10 * generation_ + starts.back() + p);
      }
      row->history.insert(row->history.end(), values.begin(), values.end());
      // Only persistent context enters the host oracle's history.
      const int retained = window()
          ? std::min(window(), static_cast<int>(row->history.size()))
          : static_cast<int>(row->history.size());
      float sum = std::accumulate(
          row->history.end() - retained, row->history.end(), 0.0f);
      for (int p = 0; p < block(); ++p) {
        const float temporary = 1000 + 100 * generation_ + 10 * row->id + p;
        values.push_back(temporary);
        sum += temporary;
      }
      means.push_back(sum / (retained + block()));
      incoming.push_back(std::move(values));
    }
    ++generation_;

    std::vector<int32_t> ids, positions;
    std::vector<float> values, expected;
    // Interleave every K/V token. Q contains only the corresponding block
    // spans (the last dummy query for block zero), in the same packed order.
    for (int p = 0; p < n; ++p) {
      for (size_t r = 0; r < rows.size(); ++r) {
        ids.push_back(rows[r]->id);
        positions.push_back(starts[r] + p);
        values.push_back(incoming[r][p]);
        if (block() ? p >= context : p == n - 1) {
          expected.push_back(means[r]);
        }
      }
    }
    ASSERT_TRUE(batch_->declare_step(ids));
    const int kv_count = static_cast<int>(values.size());
    const int q_count = static_cast<int>(expected.size());
    const auto q = zeros({1, 1, q_count, 4}, float32);
    const auto k = zeros({1, 1, kv_count, 4}, float32);
    const auto v = broadcast_to(
        array(values.data(), {1, 1, kv_count, 1}), {1, 1, kv_count, 4});
    const auto want = broadcast_to(
        array(expected.data(), {1, 1, q_count, 1}), {1, 1, q_count, 4});
    const auto out = attention_->attend(0, positions, q, k, v, 1.0f, {});
    EXPECT_EQ(out.shape(), want.shape());
    EXPECT_TRUE(allclose(out, want, 1e-4f));
    for (auto* row : rows) {
      EXPECT_EQ(batch_->pos(row->id), row->history.size());
    }
  }

  std::shared_ptr<cache::Cache> owner_;
  cache::BatchControl* batch_ = nullptr;
  MLXCache* attention_ = nullptr;
  int generation_ = 0;
};

TEST_P(MLXDFlashBatchedCacheTest, InterleavedStepsExcludePreviousDrafts) {
  if (!create_cache()) {
    return;
  }
  auto a = new_row();
  auto b = new_row();
  // Offset subsequent two-token writes so they straddle the four-slot ring.
  ASSERT_NO_FATAL_FAILURE(forward({&a, &b}, 1));
  for (int step = 0; step < 5; ++step) {
    ASSERT_NO_FATAL_FAILURE(forward({&a, &b}));
  }
  EXPECT_EQ(batch_->pos(a.id), 11);
  EXPECT_EQ(batch_->pos(b.id), 11);
  if (block()) {
    ASSERT_NO_FATAL_FAILURE(forward({&a, &b}, 0));
    ASSERT_NO_FATAL_FAILURE(forward({&a, &b}, 0));
  }
}

TEST_P(MLXDFlashBatchedCacheTest, CloneAndRewindContinueRetainedHistory) {
  if (!create_cache()) {
    return;
  }
  auto source = new_row();
  for (int step = 0; step < 4; ++step) {
    ASSERT_NO_FATAL_FAILURE(forward({&source}));
  }
  const auto full_id = batch_->seq_clone(source.id, std::nullopt);
  const auto partial_id = batch_->seq_clone(source.id, 7);
  ASSERT_TRUE(full_id);
  ASSERT_TRUE(partial_id);
  Row full{*full_id, source.history};
  Row partial{*partial_id, {source.history.begin(), source.history.end() - 1}};
  const auto older = batch_->seq_clone(source.id, 6);
  EXPECT_EQ(older.has_value(), window() == 0);
  if (older) {
    EXPECT_TRUE(batch_->seq_rm(*older));
  }
  ASSERT_NO_FATAL_FAILURE(forward({&source, &full, &partial}, 1));
  ASSERT_TRUE(batch_->rewind(source.id, 8));
  source.history.resize(8);
  ASSERT_NO_FATAL_FAILURE(forward({&source}, 1));
  EXPECT_EQ(batch_->rewind(source.id, 3), window() == 0);
  if (!window()) {
    source.history.resize(3);
  }
  EXPECT_EQ(batch_->pos(source.id), source.history.size());
  ASSERT_NO_FATAL_FAILURE(forward({&source}, 1));
  ASSERT_TRUE(batch_->seq_rm(source.id));
  ASSERT_NO_FATAL_FAILURE(forward({&full, &partial}, 1));
}

TEST_P(MLXDFlashBatchedCacheTest, InterleavedQueriesKeepTheirOwnWeights) {
  if (!create_cache()) {
    return;
  }
  using namespace ::mlx::core;
  const auto a = new_row();
  const auto b = new_row();
  const int n = 2 + block();
  std::vector<int32_t> ids, positions;
  std::vector<float> queries, keys, vals, expected;
  for (int p = 0; p < n; ++p) {
    for (int r = 0; r < 2; ++r) {
      ids.push_back(r ? b.id : a.id);
      positions.push_back(p);
      keys.push_back(0.25f * (p + 1) * (r + 1));
      vals.push_back(100 * r + 7 * p + 1);
      if (block() ? p >= 2 : p == n - 1) {
        const float query = 0.5f * (block() ? p - 1 : 1) + 0.25f * r;
        queries.push_back(query);
        float weighted = 0, weights = 0;
        for (int j = 0; j < n; ++j) {
          const float weight = std::exp(query * 0.25f * (j + 1) * (r + 1));
          weighted += weight * (100 * r + 7 * j + 1);
          weights += weight;
        }
        expected.push_back(weighted / weights);
      }
    }
  }
  const auto tensor = [](const std::vector<float>& data) {
    const int count = static_cast<int>(data.size());
    return broadcast_to(array(data.data(), {1, 1, count, 1}), {1, 1, count, 4});
  };
  ASSERT_TRUE(batch_->declare_step(ids));
  // Four identical head dimensions and scale 1/4 give exp(query * key).
  const auto out = attention_->attend(
      0, positions, tensor(queries), tensor(keys), tensor(vals), 0.25f, {});
  EXPECT_TRUE(allclose(out, tensor(expected), 1e-4f));
  EXPECT_EQ(batch_->pos(a.id), 2);
  EXPECT_EQ(batch_->pos(b.id), 2);
}

INSTANTIATE_TEST_SUITE_P(
    CachePolicies,
    MLXDFlashBatchedCacheTest,
    ::testing::Combine(
        ::testing::Values(
            cache::kind::kBatched,
            cache::kind::kBatchedCell,
            cache::kind::kBatchedSequence),
        ::testing::Values(0, 2),
        ::testing::Values(0, 3)));

// Evicting rings refuse older prefixes; cell storage retains them even when
// attention is windowed. Recent retained prefixes are available in all kinds.
TEST_P(MLXBatchedCacheTest, PartialForkHonorsWindowedRetention) {
  const int window = 4, max_write = 1; // ring of window + max_write - 1 = 4
  ASSERT_NO_FATAL_FAILURE(
      create_cache(ring_config(32, window, max_write, H, D, kHalf)));
  auto oracle = make_cache<MLXSequenceCache>(
      ring_config(32, window, max_write, H, D, kHalf));
  const int32_t a = *batch_->seq_new();

  // Decode the donor to 10, feeding the oracle only the first 6.
  std::vector<array> qs, ks, vs;
  for (int32_t p = 0; p < 10; ++p) {
    qs.push_back(randn(1));
    ks.push_back(randn(1));
    vs.push_back(randn(1));
    step({a}, {p}, qs[p], ks[p], vs[p]);
    if (p < 6) {
      alone(oracle, {p}, qs[p], ks[p], vs[p]);
    }
  }

  // Rings have overwritten history needed at position 6; cells still hold it.
  const auto older_fork = batch_->seq_clone(a, /*upto=*/6);
  EXPECT_EQ(older_fork.has_value(), GetParam() == cache::kind::kBatchedCell);
  if (older_fork) {
    EXPECT_EQ(batch_->pos(*older_fork), 6);
    EXPECT_TRUE(batch_->seq_rm(*older_fork));
  }

  // One step back is still in the ring, and reads what the donor read.
  const auto fork = batch_->seq_clone(a, /*upto=*/9);
  ASSERT_TRUE(fork);
  EXPECT_EQ(batch_->pos(*fork), 9);

  for (int32_t p = 6; p < 9; ++p) {
    alone(oracle, {p}, qs[p], ks[p], vs[p]);
  }
  array q = randn(1), k = randn(1), v = randn(1);
  array got = step({*fork}, {9}, q, k, v);
  EXPECT_TRUE(allclose(got, alone(oracle, {9}, q, k, v), 1e-2f));
}

// A fork preserves its donor's retention boundary, including non-evicting
// cell storage, so cloning again cannot recover previously evicted history.
TEST_P(MLXBatchedCacheTest, ForkOfAForkKeepsTheDonorsFloor) {
  const int window = 4, max_write = 1; // ring of window + max_write - 1 = 4
  ASSERT_NO_FATAL_FAILURE(
      create_cache(ring_config(32, window, max_write, H, D, kHalf)));
  const int32_t a = *batch_->seq_new();
  for (int32_t p = 0; p < 10; ++p) {
    array q = randn(1), k = randn(1), v = randn(1);
    step({a}, {p}, q, k, v);
  }

  const auto b = batch_->seq_clone(a, /*upto=*/9);
  ASSERT_TRUE(b);
  const auto second_fork = batch_->seq_clone(*b, /*upto=*/8);
  EXPECT_EQ(second_fork.has_value(), GetParam() == cache::kind::kBatchedCell);
  if (second_fork) {
    EXPECT_EQ(batch_->pos(*second_fork), 8);
  }
  EXPECT_EQ(batch_->pos(*b), 9);
  EXPECT_EQ(batch_->pos(a), 10);
}

TEST_P(MLXBatchedCacheTest, SnapshotSurvivesSourceAndForkWrites) {
  ASSERT_NO_FATAL_FAILURE(create_cache(clone_config()));
  const auto source = batch_->seq_new();
  ASSERT_TRUE(source);
  const auto prefix = clone_tokens(21);
  clone_forward(*attention_, batch_, *source, prefix);
  const auto snapshot = batch_->seq_clone(*source, 21);
  ASSERT_TRUE(snapshot);
  ASSERT_EQ(batch_->pos(*snapshot), 21);

  // Overwrite every source ring slot after the snapshot, then release it.
  clone_forward(*attention_, batch_, *source, clone_tokens(41), prefix.size());
  ASSERT_TRUE(batch_->seq_rm(*source));

  auto request = prefix;
  request.insert(request.end(), {50, 51, 52, 53, 54, 55, 56});
  const auto first = batch_->seq_clone(*snapshot, 21);
  ASSERT_TRUE(first);
  ASSERT_NO_FATAL_FAILURE(expect_cold_clone_match(*first, 21, request));
  ASSERT_TRUE(batch_->seq_rm(*first));

  // The previous fork also wrapped; an equal prompt still replays its last
  // token against the untouched snapshot to obtain fresh logits.
  const auto equal = batch_->seq_clone(*snapshot, 20);
  ASSERT_TRUE(equal);
  ASSERT_NO_FATAL_FAILURE(expect_cold_clone_match(*equal, 20, prefix));
  ASSERT_TRUE(batch_->seq_rm(*equal));

  request.back() = 90;
  const auto independent = batch_->seq_clone(*snapshot, 21);
  ASSERT_TRUE(independent);
  ASSERT_TRUE(batch_->seq_rm(*snapshot));
  ASSERT_NO_FATAL_FAILURE(expect_cold_clone_match(*independent, 21, request));
  EXPECT_TRUE(batch_->seq_rm(*independent));
}

TEST_P(MLXBatchedCacheTest, RetainedBoundaryAndOlderSnapshot) {
  ASSERT_NO_FATAL_FAILURE(create_cache(clone_config()));
  const auto source = batch_->seq_new();
  ASSERT_TRUE(source);
  clone_forward(*attention_, batch_, *source, clone_tokens(21));
  const auto snapshot = batch_->seq_clone(*source, 21);
  ASSERT_TRUE(snapshot);
  ASSERT_EQ(batch_->pos(*snapshot), 21);
  ASSERT_TRUE(batch_->seq_rm(*source));

  // Ring size is 4 + 3 - 1 = 6. At position 21 it retains the keys needed
  // to resume at 18; only cell storage also retains the prefix for 17.
  auto boundary = clone_tokens(18);
  boundary.insert(boundary.end(), {80, 81, 82, 83});
  const auto hit = batch_->seq_clone(*snapshot, 18);
  ASSERT_TRUE(hit);
  ASSERT_NO_FATAL_FAILURE(expect_cold_clone_match(*hit, 18, boundary));
  ASSERT_TRUE(batch_->seq_rm(*hit));

  const auto shortened = batch_->seq_clone(*snapshot, 18);
  ASSERT_TRUE(shortened);
  ASSERT_NO_FATAL_FAILURE(
      expect_cold_clone_match(*shortened, 18, clone_tokens(19)));
  ASSERT_TRUE(batch_->seq_rm(*shortened));

  auto missing = clone_tokens(17);
  missing.insert(missing.end(), {80, 81, 82});
  const auto old_prefix = batch_->seq_clone(*snapshot, 17);
  ASSERT_EQ(old_prefix.has_value(), GetParam() == cache::kind::kBatchedCell);
  if (old_prefix) {
    ASSERT_NO_FATAL_FAILURE(expect_cold_clone_match(*old_prefix, 17, missing));
    ASSERT_TRUE(batch_->seq_rm(*old_prefix));
  }
  EXPECT_EQ(batch_->pos(*snapshot), 21);

  const auto older = batch_->seq_new();
  ASSERT_TRUE(older);
  clone_forward(*attention_, batch_, *older, clone_tokens(17));
  const auto older_snapshot = batch_->seq_clone(*older, 17);
  ASSERT_TRUE(older_snapshot);
  ASSERT_EQ(batch_->pos(*older_snapshot), 17);
  ASSERT_TRUE(batch_->seq_rm(*older));
  const auto retry = batch_->seq_clone(*snapshot, 17);
  EXPECT_EQ(retry.has_value(), GetParam() == cache::kind::kBatchedCell);
  if (retry) {
    EXPECT_TRUE(batch_->seq_rm(*retry));
  }
  const auto retained = batch_->seq_clone(*older_snapshot, 17);
  ASSERT_TRUE(retained);
  ASSERT_NO_FATAL_FAILURE(expect_cold_clone_match(*retained, 17, missing));
  EXPECT_TRUE(batch_->seq_rm(*retained));
  EXPECT_TRUE(batch_->seq_rm(*older_snapshot));
  EXPECT_TRUE(batch_->seq_rm(*snapshot));
}

INSTANTIATE_TEST_SUITE_P(
    CacheKinds,
    MLXBatchedCacheTest,
    ::testing::Values(
        cache::kind::kBatched,
        cache::kind::kBatchedSequence,
        cache::kind::kBatchedCell),
    [](const ::testing::TestParamInfo<std::string>& info) {
      if (info.param == cache::kind::kBatchedCell) {
        return "BatchedCell";
      }
      return info.param == cache::kind::kBatched ? "Batched"
                                                 : "BatchedSequence";
    });

} // namespace
