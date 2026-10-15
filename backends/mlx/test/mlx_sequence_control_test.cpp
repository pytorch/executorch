/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

// Direct, op-level SequenceControl tests; requires Apple Silicon / Metal.
#include "MLXCellCache.h"
#include "MLXSequenceCacheV2.h"
#include "backend_options.h"
#include "utils.h"

#include <executorch/extension/llm/cache/cache_registry.h>
#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <vector>

using namespace ::executorch::backends::mlx;
namespace cache = ::executorch::extension::llm::cache;
using ::mlx::core::array;

namespace {

// Explicitly test-only: cell caches expose BatchControl in production. Owning
// the whole cache and exactly one live id makes free cells its extension
// budget.
class CellSequenceAdapter final : public cache::SequenceControl {
 public:
  explicit CellSequenceAdapter(const CacheArgs& args)
      : cell(args.geometry, args.config), id(cell.seq_new().value()) {}

  int capacity() const override {
    return cell.capacity();
  }
  bool can_extend(int n = 1) const override {
    return n >= 0 && n <= cell.free_cells();
  }
  bool rewind(int position) override {
    return cell.rewind(id, position);
  }
  void clear() override {
    cell.clear();
    id = cell.seq_new().value();
  }

  MLXCellCache cell;
  int32_t id;
};

enum class Implementation { Cell, Sequence };

class MLXSequenceControlTest : public ::testing::TestWithParam<Implementation> {
 protected:
  static constexpr int kHeads = 2;
  static constexpr int kDim = 8;
  static constexpr int kFloat = static_cast<int>(ScalarType::Float);
  cache::SequenceControl* control_ = nullptr;
  MLXCache* attention_ = nullptr;
  std::unique_ptr<CellSequenceAdapter> cell_;
  std::unique_ptr<MLXSequenceCache> sequence_;
  int layers_ = 0;

  CacheArgs config(bool mixed = false, int capacity = 16) {
    auto args = flat_config(capacity, mixed ? 2 : 1, kHeads, kDim, kFloat, 2);
    args.config.max_write = 3;
    if (mixed) {
      args.geometry.layers[1].policy = {cache::LayerPolicy::Kind::Ring, 3};
    }
    return args;
  }

  void create(const CacheArgs& args) {
    layers_ = static_cast<int>(args.geometry.layers.size());
    if (GetParam() == Implementation::Cell) {
      cell_ = std::make_unique<CellSequenceAdapter>(args);
      control_ = cell_.get();
      attention_ = &cell_->cell;
    } else {
      // The sequence cache implements both interfaces directly.
      sequence_ =
          std::make_unique<MLXSequenceCache>(args.geometry, args.config);
      control_ = sequence_.get();
      attention_ = sequence_.get();
    }
  }

  int length() const {
    return cell_ ? cell_->cell.pos(cell_->id) : sequence_->length();
  }

  bool admit(int n) {
    return cell_ ? cell_->cell.declare_step(
                       std::vector<int32_t>(static_cast<size_t>(n), cell_->id))
                 : sequence_->can_admit(n);
  }

  array values(const std::vector<float>& data) {
    using namespace ::mlx::core;
    const int n = static_cast<int>(data.size());
    return broadcast_to(array(data.data(), {1, 1, n, 1}), {1, kHeads, n, kDim});
  }

  // Zero Q/K makes each answer the arithmetic mean of visible values. Expected
  // means are independent of either cache's attention implementation.
  void forward(
      int position,
      const std::vector<float>& data,
      const std::vector<std::vector<float>>& expected,
      int queries = -1) {
    using namespace ::mlx::core;
    const int n = static_cast<int>(data.size());
    if (queries < 0) {
      queries = n;
    }
    ASSERT_EQ(expected.size(), static_cast<size_t>(layers_));
    if (cell_) {
      ASSERT_TRUE(cell_->cell.declare_step(
          std::vector<int32_t>(data.size(), cell_->id)));
    }
    std::vector<int32_t> positions(data.size());
    std::iota(positions.begin(), positions.end(), position);
    const auto q = zeros({1, kHeads, queries, kDim}, float32);
    const auto k = zeros({1, kHeads, n, kDim}, float32);
    for (int layer = 0; layer < layers_; ++layer) {
      SCOPED_TRACE(layer);
      const auto& means = expected[static_cast<size_t>(layer)];
      ASSERT_EQ(means.size(), static_cast<size_t>(queries));
      const auto out =
          attention_->attend(layer, positions, q, k, values(data), 1.0f, {});
      EXPECT_EQ(out.shape(), (Shape{1, kHeads, queries, kDim}));
      EXPECT_EQ(out.dtype(), float32);
      EXPECT_TRUE(allclose(out, values(means), 1e-5f));
    }
  }
};

TEST_P(MLXSequenceControlTest, NumericalPrefillDecodeAndWrappedMixedLayers) {
  for (bool mixed : {false, true}) {
    SCOPED_TRACE(mixed);
    create(config(mixed));
    auto check = [&](int position,
                     const std::vector<float>& data,
                     const std::vector<float>& flat,
                     const std::vector<float>& swa) {
      std::vector<std::vector<float>> expected{flat};
      if (mixed) {
        expected.push_back(swa);
      }
      forward(position, data, expected);
    };
    ASSERT_NO_FATAL_FAILURE(check(0, {1, 2, 3}, {1, 1.5f, 2}, {1, 1.5f, 2}));
    // The second chunk crosses the five-slot ring boundary, exercising both
    // wrapped writes and a multi-query sliding mask.
    ASSERT_NO_FATAL_FAILURE(check(3, {4, 5, 6}, {2.5f, 3, 3.5f}, {3, 4, 5}));
    ASSERT_NO_FATAL_FAILURE(check(6, {7}, {4}, {6}));
    EXPECT_EQ(length(), 7);
  }
}

TEST_P(MLXSequenceControlTest, CapacityBoundsAdmissionEvenForRingStorage) {
  create(ring_config(6, 2, 3, kHeads, kDim, kFloat));
  EXPECT_EQ(control_->capacity(), 6);
  EXPECT_TRUE(control_->can_extend(6));
  EXPECT_FALSE(control_->can_extend(-1));
  EXPECT_FALSE(control_->can_extend(7));
  EXPECT_FALSE(admit(7));
  ASSERT_NO_FATAL_FAILURE(forward(0, {1, 2, 3}, {{1, 1.5f, 2.5f}}));
  EXPECT_TRUE(control_->can_extend(3));
  EXPECT_FALSE(control_->can_extend(4));
  EXPECT_FALSE(admit(4));
  EXPECT_EQ(length(), 3);
  ASSERT_NO_FATAL_FAILURE(forward(3, {4, 5, 6}, {{3.5f, 4.5f, 5.5f}}));
  EXPECT_TRUE(control_->can_extend(0));
  EXPECT_FALSE(control_->can_extend());
  EXPECT_FALSE(admit(1));
  EXPECT_EQ(length(), 6);
  const auto q = ::mlx::core::zeros({1, kHeads, 1, kDim}, ::mlx::core::float32);
  EXPECT_ANY_THROW(attention_->attend(0, {6}, q, q, values({99}), 1.0f, {}));
  EXPECT_EQ(length(), 6);
  EXPECT_FALSE(control_->can_extend());
  ASSERT_TRUE(control_->rewind(5));
  EXPECT_TRUE(control_->can_extend());
  ASSERT_NO_FATAL_FAILURE(forward(5, {9}, {{7}}));
}

TEST_P(MLXSequenceControlTest, InvalidQueriesLeaveCacheUsable) {
  using namespace ::mlx::core;
  for (const auto& shape : std::vector<Shape>{
           {1, kHeads, 0, kDim},
           {1, kHeads, 1, kDim},
           {1, kHeads, 3, kDim},
           {1, kDim}}) {
    SCOPED_TRACE(::testing::PrintToString(shape));
    create(config());
    ASSERT_NO_FATAL_FAILURE(forward(0, {1, 3}, {{1, 2}}));
    ASSERT_TRUE(admit(2));
    const auto q = zeros(shape, float32);
    const auto k = zeros({1, kHeads, 2, kDim}, float32);
    EXPECT_ANY_THROW(
        attention_->attend(0, {2, 3}, q, k, values({99, 99}), 1.0f, {}));
    ASSERT_EQ(length(), 2);
    EXPECT_TRUE(control_->can_extend(14));
    ASSERT_NO_FATAL_FAILURE(forward(2, {5, 7}, {{3, 4}}));
    EXPECT_EQ(length(), 4);
  }
}

TEST_P(MLXSequenceControlTest, InvalidPositionsLeaveCacheUsable) {
  using namespace ::mlx::core;
  for (const auto& positions : std::vector<std::vector<int32_t>>{
           {2, 3, 5}, {2, 2, 3}, {2, 3}, {2, 3, 4, 5}}) {
    SCOPED_TRACE(::testing::PrintToString(positions));
    create(config(true));
    ASSERT_NO_FATAL_FAILURE(forward(0, {1, 3}, {{1, 2}, {1, 2}}));
    ASSERT_TRUE(admit(3));
    const auto q = zeros({1, kHeads, 3, kDim}, float32);
    EXPECT_ANY_THROW(
        attention_->attend(0, positions, q, q, values({99, 99, 99}), 1.0f, {}));
    EXPECT_EQ(length(), 2);
    EXPECT_TRUE(control_->can_extend(14));
    ASSERT_NO_FATAL_FAILURE(forward(2, {5, 7, 9}, {{3, 4, 5}, {3, 5, 7}}));
    EXPECT_EQ(length(), 5);
  }
}

TEST_P(MLXSequenceControlTest, ZeroInitialCapacitySupportsPrefillAndGrowth) {
  auto args = config(true);
  args.config.initial_capacity = 0;
  create(args);
  EXPECT_EQ(length(), 0);
  EXPECT_EQ(control_->capacity(), 16);
  ASSERT_NO_FATAL_FAILURE(forward(0, {1, 2, 3}, {{1, 1.5f, 2}, {1, 1.5f, 2}}));
  EXPECT_EQ(length(), 3);
  ASSERT_NO_FATAL_FAILURE(forward(3, {4, 5, 6}, {{2.5f, 3, 3.5f}, {3, 4, 5}}));
  EXPECT_EQ(length(), 6);
  ASSERT_NO_FATAL_FAILURE(forward(6, {7, 8, 9}, {{4, 4.5f, 5}, {6, 7, 8}}));
  EXPECT_EQ(length(), 9);
  EXPECT_TRUE(control_->can_extend(7));
  EXPECT_FALSE(control_->can_extend(8));
}

TEST_P(MLXSequenceControlTest, WeightedAttentionMatchesIndependentReference) {
  using namespace ::mlx::core;
  const std::vector<float> queries{
      0.7f, -1.1f, 1.3f, -0.9f, 1.7f, 0.6f, -1.5f, 1.1f, -0.7f, 1.4f, -1.2f};
  const std::vector<float> keys{
      1.003f,
      -0.731f,
      0.219f,
      1.337f,
      -1.009f,
      0.583f,
      -0.417f,
      1.673f,
      0.907f,
      -1.219f,
      0.311f};
  const std::vector<float> vals{
      2.01f,
      -3.007f,
      5.003f,
      0.701f,
      -6.019f,
      4.031f,
      1.117f,
      -2.009f,
      7.013f,
      -4.007f,
      3.021f};
  for (auto dtype :
       {ScalarType::Float, ScalarType::Half, ScalarType::BFloat16}) {
    SCOPED_TRACE(static_cast<int>(dtype));
    auto args = config(true);
    args.config.kv_dtype = static_cast<int>(dtype);
    create(args);
    const auto storage_dtype = dtype == ScalarType::Half ? float16
        : dtype == ScalarType::BFloat16                  ? bfloat16
                                                         : float32;
    // Only numeric conversion is shared with MLX, not cache state or attention.
    const auto rounded_keys = astype(
        astype(
            array(keys.data(), {static_cast<int>(keys.size())}), storage_dtype),
        float32);
    const auto rounded_vals = astype(
        astype(
            array(vals.data(), {static_cast<int>(vals.size())}), storage_dtype),
        float32);
    eval(rounded_keys, rounded_vals);
    const float* stored_keys = rounded_keys.data<float>();
    const float* stored_vals = rounded_vals.data<float>();
    int position = 0;
    // Small initial capacity and a five-slot ring exercise growth and wrapping.
    for (int count : {3, 3, 1, 3, 1}) {
      SCOPED_TRACE(position);
      ASSERT_TRUE(admit(count));
      std::vector<int32_t> positions(static_cast<size_t>(count));
      std::iota(positions.begin(), positions.end(), position);
      const auto chunk = [&](const std::vector<float>& data) {
        return values(std::vector<float>(
            data.begin() + position, data.begin() + position + count));
      };
      const auto q = chunk(queries), k = chunk(keys), v = chunk(vals);
      for (int layer = 0; layer < layers_; ++layer) {
        SCOPED_TRACE(layer);
        std::vector<float> expected;
        for (int p = position; p < position + count; ++p) {
          const int begin = layer == 0 ? 0 : std::max(0, p - 2);
          double weighted = 0, weights = 0;
          for (int j = begin; j <= p; ++j) {
            const double weight =
                std::exp(static_cast<double>(queries[p]) * stored_keys[j]);
            weighted += weight * stored_vals[j];
            weights += weight;
          }
          expected.push_back(static_cast<float>(weighted / weights));
        }
        // Repeated head dimensions with scale 1/D give exp(query * key).
        const auto out =
            attention_->attend(layer, positions, q, k, v, 1.0f / kDim, {});
        EXPECT_EQ(out.shape(), (Shape{1, kHeads, count, kDim}));
        EXPECT_EQ(out.dtype(), float32);
        EXPECT_TRUE(allclose(out, values(expected), 1e-5f));
      }
      position += count;
      EXPECT_EQ(length(), position);
    }
  }
}

TEST_P(MLXSequenceControlTest, HalfAndBFloat16StorageCastsBeforeAttention) {
  for (auto dtype : {ScalarType::Half, ScalarType::BFloat16}) {
    SCOPED_TRACE(static_cast<int>(dtype));
    auto args = config();
    args.config.kv_dtype = static_cast<int>(dtype);
    create(args);
    // Exact round-to-nearest storage values, deliberately different from input.
    const float first = dtype == ScalarType::Half ? 1.0029296875f : 1.0f;
    const float second = dtype == ScalarType::Half ? 2.009765625f : 2.015625f;
    ASSERT_NO_FATAL_FAILURE(forward(0, {1.003f}, {{first}}));
    ASSERT_NO_FATAL_FAILURE(forward(1, {2.01f}, {{(first + second) / 2}}));
    EXPECT_EQ(length(), 2);
  }
}

TEST_P(MLXSequenceControlTest, RewindAppendAndClearReuse) {
  create(config(true));
  ASSERT_NO_FATAL_FAILURE(forward(0, {1, 2, 3}, {{1, 1.5f, 2}, {1, 1.5f, 2}}));
  ASSERT_NO_FATAL_FAILURE(forward(3, {99}, {{26.25f}, {104.0f / 3}}));
  EXPECT_FALSE(control_->rewind(-1));
  EXPECT_FALSE(control_->rewind(5));
  EXPECT_EQ(length(), 4);
  ASSERT_TRUE(control_->rewind(2));
  EXPECT_EQ(length(), 2);
  ASSERT_NO_FATAL_FAILURE(
      forward(2, {7, 9}, {{10.0f / 3, 4.75f}, {10.0f / 3, 6}}));
  control_->clear();
  EXPECT_EQ(length(), 0);
  EXPECT_EQ(control_->capacity(), 16);
  EXPECT_TRUE(control_->can_extend(16));
  ASSERT_NO_FATAL_FAILURE(forward(0, {20, 40}, {{20, 30}, {20, 30}}));
  ASSERT_TRUE(control_->rewind(0));
  EXPECT_EQ(length(), 0);
  ASSERT_NO_FATAL_FAILURE(forward(0, {8}, {{8}, {8}}));
}

TEST_P(MLXSequenceControlTest, HistoricalOverwritePreservesLaterSuffix) {
  using namespace ::mlx::core;
  create(config(true));
  ASSERT_NO_FATAL_FAILURE(forward(0, {1, 2, 3}, {{1, 1.5f, 2}, {1, 1.5f, 2}}));
  ASSERT_NO_FATAL_FAILURE(forward(3, {4, 5, 6}, {{2.5f, 3, 3.5f}, {3, 4, 5}}));
  const auto q = zeros({1, kHeads, 1, kDim}, float32);
  if (cell_) {
    // Cell steps reject occupied positions; sequence policies allow overwrites.
    ASSERT_TRUE(cell_->cell.declare_step({cell_->id}));
    EXPECT_ANY_THROW(attention_->attend(0, {3}, q, q, values({20}), 1.0f, {}));
    EXPECT_EQ(length(), 6);
    ASSERT_NO_FATAL_FAILURE(forward(6, {7}, {{4}, {6}}));
    return;
  }

  // No rewind: both writes end before the committed frontier at six. Their
  // attention must exclude the later suffix without deleting it from storage.
  ASSERT_NO_FATAL_FAILURE(forward(3, {20}, {{6.5f}, {25.0f / 3}}));
  EXPECT_EQ(length(), 6);
  EXPECT_EQ(sequence_->reserved_length(), 6);
  EXPECT_EQ(sequence_->admission_growth(1), std::optional<int>(1));
  ASSERT_NO_FATAL_FAILURE(forward(4, {50}, {{76.0f / 5}, {73.0f / 3}}));
  EXPECT_EQ(length(), 6);
  EXPECT_EQ(sequence_->reserved_length(), 6);
  // The untouched value six still participates, alongside both replacements.
  ASSERT_NO_FATAL_FAILURE(forward(6, {7}, {{89.0f / 7}, {21}}));
  EXPECT_EQ(length(), 7);

  // The ring has now evicted this prefix. Rejecting an overwrite must leave
  // both the committed frontier and the remaining K/V usable.
  EXPECT_ANY_THROW(attention_->attend(1, {1}, q, q, values({99}), 1.0f, {}));
  EXPECT_EQ(length(), 7);
  EXPECT_EQ(sequence_->reserved_length(), 7);
  ASSERT_NO_FATAL_FAILURE(forward(7, {8}, {{97.0f / 8}, {7}}));
}

TEST_P(MLXSequenceControlTest, CellsRetainHistoryThatRingStorageEvicts) {
  create(ring_config(16, 3, 1, kHeads, kDim, kFloat));
  for (int p = 0; p < 7; ++p) {
    const float mean =
        p < 2 ? static_cast<float>(p + 2) / 2 : static_cast<float>(p);
    ASSERT_NO_FATAL_FAILURE(forward(p, {static_cast<float>(p + 1)}, {{mean}}));
  }
  // Cells narrow the attention mask without evicting historical K/V. The
  // sequence's three-slot ring lacks the prefix needed to resume at position 3.
  const bool retained = GetParam() == Implementation::Cell;
  ASSERT_EQ(control_->rewind(3), retained);
  EXPECT_EQ(length(), retained ? 3 : 7);
  if (retained) {
    ASSERT_NO_FATAL_FAILURE(forward(3, {9}, {{14.0f / 3}}));
  } else {
    ASSERT_TRUE(control_->rewind(6));
    ASSERT_NO_FATAL_FAILURE(forward(6, {9}, {{20.0f / 3}}));
  }
}

TEST_P(MLXSequenceControlTest, DFlashRejectsCellOrPersistsOnlyContext) {
  for (int block : {0, 2}) {
    SCOPED_TRACE(block);
    auto args = config(true, 4);
    args.geometry.dflash_block_size = block;
    args.geometry.layers[1].policy.window = 2;
    args.config.max_write = 4;
    if (GetParam() == Implementation::Cell) {
      try {
        MLXCellCache unsupported(args.geometry, args.config);
        FAIL() << "Cell construction accepted DFlash";
      } catch (const std::runtime_error& error) {
        EXPECT_STREQ(error.what(), "MLXCellCache: DFlash is unsupported");
      }
      continue;
    }
    create(args);
    const int queries = std::max(block, 1);
    auto step = [&](int position,
                    std::vector<float> context,
                    float temporary,
                    float flat_mean,
                    float swa_mean) {
      if (block > 0) {
        context.insert(context.end(), {temporary, temporary + 100});
      }
      forward(
          position,
          context,
          {std::vector<float>(static_cast<size_t>(queries), flat_mean),
           std::vector<float>(static_cast<size_t>(queries), swa_mean)},
          queries);
    };
    ASSERT_NO_FATAL_FAILURE(
        step(0, {1, 3}, 100, block ? 76 : 2, block ? 76 : 2));
    EXPECT_EQ(length(), 2);
    ASSERT_NO_FATAL_FAILURE(
        step(2, {5, 7}, 300, block ? 716.0f / 6 : 4, block ? 178 : 6));
    EXPECT_EQ(length(), 4); // temporary tokens never consume capacity
    EXPECT_FALSE(control_->can_extend());
    EXPECT_FALSE(sequence_->can_admit(block + 1));
    control_->clear();
    EXPECT_EQ(length(), 0);
    EXPECT_TRUE(control_->can_extend(4));
    ASSERT_NO_FATAL_FAILURE(
        step(0, {9, 11}, 500, block ? 280 : 10, block ? 280 : 10));
    EXPECT_EQ(length(), 2);
  }
}

TEST(MLXSequenceControlFactoryTest, InvalidConstructionIsRejected) {
  using namespace ::mlx::core;
  const auto valid_args =
      flat_config(8, 1, 1, 8, static_cast<int>(ScalarType::Float));
  for (bool empty_geometry : {true, false}) {
    SCOPED_TRACE(empty_geometry);
    auto args = valid_args;
    if (empty_geometry) {
      args.geometry.layers.clear();
    } else {
      args.config.initial_capacity = -1;
    }
    EXPECT_THROW(
        MLXSequenceCache(args.geometry, args.config), std::invalid_argument);
    const auto built = cache::CacheFactory::global().build(
        kMLXBackendId, cache::kind::kSingle, args.geometry, args.config);
    ASSERT_FALSE(built.ok());
    EXPECT_EQ(built.error(), executorch::runtime::Error::InvalidArgument);
  }
  auto built = cache::CacheFactory::global().build(
      kMLXBackendId,
      cache::kind::kSingle,
      valid_args.geometry,
      valid_args.config);
  ASSERT_TRUE(built.ok());
  auto* control = (*built)->as<cache::SequenceControl>();
  auto* attention = (*built)->as<MLXCache>();
  ASSERT_NE(control, nullptr);
  ASSERT_NE(attention, nullptr);
  EXPECT_EQ((*built)->as<cache::BatchControl>(), nullptr);
  EXPECT_EQ(control->capacity(), 8);
  const auto q = zeros({1, 1, 2, 8}, float32);
  const auto v = broadcast_to(array({2.0f, 6.0f}, {1, 1, 2, 1}), {1, 1, 2, 8});
  const auto expected =
      broadcast_to(array({2.0f, 4.0f}, {1, 1, 2, 1}), {1, 1, 2, 8});
  const auto out = attention->attend(0, {0, 1}, q, q, v, 1.0f, {});
  EXPECT_TRUE(allclose(out, expected, 1e-5f));
  EXPECT_TRUE(control->can_extend(6));
  EXPECT_FALSE(control->can_extend(7));
}

TEST(MLXSequenceControlFactoryTest, RejectsUnequalPersistentGrowth) {
  using namespace ::mlx::core;
  auto args = flat_config(8, 2, 1, 1, static_cast<int>(ScalarType::Float));
  args.config.max_write = 3;
  auto draft_geometry = args.geometry;
  draft_geometry.dflash_block_size = 2;
  for (int first_layer : {0, 1}) {
    SCOPED_TRACE(first_layer);
    MLXSequenceCache c(
        args.geometry,
        args.config,
        {std::make_shared<sequence::detail::Full>(),
         std::make_shared<sequence::detail::DFlash>(draft_geometry)});
    // Both factories individually admit three K/V tokens, but Full persists
    // three and DFlash persists one. Preflight must reject their disagreement.
    EXPECT_FALSE(c.admission_growth(3));
    EXPECT_FALSE(c.can_admit(3));
    EXPECT_EQ(c.length(), 0);
    EXPECT_EQ(c.reserved_length(), 0);

    const auto attend = [&](int layer, int kv_count) {
      std::vector<int32_t> positions(static_cast<size_t>(kv_count));
      std::iota(positions.begin(), positions.end(), 0);
      const auto q = zeros({1, 1, layer == 0 ? kv_count : 2, 1}, float32);
      const auto k = zeros({1, 1, kv_count, 1}, float32);
      return c.attend(
          layer, positions, q, k, ones({1, 1, kv_count, 1}, float32), 1.0f, {});
    };
    const int persistent = first_layer == 0 ? 3 : 1;
    const auto first = attend(first_layer, 3);
    EXPECT_TRUE(allclose(first, ones(first.shape(), float32), 1e-5f));
    EXPECT_EQ(c.length(), 0);
    EXPECT_EQ(c.reserved_length(), persistent);
    try {
      attend(1 - first_layer, 3);
      FAIL() << "Mismatched layer growth completed the forward";
    } catch (const std::invalid_argument& error) {
      EXPECT_STREQ(error.what(), "sequence: unequal persistent ends");
    }
    EXPECT_EQ(c.length(), 0);
    EXPECT_EQ(c.reserved_length(), persistent);
    EXPECT_FALSE(c.can_fork(persistent));
    EXPECT_FALSE(c.can_admit(1));

    // Matching persistent growth completes the existing partial forward. This
    // also checks that the rejected layer was not marked visited or published.
    const int second_layer = 1 - first_layer;
    const auto second =
        attend(second_layer, persistent + (second_layer == 1 ? 2 : 0));
    EXPECT_TRUE(allclose(second, ones(second.shape(), float32), 1e-5f));
    EXPECT_EQ(c.length(), persistent);
    EXPECT_EQ(c.reserved_length(), persistent);
    EXPECT_TRUE(c.can_fork(persistent));
    c.clear();
    EXPECT_EQ(c.length(), 0);
    EXPECT_EQ(c.reserved_length(), 0);
  }
}

class MLXSequenceStorageCloneTest : public ::testing::Test {
 protected:
  void SetUp() override {
    // Cached Metal buffers can be larger than requested. Disable reuse so
    // buffer_size assertions measure this clone's actual allocation request.
    cache_limit_ = ::mlx::core::set_cache_limit(0);
    ::mlx::core::clear_cache();
  }
  void TearDown() override {
    ::mlx::core::set_cache_limit(cache_limit_);
  }

 private:
  size_t cache_limit_ = 0;
};

TEST_F(
    MLXSequenceStorageCloneTest,
    FlatViewsPreserveSlackUnlessGrowthIsSmaller) {
  using namespace ::mlx::core;
  const struct {
    int keep, view_slots, grown_slots;
  } cases[] = {{1, 4, 4}, {4, 4, 8}, {7, 7, 14}, {8, 16, 16}, {11, 16, 16}};
  for (auto dtype :
       {ScalarType::Float, ScalarType::Half, ScalarType::BFloat16}) {
    for (int heads : {1, 2}) {
      SCOPED_TRACE(heads);
      for (const auto& test : cases) {
        const int keep = test.keep;
        SCOPED_TRACE(static_cast<int>(dtype));
        SCOPED_TRACE(keep);
        auto args = flat_config(32, 1, heads, 8, static_cast<int>(dtype), 4);
        args.config.max_write = 16;
        sequence::detail::Storage source(args.geometry.layers[0], args.config);
        const auto storage_dtype = resolve_dtype(static_cast<int8_t>(dtype));
        const auto keys = astype(
            reshape(arange(heads * 11 * 8, float32), {1, heads, 11, 8}),
            storage_dtype);
        const sequence::KV original{keys, negative(keys)};
        source.update(0, original, {});
        ASSERT_EQ(source.buffer_slots(), 16);
        const auto source_snapshot = source.read(0, 11, {});
        ASSERT_TRUE(source_snapshot);

        auto clone = source.clone(keep, 0);
        EXPECT_EQ(clone.length(), keep);
        EXPECT_EQ(clone.capacity(), 32);
        EXPECT_EQ(clone.max_write(), 16);
        EXPECT_EQ(clone.buffer_slots(), test.view_slots);
        EXPECT_THROW(source.clone(-1, 0), std::invalid_argument);
        EXPECT_THROW(source.clone(12, 0), std::invalid_argument);
        EXPECT_EQ(source.length(), 11);
        auto sibling = clone.clone(keep, 0);

        // The source graph and prefix views are still lazy. Mutate both owners
        // before evaluation, keeping the original allocation alive throughout.
        const auto replacement = full({1, heads, 1, 8}, 300, storage_dtype);
        source.update(0, sequence::KV{replacement, negative(replacement)}, {});
        const auto parent = source.read(0, 1, {});
        const auto appended = full({1, heads, 2, 8}, 400, storage_dtype);
        clone.update(keep, sequence::KV{appended, negative(appended)}, {});
        const int child_slots = test.grown_slots;
        EXPECT_EQ(clone.buffer_slots(), child_slots);

        const auto prefix = sequence::detail::tokens(keys, 0, keep, {});
        const auto snapshot = sibling.read(0, keep, {});
        ASSERT_TRUE(snapshot);
        EXPECT_TRUE(allclose(snapshot->k, prefix, 1e-5f));
        EXPECT_TRUE(allclose(snapshot->v, negative(prefix), 1e-5f));
        // A read-only child still shares the parent's original allocation.
        eval(source_snapshot->k, source_snapshot->v);
        EXPECT_EQ(
            snapshot->k.buffer().ptr(), source_snapshot->k.buffer().ptr());
        EXPECT_EQ(
            snapshot->v.buffer().ptr(), source_snapshot->v.buffer().ptr());
        const auto continued = clone.read(0, keep + 2, {});
        ASSERT_TRUE(continued);
        const auto expected = concatenate({prefix, appended}, 2);
        EXPECT_TRUE(allclose(continued->k, expected, 1e-5f));
        EXPECT_TRUE(allclose(continued->v, negative(expected), 1e-5f));
        // Check physical allocation after writing both sliced views and
        // unchanged views that preserve the parent's spare slots.
        const auto bytes = keys.nbytes() / 11 * child_slots;
        EXPECT_EQ(continued->k.buffer_size(), bytes);
        EXPECT_EQ(continued->v.buffer_size(), bytes);
        EXPECT_NE(continued->k.buffer().ptr(), snapshot->k.buffer().ptr());
        EXPECT_NE(continued->v.buffer().ptr(), snapshot->v.buffer().ptr());
        ASSERT_TRUE(parent);
        EXPECT_TRUE(allclose(parent->k, replacement, 1e-5f));
        EXPECT_TRUE(allclose(parent->v, negative(replacement), 1e-5f));
      }
    }
  }
}

TEST_F(MLXSequenceStorageCloneTest, GrowthComparisonRespectsCapacityLimit) {
  using namespace ::mlx::core;
  auto args = flat_config(13, 1, 1, 1, static_cast<int>(ScalarType::Float), 4);
  args.config.max_write = 13;
  sequence::detail::Storage source(args.geometry.layers[0], args.config);
  const auto values = reshape(arange(11, float32), {1, 1, 11, 1});
  source.update(0, sequence::KV{values, values}, {});
  ASSERT_EQ(source.buffer_slots(), 13);
  for (int keep : {6, 7, 11}) {
    SCOPED_TRACE(keep);
    auto clone = source.clone(keep, 0);
    EXPECT_EQ(clone.buffer_slots(), keep == 6 ? 6 : 13);
    const auto appended = full({1, 1, 2, 1}, 100, float32);
    clone.update(keep, sequence::KV{appended, appended}, {});
    EXPECT_EQ(clone.buffer_slots(), keep == 6 ? 12 : 13);
    const auto actual = clone.read(0, keep + 2, {});
    ASSERT_TRUE(actual);
    const auto expected = concatenate(
        {sequence::detail::tokens(values, 0, keep, {}), appended}, 2);
    EXPECT_TRUE(allclose(actual->k, expected, 1e-5f));
    EXPECT_TRUE(allclose(actual->v, expected, 1e-5f));
    const auto bytes = values.itemsize() * (keep == 6 ? 12 : 13);
    EXPECT_EQ(actual->k.buffer_size(), bytes);
    EXPECT_EQ(actual->v.buffer_size(), bytes);
  }
  const auto unchanged = source.read(0, 11, {});
  ASSERT_TRUE(unchanged);
  EXPECT_TRUE(allclose(unchanged->k, values, 1e-5f));
  EXPECT_TRUE(allclose(unchanged->v, values, 1e-5f));

  const auto tail = full({1, 1, 2, 1}, 100, float32);
  source.update(11, sequence::KV{tail, tail}, {});
  auto full_clone = source.clone(13, 0);
  EXPECT_EQ(full_clone.length(), 13);
  EXPECT_EQ(full_clone.buffer_slots(), 13);
  const auto full_snapshot = full_clone.read(0, 13, {});
  ASSERT_TRUE(full_snapshot);
  const auto full_expected = concatenate({values, tail}, 2);
  EXPECT_TRUE(allclose(full_snapshot->k, full_expected, 1e-5f));
  EXPECT_TRUE(allclose(full_snapshot->v, full_expected, 1e-5f));
}

TEST_F(
    MLXSequenceStorageCloneTest,
    KeepsAllocationFloorAndDropsEmptyCloneBuffers) {
  using namespace ::mlx::core;
  auto args = flat_config(16, 1, 1, 1, static_cast<int>(ScalarType::Float), 8);
  sequence::detail::Storage source(args.geometry.layers[0], args.config);
  const auto values = reshape(arange(4, float32), {1, 1, 4, 1});
  source.update(0, sequence::KV{values, values}, {});
  auto clone = source.clone(4, 0);
  EXPECT_EQ(clone.buffer_slots(), source.buffer_slots());
  auto empty = source.clone(0, 0);
  EXPECT_EQ(empty.length(), 0);
  EXPECT_EQ(empty.highwater(), 0);
  EXPECT_EQ(empty.buffer_slots(), 0);
  EXPECT_EQ(source.length(), 4);
  EXPECT_EQ(source.buffer_slots(), 8);
  empty.update(0, sequence::KV{values, values}, {});
  const auto actual = empty.read(0, 4, {});
  ASSERT_TRUE(actual);
  EXPECT_TRUE(allclose(actual->k, values, 1e-5f));
  EXPECT_TRUE(allclose(actual->v, values, 1e-5f));

  source.update(4, sequence::KV{values, values}, {});
  source.update(8, sequence::KV{values, values}, {});
  ASSERT_EQ(source.buffer_slots(), 16);
  // The next append fits within the floor, even though growth beyond that
  // floor would reach the parent's buffer size.
  auto below_floor = source.clone(4, 0);
  EXPECT_EQ(below_floor.buffer_slots(), 8);
  below_floor.update(4, sequence::KV{values, values}, {});
  EXPECT_EQ(below_floor.buffer_slots(), 8);
  const auto continued = below_floor.read(0, 8, {});
  ASSERT_TRUE(continued);
  const auto expected = concatenate({values, values}, 2);
  EXPECT_TRUE(allclose(continued->k, expected, 1e-5f));
  EXPECT_TRUE(allclose(continued->v, expected, 1e-5f));
  EXPECT_EQ(continued->k.buffer_size(), 2 * values.nbytes());
  EXPECT_EQ(continued->v.buffer_size(), 2 * values.nbytes());
}

TEST_F(
    MLXSequenceStorageCloneTest,
    PreservesWrappedMappingAndEvictionBoundary) {
  using namespace ::mlx::core;
  auto args = ring_config(32, 4, 1, 1, 1, static_cast<int>(ScalarType::Float));
  args.config.max_write = 3;
  sequence::detail::Storage source(args.geometry.layers[0], args.config);
  const auto write = [](sequence::detail::Storage& storage, int position) {
    const auto values =
        reshape(arange(position, position + 3, float32), {1, 1, 3, 1});
    storage.update(position, sequence::KV{values, negative(values)}, {});
  };
  for (int position = 0; position < 21; position += 3) {
    write(source, position);
  }
  ASSERT_EQ(source.slots(), 6);
  auto clone = source.clone(18, 15);
  EXPECT_EQ(clone.length(), 18);
  EXPECT_EQ(clone.highwater(), 21);
  EXPECT_EQ(clone.buffer_slots(), 6);
  EXPECT_FALSE(clone.retains(14, 18));
  EXPECT_THROW(source.clone(17, 14), std::invalid_argument);
  auto current = source.clone(21, 18);
  for (int position = 21; position < 30; position += 3) {
    write(source, position);
  }
  source.clear();
  const auto retained = clone.read(15, 18, {});
  ASSERT_TRUE(retained);
  const auto expected = reshape(arange(15, 18, float32), {1, 1, 3, 1});
  EXPECT_TRUE(allclose(retained->k, expected, 1e-5f));
  EXPECT_TRUE(allclose(retained->v, negative(expected), 1e-5f));
  write(clone, 18);
  const auto wrapped = clone.read(15, 21, {});
  ASSERT_TRUE(wrapped);
  const auto all = reshape(arange(15, 21, float32), {1, 1, 6, 1});
  EXPECT_TRUE(allclose(wrapped->k, all, 1e-5f));
  EXPECT_TRUE(allclose(wrapped->v, negative(all), 1e-5f));
  const auto untouched = current.read(15, 21, {});
  ASSERT_TRUE(untouched);
  EXPECT_TRUE(allclose(untouched->k, all, 1e-5f));
  EXPECT_TRUE(allclose(untouched->v, negative(all), 1e-5f));
}

INSTANTIATE_TEST_SUITE_P(
    Implementations,
    MLXSequenceControlTest,
    ::testing::Values(Implementation::Cell, Implementation::Sequence),
    [](const ::testing::TestParamInfo<Implementation>& info) {
      return info.param == Implementation::Cell ? "Cell" : "Sequence";
    });

} // namespace
