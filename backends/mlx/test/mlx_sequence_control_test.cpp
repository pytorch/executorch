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
#include "utils.h"

#include <gtest/gtest.h>

#include <algorithm>
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

enum class Implementation { Cell, V2 };

class MLXSequenceControlTest : public ::testing::TestWithParam<Implementation> {
 protected:
  static constexpr int kHeads = 2;
  static constexpr int kDim = 8;
  static constexpr int kFloat = static_cast<int>(ScalarType::Float);
  cache::SequenceControl* control_ = nullptr;
  MLXCache* attention_ = nullptr;
  std::unique_ptr<CellSequenceAdapter> cell_;
  std::unique_ptr<MLXSequenceCacheV2> v2_;
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
      // V2 is itself both interfaces, not a registry Cache anchor.
      v2_ = std::make_unique<MLXSequenceCacheV2>(args.geometry, args.config);
      control_ = v2_.get();
      attention_ = v2_.get();
    }
  }

  int length() const {
    return cell_ ? cell_->cell.pos(cell_->id) : v2_->length();
  }

  bool admit(int n) {
    return cell_ ? cell_->cell.declare_step(
                       std::vector<int32_t>(static_cast<size_t>(n), cell_->id))
                 : v2_->can_admit(n);
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
  ASSERT_TRUE(control_->rewind(5));
  EXPECT_TRUE(control_->can_extend());
  ASSERT_NO_FATAL_FAILURE(forward(5, {9}, {{7}}));
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
    // Cell steps reject already occupied positions; overwrites are a V2 policy.
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
  EXPECT_EQ(v2_->reserved_length(), 6);
  EXPECT_EQ(v2_->admission_growth(1), std::optional<int>(1));
  ASSERT_NO_FATAL_FAILURE(forward(4, {50}, {{76.0f / 5}, {73.0f / 3}}));
  EXPECT_EQ(length(), 6);
  EXPECT_EQ(v2_->reserved_length(), 6);
  // The untouched value six still participates, alongside both replacements.
  ASSERT_NO_FATAL_FAILURE(forward(6, {7}, {{89.0f / 7}, {21}}));
  EXPECT_EQ(length(), 7);

  // The ring has now evicted this prefix. Rejecting an overwrite must leave
  // both the committed frontier and the remaining K/V usable.
  EXPECT_ANY_THROW(attention_->attend(1, {1}, q, q, values({99}), 1.0f, {}));
  EXPECT_EQ(length(), 7);
  EXPECT_EQ(v2_->reserved_length(), 7);
  ASSERT_NO_FATAL_FAILURE(forward(7, {8}, {{97.0f / 8}, {7}}));
}

TEST_P(MLXSequenceControlTest, CellsRetainHistoryThatRingV2Evicts) {
  create(ring_config(16, 3, 1, kHeads, kDim, kFloat));
  for (int p = 0; p < 7; ++p) {
    const float mean =
        p < 2 ? static_cast<float>(p + 2) / 2 : static_cast<float>(p);
    ASSERT_NO_FATAL_FAILURE(forward(p, {static_cast<float>(p + 1)}, {{mean}}));
  }
  // Cells narrow the attention mask without evicting historical K/V. V2's
  // three-slot ring no longer has the prefix needed to resume at position 3.
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
    EXPECT_FALSE(v2_->can_admit(block + 1));
    control_->clear();
    EXPECT_EQ(length(), 0);
    EXPECT_TRUE(control_->can_extend(4));
    ASSERT_NO_FATAL_FAILURE(
        step(0, {9, 11}, 500, block ? 280 : 10, block ? 280 : 10));
    EXPECT_EQ(length(), 2);
  }
}

TEST(MLXSequenceControlFactoryTest, RejectsUnequalPersistentGrowth) {
  using namespace ::mlx::core;
  auto args = flat_config(8, 2, 1, 1, static_cast<int>(ScalarType::Float));
  args.config.max_write = 3;
  auto draft_geometry = args.geometry;
  draft_geometry.dflash_block_size = 2;
  for (int first_layer : {0, 1}) {
    SCOPED_TRACE(first_layer);
    MLXSequenceCacheV2 c(
        args.geometry,
        args.config,
        {std::make_shared<sequence_v2::Full>(),
         std::make_shared<sequence_v2::DFlash>(draft_geometry)});
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
      EXPECT_STREQ(error.what(), "sequence v2: unequal persistent ends");
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

INSTANTIATE_TEST_SUITE_P(
    Implementations,
    MLXSequenceControlTest,
    ::testing::Values(Implementation::Cell, Implementation::V2),
    [](const ::testing::TestParamInfo<Implementation>& info) {
      return info.param == Implementation::Cell ? "Cell" : "V2";
    });

} // namespace
