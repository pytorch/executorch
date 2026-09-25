/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/extension/llm/cache/cell_cache.h>
#include <executorch/extension/llm/cache/sequence_cache.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <map>
#include <memory>
#include <numeric>
#include <optional>
#include <vector>

#include <gtest/gtest.h>

namespace {

using namespace executorch::extension::llm::cache;
using Tokens = std::vector<uint64_t>;
using Logits = std::array<double, 3>;

constexpr int kWindow = 4;

const CacheGeometry kAttentionGeometry{
    {LayerGeometry{LayerPolicy{LayerPolicy::Kind::Flat, 0}, 1, 1},
     LayerGeometry{LayerPolicy{LayerPolicy::Kind::Ring, kWindow}, 1, 1}}};

CacheConfig attention_config(int max_write) {
  return CacheConfig{128, 0, 0, max_write};
}

Tokens tokens(int count) {
  Tokens result;
  for (int i = 0; i < count; ++i) {
    result.push_back((i * 7 + 3) % 19 + 1);
  }
  return result;
}

struct Slot {
  int position = -1;
  double key = 0;
  double value = 0;
};

struct Sequence {
  explicit Sequence(const CacheConfig& cfg)
      : planner(kAttentionGeometry, cfg),
        layers{
            std::vector<Slot>(cfg.capacity),
            std::vector<Slot>(kWindow + *cfg.max_write - 1)} {}

  SequenceCache planner;
  std::array<std::vector<Slot>, 2> layers;
};

// Only the byte storage is fake. Admission and every physical read/write run
// come from SequenceCache, including its high-water mark after a rewind.
class AttentionCache final : public BatchControl {
 public:
  explicit AttentionCache(int max_write)
      : cfg_(attention_config(max_write)), max_write_(max_write) {}

  int capacity() const override {
    return cfg_.capacity;
  }

  void clear() override {
    sequences_.clear();
  }

  bool declare_step(const std::vector<int32_t>& seq_ids) override {
    std::map<int32_t, int> counts;
    for (int32_t id : seq_ids) {
      ++counts[id];
    }
    for (const auto& [id, count] : counts) {
      const auto it = sequences_.find(id);
      if (it == sequences_.end() || count > max_write_ ||
          !it->second->planner.can_extend(count)) {
        return false;
      }
    }
    return true;
  }

  std::optional<int> max_seqs() const override {
    return std::nullopt;
  }

  std::optional<int32_t> seq_new() override {
    const int32_t id = next_id_++;
    sequences_.emplace(id, std::make_unique<Sequence>(cfg_));
    return id;
  }

  std::optional<int32_t> seq_clone(int32_t src, std::optional<int> upto)
      override {
    const auto it = sequences_.find(src);
    if (it == sequences_.end() || it->second->planner.length() == 0) {
      return std::nullopt;
    }
    auto copy = std::make_unique<Sequence>(*it->second);
    if (upto && !copy->planner.rewind(*upto)) {
      return std::nullopt;
    }
    const int32_t id = next_id_++;
    sequences_.emplace(id, std::move(copy));
    return id;
  }

  bool seq_rm(int32_t id) override {
    return sequences_.erase(id) != 0;
  }

  bool rewind(int32_t id, int position) override {
    const auto it = sequences_.find(id);
    return it != sequences_.end() && it->second->planner.rewind(position);
  }

  int pos(int32_t id) const override {
    const auto it = sequences_.find(id);
    return it == sequences_.end() ? -1 : it->second->planner.length();
  }

  std::vector<Logits> forward(int32_t id, const Tokens& input) {
    auto& sequence = *sequences_.at(id);
    return evaluate(sequence, input, sequence.planner.length(), id, true);
  }

  std::vector<Logits>
  replay_without_rewind(int32_t id, const Tokens& input, int position) {
    Sequence copy(*sequences_.at(id));
    return evaluate(copy, input, position, id, false);
  }

  int stale_reads() const {
    return stale_reads_;
  }

 private:
  std::vector<Logits> evaluate(
      Sequence& sequence,
      const Tokens& input,
      int begin,
      int32_t id,
      bool check_positions) {
    std::vector<Logits> result;
    for (int position = begin; position < static_cast<int>(input.size());) {
      const int count =
          std::min(max_write_, static_cast<int>(input.size()) - position);
      if (!declare_step(std::vector<int32_t>(count, id))) {
        ADD_FAILURE() << "step admission failed";
        return {};
      }
      std::vector<double> hidden(count);
      for (int i = 0; i < count; ++i) {
        hidden[i] = std::sin(input[position + i] * 0.17) +
            std::cos(input[position + i] * 0.07) + 0.01 * (position + i);
      }
      for (int layer = 0; layer < 2; ++layer) {
        const auto plan = sequence.planner.plan(layer, position, count);
        if (!plan) {
          ADD_FAILURE() << "no attention plan";
          return {};
        }
        auto& storage = sequence.layers[layer];
        int offset = 0;
        for (int run = 0; run < plan->n_write; ++run) {
          for (int row = 0; row < plan->write[run].len; ++row, ++offset) {
            const int p = position + offset;
            storage[plan->write[run].start + row] = Slot{
                p,
                hidden[offset] * 0.23 + std::sin(p * 0.13) * 0.07,
                hidden[offset] * 0.7 +
                    std::cos(input[p] * 0.31 + p * 0.11) * 0.1};
          }
        }
        for (int i = 0; i < count; ++i) {
          const int query_position = position + i;
          const double query = hidden[i] * 0.4 + 0.05;
          double weighted = 0;
          double denominator = 0;
          int logical = plan->read_base_pos;
          for (int run = 0; run < plan->n_read; ++run) {
            for (int row = 0; row < plan->read[run].len; ++row, ++logical) {
              if (logical > query_position ||
                  (layer == 1 && logical <= query_position - kWindow)) {
                continue;
              }
              const auto& slot = storage[plan->read[run].start + row];
              if (slot.position != logical) {
                ++stale_reads_;
                if (check_positions) {
                  ADD_FAILURE() << "layer " << layer << " position " << logical
                                << " reads physical row for " << slot.position;
                }
              }
              const double weight = std::exp(query * slot.key);
              weighted += weight * slot.value;
              denominator += weight;
            }
          }
          hidden[i] = std::tanh(
              0.6 * hidden[i] + 0.4 * weighted / denominator +
              0.09 * (layer + 1));
        }
        sequence.planner.commit(*plan);
      }
      for (double h : hidden) {
        result.push_back(Logits{h, 0.5 * h * h + 0.1 * h, std::sin(h)});
      }
      position += count;
    }
    return result;
  }

  CacheConfig cfg_;
  int max_write_;
  int32_t next_id_ = 0;
  int stale_reads_ = 0;
  std::map<int32_t, std::unique_ptr<Sequence>> sequences_;
};

std::vector<Logits> cold_logits(const Tokens& input, int max_write) {
  AttentionCache cold(max_write);
  const auto id = cold.seq_new();
  return cold.forward(*id, input);
}

void expect_cold_equivalent(
    AttentionCache& backend,
    int32_t source,
    const Tokens& input,
    int position,
    int max_write) {
  const auto id = backend.seq_clone(source, position);
  ASSERT_TRUE(id);
  ASSERT_EQ(backend.pos(*id), position);
  const auto actual = backend.forward(*id, input);
  // A different chunking schedule also checks the causal masks within a step.
  const auto expected = cold_logits(input, 1);
  ASSERT_EQ(actual.size(), input.size() - position);
  ASSERT_EQ(expected.size(), input.size());
  for (size_t i = 0; i < actual.size(); ++i) {
    for (size_t j = 0; j < actual[i].size(); ++j) {
      EXPECT_NEAR(actual[i][j], expected[position + i][j], 1e-12)
          << "max_write=" << max_write << ", position=" << position + i
          << ", logit=" << j;
    }
  }
  EXPECT_EQ(backend.stale_reads(), 0);
  EXPECT_TRUE(backend.seq_rm(*id));
}

class SequenceCloneAttentionTest : public ::testing::TestWithParam<int> {};

TEST_P(
    SequenceCloneAttentionTest,
    WrappedSnapshotSurvivesSourceAndBranchWrites) {
  const int max_write = GetParam();
  AttentionCache backend(max_write);
  const auto source = backend.seq_new();
  ASSERT_TRUE(source);
  const Tokens original = tokens(21);
  ASSERT_EQ(backend.forward(*source, original).size(), original.size());
  const auto snapshot = backend.seq_clone(*source, 21);
  ASSERT_TRUE(snapshot);
  ASSERT_EQ(backend.pos(*snapshot), 21);

  // Overwrite the live source's ring after its snapshot has been saved.
  ASSERT_EQ(backend.forward(*source, tokens(40)).size(), 19);
  ASSERT_TRUE(backend.seq_rm(*source));

  Tokens appended = original;
  appended.insert(appended.end(), {91, 92, 93});
  expect_cold_equivalent(backend, *snapshot, appended, 21, max_write);

  const int floor = 21 - max_write;
  Tokens branch = appended;
  branch[floor] += 31;
  expect_cold_equivalent(backend, *snapshot, branch, floor, max_write);

  expect_cold_equivalent(backend, *snapshot, original, 20, max_write);
  Tokens shorter(original.begin(), original.end() - 1);
  if (max_write == 1) {
    EXPECT_FALSE(backend.seq_clone(*snapshot, 19));
  } else {
    expect_cold_equivalent(backend, *snapshot, shorter, 19, max_write);
  }

  expect_cold_equivalent(backend, *snapshot, appended, 21, max_write);
  EXPECT_TRUE(backend.seq_rm(*snapshot));
}

TEST_P(SequenceCloneAttentionTest, OlderSnapshotRetainsAnEarlierBranchWindow) {
  const int max_write = GetParam();
  AttentionCache backend(max_write);
  const auto source = backend.seq_new();
  ASSERT_TRUE(source);
  ASSERT_EQ(backend.forward(*source, tokens(9)).size(), 9);
  const auto older = backend.seq_clone(*source, 9);
  ASSERT_TRUE(older);
  ASSERT_EQ(backend.pos(*older), 9);
  ASSERT_EQ(backend.forward(*source, tokens(21)).size(), 12);
  const auto newer = backend.seq_clone(*source, 21);
  ASSERT_TRUE(newer);
  ASSERT_EQ(backend.pos(*newer), 21);
  ASSERT_TRUE(backend.seq_rm(*source));

  Tokens branch = tokens(24);
  branch[8] += 31;
  EXPECT_FALSE(backend.seq_clone(*newer, 8));
  expect_cold_equivalent(backend, *older, branch, 8, max_write);
  EXPECT_TRUE(backend.seq_rm(*newer));
  EXPECT_TRUE(backend.seq_rm(*older));
}

TEST_P(
    SequenceCloneAttentionTest,
    MatchingWindowDoesNotReplaceMatchingEarlierHistory) {
  const int max_write = GetParam();
  AttentionCache backend(max_write);
  const auto source = backend.seq_new();
  ASSERT_TRUE(source);
  const Tokens original = tokens(21);
  ASSERT_EQ(backend.forward(*source, original).size(), original.size());
  const auto clone = backend.seq_clone(*source, 20);
  ASSERT_TRUE(clone);
  ASSERT_EQ(backend.pos(*clone), 20);

  Tokens changed = original;
  changed[0] += 31;
  // The controller knows positions, not tokens: the caller must validate the
  // whole prefix. This clone is physically valid but contains the wrong
  // history.
  const auto actual = backend.forward(*clone, changed);
  const auto changed_logits = cold_logits(changed, 1);
  ASSERT_EQ(actual.size(), 1);
  EXPECT_EQ(backend.stale_reads(), 0);
  EXPECT_GT(std::abs(actual.back()[0] - changed_logits.back()[0]), 1e-6);
  EXPECT_TRUE(backend.seq_rm(*clone));
  EXPECT_TRUE(backend.seq_rm(*source));
}

TEST_P(
    SequenceCloneAttentionTest,
    RejectsRewindWhosePhysicalRowsHaveBeenOverwritten) {
  const int max_write = GetParam();
  AttentionCache backend(max_write);
  const auto source = backend.seq_new();
  ASSERT_TRUE(source);
  const Tokens original = tokens(21);
  ASSERT_EQ(backend.forward(*source, original).size(), original.size());

  const int branch_position = 21 - max_write - 1;
  Tokens branch = tokens(24);
  branch[branch_position] += 31;
  EXPECT_FALSE(backend.rewind(*source, branch_position));
  EXPECT_FALSE(backend.seq_clone(*source, branch_position));
  EXPECT_EQ(backend.pos(*source), 21);
  expect_cold_equivalent(backend, *source, original, 21 - max_write, max_write);

  // Bypass rewind once to prove this oracle detects stale physical K/V rows.
  const auto unsafe =
      backend.replay_without_rewind(*source, branch, branch_position);
  const auto expected = cold_logits(branch, 1);
  ASSERT_FALSE(unsafe.empty());
  EXPECT_GT(backend.stale_reads(), 0);
  EXPECT_GT(std::abs(unsafe[0][0] - expected[branch_position][0]), 1e-6);
  EXPECT_TRUE(backend.seq_rm(*source));
}

INSTANTIATE_TEST_SUITE_P(
    RingWriteSizes,
    SequenceCloneAttentionTest,
    ::testing::Values(1, 3, 5));

std::vector<CellStep> append_cells(CellCache& control, int32_t id, int count) {
  std::vector<int32_t> positions(count);
  std::iota(positions.begin(), positions.end(), control.pos(id));
  if (!control.declare_step(std::vector<int32_t>(count, id))) {
    ADD_FAILURE() << "declare_step refused";
    return {};
  }
  std::vector<CellStep> steps;
  for (int layer = 0; layer < 2; ++layer) {
    const auto* step = control.place_step(layer, positions.data(), count);
    if (!step) {
      ADD_FAILURE() << "place_step refused";
      return {};
    }
    steps.push_back(*step);
  }
  return steps;
}

TEST(CellCloneTest, MixedAttentionMasksSurviveSourceAppendAndBranch) {
  CellCache control(kAttentionGeometry, attention_config(1));
  const auto source = control.seq_new();
  ASSERT_TRUE(source);
  ASSERT_EQ(append_cells(control, *source, 4).size(), 2);
  const auto snapshot = control.seq_clone(*source, 4);
  ASSERT_TRUE(snapshot);
  ASSERT_EQ(control.pos(*snapshot), 4);
  ASSERT_EQ(append_cells(control, *source, 2).size(), 2);

  const auto branch = control.seq_clone(*snapshot, 4);
  ASSERT_TRUE(branch);
  ASSERT_EQ(control.pos(*branch), 4);
  const auto steps = append_cells(control, *branch, 1);
  ASSERT_EQ(steps.size(), 2);
  EXPECT_EQ(steps[0].mask_bits, (std::vector<uint8_t>{1, 1, 1, 1, 0, 0, 1}));
  EXPECT_EQ(steps[1].mask_bits, (std::vector<uint8_t>{0, 1, 1, 1, 0, 0, 1}));
  EXPECT_EQ(control.pos(*source), 6);
  EXPECT_EQ(control.pos(*snapshot), 4);
  EXPECT_TRUE(control.seq_rm(*source));
  EXPECT_TRUE(control.seq_rm(*snapshot));
  EXPECT_EQ(control.pos(*branch), 5);
  EXPECT_EQ(control.free_cells(), control.capacity() - 5);
  EXPECT_TRUE(control.seq_rm(*branch));
  EXPECT_EQ(control.free_cells(), control.capacity());
}

TEST(CellCloneTest, SourceIdReuseDoesNotChangeClone) {
  CellCache control(kAttentionGeometry, attention_config(1));
  const auto source = control.seq_new();
  ASSERT_TRUE(source);
  ASSERT_EQ(append_cells(control, *source, 3).size(), 2);
  const auto clone = control.seq_clone(*source, 3);
  ASSERT_TRUE(clone);
  ASSERT_EQ(control.pos(*clone), 3);
  ASSERT_TRUE(control.seq_rm(*source));
  const auto reused = control.seq_new();
  ASSERT_TRUE(reused);
  ASSERT_EQ(*reused, *source);
  ASSERT_EQ(append_cells(control, *reused, 5).size(), 2);
  EXPECT_EQ(control.pos(*clone), 3);
  EXPECT_TRUE(control.seq_rm(*clone));
  EXPECT_EQ(control.pos(*reused), 5);
  EXPECT_TRUE(control.seq_rm(*reused));
  EXPECT_EQ(control.free_cells(), control.capacity());
}

} // namespace
