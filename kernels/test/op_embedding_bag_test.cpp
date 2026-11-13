/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/kernels/test/FunctionHeaderWrapper.h> // Declares the operator
#include <executorch/kernels/test/TestUtil.h>
#include <executorch/kernels/test/supported_features.h>
#include <executorch/kernels/test/supported_features_skip.h>
#include <executorch/runtime/core/exec_aten/exec_aten.h>
#include <executorch/runtime/core/exec_aten/testing_util/tensor_factory.h>
#include <executorch/runtime/core/exec_aten/testing_util/tensor_util.h>

#include <gtest/gtest.h>

#ifdef ET_USE_THREADPOOL
#include <executorch/extension/threadpool/threadpool.h>
#endif

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <limits>
#include <numeric>
#include <vector>

using namespace ::testing;
using executorch::aten::ScalarType;
using executorch::aten::Tensor;
using std::optional;
using torch::executor::testing::TensorFactory;

namespace {

// Widens the threadpool so parallel_for splits work into several tasks even on
// single-core runners, and restores the previous size afterwards.
class ScopedMinThreadCount {
 public:
  explicit ScopedMinThreadCount(uint32_t threads) {
#ifdef ET_USE_THREADPOOL
    auto* pool = ::executorch::extension::threadpool::get_threadpool();
    if (pool != nullptr && pool->get_thread_count() < threads) {
      previous_ = static_cast<uint32_t>(pool->get_thread_count());
      pool->_unsafe_reset_threadpool(threads);
    }
#else
    (void)threads;
#endif
  }
  ~ScopedMinThreadCount() {
#ifdef ET_USE_THREADPOOL
    if (previous_ != 0) {
      ::executorch::extension::threadpool::get_threadpool()
          ->_unsafe_reset_threadpool(previous_);
    }
#endif
  }
  ScopedMinThreadCount(const ScopedMinThreadCount&) = delete;
  ScopedMinThreadCount& operator=(const ScopedMinThreadCount&) = delete;

 private:
  uint32_t previous_ = 0;
};

} // namespace

class OpEmbeddingBagOutTest : public OperatorTest {
 protected:
  std::tuple<Tensor&, Tensor&, Tensor&, Tensor&> op_embedding_bag_out(
      const Tensor& weight,
      const Tensor& indices,
      const Tensor& offsets,
      int64_t mode,
      const optional<Tensor>& per_sample_weights,
      bool include_last_offset,
      int64_t padding_idx,
      Tensor& out,
      Tensor& offset2bag,
      Tensor& bag_size,
      Tensor& max_indices) {
    return torch::executor::aten::_embedding_bag_outf(
        context_,
        weight,
        indices,
        offsets,
        /*scale_grad_by_freq=*/false,
        mode,
        /*sparse=*/false,
        per_sample_weights,
        include_last_offset,
        padding_idx,
        out,
        offset2bag,
        bag_size,
        max_indices);
  }

  template <ScalarType Weight, ScalarType Index, ScalarType Offset>
  void test_dtypes() {
    TensorFactory<Weight> tf;
    TensorFactory<Index> ti;
    TensorFactory<Offset> to;
    constexpr auto OutputIndex =
        Index == ScalarType::Long || Offset == ScalarType::Long
        ? ScalarType::Long
        : ScalarType::Int;
    TensorFactory<OutputIndex> tout;
    const auto weight = tf.make({2, 2}, {1, 2, 3, 4});
    const auto indices = ti.make({3}, {0, 1, 1});
    const auto offsets = to.make({2}, {0, 1});
    for (int64_t mode = 0; mode < 3; ++mode) {
      auto out = tf.zeros({2, 2});
      const bool needs_offset2bag = mode != 0 || Weight == ScalarType::Double;
      auto offset2bag = tout.full({needs_offset2bag ? 3 : 0}, -99);
      auto bag_size = tout.zeros({2});
      auto max_indices = mode == 2 ? tout.zeros({2, 2}) : tout.zeros({2});
      op_embedding_bag_out(
          weight,
          indices,
          offsets,
          mode,
          {},
          false,
          -1,
          out,
          offset2bag,
          bag_size,
          max_indices);
      EXPECT_TENSOR_EQ(
          out,
          mode == 0 ? tf.make({2, 2}, {1, 2, 6, 8})
                    : tf.make({2, 2}, {1, 2, 3, 4}));
      EXPECT_TENSOR_EQ(
          bag_size,
          mode == 0 ? tout.make({2}, {0, 0}) : tout.make({2}, {1, 2}));
      if (needs_offset2bag) {
        EXPECT_TENSOR_EQ(offset2bag, tout.make({3}, {0, 1, 1}));
      }
      if (mode == 2) {
        EXPECT_TENSOR_EQ(max_indices, tout.make({2, 2}, {0, 0, 1, 1}));
      }
    }
  }

  template <ScalarType DTYPE>
  void test_large_bag() {
    TensorFactory<DTYPE> tf;
    TensorFactory<ScalarType::Long> ti;
    auto weight = tf.ones({1, 1});
    auto indices = ti.make({4096}, std::vector<int64_t>(4096, 0));
    auto offsets = ti.make({1}, {0});
    auto scales = tf.ones({4096});
    for (int64_t mode = 0; mode < 2; ++mode) {
      auto out = tf.zeros({1, 1});
      auto offset2bag = ti.zeros({mode == 0 ? 0 : 4096});
      auto bag_size = ti.zeros({1});
      auto max_indices = ti.zeros({1});
      optional<Tensor> psw =
          mode == 0 ? optional<Tensor>(scales) : std::nullopt;
      op_embedding_bag_out(
          weight,
          indices,
          offsets,
          mode,
          psw,
          false,
          -1,
          out,
          offset2bag,
          bag_size,
          max_indices);
      EXPECT_TENSOR_EQ(out, tf.full({1, 1}, mode == 0 ? 4096 : 1));
    }
  }

  // Index outputs use the offsets dtype, as export allocates them, so the
  // optimized kernel takes its fast path whenever the weight allows it.
  template <ScalarType Index, ScalarType Offset>
  void expect_invalid_with_dtypes(
      const std::vector<int64_t>& index_values,
      const std::vector<int64_t>& offset_values,
      int64_t mode,
      int64_t dim,
      bool include_last,
      const optional<Tensor>& psw) {
    using IndexType = typename TensorFactory<Index>::ctype;
    using OffsetType = typename TensorFactory<Offset>::ctype;
    TensorFactory<ScalarType::Float> tf;
    TensorFactory<Index> ti;
    TensorFactory<Offset> to;
    auto weight = tf.ones({2, static_cast<int32_t>(dim)});
    auto indices = ti.make(
        {static_cast<int32_t>(index_values.size())},
        std::vector<IndexType>(index_values.begin(), index_values.end()));
    auto offsets = to.make(
        {static_cast<int32_t>(offset_values.size())},
        std::vector<OffsetType>(offset_values.begin(), offset_values.end()));
    int32_t bags =
        static_cast<int32_t>(offset_values.size()) - (include_last ? 1 : 0);
    auto out = tf.zeros({bags, static_cast<int32_t>(dim)});
    auto offset2bag =
        to.zeros({mode == 0 ? 0 : static_cast<int32_t>(index_values.size())});
    auto bag_size = to.zeros({bags});
    auto max_indices = mode == 2 ? to.zeros({bags, static_cast<int32_t>(dim)})
                                 : to.zeros({bags});
    ET_EXPECT_KERNEL_FAILURE(
        context_,
        op_embedding_bag_out(
            weight,
            indices,
            offsets,
            mode,
            psw,
            include_last,
            -1,
            out,
            offset2bag,
            bag_size,
            max_indices));
  }

  void expect_invalid(
      const std::vector<int64_t>& index_values,
      const std::vector<int64_t>& offset_values,
      int64_t mode = 0,
      int64_t dim = 1,
      bool include_last = false,
      const optional<Tensor>& psw = std::nullopt) {
    {
      SCOPED_TRACE("Long indices, Long offsets");
      expect_invalid_with_dtypes<ScalarType::Long, ScalarType::Long>(
          index_values, offset_values, mode, dim, include_last, psw);
    }
    context_.fail(torch::executor::Error::Ok);
    {
      SCOPED_TRACE("Int indices, Int offsets");
      expect_invalid_with_dtypes<ScalarType::Int, ScalarType::Int>(
          index_values, offset_values, mode, dim, include_last, psw);
    }
    context_.fail(torch::executor::Error::Ok);
    {
      SCOPED_TRACE("Int indices, Long offsets");
      expect_invalid_with_dtypes<ScalarType::Int, ScalarType::Long>(
          index_values, offset_values, mode, dim, include_last, psw);
    }
    // ATen requires the promoted Long dtype for the index outputs.
    if (!torch::executor::testing::SupportedFeatures::get()->is_aten) {
      context_.fail(torch::executor::Error::Ok);
      SCOPED_TRACE("Long indices, Int offsets");
      expect_invalid_with_dtypes<ScalarType::Long, ScalarType::Int>(
          index_values, offset_values, mode, dim, include_last, psw);
    }
  }

  template <ScalarType DTYPE>
  void test_partial_final_offset() {
    TensorFactory<DTYPE> tf;
    TensorFactory<ScalarType::Long> ti;
    auto weight = tf.make({2, 1}, {2, 7});
    auto indices = ti.make({2}, {0, 1});
    auto offsets = ti.make({2}, {0, 1});
    // ATen CPU references for the partial final offset, in sum/mean/max order.
    const std::vector<float> no_padding = DTYPE == ScalarType::Double
        ? std::vector<float>{9, 4.5, 7}
        : std::vector<float>{2, 1, 7};
    for (int64_t padding_idx : {-1, 0}) {
      for (int64_t mode = 0; mode < 3; ++mode) {
        SCOPED_TRACE(static_cast<int>(DTYPE));
        SCOPED_TRACE(padding_idx);
        SCOPED_TRACE(mode);
        auto out = tf.zeros({1, 1});
        const bool needs_offset2bag =
            mode != 0 || padding_idx == 0 || DTYPE == ScalarType::Double;
        auto offset2bag = ti.full({needs_offset2bag ? 2 : 0}, -99);
        auto bag_size = ti.full({1}, -99);
        auto max_indices = mode == 2 ? ti.full({1, 1}, -99) : ti.full({1}, -99);
        op_embedding_bag_out(
            weight,
            indices,
            offsets,
            mode,
            {},
            true,
            padding_idx,
            out,
            offset2bag,
            bag_size,
            max_indices);
        ASSERT_EQ(context_.failure_state(), torch::executor::Error::Ok);
        EXPECT_TENSOR_EQ(
            out, tf.full({1, 1}, padding_idx == 0 ? 7 : no_padding[mode]));
        if (needs_offset2bag) {
          EXPECT_TENSOR_EQ(offset2bag, ti.make({2}, {0, 0}));
        }
        const int64_t count = padding_idx == 0 ? 1 : 2;
        EXPECT_TENSOR_EQ(bag_size, ti.full({1}, mode == 0 ? 0 : count));
        EXPECT_TENSOR_EQ(
            max_indices,
            mode == 2 ? ti.full({1, 1}, 1)
                      : ti.full({1}, mode == 0 ? 0 : count));
      }
    }
  }

  template <ScalarType DTYPE>
  void test_sum_static_offset2bag() {
    TensorFactory<DTYPE> tf;
    TensorFactory<ScalarType::Long> ti;
    auto weight = tf.make({2, 2}, {1, 2, 3, 4});
    auto indices = ti.make({3}, {0, 1, 1});
    auto offsets = ti.make({2}, {0, 1});
    auto out = tf.zeros({2, 2});
    // Export sizes offset2bag with meta_embedding_bag, which allocates
    // num_indices even where eager ATen skips it.
    auto offset2bag = ti.full({3}, -99);
    auto bag_size = ti.full({2}, -99);
    auto max_indices = ti.full({2}, -99);
    op_embedding_bag_out(
        weight,
        indices,
        offsets,
        0,
        {},
        false,
        -1,
        out,
        offset2bag,
        bag_size,
        max_indices);
    ASSERT_EQ(context_.failure_state(), torch::executor::Error::Ok);
    EXPECT_TENSOR_EQ(out, tf.make({2, 2}, {1, 2, 6, 8}));
    EXPECT_TENSOR_EQ(offset2bag, ti.make({3}, {0, 1, 1}));
    EXPECT_TENSOR_EQ(bag_size, ti.zeros({2}));
    EXPECT_TENSOR_EQ(max_indices, ti.zeros({2}));
  }

  template <ScalarType DTYPE>
  void test_reductions(
      int32_t width,
      int64_t mode,
      bool last,
      bool padding,
      bool weighted = false) {
    TensorFactory<ScalarType::Float> tf;
    TensorFactory<DTYPE> ti;
    const std::array<std::array<float, 3>, 4> rows = {
        {{1, -2, 3}, {4, 5, -6}, {7, 8, 9}, {10, -11, 12}}};
    // Exact in float, so fused and unfused multiply-adds agree.
    const std::array<float, 4> scales = {0.5, -1, 2, 3};
    std::vector<float> weights;
    for (const auto& row : rows) {
      for (int32_t d = 0; d < width; ++d) {
        weights.push_back(row[d % 3]);
      }
    }
    auto weight = tf.make({4, width}, weights);
    auto indices = ti.make({4}, {1, 2, 1, 3});
    auto offsets = last ? ti.make({4}, {0, 2, 2, 4}) : ti.make({3}, {0, 2, 2});
    auto out = tf.zeros({3, width});
    auto offset2bag = ti.zeros({mode == 0 && !padding ? 0 : 4});
    auto bag_size = ti.zeros({3});
    auto max_indices = mode == 2 ? ti.zeros({3, width}) : ti.zeros({3});
    optional<Tensor> psw;
    if (weighted) {
      psw = tf.make({4}, std::vector<float>(scales.begin(), scales.end()));
    }
    op_embedding_bag_out(
        weight,
        indices,
        offsets,
        mode,
        psw,
        last,
        padding ? 1 : -1,
        out,
        offset2bag,
        bag_size,
        max_indices);
    ASSERT_EQ(context_.failure_state(), torch::executor::Error::Ok);
    std::vector<float> expected;
    for (int32_t bag = 0; bag < 3; ++bag) {
      // Bags 0 and 2 hold positions {0, 1} and {2, 3}; row 1 comes first.
      const size_t first = bag == 0 ? 0 : 2;
      const float scale_a = weighted ? scales[first] : 1;
      const float scale_b = weighted ? scales[first + 1] : 1;
      for (int32_t d = 0; d < width; ++d) {
        const float a = rows[1][d % 3];
        const float b = rows[bag == 0 ? 2 : 3][d % 3];
        float value = 0;
        if (bag == 1) {
          value = 0;
        } else if (padding) {
          value = b * scale_b;
        } else if (mode == 2) {
          value = std::max(a, b);
        } else {
          value = (a * scale_a + b * scale_b) / (mode == 1 ? 2 : 1);
        }
        expected.push_back(value);
      }
    }
    EXPECT_TENSOR_EQ(out, tf.make({3, width}, expected));
    if (offset2bag.numel() != 0) {
      EXPECT_TENSOR_EQ(offset2bag, ti.make({4}, {0, 0, 2, 2}));
    }
    EXPECT_TENSOR_EQ(
        bag_size,
        mode == 0     ? ti.zeros({3})
            : padding ? ti.make({3}, {1, 0, 1})
                      : ti.make({3}, {2, 0, 2}));
    if (mode != 2) {
      EXPECT_TENSOR_EQ(max_indices, bag_size);
    } else {
      const auto* actual =
          max_indices.template const_data_ptr<typename decltype(ti)::ctype>();
      for (int32_t d = 0; d < width; ++d) {
        EXPECT_EQ(actual[d], 2);
        EXPECT_EQ(actual[width + d], 0);
        EXPECT_EQ(actual[2 * width + d], padding || d % 3 != 1 ? 3 : 1);
      }
    }
  }

  template <ScalarType Index, ScalarType Offset>
  void test_weighted_sum_mixed_index_types() {
    TensorFactory<ScalarType::Float> tf;
    TensorFactory<Index> ti;
    TensorFactory<Offset> to;
    auto weight = tf.make({3, 2}, {1, 2, 3, 4, 5, 6});
    auto indices = ti.make({4}, {0, 1, 1, 2});
    auto offsets = to.make({3}, {0, 2, 4});
    auto scales = tf.make({4}, {0.5, -1, 2, 3});
    auto out = tf.zeros({2, 2});
    auto offset2bag = to.zeros({4});
    auto bag_size = to.zeros({2});
    auto max_indices = to.zeros({2});
    op_embedding_bag_out(
        weight,
        indices,
        offsets,
        0,
        scales,
        true,
        1,
        out,
        offset2bag,
        bag_size,
        max_indices);
    ASSERT_EQ(context_.failure_state(), torch::executor::Error::Ok);
    EXPECT_TENSOR_EQ(out, tf.make({2, 2}, {0.5, 1, 15, 18}));
    EXPECT_TENSOR_EQ(offset2bag, to.make({4}, {0, 0, 1, 1}));
    EXPECT_TENSOR_EQ(bag_size, to.zeros({2}));
    EXPECT_TENSOR_EQ(max_indices, to.zeros({2}));
  }

  // weight[r][c] = r * 3 + c, so a row is identifiable from its values.
  static Tensor ramp_weight(TensorFactory<ScalarType::Float>& tf) {
    std::vector<float> values(15);
    std::iota(values.begin(), values.end(), 0.0f);
    return tf.make({5, 3}, values);
  }
};

// All references come from torch.ops.aten._embedding_bag.

TEST_F(OpEmbeddingBagOutTest, Sum) {
  TensorFactory<ScalarType::Float> tf;
  TensorFactory<ScalarType::Long> tf_long;

  Tensor weight = ramp_weight(tf);
  Tensor indices = tf_long.make({4}, {4, 0, 1, 2});
  Tensor offsets = tf_long.make({2}, {0, 1});

  Tensor out = tf.zeros({2, 3});
  Tensor offset2bag = tf_long.zeros({0});
  Tensor bag_size = tf_long.zeros({2});
  Tensor max_indices = tf_long.zeros({2});

  op_embedding_bag_out(
      weight,
      indices,
      offsets,
      0,
      {},
      false,
      -1,
      out,
      offset2bag,
      bag_size,
      max_indices);

  EXPECT_TENSOR_EQ(out, tf.make({2, 3}, {12, 13, 14, 9, 12, 15}));
  // The sum mode leaves these for the backward pass to not need.
  EXPECT_TENSOR_EQ(offset2bag, tf_long.make({0}, {}));
  EXPECT_TENSOR_EQ(bag_size, tf_long.make({2}, {0, 0}));
  EXPECT_TENSOR_EQ(max_indices, tf_long.make({2}, {0, 0}));
}

TEST_F(OpEmbeddingBagOutTest, Mean) {
  TensorFactory<ScalarType::Float> tf;
  TensorFactory<ScalarType::Long> tf_long;

  Tensor weight = ramp_weight(tf);
  Tensor indices = tf_long.make({4}, {4, 0, 1, 2});
  Tensor offsets = tf_long.make({2}, {0, 1});

  Tensor out = tf.zeros({2, 3});
  Tensor offset2bag = tf_long.zeros({4});
  Tensor bag_size = tf_long.zeros({2});
  Tensor max_indices = tf_long.zeros({2});

  op_embedding_bag_out(
      weight,
      indices,
      offsets,
      1,
      {},
      false,
      -1,
      out,
      offset2bag,
      bag_size,
      max_indices);

  EXPECT_TENSOR_EQ(out, tf.make({2, 3}, {12, 13, 14, 3, 4, 5}));
  EXPECT_TENSOR_EQ(offset2bag, tf_long.make({4}, {0, 1, 1, 1}));
  EXPECT_TENSOR_EQ(bag_size, tf_long.make({2}, {1, 3}));
  EXPECT_TENSOR_EQ(max_indices, tf_long.make({2}, {1, 3}));
}

TEST_F(OpEmbeddingBagOutTest, Max) {
  TensorFactory<ScalarType::Float> tf;
  TensorFactory<ScalarType::Long> tf_long;

  Tensor weight = ramp_weight(tf);
  Tensor indices = tf_long.make({4}, {4, 0, 1, 2});
  Tensor offsets = tf_long.make({2}, {0, 1});

  Tensor out = tf.zeros({2, 3});
  Tensor offset2bag = tf_long.zeros({4});
  Tensor bag_size = tf_long.zeros({2});
  Tensor max_indices = tf_long.zeros({2, 3});

  op_embedding_bag_out(
      weight,
      indices,
      offsets,
      2,
      {},
      false,
      -1,
      out,
      offset2bag,
      bag_size,
      max_indices);

  EXPECT_TENSOR_EQ(out, tf.make({2, 3}, {12, 13, 14, 6, 7, 8}));
  EXPECT_TENSOR_EQ(offset2bag, tf_long.make({4}, {0, 1, 1, 1}));
  EXPECT_TENSOR_EQ(bag_size, tf_long.make({2}, {1, 3}));
  // max_indices holds the weight row each feature's maximum came from.
  EXPECT_TENSOR_EQ(max_indices, tf_long.make({2, 3}, {4, 4, 4, 2, 2, 2}));
}

TEST_F(OpEmbeddingBagOutTest, SumWithPerSampleWeights) {
  TensorFactory<ScalarType::Float> tf;
  TensorFactory<ScalarType::Long> tf_long;

  Tensor weight = ramp_weight(tf);
  Tensor indices = tf_long.make({4}, {4, 0, 1, 2});
  Tensor offsets = tf_long.make({2}, {0, 1});
  Tensor per_sample_weights = tf.make({4}, {2.0, 1.0, 0.5, -1.0});

  Tensor out = tf.zeros({2, 3});
  Tensor offset2bag = tf_long.zeros({0});
  Tensor bag_size = tf_long.zeros({2});
  Tensor max_indices = tf_long.zeros({2});

  op_embedding_bag_out(
      weight,
      indices,
      offsets,
      0,
      per_sample_weights,
      false,
      -1,
      out,
      offset2bag,
      bag_size,
      max_indices);

  EXPECT_TENSOR_CLOSE(out, tf.make({2, 3}, {24, 26, 28, -4.5, -4, -3.5}));
}

TEST_F(OpEmbeddingBagOutTest, PaddingIdxIsSkippedAndNotCounted) {
  TensorFactory<ScalarType::Float> tf;
  TensorFactory<ScalarType::Long> tf_long;

  Tensor weight = ramp_weight(tf);
  Tensor indices = tf_long.make({4}, {4, 0, 1, 2});
  Tensor offsets = tf_long.make({2}, {0, 1});

  Tensor out = tf.zeros({2, 3});
  Tensor offset2bag = tf_long.zeros({4});
  Tensor bag_size = tf_long.zeros({2});
  Tensor max_indices = tf_long.zeros({2});

  op_embedding_bag_out(
      weight,
      indices,
      offsets,
      1,
      {},
      false,
      /*padding_idx=*/0,
      out,
      offset2bag,
      bag_size,
      max_indices);

  // Row 0 drops out of the second bag, so it averages rows 1 and 2.
  EXPECT_TENSOR_CLOSE(out, tf.make({2, 3}, {12, 13, 14, 4.5, 5.5, 6.5}));
  EXPECT_TENSOR_EQ(bag_size, tf_long.make({2}, {1, 2}));
}

TEST_F(OpEmbeddingBagOutTest, IncludeLastOffset) {
  TensorFactory<ScalarType::Float> tf;
  TensorFactory<ScalarType::Long> tf_long;

  Tensor weight = ramp_weight(tf);
  Tensor indices = tf_long.make({4}, {4, 0, 1, 2});
  Tensor offsets = tf_long.make({3}, {0, 1, 4});

  Tensor out = tf.zeros({2, 3});
  Tensor offset2bag = tf_long.zeros({0});
  Tensor bag_size = tf_long.zeros({2});
  Tensor max_indices = tf_long.zeros({2});

  op_embedding_bag_out(
      weight,
      indices,
      offsets,
      0,
      {},
      /*include_last_offset=*/true,
      -1,
      out,
      offset2bag,
      bag_size,
      max_indices);

  EXPECT_TENSOR_EQ(out, tf.make({2, 3}, {12, 13, 14, 9, 12, 15}));
}

TEST_F(OpEmbeddingBagOutTest, EmptyBag) {
  TensorFactory<ScalarType::Float> tf;
  TensorFactory<ScalarType::Long> tf_long;

  Tensor weight = ramp_weight(tf);
  Tensor indices = tf_long.make({5}, {0, 1, 2, 3, 4});
  Tensor offsets = tf_long.make({3}, {0, 2, 2});

  Tensor out = tf.zeros({3, 3});
  Tensor offset2bag = tf_long.zeros({5});
  Tensor bag_size = tf_long.zeros({3});
  Tensor max_indices = tf_long.zeros({3});

  op_embedding_bag_out(
      weight,
      indices,
      offsets,
      1,
      {},
      false,
      -1,
      out,
      offset2bag,
      bag_size,
      max_indices);

  EXPECT_TENSOR_CLOSE(
      out, tf.make({3, 3}, {1.5, 2.5, 3.5, 0, 0, 0, 9, 10, 11}));
  EXPECT_TENSOR_EQ(offset2bag, tf_long.make({5}, {0, 0, 2, 2, 2}));
  EXPECT_TENSOR_EQ(bag_size, tf_long.make({3}, {2, 0, 3}));

  Tensor max_out = tf.zeros({3, 3});
  Tensor max_indices_2d = tf_long.zeros({3, 3});
  op_embedding_bag_out(
      weight,
      indices,
      offsets,
      2,
      {},
      false,
      -1,
      max_out,
      offset2bag,
      bag_size,
      max_indices_2d);

  EXPECT_TENSOR_EQ(max_out, tf.make({3, 3}, {3, 4, 5, 0, 0, 0, 12, 13, 14}));
  EXPECT_TENSOR_EQ(
      max_indices_2d, tf_long.make({3, 3}, {1, 1, 1, 0, 0, 0, 4, 4, 4}));
}

TEST_F(OpEmbeddingBagOutTest, PerSampleWeightsRejectedForMean) {
  TensorFactory<ScalarType::Float> tf;
  TensorFactory<ScalarType::Long> tf_long;

  Tensor weight = ramp_weight(tf);
  Tensor indices = tf_long.make({4}, {4, 0, 1, 2});
  Tensor offsets = tf_long.make({2}, {0, 1});
  Tensor per_sample_weights = tf.make({4}, {1, 1, 1, 1});

  Tensor out = tf.zeros({2, 3});
  Tensor offset2bag = tf_long.zeros({4});
  Tensor bag_size = tf_long.zeros({2});
  Tensor max_indices = tf_long.zeros({2});

  ET_EXPECT_KERNEL_FAILURE(
      context_,
      op_embedding_bag_out(
          weight,
          indices,
          offsets,
          1,
          per_sample_weights,
          false,
          -1,
          out,
          offset2bag,
          bag_size,
          max_indices));
}

TEST_F(OpEmbeddingBagOutTest, NegativeIndexWithoutPaddingRejected) {
  TensorFactory<ScalarType::Float> tf;
  TensorFactory<ScalarType::Long> ti;
  auto weight = tf.ones({2, 1});
  auto indices = ti.make({1}, {-1});
  auto offsets = ti.make({1}, {0});
  auto out = tf.zeros({1, 1});
  auto offset2bag = ti.zeros({0});
  auto bag_size = ti.zeros({1});
  auto max_indices = ti.zeros({1});
  ET_EXPECT_KERNEL_FAILURE(
      context_,
      op_embedding_bag_out(
          weight,
          indices,
          offsets,
          0,
          {},
          false,
          -1,
          out,
          offset2bag,
          bag_size,
          max_indices));
}

TEST_F(OpEmbeddingBagOutTest, NonzeroFirstOffsetRejected) {
  TensorFactory<ScalarType::Float> tf;
  TensorFactory<ScalarType::Long> ti;
  auto weight = tf.ones({2, 1});
  auto indices = ti.make({1}, {0});
  auto offsets = ti.make({1}, {1});
  auto out = tf.zeros({1, 1});
  auto offset2bag = ti.zeros({0});
  auto bag_size = ti.zeros({1});
  auto max_indices = ti.zeros({1});
  ET_EXPECT_KERNEL_FAILURE(
      context_,
      op_embedding_bag_out(
          weight,
          indices,
          offsets,
          0,
          {},
          false,
          -1,
          out,
          offset2bag,
          bag_size,
          max_indices));
}

TEST_F(OpEmbeddingBagOutTest, SupportedDtypes) {
#define TEST_ENTRY(CTYPE, DTYPE)                                        \
  test_dtypes<ScalarType::DTYPE, ScalarType::Int, ScalarType::Int>();   \
  test_dtypes<ScalarType::DTYPE, ScalarType::Long, ScalarType::Long>(); \
  test_dtypes<ScalarType::DTYPE, ScalarType::Int, ScalarType::Long>();  \
  test_dtypes<ScalarType::DTYPE, ScalarType::Long, ScalarType::Int>();
  ET_FORALL_FLOATHBF16_TYPES(TEST_ENTRY);
#undef TEST_ENTRY
}

TEST_F(OpEmbeddingBagOutTest, LargeHalfBagPrecision) {
  test_large_bag<ScalarType::Half>();
}
TEST_F(OpEmbeddingBagOutTest, LargeBFloat16BagPrecision) {
  test_large_bag<ScalarType::BFloat16>();
}
TEST_F(OpEmbeddingBagOutTest, IndexAtRowCountRejected) {
  expect_invalid({2}, {0});
}
TEST_F(OpEmbeddingBagOutTest, NegativeIndexRejectedForMax) {
  expect_invalid({-2}, {0}, 2);
}
TEST_F(OpEmbeddingBagOutTest, InvalidIndexWithZeroWidthRejected) {
  expect_invalid({2}, {0}, 0, 0);
}
TEST_F(OpEmbeddingBagOutTest, NegativeOffsetRejected) {
  expect_invalid({0, 1}, {-1});
}
TEST_F(OpEmbeddingBagOutTest, DecreasingOffsetsRejected) {
  expect_invalid({0, 1}, {0, 2, 1});
}
TEST_F(OpEmbeddingBagOutTest, OffsetPastEndRejected) {
  expect_invalid({0, 1}, {0, 3});
}
TEST_F(OpEmbeddingBagOutTest, InvalidFinalOffsetRejected) {
  expect_invalid({0, 1}, {0, 3}, 0, 1, true);
}
TEST_F(OpEmbeddingBagOutTest, InvalidModeRejected) {
  expect_invalid({0}, {0}, 3);
}
TEST_F(OpEmbeddingBagOutTest, WrongPerSampleWeightsLengthRejected) {
  TensorFactory<ScalarType::Float> tf;
  expect_invalid({0, 1}, {0}, 0, 1, false, tf.ones({1}));
}
TEST_F(OpEmbeddingBagOutTest, EmptyPerSampleWeightsRejectedForNonemptyIndices) {
  TensorFactory<ScalarType::Float> tf;
  expect_invalid({0}, {0}, 0, 1, false, tf.zeros({0}));
}
TEST_F(OpEmbeddingBagOutTest, WrongPerSampleWeightsRankRejected) {
  TensorFactory<ScalarType::Float> tf;
  expect_invalid({0, 1}, {0}, 0, 1, false, tf.ones({1, 2}));
}
TEST_F(OpEmbeddingBagOutTest, WrongIndicesDtypeRejected) {
  ET_SKIP_IF(
      torch::executor::testing::SupportedFeatures::get()->is_aten,
      "ATen promotes Short indices to the offsets dtype");
  expect_invalid_with_dtypes<ScalarType::Short, ScalarType::Long>(
      {0}, {0}, 0, 1, false, std::nullopt);
}
TEST_F(OpEmbeddingBagOutTest, WrongOffsetsDtypeRejected) {
  expect_invalid_with_dtypes<ScalarType::Long, ScalarType::Short>(
      {0}, {0}, 0, 1, false, std::nullopt);
}

TEST_F(OpEmbeddingBagOutTest, WrongAuxiliaryDtypeRejected) {
  TensorFactory<ScalarType::Float> tf;
  TensorFactory<ScalarType::Long> ti;
  auto weight = tf.ones({2, 1});
  auto indices = ti.make({1}, {0});
  auto offsets = ti.make({1}, {0});
  for (int which = 0; which < 3; ++which) {
    context_.fail(torch::executor::Error::Ok);
    auto out = tf.zeros({1, 1});
    auto offset2bag = which == 0 ? tf.zeros({0}) : ti.zeros({0});
    auto bag_size = which == 1 ? tf.zeros({1}) : ti.zeros({1});
    auto max_indices = which == 2 ? tf.zeros({1}) : ti.zeros({1});
    ET_EXPECT_KERNEL_FAILURE(
        context_,
        op_embedding_bag_out(
            weight,
            indices,
            offsets,
            0,
            {},
            false,
            -1,
            out,
            offset2bag,
            bag_size,
            max_indices));
  }
}

TEST_F(OpEmbeddingBagOutTest, PartialFinalOffsetIsValid) {
  TensorFactory<ScalarType::Float> tf;
  TensorFactory<ScalarType::Long> ti;
  auto weight = tf.make({2, 1}, {2, 7});
  auto indices = ti.make({2}, {0, 1});
  auto offsets = ti.make({2}, {0, 1});
  auto out = tf.zeros({1, 1});
  auto offset2bag = ti.zeros({0});
  auto bag_size = ti.zeros({1});
  auto max_indices = ti.zeros({1});
  op_embedding_bag_out(
      weight,
      indices,
      offsets,
      0,
      {},
      true,
      -1,
      out,
      offset2bag,
      bag_size,
      max_indices);
  EXPECT_TENSOR_EQ(out, tf.make({1, 1}, {2}));
}

TEST_F(OpEmbeddingBagOutTest, EmptyInputsWithPerSampleWeights) {
  TensorFactory<ScalarType::Float> tf;
  TensorFactory<ScalarType::Long> ti;
  auto weight = tf.ones({2, 1});
  auto indices = ti.zeros({0});
  auto offsets = ti.make({1}, {0});
  auto psw = tf.zeros({0});
  auto out = tf.ones({1, 1});
  auto offset2bag = ti.zeros({0});
  auto bag_size = ti.zeros({1});
  auto max_indices = ti.zeros({1});
  op_embedding_bag_out(
      weight,
      indices,
      offsets,
      0,
      psw,
      false,
      -1,
      out,
      offset2bag,
      bag_size,
      max_indices);
  EXPECT_TENSOR_EQ(out, tf.zeros({1, 1}));
}

TEST_F(OpEmbeddingBagOutTest, SumWithPaddingStaticAuxiliaryOutputs) {
  TensorFactory<ScalarType::Float> tf;
  TensorFactory<ScalarType::Long> ti;
  auto weight = tf.make({3, 2}, {10, 20, 1, 2, 3, 4});
  auto indices = ti.make({3}, {0, 1, 2});
  auto offsets = ti.make({1}, {0});
  auto scales = tf.make({3}, {100, 2, 3});
  for (bool weighted : {false, true}) {
    SCOPED_TRACE(weighted);
    auto out = tf.zeros({1, 2});
    // Match the static auxiliary shape allocated by export, including padding.
    auto offset2bag = ti.full({3}, -99);
    auto bag_size = ti.full({1}, -99);
    auto max_indices = ti.full({1}, -99);
    op_embedding_bag_out(
        weight,
        indices,
        offsets,
        0,
        weighted ? optional<Tensor>(scales) : std::nullopt,
        false,
        0,
        out,
        offset2bag,
        bag_size,
        max_indices);
    ASSERT_EQ(context_.failure_state(), torch::executor::Error::Ok);
    EXPECT_TENSOR_EQ(
        out, weighted ? tf.make({1, 2}, {11, 16}) : tf.make({1, 2}, {4, 6}));
    EXPECT_TENSOR_EQ(offset2bag, ti.make({3}, {0, 0, 0}));
    EXPECT_TENSOR_EQ(bag_size, ti.zeros({1}));
    EXPECT_TENSOR_EQ(max_indices, ti.zeros({1}));
  }
}

TEST_F(OpEmbeddingBagOutTest, TransposedWeightAllModes) {
  TensorFactory<ScalarType::Float> tf;
  TensorFactory<ScalarType::Long> ti;
  // Equivalent to tensor([[1, 2, 3], [4, 5, 6]]).t(): strides (1, 3).
  auto weight = tf.make_with_dimorder({3, 2}, {1, 2, 3, 4, 5, 6}, {1, 0});
  auto indices = ti.make({2}, {0, 2});
  auto offsets = ti.make({1}, {0});
  auto scales = tf.make({2}, {2, 3});
  for (int64_t mode = 0; mode < 3; ++mode) {
    SCOPED_TRACE(mode);
    auto out = tf.zeros({1, 2});
    auto offset2bag = ti.full({2}, -99);
    auto bag_size = ti.full({1}, -99);
    auto max_indices = mode == 2 ? ti.full({1, 2}, -99) : ti.full({1}, -99);
    op_embedding_bag_out(
        weight,
        indices,
        offsets,
        mode,
        {},
        false,
        -1,
        out,
        offset2bag,
        bag_size,
        max_indices);
    ASSERT_EQ(context_.failure_state(), torch::executor::Error::Ok);
    EXPECT_TENSOR_EQ(
        out,
        mode == 0       ? tf.make({1, 2}, {4, 10})
            : mode == 1 ? tf.make({1, 2}, {2, 5})
                        : tf.make({1, 2}, {3, 6}));
    EXPECT_TENSOR_EQ(offset2bag, ti.make({2}, {0, 0}));
    EXPECT_TENSOR_EQ(bag_size, ti.full({1}, mode == 0 ? 0 : 2));
    EXPECT_TENSOR_EQ(
        max_indices,
        mode == 2 ? ti.make({1, 2}, {2, 2}) : ti.full({1}, mode == 0 ? 0 : 2));
    if (mode == 0) {
      op_embedding_bag_out(
          weight,
          indices,
          offsets,
          mode,
          scales,
          false,
          -1,
          out,
          offset2bag,
          bag_size,
          max_indices);
      EXPECT_TENSOR_EQ(out, tf.make({1, 2}, {11, 26}));
    }
  }
}

TEST_F(OpEmbeddingBagOutTest, PartialFinalOffsetAllModesAndDtypes) {
#define TEST_ENTRY(CTYPE, DTYPE) test_partial_final_offset<ScalarType::DTYPE>();
  ET_FORALL_FLOATHBF16_TYPES(TEST_ENTRY);
#undef TEST_ENTRY
}

TEST_F(OpEmbeddingBagOutTest, PartialFinalOffsetMaxWithEmptyBag) {
  TensorFactory<ScalarType::Float> tf;
  TensorFactory<ScalarType::Long> ti;
  auto weight = tf.make({2, 1}, {2, 7});
  auto indices = ti.make({2}, {0, 1});
  auto offsets = ti.make({3}, {0, 0, 1});
  auto out = tf.full({2, 1}, -99);
  auto offset2bag = ti.full({2}, -99);
  auto bag_size = ti.full({2}, -99);
  auto max_indices = ti.full({2, 1}, -99);
  op_embedding_bag_out(
      weight,
      indices,
      offsets,
      2,
      {},
      true,
      -1,
      out,
      offset2bag,
      bag_size,
      max_indices);
  ASSERT_EQ(context_.failure_state(), torch::executor::Error::Ok);
  EXPECT_TENSOR_EQ(out, tf.make({2, 1}, {0, 7}));
  EXPECT_TENSOR_EQ(offset2bag, ti.make({2}, {1, 1}));
  EXPECT_TENSOR_EQ(bag_size, ti.make({2}, {0, 2}));
  EXPECT_TENSOR_EQ(max_indices, ti.make({2, 1}, {0, 1}));
}

TEST_F(OpEmbeddingBagOutTest, SumStaticOffset2BagAllDtypes) {
  ET_SKIP_IF(
      torch::executor::testing::SupportedFeatures::get()->is_aten,
      "ATen resizes offset2bag to [0] for sum on its fast path");
#define TEST_ENTRY(CTYPE, DTYPE) \
  test_sum_static_offset2bag<ScalarType::DTYPE>();
  ET_FORALL_FLOATHBF16_TYPES(TEST_ENTRY);
#undef TEST_ENTRY
}

TEST_F(OpEmbeddingBagOutTest, ZeroBagsWithIndicesFillOffset2Bag) {
  TensorFactory<ScalarType::Float> tf;
  TensorFactory<ScalarType::Long> ti;
  auto weight = tf.make({2, 2}, {1, 2, 3, 4});
  auto indices = ti.make({4}, {0, 1, 1, 0});
  auto offsets = ti.make({1}, {0});
  auto out = tf.zeros({0, 2});
  auto offset2bag = ti.full({4}, -99);
  auto bag_size = ti.zeros({0});
  auto max_indices = ti.zeros({0});
  op_embedding_bag_out(
      weight,
      indices,
      offsets,
      1,
      {},
      true,
      -1,
      out,
      offset2bag,
      bag_size,
      max_indices);
  ASSERT_EQ(context_.failure_state(), torch::executor::Error::Ok);
  EXPECT_TENSOR_EQ(offset2bag, ti.full({4}, -1));
}

TEST_F(OpEmbeddingBagOutTest, NonFloatingWeightRejectedWithEmptyOutput) {
  TensorFactory<ScalarType::Long> ti;
  auto weight = ti.ones({2, 1});
  auto indices = ti.make({1}, {0});
  auto offsets = ti.make({1}, {0});
  auto out = ti.zeros({0, 1});
  auto offset2bag = ti.zeros({1});
  auto bag_size = ti.zeros({0});
  auto max_indices = ti.zeros({0});
  ET_EXPECT_KERNEL_FAILURE(
      context_,
      op_embedding_bag_out(
          weight,
          indices,
          offsets,
          1,
          {},
          true,
          -1,
          out,
          offset2bag,
          bag_size,
          max_indices));
}

TEST_F(OpEmbeddingBagOutTest, OffsetsDtypeAuxiliaryOutputs) {
  ET_SKIP_IF(
      torch::executor::testing::SupportedFeatures::get()->is_aten,
      "ATen requires the promoted indices/offsets dtype");
  TensorFactory<ScalarType::Float> tf;
  TensorFactory<ScalarType::Long> ti;
  TensorFactory<ScalarType::Int> to;
  auto weight = tf.make({2, 2}, {1, 2, 3, 4});
  auto indices = ti.make({3}, {0, 1, 1});
  auto offsets = to.make({2}, {0, 1});
  for (int64_t mode = 0; mode < 3; ++mode) {
    SCOPED_TRACE(mode);
    auto out = tf.zeros({2, 2});
    // Export allocates these with the offsets dtype, not the promoted one.
    auto offset2bag = to.full({3}, -99);
    auto bag_size = to.zeros({2});
    auto max_indices = mode == 2 ? to.zeros({2, 2}) : to.zeros({2});
    op_embedding_bag_out(
        weight,
        indices,
        offsets,
        mode,
        {},
        false,
        -1,
        out,
        offset2bag,
        bag_size,
        max_indices);
    ASSERT_EQ(context_.failure_state(), torch::executor::Error::Ok);
    EXPECT_TENSOR_EQ(
        out,
        mode == 0 ? tf.make({2, 2}, {1, 2, 6, 8})
                  : tf.make({2, 2}, {1, 2, 3, 4}));
    EXPECT_TENSOR_EQ(offset2bag, to.make({3}, {0, 1, 1}));
    EXPECT_TENSOR_EQ(
        bag_size, mode == 0 ? to.make({2}, {0, 0}) : to.make({2}, {1, 2}));
    if (mode == 2) {
      EXPECT_TENSOR_EQ(max_indices, to.make({2, 2}, {0, 0, 1, 1}));
    }
  }
}

TEST_F(OpEmbeddingBagOutTest, ReductionsAcrossVectorWidths) {
  for (int32_t width : {0, 1, 3, 4, 7, 8, 9, 16, 17, 65}) {
    // ATen's FBGEMM path raises a spurious index error for zero-width weights.
    if (width == 0 &&
        torch::executor::testing::SupportedFeatures::get()->is_aten) {
      continue;
    }
    for (int64_t mode = 0; mode < 3; ++mode) {
      for (bool last : {false, true}) {
        for (bool padding : {false, true}) {
          SCOPED_TRACE(
              ::testing::Message()
              << width << "," << mode << "," << last << "," << padding);
          test_reductions<ScalarType::Int>(width, mode, last, padding);
          test_reductions<ScalarType::Long>(width, mode, last, padding);
          if (mode == 0) {
            SCOPED_TRACE("weighted");
            test_reductions<ScalarType::Int>(width, mode, last, padding, true);
            test_reductions<ScalarType::Long>(width, mode, last, padding, true);
          }
        }
      }
    }
  }
}

TEST_F(OpEmbeddingBagOutTest, WeightedSumMixedIndexTypes) {
  test_weighted_sum_mixed_index_types<ScalarType::Int, ScalarType::Long>();
  // ATen requires the promoted Long dtype for these outputs.
  if (!torch::executor::testing::SupportedFeatures::get()->is_aten) {
    test_weighted_sum_mixed_index_types<ScalarType::Long, ScalarType::Int>();
  }
}

TEST_F(OpEmbeddingBagOutTest, PaddingOnlyAndEmptyBags) {
  TensorFactory<ScalarType::Float> tf;
  TensorFactory<ScalarType::Long> ti;
  auto weight = tf.ones({2, 9});
  for (int64_t mode = 0; mode < 3; ++mode) {
    for (bool empty : {false, true}) {
      SCOPED_TRACE(::testing::Message() << mode << "," << empty);
      auto indices = empty ? ti.zeros({0}) : ti.make({2}, {1, 1});
      auto offsets = ti.make({2}, {0, 0});
      auto out = tf.ones({2, 9});
      auto offset2bag = ti.zeros({empty ? 0 : 2});
      auto bag_size = ti.ones({2});
      auto max_indices = mode == 2 ? ti.ones({2, 9}) : ti.ones({2});
      op_embedding_bag_out(
          weight,
          indices,
          offsets,
          mode,
          {},
          false,
          1,
          out,
          offset2bag,
          bag_size,
          max_indices);
      ASSERT_EQ(context_.failure_state(), torch::executor::Error::Ok);
      EXPECT_TENSOR_EQ(out, tf.zeros({2, 9}));
      EXPECT_TENSOR_EQ(bag_size, ti.zeros({2}));
      EXPECT_TENSOR_EQ(
          max_indices, mode == 2 ? ti.zeros({2, 9}) : ti.zeros({2}));
    }
  }
}

TEST_F(OpEmbeddingBagOutTest, MaxPreservesFirstTieAndNan) {
  TensorFactory<ScalarType::Float> tf;
  TensorFactory<ScalarType::Long> ti;
  const float nan = std::numeric_limits<float>::quiet_NaN();
  std::vector<float> values(18, 2);
  values[0] = nan;
  values[10] = nan;
  auto weight = tf.make({2, 9}, values);
  auto indices = ti.make({2}, {0, 1});
  auto offsets = ti.make({1}, {0});
  auto out = tf.zeros({1, 9});
  auto offset2bag = ti.zeros({2});
  auto bag_size = ti.zeros({1});
  auto max_indices = ti.ones({1, 9});
  op_embedding_bag_out(
      weight,
      indices,
      offsets,
      2,
      {},
      false,
      -1,
      out,
      offset2bag,
      bag_size,
      max_indices);
  ASSERT_EQ(context_.failure_state(), torch::executor::Error::Ok);
  EXPECT_TRUE(std::isnan(out.const_data_ptr<float>()[0]));
  for (int d = 1; d < 9; ++d) {
    EXPECT_EQ(out.const_data_ptr<float>()[d], 2);
  }
  EXPECT_TENSOR_EQ(max_indices, ti.zeros({1, 9}));
}

TEST_F(OpEmbeddingBagOutTest, DynamicBoundOutputsResize) {
  TensorFactory<ScalarType::Float> tf;
  TensorFactory<ScalarType::Long> ti;
  const auto bounded = torch::executor::TensorShapeDynamism::DYNAMIC_BOUND;
  auto weight = tf.make({2, 2}, {1, 4, 3, 2});
  auto indices = ti.make({2}, {0, 1});
  auto offsets = ti.make({1}, {0});
  auto out = tf.zeros({3, 3}, bounded);
  auto offset2bag = ti.zeros({4}, bounded);
  auto bag_size = ti.zeros({3}, bounded);
  auto max_indices = ti.zeros({3, 3}, bounded);
  op_embedding_bag_out(
      weight,
      indices,
      offsets,
      2,
      {},
      false,
      -1,
      out,
      offset2bag,
      bag_size,
      max_indices);
  ASSERT_EQ(context_.failure_state(), torch::executor::Error::Ok);
  EXPECT_TENSOR_EQ(out, tf.make({1, 2}, {3, 4}));
  EXPECT_TENSOR_EQ(offset2bag, ti.make({2}, {0, 0}));
  EXPECT_TENSOR_EQ(bag_size, ti.make({1}, {2}));
  EXPECT_TENSOR_EQ(max_indices, ti.make({1, 2}, {1, 0}));
}

TEST_F(OpEmbeddingBagOutTest, TransposedMaxIndices) {
  TensorFactory<ScalarType::Float> tf;
  TensorFactory<ScalarType::Long> ti;
  auto weight = tf.make({3, 2}, {1, 4, 3, 2, 0, 5});
  auto indices = ti.make({4}, {0, 1, 1, 2});
  auto offsets = ti.make({2}, {0, 2});
  auto out = tf.zeros({2, 2});
  auto offset2bag = ti.zeros({4});
  auto bag_size = ti.zeros({2});
  // Strides (1, 2), like a transposed contiguous tensor.
  auto max_indices = ti.make_with_dimorder({2, 2}, {0, 0, 0, 0}, {1, 0});
  op_embedding_bag_out(
      weight,
      indices,
      offsets,
      2,
      {},
      false,
      -1,
      out,
      offset2bag,
      bag_size,
      max_indices);
  ASSERT_EQ(context_.failure_state(), torch::executor::Error::Ok);
  EXPECT_TENSOR_EQ(out, tf.make({2, 2}, {3, 4, 3, 5}));
  // Equivalent to tensor([[1, 0], [1, 2]]).
  EXPECT_TENSOR_EQ(
      max_indices, ti.make_with_dimorder({2, 2}, {1, 1, 0, 2}, {1, 0}));
}

TEST_F(OpEmbeddingBagOutTest, ManyBagsAllModes) {
  ScopedMinThreadCount threads(4);
  TensorFactory<ScalarType::Float> tf;
  TensorFactory<ScalarType::Long> ti;
  constexpr int32_t kRows = 7;
  constexpr int32_t kWidth = 65;
  // Enough work to split the bags across parallel tasks.
  constexpr int32_t kBags = 512;
  std::vector<float> weights;
  for (int32_t row = 0; row < kRows; ++row) {
    for (int32_t d = 0; d < kWidth; ++d) {
      weights.push_back(static_cast<float>(row * 100 + d - 50));
    }
  }
  auto weight = tf.make({kRows, kWidth}, weights);
  // Bag b repeats row b % kRows (b % 4 + 1) times.
  std::vector<int64_t> indices_data;
  std::vector<int64_t> offsets_data;
  std::vector<int64_t> offset2bag_data;
  std::vector<int64_t> counts;
  for (int32_t bag = 0; bag < kBags; ++bag) {
    offsets_data.push_back(static_cast<int64_t>(indices_data.size()));
    counts.push_back(bag % 4 + 1);
    for (int32_t i = 0; i < counts.back(); ++i) {
      indices_data.push_back(bag % kRows);
      offset2bag_data.push_back(bag);
    }
  }
  const auto num_indices = static_cast<int32_t>(indices_data.size());
  auto indices = ti.make({num_indices}, indices_data);
  auto offsets = ti.make({kBags}, offsets_data);
  for (int64_t mode = 0; mode < 3; ++mode) {
    SCOPED_TRACE(mode);
    auto out = tf.zeros({kBags, kWidth});
    auto offset2bag = ti.zeros({mode == 0 ? 0 : num_indices});
    auto bag_size = ti.zeros({kBags});
    auto max_indices =
        mode == 2 ? ti.zeros({kBags, kWidth}) : ti.zeros({kBags});
    op_embedding_bag_out(
        weight,
        indices,
        offsets,
        mode,
        {},
        false,
        -1,
        out,
        offset2bag,
        bag_size,
        max_indices);
    ASSERT_EQ(context_.failure_state(), torch::executor::Error::Ok);
    std::vector<float> expected_out;
    std::vector<int64_t> expected_winners;
    for (int32_t bag = 0; bag < kBags; ++bag) {
      for (int32_t d = 0; d < kWidth; ++d) {
        const float value = weights[(bag % kRows) * kWidth + d];
        expected_out.push_back(
            mode == 0 ? value * static_cast<float>(counts[bag]) : value);
        expected_winners.push_back(bag % kRows);
      }
    }
    EXPECT_TENSOR_EQ(out, tf.make({kBags, kWidth}, expected_out));
    if (mode != 0) {
      EXPECT_TENSOR_EQ(offset2bag, ti.make({num_indices}, offset2bag_data));
    }
    EXPECT_TENSOR_EQ(
        bag_size, mode == 0 ? ti.zeros({kBags}) : ti.make({kBags}, counts));
    EXPECT_TENSOR_EQ(
        max_indices,
        mode == 0       ? ti.zeros({kBags})
            : mode == 1 ? ti.make({kBags}, counts)
                        : ti.make({kBags, kWidth}, expected_winners));
  }
}

TEST_F(OpEmbeddingBagOutTest, ManyBagsPartialFinalOffset) {
  // Four threads split the bags below into four tasks, so the last bag and
  // its partial final offset are handled away from the first task.
  ScopedMinThreadCount threads(4);
  TensorFactory<ScalarType::Float> tf;
  TensorFactory<ScalarType::Long> ti;
  constexpr int32_t kRows = 7;
  constexpr int32_t kWidth = 64;
  constexpr int32_t kBags = 1024;
  // Indices past the final offset.
  constexpr int32_t kTail = 3;
  constexpr int64_t kPaddingRow = 3;
  std::vector<float> weights;
  for (int32_t row = 0; row < kRows; ++row) {
    for (int32_t d = 0; d < kWidth; ++d) {
      weights.push_back(static_cast<float>(row * 100 + d - 50));
    }
  }
  auto weight = tf.make({kRows, kWidth}, weights);
  // Bag b holds rows (b + j) % kRows for j < b % 4 + 1.
  std::vector<int64_t> indices_data;
  std::vector<int64_t> offsets_data;
  for (int32_t bag = 0; bag < kBags; ++bag) {
    offsets_data.push_back(static_cast<int64_t>(indices_data.size()));
    for (int32_t j = 0; j <= bag % 4; ++j) {
      indices_data.push_back((bag + j) % kRows);
    }
  }
  offsets_data.push_back(static_cast<int64_t>(indices_data.size()));
  for (int32_t j = 0; j < kTail; ++j) {
    indices_data.push_back((j * 2) % kRows);
  }
  const auto num_indices = static_cast<int32_t>(indices_data.size());
  // Multiples of 0.5, so every weighted sum is exact.
  std::vector<float> scales_data;
  for (int32_t i = 0; i < num_indices; ++i) {
    scales_data.push_back(static_cast<float>(i % 5 - 2) * 0.5f);
  }
  auto indices = ti.make({num_indices}, indices_data);
  auto offsets = ti.make({kBags + 1}, offsets_data);
  auto scales = tf.make({num_indices}, scales_data);

  for (int64_t mode = 0; mode < 3; ++mode) {
    for (bool padded : {false, true}) {
      for (bool weighted : {false, true}) {
        if (weighted && mode != 0) {
          continue;
        }
        SCOPED_TRACE(
            ::testing::Message() << mode << "," << padded << "," << weighted);
        const int64_t padding_idx = padded ? kPaddingRow : -1;
        // ATen's FP32 fast path (no padding) stops the last sum/mean bag at
        // the final offset; otherwise it takes every remaining index.
        const bool truncate = !padded && mode != 2;
        std::vector<float> expected_out(kBags * kWidth, 0);
        std::vector<int64_t> expected_offset2bag(num_indices);
        std::vector<int64_t> expected_counts(kBags);
        std::vector<int64_t> expected_winners(kBags * kWidth, 0);
        for (int32_t bag = 0; bag < kBags; ++bag) {
          const int64_t begin = offsets_data[bag];
          const int64_t end =
              bag + 1 < kBags ? offsets_data[bag + 1] : num_indices;
          const int64_t reduce_end = truncate ? offsets_data[bag + 1] : end;
          int64_t count = 0;
          for (int64_t i = begin; i < end; ++i) {
            expected_offset2bag[i] = bag;
            const int64_t row = indices_data[i];
            if (row == padding_idx) {
              continue;
            }
            ++count;
            if (i >= reduce_end) {
              continue;
            }
            for (int32_t d = 0; d < kWidth; ++d) {
              const float value = weights[row * kWidth + d];
              float& acc = expected_out[bag * kWidth + d];
              if (mode == 2) {
                if (count == 1 || value > acc) {
                  acc = value;
                  expected_winners[bag * kWidth + d] = row;
                }
              } else {
                acc += value * (weighted ? scales_data[i] : 1.0f);
              }
            }
          }
          expected_counts[bag] = count;
          if (mode == 1 && count != 0) {
            for (int32_t d = 0; d < kWidth; ++d) {
              expected_out[bag * kWidth + d] /= static_cast<float>(count);
            }
          }
        }

        auto out = tf.zeros({kBags, kWidth});
        auto offset2bag =
            ti.full({mode == 0 && !padded ? 0 : num_indices}, -99);
        auto bag_size = ti.full({kBags}, -99);
        auto max_indices =
            mode == 2 ? ti.full({kBags, kWidth}, -99) : ti.full({kBags}, -99);
        op_embedding_bag_out(
            weight,
            indices,
            offsets,
            mode,
            weighted ? optional<Tensor>(scales) : std::nullopt,
            true,
            padding_idx,
            out,
            offset2bag,
            bag_size,
            max_indices);
        ASSERT_EQ(context_.failure_state(), torch::executor::Error::Ok);
        EXPECT_TENSOR_CLOSE(out, tf.make({kBags, kWidth}, expected_out));
        if (offset2bag.numel() != 0) {
          EXPECT_TENSOR_EQ(
              offset2bag, ti.make({num_indices}, expected_offset2bag));
        }
        EXPECT_TENSOR_EQ(
            bag_size,
            mode == 0 ? ti.zeros({kBags}) : ti.make({kBags}, expected_counts));
        EXPECT_TENSOR_EQ(
            max_indices,
            mode == 0       ? ti.zeros({kBags})
                : mode == 1 ? ti.make({kBags}, expected_counts)
                            : ti.make({kBags, kWidth}, expected_winners));
      }
    }
  }
}

TEST_F(OpEmbeddingBagOutTest, RejectedInputsLeaveDynamicOutputsUnresized) {
  ET_SKIP_IF(
      torch::executor::testing::SupportedFeatures::get()->is_aten,
      "ATen tensors have no dynamic-bound shape");
  TensorFactory<ScalarType::Float> tf;
  TensorFactory<ScalarType::Long> ti;
  TensorFactory<ScalarType::Short> ts;
  const auto bounded = torch::executor::TensorShapeDynamism::DYNAMIC_BOUND;
  auto weight = tf.ones({2, 2});
  struct Case {
    const char* name;
    Tensor indices;
    Tensor offsets;
  };
  const std::vector<Case> cases = {
      {"index past the last row", ti.make({1}, {5}), ti.make({1}, {0})},
      {"offset past the last index",
       ti.make({4}, {0, 1, 0, 1}),
       ti.make({2}, {0, 5})},
      {"Short indices", ts.make({1}, {0}), ti.make({1}, {0})},
  };
  for (const auto& c : cases) {
    SCOPED_TRACE(c.name);
    context_.fail(torch::executor::Error::Ok);
    auto out = tf.zeros({3, 3}, bounded);
    auto offset2bag = ti.zeros({4}, bounded);
    auto bag_size = ti.zeros({3}, bounded);
    auto max_indices = ti.zeros({3}, bounded);
    ET_EXPECT_KERNEL_FAILURE(
        context_,
        op_embedding_bag_out(
            weight,
            c.indices,
            c.offsets,
            1,
            {},
            false,
            -1,
            out,
            offset2bag,
            bag_size,
            max_indices));
    EXPECT_EQ(out.size(0), 3);
    EXPECT_EQ(out.size(1), 3);
    EXPECT_EQ(offset2bag.size(0), 4);
    EXPECT_EQ(bag_size.size(0), 3);
    EXPECT_EQ(max_indices.size(0), 3);
  }
}
