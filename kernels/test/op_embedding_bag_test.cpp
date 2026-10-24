/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/kernels/test/FunctionHeaderWrapper.h> // Declares the operator
#include <executorch/kernels/test/TestUtil.h>
#include <executorch/runtime/core/exec_aten/exec_aten.h>
#include <executorch/runtime/core/exec_aten/testing_util/tensor_factory.h>
#include <executorch/runtime/core/exec_aten/testing_util/tensor_util.h>

#include <gtest/gtest.h>

#include <numeric>
#include <vector>

using namespace ::testing;
using executorch::aten::ScalarType;
using executorch::aten::Tensor;
using std::optional;
using torch::executor::testing::TensorFactory;

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
      auto offset2bag = tout.zeros({mode == 0 ? 0 : 3});
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
      if (mode != 0) {
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

  void expect_invalid(
      const std::vector<int64_t>& index_values,
      const std::vector<int64_t>& offset_values,
      int64_t mode = 0,
      int64_t dim = 1,
      bool include_last = false,
      const optional<Tensor>& psw = std::nullopt) {
    TensorFactory<ScalarType::Float> tf;
    TensorFactory<ScalarType::Long> ti;
    auto weight = tf.ones({2, static_cast<int32_t>(dim)});
    auto indices =
        ti.make({static_cast<int32_t>(index_values.size())}, index_values);
    auto offsets =
        ti.make({static_cast<int32_t>(offset_values.size())}, offset_values);
    int32_t bags =
        static_cast<int32_t>(offset_values.size()) - (include_last ? 1 : 0);
    auto out = tf.zeros({bags, static_cast<int32_t>(dim)});
    auto offset2bag =
        ti.zeros({mode == 0 ? 0 : static_cast<int32_t>(index_values.size())});
    auto bag_size = ti.zeros({bags});
    auto max_indices = mode == 2 ? ti.zeros({bags, static_cast<int32_t>(dim)})
                                 : ti.zeros({bags});
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
