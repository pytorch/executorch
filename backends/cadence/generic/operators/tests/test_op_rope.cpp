/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <gtest/gtest.h>

#include <executorch/kernels/test/TestUtil.h>
#include <executorch/runtime/core/exec_aten/exec_aten.h>
#include <executorch/runtime/core/exec_aten/testing_util/tensor_factory.h>
#include <executorch/runtime/core/exec_aten/testing_util/tensor_util.h>

#include <executorch/backends/cadence/generic/operators/op_rope.h>

namespace impl {
namespace generic {
namespace native {
namespace {

using ::executorch::aten::ScalarType;
using ::executorch::aten::Tensor;
using ::executorch::runtime::testing::TensorFactory;

// Expected values are aot/ref_implementations.py's. sin and cos are small
// exact values rather than real angles, so every product is exact. The input is
// [1, seq=2, heads=1, head_dim=4], and sin and cos have one row per token.
class GenericRopeTest : public OperatorTest {
 protected:
  TensorFactory<ScalarType::Float> tf_;
  const Tensor input_ =
      tf_.make({1, 2, 1, 4}, {1.0, 2.0, 3.0, 4.0, -1.0, 0.5, 2.0, -2.0});
  const Tensor sin_ = tf_.make({3, 2}, {0.0, 1.0, 0.5, -1.0, 1.0, 0.25});
  const Tensor cos_ = tf_.make({3, 2}, {1.0, 0.0, 0.5, 0.0, 0.0, 2.0});
  const Tensor sin_2_ = tf_.make({2, 2}, {0.0, 1.0, 0.5, -1.0});
  const Tensor cos_2_ = tf_.make({2, 2}, {1.0, 0.0, 0.5, 0.0});
};

TEST_F(GenericRopeTest, InterleavedPairs) {
  Tensor out = tf_.zeros({1, 2, 1, 4});

  rope_out(context_, input_, sin_2_, cos_2_, std::nullopt, out);

  EXPECT_TENSOR_EQ(
      out,
      tf_.make({1, 2, 1, 4}, {1.0, 2.0, -4.0, 3.0, -0.75, -0.25, -2.0, -2.0}));
}

// Token 0 takes row 2 of sin and cos, and token 1 takes row 0.
TEST_F(GenericRopeTest, PositionsSelectTheRows) {
  const Tensor expected =
      tf_.make({1, 2, 1, 4}, {-2.0, 1.0, 5.0, 8.75, -1.0, 0.5, 2.0, 2.0});
  TensorFactory<ScalarType::Int> tf_int;
  TensorFactory<ScalarType::Long> tf_long;
  Tensor out = tf_.zeros({1, 2, 1, 4});

  rope_out(context_, input_, sin_, cos_, tf_int.make({2}, {2, 0}), out);
  EXPECT_TENSOR_EQ(out, expected);

  out = tf_.zeros({1, 2, 1, 4});
  rope_out(context_, input_, sin_, cos_, tf_long.make({2}, {2, 0}), out);
  EXPECT_TENSOR_EQ(out, expected);
}

// Each head's first half x0 rotates with its second half x1: the output is
// [x0 * cos - x1 * sin, x1 * cos + x0 * sin]. These values are worked out by
// hand: ref_implementations.py's rope_rotate_stacked_halves mixes tokens when
// seq * heads > 1. Token 0 has x0 = [1, 2], x1 = [3, 4], sin = [0, 1] and
// cos = [1, 0], so it gives [1, -4, 3, 2].
TEST_F(GenericRopeTest, StackedHalves) {
  Tensor out = tf_.zeros({1, 2, 1, 4});

  rope_rotate_stacked_halves_out(
      context_, input_, sin_2_, cos_2_, std::nullopt, out);

  EXPECT_TENSOR_EQ(
      out,
      tf_.make({1, 2, 1, 4}, {1.0, -4.0, 3.0, 2.0, -1.5, -2.0, 0.5, -0.5}));
}

TEST_F(GenericRopeTest, StackedHalvesWithPositions) {
  TensorFactory<ScalarType::Long> tf_long;
  Tensor out = tf_.zeros({1, 2, 1, 4});

  rope_rotate_stacked_halves_out(
      context_, input_, sin_, cos_, tf_long.make({2}, {2, 0}), out);

  EXPECT_TENSOR_EQ(
      out, tf_.make({1, 2, 1, 4}, {-3.0, 3.0, 1.0, 8.5, -1.0, 2.0, 2.0, 0.5}));
}

} // namespace
} // namespace native
} // namespace generic
} // namespace impl
