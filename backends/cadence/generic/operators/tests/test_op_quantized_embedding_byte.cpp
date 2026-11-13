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

#include <executorch/backends/cadence/generic/operators/op_quantized_embedding_byte.h>

namespace impl {
namespace generic {
namespace native {
namespace {

using ::executorch::aten::ScalarType;
using ::executorch::aten::Tensor;
using ::executorch::runtime::testing::TensorFactory;

// Expected values are aot/ref_implementations.py's.
class GenericQuantizedEmbeddingByteTest : public OperatorTest {
 protected:
  TensorFactory<ScalarType::Char> tf_char_;
  TensorFactory<ScalarType::Float> tf_float_;
  TensorFactory<ScalarType::Long> tf_long_;
  const Tensor weight_ =
      tf_char_.make({3, 4}, {10, 20, 30, 40, -8, 0, 8, 16, 127, -128, 1, -1});
};

// One scale and zero point per row; an index can repeat.
TEST_F(GenericQuantizedEmbeddingByteTest, PerRowScalesAndZeroPoints) {
  Tensor out = tf_float_.zeros({3, 4});

  quantized_embedding_byte_out(
      context_,
      weight_,
      tf_float_.make({3}, {0.5, 0.25, 2.0}),
      tf_char_.make({3}, {2, -4, 1}),
      tf_long_.make({3}, {2, 0, 0}),
      /*pruned_weights=*/false,
      out);

  EXPECT_TENSOR_EQ(
      out,
      tf_float_.make(
          {3, 4},
          {252.0,
           -258.0,
           0.0,
           -4.0,
           4.0,
           9.0,
           14.0,
           19.0,
           4.0,
           9.0,
           14.0,
           19.0}));
}

// Two groups of two columns per row, each with its own scale, and no zero
// points.
TEST_F(GenericQuantizedEmbeddingByteTest, GroupedScalesWithoutZeroPoints) {
  Tensor out = tf_float_.zeros({2, 4});

  quantized_embedding_byte_out(
      context_,
      weight_,
      tf_float_.make({3, 2}, {0.5, 2.0, 0.25, 1.0, 1.0, 0.125}),
      std::nullopt,
      tf_long_.make({2}, {1, 2}),
      /*pruned_weights=*/false,
      out);

  EXPECT_TENSOR_EQ(
      out,
      tf_float_.make(
          {2, 4}, {-2.0, 0.0, 8.0, 16.0, 127.0, -128.0, 0.125, -0.125}));
}

TEST_F(GenericQuantizedEmbeddingByteTest, Uint8) {
  TensorFactory<ScalarType::Byte> tf_byte;
  Tensor out = tf_float_.zeros({2, 4});

  quantized_embedding_byte_out(
      context_,
      tf_byte.make({2, 4}, {128, 130, 126, 255, 0, 1, 2, 3}),
      tf_float_.make({2}, {0.5, 4.0}),
      tf_byte.make({2}, {128, 1}),
      tf_long_.make({2}, {0, 1}),
      /*pruned_weights=*/false,
      out);

  EXPECT_TENSOR_EQ(
      out, tf_float_.make({2, 4}, {0.0, 1.0, -1.0, 63.5, -4.0, 0.0, 4.0, 8.0}));
}

} // namespace
} // namespace native
} // namespace generic
} // namespace impl
