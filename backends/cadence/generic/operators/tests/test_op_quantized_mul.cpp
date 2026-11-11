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

#include <executorch/backends/cadence/generic/operators/op_quantized_mul.h>

namespace impl {
namespace generic {
namespace native {
namespace {

using ::executorch::aten::ScalarType;
using ::executorch::aten::Tensor;
using ::executorch::runtime::testing::TensorFactory;

// Expected values are aot/ref_implementations.py's. Scales are powers of two,
// so every intermediate value is exact.
class GenericQuantizedMulTest : public OperatorTest {};

// Y broadcasts along dim 0.
TEST_F(GenericQuantizedMulTest, PerTensorBroadcasts) {
  TensorFactory<ScalarType::Char> tf;
  const Tensor x = tf.make({2, 3}, {2, -3, 4, 5, 6, -7});
  const Tensor y = tf.make({1, 3}, {2, 4, -2});
  Tensor out = tf.zeros({2, 3});

  quantized_mul_per_tensor_out(context_, x, 0.5, 0, y, 0.5, 0, 0.25, 0, out);

  EXPECT_TENSOR_EQ(out, tf.make({2, 3}, {4, -12, -8, 10, 24, 14}));
}

TEST_F(GenericQuantizedMulTest, PerTensorSaturatesWithZeroPoints) {
  TensorFactory<ScalarType::Char> tf;
  const Tensor x = tf.make({1, 3}, {100, -100, 3});
  const Tensor y = tf.make({1, 3}, {100, 100, 4});
  Tensor out = tf.zeros({1, 3});

  quantized_mul_per_tensor_out(context_, x, 0.5, 0, y, 0.5, 2, 0.5, -1, out);

  EXPECT_TENSOR_EQ(out, tf.make({1, 3}, {127, -128, 2}));
}

} // namespace
} // namespace native
} // namespace generic
} // namespace impl
