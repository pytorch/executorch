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

#include <executorch/backends/cadence/generic/operators/op_quantized_add.h>

namespace impl {
namespace generic {
namespace native {
namespace {

using ::executorch::aten::ScalarType;
using ::executorch::aten::Tensor;
using ::executorch::runtime::testing::TensorFactory;

// Expected values are aot/ref_implementations.py's. Scales are powers of two,
// so every intermediate value is exact.
class GenericQuantizedAddTest : public OperatorTest {};

TEST_F(GenericQuantizedAddTest, PerTensor) {
  TensorFactory<ScalarType::Char> tf;
  const Tensor x = tf.make({2, 3}, {-10, 0, 20, 5, -5, 50});
  const Tensor y = tf.make({2, 3}, {10, 20, 30, 1, 2, 3});
  Tensor out = tf.zeros({2, 3});

  quantized_add_per_tensor_out(context_, x, 0.5, 0, y, 0.25, 2, 0.25, -3, out);

  EXPECT_TENSOR_EQ(out, tf.make({2, 3}, {-15, 15, 65, 6, -13, 98}));
}

// Y broadcasts along dim 0, and sums past the int8 range saturate.
TEST_F(GenericQuantizedAddTest, Asym8sBroadcastsAndSaturates) {
  TensorFactory<ScalarType::Char> tf;
  const Tensor x = tf.make({2, 3}, {100, -100, 60, 127, -128, 0});
  const Tensor y = tf.make({1, 3}, {100, -100, 10});
  Tensor out = tf.zeros({2, 3});

  quantized_add_asym8sxasym8s_asym8s_per_tensor_out(
      context_, x, 0.5, 0, y, 0.5, 0, 0.5, 0, out);

  EXPECT_TENSOR_EQ(out, tf.make({2, 3}, {127, -128, 70, 127, -128, 10}));
}

TEST_F(GenericQuantizedAddTest, Asym8uSaturatesAtBothEnds) {
  TensorFactory<ScalarType::Byte> tf;
  const Tensor x = tf.make({2, 3}, {130, 120, 255, 0, 128, 200});
  const Tensor y = tf.make({2, 3}, {140, 110, 255, 0, 128, 10});
  Tensor out = tf.zeros({2, 3});

  quantized_add_asym8uxasym8u_asym8u_per_tensor_out(
      context_, x, 0.25, 128, y, 0.25, 128, 0.25, 128, out);

  EXPECT_TENSOR_EQ(out, tf.make({2, 3}, {142, 102, 255, 0, 128, 82}));
}

} // namespace
} // namespace native
} // namespace generic
} // namespace impl
