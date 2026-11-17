/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/backends/cadence/generic/operators/op_fully_connected.h>

#include <optional>

#include <executorch/kernels/test/TestUtil.h>
#include <executorch/runtime/core/exec_aten/exec_aten.h>
#include <executorch/runtime/core/exec_aten/testing_util/tensor_factory.h>
#include <executorch/runtime/core/exec_aten/testing_util/tensor_util.h>
#include <gtest/gtest.h>

namespace impl {
namespace generic {
namespace native {
namespace {

using ::executorch::aten::ScalarType;
using ::executorch::aten::Tensor;
using ::executorch::runtime::testing::TensorFactory;

class GenericFullyConnectedTest : public OperatorTest {
 protected:
  TensorFactory<ScalarType::Float> tf_;
};

// Each of the 2 leading rows gets its own output row.
TEST_F(GenericFullyConnectedTest, ComputesEveryLeadingRow) {
  Tensor input = tf_.make({2, 1, 3}, {1, 2, 3, 4, 5, 6});
  Tensor weight = tf_.make({2, 3}, {1, 0, -1, 0.5, 0.5, 0.5});
  Tensor bias = tf_.make({2}, {10, 20});
  Tensor out = tf_.zeros({2, 1, 2});

  fully_connected_out(context_, input, weight, bias, out);

  EXPECT_TENSOR_EQ(out, tf_.make({2, 1, 2}, {8, 23, 8, 27.5}));
}

TEST_F(GenericFullyConnectedTest, ComputesWithoutBias) {
  Tensor input = tf_.make({2, 3}, {1, 2, 3, 4, 5, 6});
  Tensor weight = tf_.make({2, 3}, {1, 0, -1, 0.5, 0.5, 0.5});
  Tensor out = tf_.zeros({2, 2});

  fully_connected_out(context_, input, weight, std::nullopt, out);

  EXPECT_TENSOR_EQ(out, tf_.make({2, 2}, {-2, 3, -2, 7.5}));
}

TEST_F(GenericFullyConnectedTest, RejectsNonFloatDtype) {
  TensorFactory<ScalarType::Double> tf_double;
  Tensor input = tf_double.make({1, 3}, {1, 2, 3});
  Tensor weight = tf_double.make({2, 3}, {1, 0, -1, 0.5, 0.5, 0.5});
  Tensor out = tf_double.zeros({1, 2});

  ET_EXPECT_KERNEL_FAILURE(
      context_,
      fully_connected_out(context_, input, weight, std::nullopt, out));
}

TEST_F(GenericFullyConnectedTest, RejectsMismatchedInnerDim) {
  Tensor input = tf_.make({1, 4}, {1, 2, 3, 4});
  Tensor weight = tf_.make({2, 3}, {1, 0, -1, 0.5, 0.5, 0.5});
  Tensor out = tf_.zeros({1, 2});

  ET_EXPECT_KERNEL_FAILURE(
      context_,
      fully_connected_out(context_, input, weight, std::nullopt, out));
}

} // namespace
} // namespace native
} // namespace generic
} // namespace impl
