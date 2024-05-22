/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/extension/training/optimizer/sgd.h>
#include <executorch/runtime/core/exec_aten/exec_aten.h>
#include <executorch/runtime/core/exec_aten/testing_util/tensor_factory.h>
#include <executorch/runtime/platform/runtime.h>

#include <gtest/gtest.h>

using namespace ::testing;
using namespace torch::executor::optimizer;
using exec_aten::ScalarType;
using exec_aten::Tensor;
using torch::executor::testing::TensorFactory;

class SGDOptimizerTest : public ::testing::Test {
 protected:
  void SetUp() override {
    torch::executor::runtime_init();
  }
};

TEST_F(SGDOptimizerTest, SGDParamStateTest) {
  TensorFactory<ScalarType::Int> tf;
  Tensor momentum_buffer = tf.make({2, 2}, {1, 2, 3, 4});
  SGDParamState state(momentum_buffer);

  auto data_p = state.momentum_buffer().const_data_ptr<int32_t>();

  ASSERT_EQ(data_p[0], 1);
  ASSERT_EQ(data_p[1], 2);
  ASSERT_EQ(data_p[2], 3);
  ASSERT_EQ(data_p[3], 4);
}

TEST_F(SGDOptimizerTest, SGDOptionsNonDefaultValuesTest) {
  SGDOptions options(0.1, 1.0, 2.0, 3.0, true);

  EXPECT_EQ(options.lr(), 0.1);
  EXPECT_EQ(options.momentum(), 1.0);
  EXPECT_EQ(options.dampening(), 2.0);
  EXPECT_EQ(options.weight_decay(), 3.0);
  EXPECT_TRUE(options.nesterov());
}

TEST_F(SGDOptimizerTest, SGDOptionsDefaultValuesTest) {
  SGDOptions options(0.1);

  EXPECT_EQ(options.lr(), 0.1);
  EXPECT_EQ(options.momentum(), 0);
  EXPECT_EQ(options.dampening(), 0);
  EXPECT_EQ(options.weight_decay(), 0);
  EXPECT_TRUE(!options.nesterov());
}
