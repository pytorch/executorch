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
#include <executorch/runtime/core/span.h>
#include <executorch/runtime/platform/runtime.h>

#include <gtest/gtest.h>

// @lint-ignore-every CLANGTIDY facebook-hte-CArray

using namespace ::testing;
using exec_aten::ScalarType;
using exec_aten::Tensor;
using ::executorch::extension::training::optimizer::SGD;
using ::executorch::extension::training::optimizer::SGDOptions;
using ::executorch::extension::training::optimizer::SGDParamState;
using ::executorch::runtime::Error;
using ::executorch::runtime::testing::TensorFactory;

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

TEST_F(SGDOptimizerTest, SGDOptimizerSimple) {
  TensorFactory<ScalarType::Float> tf;

  std::map<exec_aten::string_view, exec_aten::Tensor> named_parameters;
  std::map<exec_aten::string_view, exec_aten::Tensor> named_gradients;

  named_parameters.insert({"param1", tf.make({1, 1}, {1})});

  // dummy gradient of -1 for all epochs
  named_gradients.insert({"param1", tf.make({1, 1}, {-1})});

  SGD optimizer(named_parameters, SGDOptions{0.1});

  for (int i = 0; i < 10; ++i) {
    optimizer.step(named_gradients);
  }

  auto p1 = static_cast<const float*>(
      named_parameters.at("param1").unsafeGetTensorImpl()->data());
  EXPECT_NEAR(p1[0], 2.0, 0.1);
}

TEST_F(SGDOptimizerTest, SGDOptimizerComplex) {
  TensorFactory<ScalarType::Float> tf;

  std::map<exec_aten::string_view, exec_aten::Tensor> named_parameters;

  named_parameters.insert({"param1", tf.make({1, 1}, {1.0})});
  named_parameters.insert({"param2", tf.make({1, 1}, {2.0})});

  SGD optimizer(named_parameters, SGDOptions{0.1, 0.1, 0, 2, true});

  for (int i = 0; i < 10; ++i) {
    std::map<exec_aten::string_view, exec_aten::Tensor> named_gradients;
    // dummy gradient of -1 for all epochs
    named_gradients.insert({"param1", tf.make({1, 1}, {-1})});
    named_gradients.insert({"param2", tf.make({1, 1}, {-1})});
    optimizer.step(named_gradients);
  }

  auto p1 =
      static_cast<const float*>(named_parameters.at("param1").const_data_ptr());
  auto p2 =
      static_cast<const float*>(named_parameters.at("param2").const_data_ptr());
  EXPECT_NEAR(p1[0], 0.540303, 0.1);
  EXPECT_NEAR(p2[0], 0.620909, 0.1);
}
