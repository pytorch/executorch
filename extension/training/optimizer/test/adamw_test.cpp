/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/extension/training/optimizer/adamw.h>
#include <executorch/runtime/core/exec_aten/exec_aten.h>
#include <executorch/runtime/core/exec_aten/testing_util/tensor_factory.h>
#include <executorch/runtime/core/span.h>
#include <executorch/runtime/platform/runtime.h>

#include <gtest/gtest.h>

// @lint-ignore-every CLANGTIDY facebook-hte-CArray

using namespace ::testing;
using executorch::aten::ScalarType;
using executorch::aten::Tensor;
using ::executorch::extension::training::optimizer::AdamW;
using ::executorch::extension::training::optimizer::AdamWOptions;
using ::executorch::extension::training::optimizer::AdamWParamState;
using ::executorch::runtime::Error;
using ::executorch::runtime::testing::TensorFactory;

class AdamWOptimizerTest : public ::testing::Test {
 protected:
  void SetUp() override {
    torch::executor::runtime_init();
  }
};

TEST_F(AdamWOptimizerTest, AdamWParamStateTest) {
  TensorFactory<ScalarType::Float> tf;
  Tensor exp_avg = tf.make({2, 2}, {0, 0, 0, 0});
  Tensor exp_avg_sq = tf.make({2, 2}, {0, 0, 0, 0});
  AdamWParamState state(exp_avg, exp_avg_sq);

  EXPECT_EQ(state.step_count(), 0);
  state.increment_step_count();
  EXPECT_EQ(state.step_count(), 1);
}

TEST_F(AdamWOptimizerTest, AdamWOptionsDefaultValuesTest) {
  AdamWOptions options;

  EXPECT_DOUBLE_EQ(options.lr(), 1e-3);
  EXPECT_DOUBLE_EQ(options.beta1(), 0.9);
  EXPECT_DOUBLE_EQ(options.beta2(), 0.999);
  EXPECT_DOUBLE_EQ(options.eps(), 1e-8);
  EXPECT_DOUBLE_EQ(options.weight_decay(), 1e-2);
}

TEST_F(AdamWOptimizerTest, AdamWOptionsNonDefaultValuesTest) {
  AdamWOptions options(0.1, 0.8, 0.99, 1e-6, 0.5);

  EXPECT_DOUBLE_EQ(options.lr(), 0.1);
  EXPECT_DOUBLE_EQ(options.beta1(), 0.8);
  EXPECT_DOUBLE_EQ(options.beta2(), 0.99);
  EXPECT_DOUBLE_EQ(options.eps(), 1e-6);
  EXPECT_DOUBLE_EQ(options.weight_decay(), 0.5);
}

TEST_F(AdamWOptimizerTest, AdamWOptimizerSimple) {
  TensorFactory<ScalarType::Float> tf;

  std::map<std::string_view, executorch::aten::Tensor> named_parameters;
  named_parameters.insert({"param1", tf.make({1, 1}, {1.0})});

  // lr=0.1, defaults otherwise, wd=0 to isolate the moment-based update.
  AdamW optimizer(named_parameters, AdamWOptions{0.1, 0.9, 0.999, 1e-8, 0.0});

  for (int i = 0; i < 10; ++i) {
    std::map<std::string_view, executorch::aten::Tensor> named_gradients;
    named_gradients.insert({"param1", tf.make({1, 1}, {-1.0})});
    optimizer.step(named_gradients);
  }

  auto p1 = static_cast<const float*>(
      named_parameters.at("param1").const_data_ptr());
  // With a constant gradient of -1 and no weight decay, the bias-corrected
  // m_hat / sqrt(v_hat) is ~= -1 at every step, so each step shifts p by
  // +lr. After 10 steps of lr=0.1, p should be near 2.0.
  EXPECT_NEAR(p1[0], 2.0, 0.1);
}

TEST_F(AdamWOptimizerTest, AdamWOptimizerDecoupledWeightDecay) {
  TensorFactory<ScalarType::Float> tf;

  std::map<std::string_view, executorch::aten::Tensor> named_parameters;
  named_parameters.insert({"param1", tf.make({1, 1}, {1.0})});

  // lr=0.1, wd=0.5. With a ZERO gradient, the moment update contributes
  // nothing (m stays 0, v stays 0 -> m_hat/sqrt(v_hat+eps) ~= 0), so only
  // the decoupled weight-decay term moves the parameter:
  //   p <- p * (1 - lr * wd) = 1.0 * (1 - 0.05) = 0.95
  // This is the test that distinguishes AdamW from Adam-with-L2.
  AdamW optimizer(named_parameters, AdamWOptions{0.1, 0.9, 0.999, 1e-8, 0.5});

  std::map<std::string_view, executorch::aten::Tensor> named_gradients;
  named_gradients.insert({"param1", tf.make({1, 1}, {0.0})});
  optimizer.step(named_gradients);

  auto p1 = static_cast<const float*>(
      named_parameters.at("param1").const_data_ptr());
  EXPECT_NEAR(p1[0], 0.95, 1e-5);
}

TEST_F(AdamWOptimizerTest, AdamWOptimizerMultipleParams) {
  TensorFactory<ScalarType::Float> tf;

  std::map<std::string_view, executorch::aten::Tensor> named_parameters;
  named_parameters.insert({"param1", tf.make({1, 1}, {1.0})});
  named_parameters.insert({"param2", tf.make({1, 1}, {2.0})});

  AdamW optimizer(named_parameters, AdamWOptions{0.1, 0.9, 0.999, 1e-8, 0.0});

  for (int i = 0; i < 5; ++i) {
    std::map<std::string_view, executorch::aten::Tensor> named_gradients;
    named_gradients.insert({"param1", tf.make({1, 1}, {-1.0})});
    named_gradients.insert({"param2", tf.make({1, 1}, {1.0})});
    optimizer.step(named_gradients);
  }

  auto p1 = static_cast<const float*>(
      named_parameters.at("param1").const_data_ptr());
  auto p2 = static_cast<const float*>(
      named_parameters.at("param2").const_data_ptr());
  // Each param sees a constant gradient of +/- 1 for 5 steps -> p shifts by
  // roughly +/- 5 * lr = +/- 0.5. State is tracked independently per param.
  EXPECT_NEAR(p1[0], 1.5, 0.1);
  EXPECT_NEAR(p2[0], 1.5, 0.1);
}
