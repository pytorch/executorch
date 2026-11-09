/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/extension/tensor/tensor_ptr.h>
#include <executorch/extension/training/optimizer/adamw.h>
#include <executorch/runtime/core/exec_aten/exec_aten.h>
#include <executorch/runtime/core/exec_aten/testing_util/tensor_factory.h>
#include <executorch/runtime/platform/runtime.h>

#include <utility>
#include <vector>

#include <gtest/gtest.h>

// @lint-ignore-every CLANGTIDY facebook-hte-CArray

using executorch::aten::ScalarType;
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

namespace {

template <ScalarType DTYPE>
void test_adamw_constant_gradient() {
  TensorFactory<DTYPE> tf;

  std::map<std::string_view, executorch::aten::Tensor> named_parameters;
  named_parameters.insert({"param1", tf.make({1, 1}, {1.0})});

  AdamW optimizer(named_parameters, AdamWOptions{0.1, 0.9, 0.999, 1e-8, 0.0});

  for (int i = 0; i < 10; ++i) {
    std::map<std::string_view, executorch::aten::Tensor> named_gradients;
    named_gradients.insert({"param1", tf.make({1, 1}, {-1.0})});
    ASSERT_EQ(optimizer.step(named_gradients), Error::Ok);
  }

  auto p1 = named_parameters.at("param1")
                .const_data_ptr<typename TensorFactory<DTYPE>::true_ctype>();
  EXPECT_NEAR(static_cast<float>(p1[0]), 2.0, 0.1);
}

template <ScalarType DTYPE>
void test_adamw_varying_gradient(
    const float expected_without_decay,
    const float expected_with_decay,
    const float tolerance) {
  TensorFactory<DTYPE> tf;
  const std::vector<float> grads = {
      0.5, -1.5, 2.0, -0.25, 1.0, 0.75, -2.5, 0.1};

  for (const auto [weight_decay, expected] : {
           std::pair{0.0, expected_without_decay},
           std::pair{0.1, expected_with_decay},
       }) {
    std::map<std::string_view, executorch::aten::Tensor> named_parameters;
    named_parameters.insert({"param1", tf.make({1, 1}, {1.0})});
    AdamW optimizer(
        named_parameters, AdamWOptions{0.1, 0.9, 0.999, 1e-8, weight_decay});

    for (float g : grads) {
      std::map<std::string_view, executorch::aten::Tensor> named_gradients;
      named_gradients.insert({"param1", tf.make({1, 1}, {g})});
      ASSERT_EQ(optimizer.step(named_gradients), Error::Ok);
    }

    auto p = named_parameters.at("param1")
                 .const_data_ptr<typename TensorFactory<DTYPE>::true_ctype>();
    EXPECT_NEAR(static_cast<float>(p[0]), expected, tolerance);
  }
}

} // namespace

TEST_F(AdamWOptimizerTest, AdamWParamStateTest) {
  auto exp_avg =
      executorch::extension::make_tensor_ptr({2, 2}, {0.f, 0.f, 0.f, 0.f});
  auto exp_avg_sq =
      executorch::extension::make_tensor_ptr({2, 2}, {0.f, 0.f, 0.f, 0.f});
  AdamWParamState state(std::move(exp_avg), std::move(exp_avg_sq));

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
  test_adamw_constant_gradient<ScalarType::Float>();
}

TEST_F(AdamWOptimizerTest, AdamWOptimizerHalf) {
  test_adamw_constant_gradient<ScalarType::Half>();
}

TEST_F(AdamWOptimizerTest, AdamWOptimizerVaryingGradient) {
  // A constant gradient makes Adam's normalized update very nearly sign-only,
  // so the tests above would still pass with a broken moment recurrence or
  // bias correction. A varying gradient exercises both. The float expected
  // values are from torch.optim.AdamW; the half values account for fp16 state
  // rounding at each step.
  test_adamw_varying_gradient<ScalarType::Float>(0.84547687, 0.77574724, 1e-6);
  test_adamw_varying_gradient<ScalarType::Half>(0.84619141, 0.77685547, 1e-3);
}

TEST_F(AdamWOptimizerTest, AdamWOptimizerRejectsInvalidDtypes) {
  TensorFactory<ScalarType::Float> float_tf;
  TensorFactory<ScalarType::Half> half_tf;
  TensorFactory<ScalarType::Double> double_tf;

  std::map<std::string_view, executorch::aten::Tensor> named_parameters;
  named_parameters.insert({"param1", float_tf.make({1, 1}, {1.0})});
  AdamW optimizer(named_parameters, AdamWOptions{});

  std::map<std::string_view, executorch::aten::Tensor> mismatched_gradients;
  mismatched_gradients.insert({"param1", half_tf.make({1, 1}, {1.0})});

  EXPECT_EQ(optimizer.step(mismatched_gradients), Error::InvalidArgument);

  std::map<std::string_view, executorch::aten::Tensor> double_parameters;
  double_parameters.insert({"param1", double_tf.make({1, 1}, {1.0})});
  AdamW double_optimizer(double_parameters, AdamWOptions{});

  std::map<std::string_view, executorch::aten::Tensor> double_gradients;
  double_gradients.insert({"param1", double_tf.make({1, 1}, {1.0})});

  EXPECT_EQ(double_optimizer.step(double_gradients), Error::InvalidArgument);
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
  ASSERT_EQ(optimizer.step(named_gradients), Error::Ok);

  auto p1 =
      static_cast<const float*>(named_parameters.at("param1").const_data_ptr());
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
    ASSERT_EQ(optimizer.step(named_gradients), Error::Ok);
  }

  auto p1 =
      static_cast<const float*>(named_parameters.at("param1").const_data_ptr());
  auto p2 =
      static_cast<const float*>(named_parameters.at("param2").const_data_ptr());
  // Each param sees a constant gradient of +/- 1 for 5 steps -> p shifts by
  // roughly +/- 5 * lr = +/- 0.5. State is tracked independently per param.
  EXPECT_NEAR(p1[0], 1.5, 0.1);
  EXPECT_NEAR(p2[0], 1.5, 0.1);
}
