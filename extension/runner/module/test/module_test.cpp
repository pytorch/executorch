/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <gtest/gtest.h>

#include <executorch/extension/runner/module/module.h>
#include <executorch/runtime/platform/runtime.h>

using namespace ::testing;

namespace torch::executor {

class ModuleTest : public ::testing::Test {
 public:
  void SetUp() override {
    torch::executor::runtime_init();
  }
};

TEST_F(ModuleTest, test) {
  Module module(std::getenv("RESOURCES_PATH") + std::string("/model.pte"));

  EXPECT_EQ(module.methodNames(), std::vector<std::string>{"forward"});

  float input[] = {1, 2};
  int32_t sizes[] = {1, 2};
  TensorImpl tensorImpl(ScalarType::Float, std::size(sizes), sizes, input);
  std::vector<EValue> inputs = {EValue(Tensor(&tensorImpl))};
  std::vector<EValue> outputs;

  const auto error = module.forward(inputs, outputs);

  EXPECT_EQ(error, Error::Ok);

  const auto outputTensor = outputs[0].toTensor();
  const auto data = outputTensor.const_data_ptr<float>();

  EXPECT_NEAR(data[0], 1.5, 1e-5);
}

} // namespace torch::executor
