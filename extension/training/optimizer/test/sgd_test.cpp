/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/extension/training/optimizer/sgd.h>

#include <gtest/gtest.h>

using namespace ::testing;
using namespace torch::executor::optimizer;

class SGDOptimizerTest : public ::testing::Test {};

TEST_F(SGDOptimizerTest, InstantiateTypes) {
  SGDParamState state;
  SGDOptions options;
  SGDParamGroup param_group;
  SGD sgd;

  EXPECT_TRUE(dynamic_cast<SGDParamState*>(&state) != nullptr);
  EXPECT_TRUE(dynamic_cast<SGDOptions*>(&options) != nullptr);
  EXPECT_TRUE(dynamic_cast<SGDParamGroup*>(&param_group) != nullptr);
  EXPECT_TRUE(dynamic_cast<SGD*>(&sgd) != nullptr);
}
