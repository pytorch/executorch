/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/extension/llm/sampler/sampler.h>

#include <gtest/gtest.h>
#include <torch/torch.h>

using namespace ::testing;
using ::executorch::extension::llm::Sampler;

TEST(SamplerTest, TestArgMax) {
  Sampler sampler{
      /*vocab_size*/ 32000,
      /*temperature*/ 0.0f,
      /*topp*/ 0.9f,
      /*rng_seed*/ 0};
  // tensor([[[-12.9832,  -7.4133,  -0.4327,  ...,  -6.8297,  -8.0880,
  // -7.5863]]])
  torch::Tensor input = torch::rand({1, 1, 32000}, at::kFloat);
  input[0][0][396] = 1.0f;
  EXPECT_EQ(sampler.sample(input.data_ptr<float>()), 396);
}

TEST(SamplerTest, TestArgMaxWithFP16) {
  Sampler sampler{
      /*vocab_size*/ 32000,
      /*temperature*/ 0.0f,
      /*topp*/ 0.9f,
      /*rng_seed*/ 0};
  // tensor([[[-12.9832,  -7.4133,  -0.4327,  ...,  -6.8297,  -8.0880,
  // -7.5863]]])
  torch::Tensor input = torch::rand({1, 1, 32000}, at::kHalf);
  input[0][0][396] = 1.0f;
  EXPECT_EQ(sampler.sample(input.data_ptr<c10::Half>()), 396);
}
