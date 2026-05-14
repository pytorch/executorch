/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/extension/llm/sampler/sampler.h>

#include <set>

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

TEST(SamplerTest, TestTopKRestrictsToCandidates) {
  // With topk=3, sampling must always return one of the top-3 indices,
  // regardless of the random draw.
  Sampler sampler{
      /*vocab_size*/ 100,
      /*temperature*/ 1.0f,
      /*topp*/ 0.0f, // disable top-p so we exercise top-k alone
      /*rng_seed*/ 42};
  sampler.set_topk(3);

  // Construct logits where indices {7, 13, 42} dominate.
  torch::Tensor input = torch::full({100}, -10.0f, at::kFloat);
  input[7] = 5.0f;
  input[13] = 4.5f;
  input[42] = 4.0f;

  std::set<int32_t> allowed = {7, 13, 42};
  for (int trial = 0; trial < 50; ++trial) {
    // Re-fill logits each trial because sample() mutates them in place.
    torch::Tensor logits = input.clone();
    int32_t out = sampler.sample(logits.data_ptr<float>());
    EXPECT_TRUE(allowed.count(out)) << "trial " << trial << " got " << out;
  }
}

TEST(SamplerTest, TestTopKDisabledByZero) {
  // topk=0 means disabled. With topp disabled, sampling collapses to
  // multinomial over the full vocab, but the dominant token should still
  // win the vast majority of the time.
  Sampler sampler{
      /*vocab_size*/ 50,
      /*temperature*/ 1.0f,
      /*topp*/ 0.0f,
      /*rng_seed*/ 7};
  sampler.set_topk(0); // disabled

  torch::Tensor input = torch::full({50}, -10.0f, at::kFloat);
  input[11] = 20.0f; // dominant

  int hits = 0;
  for (int trial = 0; trial < 20; ++trial) {
    torch::Tensor logits = input.clone();
    if (sampler.sample(logits.data_ptr<float>()) == 11) {
      hits++;
    }
  }
  EXPECT_GE(hits, 18); // dominant token should win nearly every time
}

TEST(SamplerTest, TestTopKWithFP16) {
  // Smoke test the FP16 template instantiation of the top-k path.
  Sampler sampler{
      /*vocab_size*/ 50,
      /*temperature*/ 1.0f,
      /*topp*/ 0.0f,
      /*rng_seed*/ 99};
  sampler.set_topk(2);

  torch::Tensor input = torch::full({50}, -10.0f, at::kHalf);
  input[3] = 5.0f;
  input[8] = 4.5f;

  std::set<int32_t> allowed = {3, 8};
  for (int trial = 0; trial < 30; ++trial) {
    torch::Tensor logits = input.clone();
    int32_t out = sampler.sample(logits.data_ptr<c10::Half>());
    EXPECT_TRUE(allowed.count(out)) << "trial " << trial << " got " << out;
  }
}

TEST(SamplerTest, TestTopKEqualsOneIsArgmax) {
  // topk=1 should behave like greedy argmax even with temperature > 0.
  Sampler sampler{
      /*vocab_size*/ 100,
      /*temperature*/ 1.0f,
      /*topp*/ 0.0f,
      /*rng_seed*/ 123};
  sampler.set_topk(1);

  torch::Tensor input = torch::rand({100}, at::kFloat);
  input[57] = 100.0f; // make 57 the unambiguous max

  for (int trial = 0; trial < 10; ++trial) {
    torch::Tensor logits = input.clone();
    EXPECT_EQ(sampler.sample(logits.data_ptr<float>()), 57);
  }
}

TEST(SamplerTest, TestTopKTakesPrecedenceOverTopP) {
  // When both top-k and top-p are set, top-k should restrict the candidate
  // set; top-p alone would admit a third token that top-k=2 must exclude.
  Sampler sampler{
      /*vocab_size*/ 100,
      /*temperature*/ 1.0f,
      /*topp*/ 0.99f, // would keep nearly the whole vocab on its own
      /*rng_seed*/ 99};
  sampler.set_topk(2);

  torch::Tensor input = torch::full({100}, -10.0f, at::kFloat);
  input[3] = 5.0f;
  input[8] = 4.5f;
  input[19] = 4.0f; // would be in the top-p set but is excluded by top-k=2

  std::set<int32_t> allowed = {3, 8};
  for (int trial = 0; trial < 50; ++trial) {
    torch::Tensor logits = input.clone();
    int32_t out = sampler.sample(logits.data_ptr<float>());
    EXPECT_TRUE(allowed.count(out)) << "trial " << trial << " got " << out;
  }
}
