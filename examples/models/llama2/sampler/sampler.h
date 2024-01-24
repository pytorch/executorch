/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <cctype>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>

namespace torch {
namespace executor {
// A simple llama2 sampler.

struct ProbIndex {
  float prob;
  int32_t index;
}; // struct used when sorting probabilities during top-p sampling

class Sampler {
 public:
  Sampler(
      int32_t vocab_size,
      float temperature,
      float topp,
      unsigned long long rng_seed);

  int32_t sample(float* logits);

 private:
  int32_t sample_topp(float* probabilities, float coin);
  int32_t sample_mult(float* probabilities, float coin);
  int32_t sample_argmax(float* probabilities);

 private:
  int32_t vocab_size_;
  std::unique_ptr<ProbIndex[]> probindex_; // buffer used in top-p sampling
  float temperature_;
  float topp_;
  unsigned long long rng_state_;
};

} // namespace executor
} // namespace torch
