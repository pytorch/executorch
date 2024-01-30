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
#ifdef USE_ATEN_LIB
#include <torch/torch.h>
#endif

#include <executorch/runtime/core/exec_aten/exec_aten.h>

namespace torch {
namespace executor {
// A simple llama2 sampler.

template <typename T>
struct ProbIndex {
  T prob;
  int32_t index;
}; // struct used when sorting probabilities during top-p sampling

class Sampler {
 public:
  Sampler(
      int32_t vocab_size,
      float temperature,
      float topp,
      unsigned long long rng_seed);

  template <typename T>
  int32_t sample(T* logits);

 private:
  template <typename T>
  int32_t sample_topp(T* probabilities, float coin);
  template <typename T>
  int32_t sample_mult(T* probabilities, float coin);
  template <typename T>
  int32_t sample_argmax(T* probabilities);

 private:
  int32_t vocab_size_;
  float temperature_;
  float topp_;
  unsigned long long rng_state_;
};

} // namespace executor
} // namespace torch
