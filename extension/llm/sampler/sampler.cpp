/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

// This is a modified version of https://github.com/karpathy/llama2.c.git
// @lint-ignore-every LICENSELINT
/**
 * MIT License
 *
 * Copyright (c) 2023 Andrej
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */

#include <executorch/extension/llm/sampler/sampler.h>
#include <algorithm>

namespace executorch {
namespace extension {
namespace llm {

// sampler stuff
template <typename T>
int32_t Sampler::sample_argmax(T* probabilities) {
  // return the index that has the highest probability
  int max_i = 0;
  T max_p = probabilities[0];
  for (int i = 1; i < vocab_size_; i++) {
    if (probabilities[i] > max_p) {
      max_i = i;
      max_p = probabilities[i];
    }
  }
  return max_i;
}

template <typename T>
int32_t Sampler::sample_mult(T* probabilities, float coin) {
  // sample index from probabilities (they must sum to 1!)
  // coin is a random number in [0, 1), usually from random_f32()
  T cdf = 0.0;
  for (int i = 0; i < vocab_size_; i++) {
    cdf += probabilities[i];
    if (coin < cdf) {
      return i;
    }
  }
  return vocab_size_ - 1; // in case of rounding errors
}

template <typename T>
int32_t Sampler::sample_topp(T* probabilities, float coin) {
  // top-p sampling (or "nucleus sampling") samples from the smallest set of
  // tokens that exceed probability topp. This way we never sample tokens that
  // have very low probabilities and are less likely to go "off the rails".
  // coin is a random number in [0, 1), usually from random_f32()
  int n = vocab_size_;
  int n0 = 0;
  // quicksort indices in descending order of probabilities
  // values smaller than (1 - topp) / (n - 1) cannot be part of the result
  // so for efficiency we crop these out as candidates before sorting
  std::unique_ptr<ProbIndex<T>[]> probindex =
      std::make_unique<ProbIndex<T>[]>(vocab_size_);

  const float cutoff = (1.0f - topp_) / (n - 1);
  for (int i = 0; i < n; i++) {
    if (probabilities[i] >= cutoff) {
      probindex[n0].index = i;
      probindex[n0].prob = probabilities[i];
      n0++;
    }
  }

  auto compare = [](const ProbIndex<T>& a, const ProbIndex<T>& b) {
    return a.prob > b.prob;
  };
  std::sort(probindex.get(), probindex.get() + n0, compare);

  // truncate the list where cumulative probability exceeds topp
  T cumulative_prob = 0;
  int last_idx = n0 - 1; // in case of rounding errors consider all elements
  for (int i = 0; i < n0; i++) {
    cumulative_prob += probindex[i].prob;
    if (cumulative_prob > topp_) {
      last_idx = i;
      break; // we've exceeded topp by including last_idx
    }
  }

  // sample from the truncated list
  const T& r = coin * cumulative_prob;
  T cdf = 0;
  for (int i = 0; i <= last_idx; i++) {
    cdf += probindex[i].prob;
    if (r < cdf) {
      return probindex[i].index;
    }
  }
  return probindex[last_idx].index; // in case of rounding errors
}

Sampler::Sampler(
    int vocab_size,
    float temperature,
    float topp,
    unsigned long long rng_seed)
    : vocab_size_(vocab_size),
      inv_temperature_(static_cast<bool>(temperature) ? 1.0f / temperature : 0),
      topp_(topp),
      rng_state_(rng_seed) {}

template <typename T>
static void softmax(T* x, int size) {
  // find max value (for numerical stability)
  T max_val = x[0];
  for (int i = 1; i < size; i++) {
    if (x[i] > max_val) {
      max_val = x[i];
    }
  }
  // exp and sum
  T sum = 0;
  for (int i = 0; i < size; i++) {
    x[i] = expf(x[i] - max_val);
    sum += x[i];
  }
  // normalize
  for (int i = 0; i < size; i++) {
    x[i] /= sum;
  }
}

static unsigned int random_u32(unsigned long long* state) {
  // xorshift rng: https://en.wikipedia.org/wiki/Xorshift#xorshift.2A
  *state ^= *state >> 12;
  *state ^= *state << 25;
  *state ^= *state >> 27;
  return (*state * 0x2545F4914F6CDD1Dull) >> 32;
}

static float random_f32(unsigned long long* state) { // random float32 in [0,1)
  return (random_u32(state) >> 8) / 16777216.0f;
}

template <typename T>
int32_t Sampler::sample(T* logits) {
  // sample the token given the logits and some hyperparameters
  int next;
  if (inv_temperature_ == 0.0f) {
    // greedy argmax sampling: take the token with the highest probability
    next = sample_argmax(logits);
  } else {
    // apply the temperature to the logits
    for (int q = 0; q < vocab_size_; q++) {
      logits[q] *= inv_temperature_;
    }
    // apply softmax to the logits to get the probabilities for next token
    softmax(logits, vocab_size_);
    // flip a (float) coin (this is our source of entropy for sampling)
    float coin = random_f32(&rng_state_);
    // we sample from this distribution to get the next token
    if (topp_ <= 0 || topp_ >= 1) {
      // simply sample from the predicted probability distribution
      next = sample_mult(logits, coin);
    } else {
      // top-p (nucleus) sampling, clamping the least likely tokens to zero
      next = sample_topp(logits, coin);
    }
  }
  return next;
}

template int32_t Sampler::sample<float>(float* logits);
template int32_t Sampler::sample<exec_aten::Half>(exec_aten::Half* logits);
template int32_t Sampler::sample<exec_aten::BFloat16>(
    exec_aten::BFloat16* logits);

} // namespace llm
} // namespace extension
} // namespace executorch
