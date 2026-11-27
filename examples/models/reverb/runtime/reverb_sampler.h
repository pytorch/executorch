/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <cstdint>
#include <random>
#include <vector>

namespace reverb {

constexpr int kCodebooks = 8;
constexpr int kMainCodebooks = 2;
constexpr int64_t kMask = 1024;

// How takes are sampled. The defaults are the tuned setup of the reference
// implementation.
struct SamplerConfig {
  int steps = 8; // main passes, which decide codebooks 0-1
  bool remasking = true; // whether main passes withdraw doubted commits
  float remask_ratio = 0.25f; // commits withdrawn per new commit
  float t_shift = 0.1f; // warps the schedule: most commits in late passes
  float codebook_penalty = 5.0f; // per codebook index: codebook 0 goes first
  float position_temperature = 5.0f; // confidence scale against the noise
};

void validate(const SamplerConfig& config);

// The commit schedule t[0..steps], warped by t_shift; pass s commits about
// 2 * frames * (t[s + 1] - t[s]) slots. Computed in fp32 like the reference.
std::vector<float> commit_schedule(const SamplerConfig& config);

// What one main pass says about codebooks 0-1, each [2, frames] row-major.
struct MainPass {
  const float* confidence; // log-probability of the best code per slot
  const int64_t* prediction; // the best code per slot
  const float* held; // log-probability of the code a slot holds
};

// Uniform noise in [0, 1) from a seeded 64-bit Mersenne Twister. Seeds give
// the same takes on every platform, but not the takes of the PyTorch package.
class Noise {
 public:
  explicit Noise(uint64_t seed) : engine_(seed) {}
  float uniform();

 private:
  std::mt19937_64 engine_;
};

// Applies main pass `step` to `tokens` ([8, frames] row-major): commits the
// most confident masked slots of codebooks 0-1 and, on every pass but the
// last, remasks the held codes the model now finds least likely.
void apply_main_pass(
    std::vector<int64_t>& tokens,
    int frames,
    const MainPass& pass,
    int step,
    const SamplerConfig& config,
    const std::vector<float>& schedule,
    Noise& noise);

} // namespace reverb
