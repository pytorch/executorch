/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/extension/diffusers/runner/torch_rng.h>

#include <executorch/runtime/core/exec_aten/exec_aten.h>
#include <executorch/runtime/platform/assert.h>

#include <algorithm>
#include <cmath>
#include <type_traits>

namespace executorch::extension::diffusers {

namespace {

constexpr double kPi = 3.14159265358979323846;

// Box-Muller over 8 pairs (data[j], data[j + 8]), as in torch's NormalFill16.
void normal_fill_16(float* data) {
  for (int j = 0; j < 8; ++j) {
    const float u1 = 1 - data[j];
    const float u2 = data[j + 8];
    const float radius = std::sqrt(-2 * std::log(u1));
    const float theta = 2.0f * kPi * u2;
    data[j] = radius * std::cos(theta);
    data[j + 8] = radius * std::sin(theta);
  }
}

} // namespace

TorchGenerator::TorchGenerator(uint64_t seed)
    : engine_(static_cast<uint32_t>(seed)) {}

float TorchGenerator::uniform() {
  constexpr uint32_t kMask = (1u << 24) - 1;
  return static_cast<float>(engine_() & kMask) / (1u << 24);
}

template <typename T>
std::vector<T> TorchGenerator::randn(size_t numel) {
  ET_CHECK_MSG(numel >= 16, "randn needs at least 16 elements, got %zu", numel);
  std::vector<T> out(numel);
  if constexpr (std::is_same_v<T, float>) {
    std::generate(out.begin(), out.end(), [this] { return uniform(); });
    for (size_t i = 0; i + 16 <= numel; i += 16) {
      normal_fill_16(out.data() + i);
    }
    if (numel % 16 != 0) {
      float* tail = out.data() + numel - 16;
      for (int j = 0; j < 16; ++j) {
        tail[j] = uniform();
      }
      normal_fill_16(tail);
    }
  } else {
    float buffer[16];
    auto fill = [&](size_t offset) {
      for (int j = 0; j < 16; ++j) {
        buffer[j] = uniform();
      }
      normal_fill_16(buffer);
      for (int j = 0; j < 16; ++j) {
        out[offset + j] = static_cast<T>(buffer[j]);
      }
    };
    for (size_t i = 0; i + 16 <= numel; i += 16) {
      fill(i);
    }
    if (numel % 16 != 0) {
      fill(numel - 16);
    }
  }
  return out;
}

template std::vector<float> TorchGenerator::randn<float>(size_t);
template std::vector<executorch::aten::Half>
    TorchGenerator::randn<executorch::aten::Half>(size_t);
template std::vector<executorch::aten::BFloat16>
    TorchGenerator::randn<executorch::aten::BFloat16>(size_t);

} // namespace executorch::extension::diffusers
