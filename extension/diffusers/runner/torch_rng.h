/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <cstddef>
#include <cstdint>
#include <random>
#include <vector>

// Port of PyTorch's CPU generator and normal sampling
// (aten/src/ATen/core/MT19937RNGEngine.h,
// aten/src/ATen/core/DistributionsHelper.h,
// aten/src/ATen/native/cpu/DistributionTemplates.h).

namespace executorch::extension::diffusers {

/**
 * Reproduces torch.Generator().manual_seed(seed) followed by torch.randn on
 * CPU, so a seed yields the same noise as the Python runner. Successive calls
 * continue the same stream, as with a reused torch.Generator.
 */
class TorchGenerator {
 public:
  explicit TorchGenerator(uint64_t seed);

  /**
   * Draws numel standard normal values, matching torch.randn(numel,
   * dtype=...) for a contiguous output of at least 16 elements. T is float,
   * executorch::aten::Half or executorch::aten::BFloat16.
   */
  template <typename T>
  std::vector<T> randn(size_t numel);

 private:
  float uniform();

  std::mt19937 engine_;
};

} // namespace executorch::extension::diffusers
