/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <vector>

namespace executorch::extension::diffusers {

/** torch.linspace for float32 or float64. */
template <typename T>
std::vector<T> torch_linspace(T start, T end, int steps) {
  if (steps == 1) {
    return {start};
  }
  const T step = (end - start) / static_cast<T>(steps - 1);
  const int halfway = steps / 2;
  std::vector<T> values(steps);
  for (int i = 0; i < steps; ++i) {
    values[i] = i < halfway ? start + step * static_cast<T>(i)
                            : end - step * static_cast<T>(steps - i - 1);
  }
  return values;
}

} // namespace executorch::extension::diffusers
