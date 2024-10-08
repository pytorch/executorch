/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/extension/llm/custom_ops/spinquant/test/fast_hadamard_transform_test_impl.h>
#include <executorch/extension/llm/custom_ops/spinquant/third-party/FFHT/dumb_fht.h>

#include <cmath>
#include <random>
#include <vector>

namespace executorch::runtime::testing {

void reference_fht_impl(float* buf, int n) {
  dumb_fht(buf, std::log2<int>(n));
  const auto root_n = std::sqrt(n);
  for (int ii = 0; ii < n; ++ii) {
    buf[ii] /= root_n;
  }
}

std::vector<float> random_floats(int howMany) {
  std::random_device rd;
  std::mt19937 gen(rd());
  std::normal_distribution<float> dist;
  std::vector<float> data(howMany);
  for (int ii = 0; ii < data.size(); ++ii) {
    data[ii] = dist(gen);
  }
  return data;
}

} // namespace executorch::runtime::testing
