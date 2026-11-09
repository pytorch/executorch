// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <executorch/backends/vulkan/test/ops/benchmark_v2/operators/gemm/GemmReference.h>

#include <algorithm>
#include <array>

namespace vkcompute::shader_profiling::reference {

Tensor linear_projection(
    const Tensor& input,
    const Tensor& weight,
    const Tensor* bias,
    size_t threads) {
  require(
      !input.sizes.empty() && weight.sizes.size() == 2,
      "Linear requires input rank >= 1 and weight rank 2");
  const size_t k = input.sizes.back();
  const size_t n = weight.sizes[0];
  require(
      k == size_t(weight.sizes[1]), "Linear input/weight dimensions differ");
  require(
      !bias || bias->values.size() == n,
      "Linear bias size must equal output features");
  const size_t m = input.values.size() / k;
  Tensor output{input.sizes, std::vector<double>(m * n)};
  output.sizes.back() = n;
  std::vector<float> packed(k * n);
  parallel_for(n, threads, [&](size_t column) {
    for (size_t inner = 0; inner < k; ++inner) {
      packed[inner * n + column] = weight.values[column * k + inner];
    }
  });
  std::vector<float> activations(input.values.begin(), input.values.end());
  constexpr size_t kRows = 4;
  constexpr size_t kColumns = 64;
  const size_t column_tiles = (n + kColumns - 1) / kColumns;
  const size_t row_tiles = (m + kRows - 1) / kRows;
  parallel_for(row_tiles * column_tiles, threads, [&](size_t tile) {
    const size_t row = (tile / column_tiles) * kRows;
    const size_t column = (tile % column_tiles) * kColumns;
    const size_t rows = std::min(kRows, m - row);
    const size_t columns = std::min(kColumns, n - column);
    std::array<std::array<float, kColumns>, kRows> accumulators{};
    for (size_t inner = 0; inner < k; ++inner) {
      const float* weights = packed.data() + inner * n + column;
      for (size_t i = 0; i < rows; ++i) {
        const float activation = activations[(row + i) * k + inner];
        for (size_t j = 0; j < columns; ++j) {
          accumulators[i][j] += activation * weights[j];
        }
      }
    }
    for (size_t i = 0; i < rows; ++i) {
      for (size_t j = 0; j < columns; ++j) {
        output.values[(row + i) * n + column + j] = accumulators[i][j] +
            (bias ? float(bias->values[column + j]) : 0.0f);
      }
    }
  });
  return output;
}

} // namespace vkcompute::shader_profiling::reference
