/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "MLXInterpreter.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <random>
#include <tuple>
#include <vector>

namespace mx = ::mlx::core;
using namespace ::executorch::backends::mlx;

namespace {

std::vector<float> read_floats(const mx::array& input) {
  auto a = mx::contiguous(mx::astype(input, mx::float32));
  mx::eval(a);
  return {a.data<float>(), a.data<float>() + a.size()};
}

mx::array input_array(const mx::Shape& shape, mx::Dtype dtype, uint32_t seed) {
  size_t size = 1;
  for (int dim : shape) {
    size *= dim;
  }
  std::mt19937 rng(seed);
  std::vector<float> data(size);
  for (auto& value : data) {
    // Exactly representable in all three activation dtypes.
    value = static_cast<float>(static_cast<int>(rng() % 33) - 16) / 16;
  }
  return mx::astype(mx::array(data.data(), shape), dtype);
}

void expect_close(
    const mx::array& output,
    const std::vector<float>& expected,
    float tolerance) {
  auto actual = read_floats(output);
  ASSERT_EQ(actual.size(), expected.size());
  double worst = 0;
  size_t index = 0;
  for (size_t i = 0; i < actual.size(); ++i) {
    ASSERT_TRUE(std::isfinite(actual[i])) << "element " << i;
    const double error = std::abs(actual[i] - expected[i]) /
        (tolerance + tolerance * std::abs(expected[i]));
    if (error > worst) {
      worst = error;
      index = i;
    }
  }
  EXPECT_LE(worst, 1.0) << "element " << index << ": actual " << actual[index]
                        << ", expected " << expected[index];
}

std::vector<float> attention_reference(
    const mx::array& q,
    const mx::array& k,
    const mx::array& v,
    float scale,
    int window,
    bool causal) {
  const auto queries = read_floats(q);
  const auto keys = read_floats(k);
  const auto values = read_floats(v);
  const int batches = q.shape(0), heads = q.shape(1), rows = q.shape(2);
  const int dim = q.shape(3), kv_heads = k.shape(1), columns = k.shape(2);
  std::vector<float> expected(q.size());
  std::vector<double> scores(columns);
  for (int b = 0; b < batches; ++b) {
    for (int h = 0; h < heads; ++h) {
      const int kv_head = h / (heads / kv_heads);
      const size_t kv_base = (b * kv_heads + kv_head) * columns * dim;
      for (int row = 0; row < rows; ++row) {
        const size_t q_base = ((b * heads + h) * rows + row) * dim;
        const int position = columns - rows + row;
        double maximum = -std::numeric_limits<double>::infinity();
        for (int col = 0; col < columns; ++col) {
          double score = -std::numeric_limits<double>::infinity();
          if ((!causal || col <= position) &&
              (window < 0 || std::abs(position - col) < window)) {
            score = 0;
            for (int d = 0; d < dim; ++d) {
              score +=
                  double(queries[q_base + d]) * keys[kv_base + col * dim + d];
            }
            score *= scale;
          }
          scores[col] = score;
          maximum = std::max(maximum, score);
        }
        double denominator = 0;
        for (auto& score : scores) {
          score = std::exp(score - maximum);
          denominator += score;
        }
        for (int d = 0; d < dim; ++d) {
          double sum = 0;
          for (int col = 0; col < columns; ++col) {
            sum += scores[col] * values[kv_base + col * dim + d];
          }
          expected[q_base + d] = static_cast<float>(sum / denominator);
        }
      }
    }
  }
  return expected;
}

class MLXSpeculativeOpsTest : public ::testing::TestWithParam<mx::Dtype> {
 protected:
  float tolerance() const {
    if (GetParam() == mx::float32) {
      // NAX attention may use TF32; CTest also runs with it disabled.
      return mx::env::enable_tf32() ? 1e-3f : 1e-4f;
    }
    return GetParam() == mx::bfloat16 ? 2e-2f : 5e-3f;
  }
};

TEST_P(MLXSpeculativeOpsTest, SlidingWindowMatchesHostReference) {
  constexpr int dim = 128;
  const float scale = 1.0f / std::sqrt(float(dim));
  for (int rows : {1, 4, 16}) {
    for (int window : {1, 3, 64}) {
      for (bool causal : {false, true}) {
        SCOPED_TRACE(
            (::testing::Message() << "rows=" << rows << " window=" << window
                                  << " causal=" << causal));
        const auto q = input_array({2, 8, rows, dim}, GetParam(), 1);
        const auto k = input_array({2, 2, 33, dim}, GetParam(), 2);
        const auto v = input_array({2, 2, 33, dim}, GetParam(), 3);
        const auto expected =
            attention_reference(q, k, v, scale, window, causal);
        const auto out = mx::fast::scaled_dot_product_attention(
            q,
            k,
            v,
            scale,
            window,
            causal ? "causal" : "",
            std::nullopt,
            std::nullopt,
            true,
            mx::Device::gpu);
        EXPECT_EQ(out.shape(), q.shape());
        EXPECT_EQ(out.dtype(), GetParam());
        expect_close(out, expected, tolerance());
      }
    }
  }
}

TEST_P(MLXSpeculativeOpsTest, SlidingWindowMovesWithEachDraftPosition) {
  constexpr int rows = 16, dim = 128, window = 2048;
  for (int context : {2047, 2048, 2049}) {
    for (bool causal : {false, true}) {
      SCOPED_TRACE(
          (::testing::Message()
           << "context=" << context << " causal=" << causal));
      const int columns = context + rows;
      const auto q = mx::zeros({1, 4, rows, dim}, GetParam());
      const auto k = mx::zeros({1, 2, columns, dim}, GetParam());
      std::vector<float> values(2 * columns * dim, 0);
      for (int h = 0; h < 2; ++h) {
        // Impulses make one-key window errors visible even in bfloat16.
        for (int col = 0; col <= rows; ++col) {
          values[(h * columns + col) * dim + col] = window;
        }
        values[(h * columns + columns - 1) * dim + rows + 1] = window;
      }
      const auto v =
          mx::astype(mx::array(values.data(), k.shape()), GetParam());
      const auto expected = attention_reference(q, k, v, 1, window, causal);
      const auto out = mx::fast::scaled_dot_product_attention(
          q,
          k,
          v,
          1,
          window,
          causal ? "causal" : "",
          std::nullopt,
          std::nullopt,
          true,
          mx::Device::gpu);
      expect_close(out, expected, tolerance());
    }
  }
}

TEST_P(MLXSpeculativeOpsTest, ShortQueriesAndStridedCachesMatchHostReference) {
  constexpr int dim = 128, kv_heads = 4;
  const float scale = 1.0f / std::sqrt(float(dim));
  // Six/eight queries with GQA > 4 exercise the new split-attention kernels.
  for (const auto& [rows, columns, heads] :
       {std::tuple{4, 129, 16},
        std::tuple{6, 128, 24},
        std::tuple{8, 1985, 28},
        std::tuple{16, 257, 16}}) {
    for (bool causal : {false, true}) {
      SCOPED_TRACE(
          (::testing::Message() << "rows=" << rows << " columns=" << columns
                                << " heads=" << heads << " causal=" << causal));
      const auto q = mx::transpose(
          input_array({1, rows, heads, dim}, GetParam(), 4), {0, 2, 1, 3});
      const auto k = mx::slice(
          input_array({1, kv_heads, columns + 64, dim}, GetParam(), 5),
          {0, 0, 0, 0},
          {1, kv_heads, columns, dim});
      const auto v = mx::slice(
          input_array({1, kv_heads, columns + 64, dim}, GetParam(), 6),
          {0, 0, 0, 0},
          {1, kv_heads, columns, dim});
      const auto expected = attention_reference(q, k, v, scale, -1, causal);
      const auto fused = mx::fast::scaled_dot_product_attention(
          q,
          k,
          v,
          scale,
          -1,
          causal ? "causal" : "",
          std::nullopt,
          std::nullopt,
          true,
          mx::Device::gpu);
      expect_close(fused, expected, tolerance());

      MLXProgram program;
      program.num_input_tensors = 3;
      program.num_output_tensors = 1;
      ConstantData constants;
      MutableBufferData buffers;
      ExecutionState state;
      state.bind(program, constants, buffers);
      state.set_tensor(Tid{0}, q);
      state.set_tensor(Tid{1}, k);
      state.set_tensor(Tid{2}, v);
      SdpaNode node{
          Tid{0}, Tid{1}, Tid{2}, Tid{3}, scale, std::nullopt, causal};
      ops::exec_sdpa(node, state, mx::Device::gpu);
      expect_close(state.const_tensor_ref(Tid{3}), expected, tolerance());
    }
  }
}

TEST_P(MLXSpeculativeOpsTest, AffineMatmulFewRowsMatchesHostReference) {
  constexpr int max_rows = 16, group_size = 64;
  for (const auto& [inputs, outputs] :
       {std::pair{512, 4096}, std::pair{1024, 2048}}) {
    const auto x = input_array({max_rows, inputs}, GetParam(), 7);
    const auto host_x = read_floats(x);
    std::mt19937 rng(8);
    std::vector<uint32_t> packed(outputs * inputs / 8);
    for (auto& word : packed) {
      word = rng();
    }
    std::vector<float> scales(outputs * inputs / group_size),
        biases(scales.size());
    for (size_t i = 0; i < scales.size(); ++i) {
      scales[i] = float(1 + rng() % 4) / 32;
      biases[i] = -8 * scales[i];
    }
    std::vector<float> expected(max_rows * outputs);
    for (int row = 0; row < max_rows; ++row) {
      for (int col = 0; col < outputs; ++col) {
        double sum = 0;
        for (int i = 0; i < inputs; ++i) {
          const int q =
              (packed[col * inputs / 8 + i / 8] >> (4 * (i % 8))) & 15;
          const int group = col * inputs / group_size + i / group_size;
          sum += double(host_x[row * inputs + i]) *
              (q * scales[group] + biases[group]);
        }
        expected[row * outputs + col] = static_cast<float>(sum);
      }
    }
    for (int rows : {3, 4, 8, 9, 16}) {
      SCOPED_TRACE(
          (::testing::Message()
           << "rows=" << rows << " K=" << inputs << " N=" << outputs));
      MLXProgram program;
      program.num_input_tensors = 4;
      program.num_output_tensors = 1;
      ConstantData constants;
      MutableBufferData buffers;
      ExecutionState state;
      state.bind(program, constants, buffers);
      state.set_tensor(Tid{0}, mx::slice(x, {0, 0}, {rows, inputs}));
      state.set_tensor(Tid{1}, mx::array(packed.data(), {outputs, inputs / 8}));
      state.set_tensor(
          Tid{2},
          mx::astype(
              mx::array(scales.data(), {outputs, inputs / group_size}),
              GetParam()));
      state.set_tensor(
          Tid{3},
          mx::astype(
              mx::array(biases.data(), {outputs, inputs / group_size}),
              GetParam()));
      QuantizedMatmulNode node{
          Tid{0},
          Tid{1},
          Tid{2},
          Tid{4},
          Tid{3},
          group_size,
          4,
          "affine",
          true};
      ops::exec_quantized_matmul(node, state, mx::Device::gpu);
      const auto& out = state.const_tensor_ref(Tid{4});
      EXPECT_EQ(out.shape(), (mx::Shape{rows, outputs}));
      EXPECT_EQ(out.dtype(), GetParam());
      expect_close(
          out,
          {expected.begin(), expected.begin() + rows * outputs},
          tolerance());
    }
  }
}

INSTANTIATE_TEST_SUITE_P(
    Dtypes,
    MLXSpeculativeOpsTest,
    ::testing::Values(mx::float32, mx::float16, mx::bfloat16),
    [](const ::testing::TestParamInfo<mx::Dtype>& info) {
      return info.param == mx::float32 ? "Float32"
          : info.param == mx::float16  ? "Float16"
                                       : "BFloat16";
    });

} // namespace
