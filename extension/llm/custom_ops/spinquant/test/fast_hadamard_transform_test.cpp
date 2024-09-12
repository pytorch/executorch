/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <algorithm>
#include <cmath>
#include <iostream>
#include <random>

#include <gtest/gtest.h>

#include <executorch/extension/llm/custom_ops/spinquant/fast_hadamard_transform.h>
#include <executorch/extension/llm/custom_ops/spinquant/test/fast_hadamard_transform_special_unstrided_cpu.h>
#include <executorch/extension/llm/custom_ops/spinquant/third-party/FFHT/dumb_fht.h>

namespace {
void reference_fht_impl(float* buf, int n) {
  dumb_fht(buf, std::log2<int>(n));
  const auto root_n = std::sqrt(n);
  for (int ii = 0; ii < n; ++ii) {
    buf[ii] /= root_n;
  }
}

// Alternate implementation of fast_hadamard_transform_28N to mutation
// test against. Benchmarking suggests this one is slower, which is
// why it's in the test and the strided implementation is in the
// header.
template <typename T>
void fast_hadamard_transform_28N_with_transpose(T* vec, int log2_vec_size) {
  const int vec_size = (1 << log2_vec_size);
  for (int ii = 0; ii < 28; ++ii) {
    executorch::fast_hadamard_transform(&vec[ii * vec_size], log2_vec_size);
  }
  std::unique_ptr<T[]> transposed = std::make_unique<T[]>(28 * vec_size);
  for (int ii = 0; ii < 28; ++ii) {
    for (int jj = 0; jj < vec_size; ++jj) {
      transposed[jj * 28 + ii] = vec[ii * vec_size + jj];
    }
  }
  for (int ii = 0; ii < vec_size; ++ii) {
    hadamard_mult_28(&transposed[ii * 28]);
  }
  for (int jj = 0; jj < vec_size; ++jj) {
    for (int ii = 0; ii < 28; ++ii) {
      vec[ii * vec_size + jj] = transposed[jj * 28 + ii];
    }
  }
}

std::vector<float> randomFloats(int howMany) {
  std::random_device rd;
  std::mt19937 gen(rd());
  std::normal_distribution<float> dist;
  std::vector<float> data(howMany);
  for (int ii = 0; ii < data.size(); ++ii) {
    data[ii] = dist(gen);
  }
  return data;
}
} // namespace

TEST(FastHadamardTransformTest, SingleElement) {
  // FHT of a single element is a no-op.
  float data[1] = {42};
  executorch::fast_hadamard_transform(data, 0);
  EXPECT_EQ(data[0], 42);
}

TEST(FastHadamardTransformTest, LargerInput) {
  std::vector<float> data = randomFloats(4096);

  auto expected = data;
  reference_fht_impl(expected.data(), expected.size());

  auto actual = data;
  executorch::fast_hadamard_transform(actual.data(), 12);

  for (int ii = 0; ii < expected.size(); ++ii) {
    EXPECT_FLOAT_EQ(actual[ii], expected[ii]);
  }
}

TEST(FastHadamardTransform28NTest, Basic) {
  std::vector<float> data = randomFloats(1024 * 28);

  auto expected = data;
  fast_hadamard_transform_28N_with_transpose(expected.data(), 10);

  auto actual = data;
  executorch::fast_hadamard_transform_28N(actual.data(), 10);

  for (int ii = 0; ii < actual.size(); ++ii) {
    EXPECT_FLOAT_EQ(actual[ii], expected[ii]);
  }
}

namespace {
constexpr int32_t qmin = -(1 << 15) + 1;
constexpr int32_t qmax = -qmin;

int16_t quantize(float x, float scale) {
  float scaled = x / scale;
  // XXX: Supposed to round ties to even, but this is just test code.
  int32_t scaled_int =
      std::clamp((int32_t)std::lround<int32_t>(scaled), qmin, qmax);
  return static_cast<int16_t>(scaled_int);
}

template <typename T>
std::vector<T> quantize(const std::vector<float>& data, float scale) {
  std::vector<T> result;
  result.reserve(data.size());
  for (const float unquant : data) {
    result.push_back(quantize(unquant, scale));
  }
  return result;
}

template <typename T>
std::pair<std::vector<T>, float> quantize(const std::vector<float>& data) {
  auto [minIt, maxIt] = std::minmax_element(data.begin(), data.end());
  float scale = (*maxIt - *minIt) / (qmax - qmin);
  return {quantize<T>(data, scale), scale};
}

template <typename T>
float dequantize(T x, float scale) {
  return x * scale;
}

template <typename T>
std::vector<float> dequantize(const std::vector<T>& data, float scale) {
  static_assert(!std::is_same_v<T, float>);
  std::vector<float> result;
  result.reserve(data.size());
  for (const T quant : data) {
    result.push_back(dequantize(quant, scale));
  }
  return result;
}

#define EXPECT_CLOSE_IMPL(a, b, atol, rtol)             \
  EXPECT_LE(std::abs(a - b), atol + rtol * std::abs(b)) \
      << "a: " << a << ", b: " << b
#define EXPECT_CLOSE(a, b) EXPECT_CLOSE_IMPL(a, b, 2e-4, 1e-4)

void testQuantizedFastHadamardTransform(int logN) {
  std::vector<float> data = randomFloats(1 << logN);

  auto [qdata, scale] = quantize<int16_t>(data);

  auto expected_unquant = dequantize(qdata, scale);
  reference_fht_impl(expected_unquant.data(), expected_unquant.size());
  auto expected = quantize<int16_t>(expected_unquant, scale);

  auto actual = qdata;
  executorch::fast_hadamard_transform_symmetric_quantized_s16(
      actual.data(), logN);

  for (int ii = 0; ii < expected.size(); ++ii) {
    EXPECT_CLOSE(
        dequantize(actual[ii], scale), dequantize(expected[ii], scale));
  }
}

} // namespace

TEST(QuantizedFastHadamardTransformTest, Basic) {
  testQuantizedFastHadamardTransform(12); // 4096
}

TEST(QuantizedFastHadamardTransformTest, OddLogN) {
  testQuantizedFastHadamardTransform(11); // 2048
}

TEST(QuantizedFastHadamardTransform28NTest, Basic) {
  std::vector<float> data = randomFloats(1024 * 28);

  auto [qdata, scale] = quantize<int16_t>(data);

  auto expected_unquant = dequantize(qdata, scale);
  fast_hadamard_transform_28N_with_transpose(expected_unquant.data(), 10);
  auto expected = quantize<int16_t>(expected_unquant, scale);

  auto actual = qdata;
  executorch::fast_hadamard_transform_symmetric_quantized_s16_28N(
      actual.data(), 10);

  for (int ii = 0; ii < expected.size(); ++ii) {
    std::cerr << "element " << ii << ": actual: " << actual[ii]
              << ", expected: " << expected[ii] << std::endl;
    EXPECT_CLOSE(
        dequantize(actual[ii], scale), dequantize(expected[ii], scale));
  }
}
