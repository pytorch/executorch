/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <cuda_runtime.h>
#include <executorch/backends/cuda/runtime/shims/int4_plain_mm.h>
#include <executorch/backends/cuda/runtime/shims/int5_plain_mm.h>
#include <executorch/backends/cuda/runtime/shims/int6_plain_mm.h>
#include <executorch/backends/cuda/runtime/shims/memory.h>
#include <executorch/runtime/platform/platform.h>
#include <gtest/gtest.h>

#include <algorithm>
#include <cstring>
#include <memory>
#include <tuple>
#include <vector>

namespace {
namespace cuda = executorch::backends::cuda;
namespace slim = executorch::backends::aoti::slim;
using Tensor = slim::SlimTensor;
using executorch::runtime::Error;

uint16_t bf16(float value) {
  uint32_t bits;
  std::memcpy(&bits, &value, sizeof(bits));
  return (bits + 0x7fff + ((bits >> 16) & 1)) >> 16;
}

class SpeculativePlainMMTest
    : public ::testing::TestWithParam<std::tuple<int, int>> {
 protected:
  void SetUp() override {
    et_pal_init();
    int count = 0;
    if (cudaGetDeviceCount(&count) != cudaSuccess || count == 0) {
      GTEST_SKIP() << "CUDA not available";
    }
  }

  template <typename T>
  std::unique_ptr<Tensor> upload(
      std::vector<int64_t> sizes,
      slim::c10::ScalarType dtype,
      const std::vector<T>& values) {
    Tensor* tensor = nullptr;
    EXPECT_EQ(
        cuda::aoti_torch_empty_strided(
            sizes.size(),
            sizes.data(),
            nullptr,
            static_cast<int32_t>(dtype),
            static_cast<int32_t>(slim::c10::DeviceType::CUDA),
            0,
            &tensor),
        Error::Ok);
    EXPECT_EQ(
        cudaMemcpy(
            tensor->data_ptr(),
            values.data(),
            values.size() * sizeof(T),
            cudaMemcpyHostToDevice),
        cudaSuccess);
    return std::unique_ptr<Tensor>(tensor);
  }
};

// Binary-exact inputs make the independent CPU dot product an exact reference.
// Each activation block includes 127 so its Q8 scale is known without rounding.
TEST_P(SpeculativePlainMMTest, MatchesIndependentDotProducts) {
  const auto [bits, rows] = GetParam();
  constexpr int N = 19, K = 768;
  const int group = bits == 6 ? 16 : 32;
  using D = slim::c10::ScalarType;
  std::vector<uint16_t> activations(rows * K), expected(rows * N);
  std::vector<float> a_ref(rows * K), w_ref(N * K);
  std::vector<uint8_t> low_values(N * K / 2),
      high_values(N * K / (bits == 5 ? 8 : 4));
  std::vector<uint8_t> codes(N * K / group), zeros(N * K / group);
  for (int row = 0; row < rows; ++row) {
    for (int k = 0; k < K; ++k) {
      const int q = k % 32 == 0 ? 127 : (k * 17 + row * 31) % 255 - 127;
      a_ref[row * K + k] = row == 0 ? 0.0f : q * (1 << (row % 3)) / 128.0f;
      activations[row * K + k] = bf16(a_ref[row * K + k]);
    }
  }
  for (int n = 0; n < N; ++n) {
    for (int g = 0; g < K / group; ++g) {
      codes[n * K / group + g] = bits == 6 ? (n + g) % 7 - 3 : (n + g) % 5 + 1;
      zeros[n * K / group + g] = (n + g) % 4 + 1;
    }
    for (int k = 0; k < K; ++k) {
      const int q = (n * 13 + k * 7) % (1 << bits);
      low_values[n * K / 2 + k / 2] |= (q & 15) << ((k % 2) * 4);
      if (bits == 5) {
        high_values[n * K / 8 + k / 8] |= (q >> 4)
            << ((k % 8) / 2 + (k % 2) * 4);
      } else if (bits == 6) {
        high_values[n * K / 4 + k / 32 * 8 + (k % 32) / 8 + (k % 2) * 4] |=
            (q >> 4) << ((k % 8) / 2 * 2);
      }
      const int g = n * K / group + k / group;
      const float scale =
          (bits == 6 ? static_cast<int8_t>(codes[g]) : codes[g]) / 32.0f;
      w_ref[n * K + k] = (bits == 6 ? q - 32.0f : q - zeros[g] / 4.0f) * scale;
    }
  }
  for (int row = 0; row < rows; ++row) {
    for (int n = 0; n < N; ++n) {
      float sum = 0;
      for (int k = 0; k < K; ++k) {
        sum += a_ref[row * K + k] * w_ref[n * K + k];
      }
      expected[row * N + n] = bf16(sum);
    }
  }
  auto a = upload({rows, K}, D::BFloat16, activations);
  auto low = upload({N, K / 2}, D::Byte, low_values);
  auto scale = upload({N, K / group}, bits == 6 ? D::Char : D::Byte, codes);
  auto zero = upload({N, K / group}, D::Byte, zeros);
  auto step =
      upload({N, K / 256}, D::Half, std::vector<uint16_t>(N * K / 256, 0x2800));
  auto zero_step =
      upload({N, K / 256}, D::Half, std::vector<uint16_t>(N * K / 256, 0x3400));
  Tensor* raw_output = nullptr;
  if (bits == 4) {
    EXPECT_EQ(
        cuda::aoti_torch_cuda_int4_plain_mm(
            a.get(),
            low.get(),
            scale.get(),
            step.get(),
            zero.get(),
            zero_step.get(),
            group,
            &raw_output),
        Error::Ok);
  } else {
    const int divisor = bits == 5 ? 8 : 4;
    auto high = upload({N, K / divisor}, D::Byte, high_values);
    if (bits == 5) {
      EXPECT_EQ(
          cuda::aoti_torch_cuda_int5_plain_mm(
              a.get(),
              low.get(),
              high.get(),
              scale.get(),
              step.get(),
              zero.get(),
              zero_step.get(),
              group,
              &raw_output),
          Error::Ok);
    } else {
      EXPECT_EQ(
          cuda::aoti_torch_cuda_int6_plain_mm(
              a.get(),
              low.get(),
              high.get(),
              scale.get(),
              step.get(),
              group,
              &raw_output),
          Error::Ok);
    }
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
  }
  std::unique_ptr<Tensor> output(raw_output);
  ASSERT_NE(output, nullptr);
  EXPECT_EQ(output->size(0), rows);
  EXPECT_EQ(output->size(1), N);
  std::vector<uint16_t> actual(rows * N);
  ASSERT_EQ(
      cudaMemcpy(
          actual.data(),
          output->data_ptr(),
          actual.size() * sizeof(uint16_t),
          cudaMemcpyDeviceToHost),
      cudaSuccess);
  EXPECT_EQ(actual, expected);
}

INSTANTIATE_TEST_SUITE_P(
    DFlash2,
    SpeculativePlainMMTest,
    ::testing::Combine(::testing::Values(4, 5, 6), ::testing::Range(1, 17)));
} // namespace
