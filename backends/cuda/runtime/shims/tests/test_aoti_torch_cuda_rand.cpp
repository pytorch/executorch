/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <cuda_runtime.h>
#include <executorch/backends/aoti/slim/c10/core/DeviceType.h>
#include <executorch/backends/aoti/slim/c10/core/ScalarType.h>
#include <executorch/backends/cuda/runtime/shims/memory.h>
#include <executorch/backends/cuda/runtime/shims/rand.h>
#include <executorch/runtime/core/error.h>
#include <executorch/runtime/platform/platform.h>
#include <gtest/gtest.h>

#include <cstdint>
#include <cstring>
#include <limits>
#include <vector>

// Use explicit types to avoid ambiguity between different Tensor typedefs
using executorch::backends::cuda::aoti_torch_cuda_rand;
using executorch::backends::cuda::aoti_torch_cuda_randint_low_out;
using executorch::backends::cuda::aoti_torch_empty_strided;
using executorch::backends::cuda::AOTITorchError;
using executorch::runtime::Error;
namespace slim_c10 = executorch::backends::aoti::slim::c10;

// Tensor type definition using SlimTensor
using Tensor = executorch::backends::aoti::slim::SlimTensor;

namespace {

// Helper: convert raw bfloat16 bits (uint16_t) to float for value checks.
float bfloat16_bits_to_float(uint16_t bits) {
  uint32_t expanded = static_cast<uint32_t>(bits) << 16;
  float result;
  std::memcpy(&result, &expanded, sizeof(float));
  return result;
}

} // namespace

// Test fixture for aoti_torch_cuda_rand tests
class AOTITorchCudaRandTest : public ::testing::Test {
 protected:
  void SetUp() override {
    // Initialize ExecuTorch Platform Abstraction Layer
    et_pal_init();

    // Check if CUDA is available
    int device_count = 0;
    cudaError_t err = cudaGetDeviceCount(&device_count);
    if (err != cudaSuccess || device_count == 0) {
      GTEST_SKIP() << "CUDA not available, skipping CUDA tests";
    }
  }

  // Helper: create a pre-allocated Int64 (Long) CUDA tensor used as the
  // `out` argument for randint_low_out.
  Tensor* create_int64_tensor(const std::vector<int64_t>& sizes) {
    Tensor* tensor = nullptr;
    AOTITorchError error = aoti_torch_empty_strided(
        sizes.size(),
        sizes.data(),
        nullptr, // default contiguous strides
        static_cast<int32_t>(slim_c10::ScalarType::Long),
        static_cast<int32_t>(slim_c10::DeviceType::CUDA),
        0, // device index
        &tensor);
    return (error == Error::Ok) ? tensor : nullptr;
  }

  // Helper: copy a CUDA tensor's raw bytes back to host.
  template <typename T>
  std::vector<T> copy_to_host(const Tensor* tensor, int64_t numel) {
    std::vector<T> host(static_cast<size_t>(numel));
    cudaError_t err = cudaMemcpy(
        host.data(),
        tensor->data_ptr(),
        static_cast<size_t>(numel) * sizeof(T),
        cudaMemcpyDeviceToHost);
    EXPECT_EQ(err, cudaSuccess) << "cudaMemcpy D2H failed";
    return host;
  }
};

// ----------------------------------------------------------------------------
// aoti_torch_cuda_rand tests
// ----------------------------------------------------------------------------

// Basic float32 rand: produces a tensor in [0, 1).
TEST_F(AOTITorchCudaRandTest, RandFloat32Basic) {
  std::vector<int64_t> sizes = {4, 8};
  int64_t numel = 4 * 8;
  int32_t dtype = static_cast<int32_t>(slim_c10::ScalarType::Float);

  Tensor* out = nullptr;
  AOTITorchError error = aoti_torch_cuda_rand(
      sizes.data(),
      static_cast<int64_t>(sizes.size()),
      &dtype,
      /*layout=*/nullptr,
      /*device=*/nullptr,
      /*device_index_=*/0,
      /*pin_memory=*/nullptr,
      &out);

  ASSERT_EQ(error, Error::Ok) << "aoti_torch_cuda_rand should succeed";
  ASSERT_NE(out, nullptr);
  EXPECT_EQ(out->dim(), 2);
  EXPECT_EQ(out->size(0), 4);
  EXPECT_EQ(out->size(1), 8);
  EXPECT_EQ(out->numel(), numel);
  ASSERT_NE(out->data_ptr(), nullptr);

  cudaDeviceSynchronize();
  auto host = copy_to_host<float>(out, numel);
  for (int64_t i = 0; i < numel; ++i) {
    EXPECT_GE(host[i], 0.0f) << "value at " << i << " < 0";
    EXPECT_LT(host[i], 1.0f) << "value at " << i << " >= 1";
  }
}

// Default dtype path: when dtype is null the shim defaults to float32.
TEST_F(AOTITorchCudaRandTest, RandDefaultDtypeIsFloat) {
  std::vector<int64_t> sizes = {16};
  Tensor* out = nullptr;
  AOTITorchError error = aoti_torch_cuda_rand(
      sizes.data(),
      static_cast<int64_t>(sizes.size()),
      /*dtype=*/nullptr,
      /*layout=*/nullptr,
      /*device=*/nullptr,
      /*device_index_=*/0,
      /*pin_memory=*/nullptr,
      &out);

  ASSERT_EQ(error, Error::Ok);
  ASSERT_NE(out, nullptr);
  EXPECT_EQ(out->dtype(), slim_c10::ScalarType::Float);
  EXPECT_EQ(out->numel(), 16);
}

// BFloat16 rand: values must lie in [0, 1).
TEST_F(AOTITorchCudaRandTest, RandBFloat16Basic) {
  std::vector<int64_t> sizes = {32};
  int64_t numel = 32;
  int32_t dtype = static_cast<int32_t>(slim_c10::ScalarType::BFloat16);

  Tensor* out = nullptr;
  AOTITorchError error = aoti_torch_cuda_rand(
      sizes.data(),
      static_cast<int64_t>(sizes.size()),
      &dtype,
      /*layout=*/nullptr,
      /*device=*/nullptr,
      /*device_index_=*/0,
      /*pin_memory=*/nullptr,
      &out);

  ASSERT_EQ(error, Error::Ok);
  ASSERT_NE(out, nullptr);
  EXPECT_EQ(out->dtype(), slim_c10::ScalarType::BFloat16);
  EXPECT_EQ(out->numel(), numel);

  cudaDeviceSynchronize();
  auto host = copy_to_host<uint16_t>(out, numel);
  for (int64_t i = 0; i < numel; ++i) {
    float v = bfloat16_bits_to_float(host[i]);
    EXPECT_GE(v, 0.0f) << "bf16 value at " << i << " < 0";
    EXPECT_LT(v, 1.0f) << "bf16 value at " << i << " >= 1";
  }
}

// Empty tensor: numel == 0 should be a no-op success.
TEST_F(AOTITorchCudaRandTest, RandEmptyTensor) {
  std::vector<int64_t> sizes = {0, 4};
  int32_t dtype = static_cast<int32_t>(slim_c10::ScalarType::Float);

  Tensor* out = nullptr;
  AOTITorchError error = aoti_torch_cuda_rand(
      sizes.data(),
      static_cast<int64_t>(sizes.size()),
      &dtype,
      /*layout=*/nullptr,
      /*device=*/nullptr,
      /*device_index_=*/0,
      /*pin_memory=*/nullptr,
      &out);

  ASSERT_EQ(error, Error::Ok);
  ASSERT_NE(out, nullptr);
  EXPECT_EQ(out->numel(), 0);
}

// Unsupported dtype should return NotSupported.
TEST_F(AOTITorchCudaRandTest, RandUnsupportedDtypeFails) {
  std::vector<int64_t> sizes = {8};
  // Long is not supported by aoti_torch_cuda_rand.
  int32_t dtype = static_cast<int32_t>(slim_c10::ScalarType::Long);

  Tensor* out = nullptr;
  AOTITorchError error = aoti_torch_cuda_rand(
      sizes.data(),
      static_cast<int64_t>(sizes.size()),
      &dtype,
      /*layout=*/nullptr,
      /*device=*/nullptr,
      /*device_index_=*/0,
      /*pin_memory=*/nullptr,
      &out);

  EXPECT_EQ(error, Error::NotSupported);
}

// Null ret0 should fail with InvalidArgument.
TEST_F(AOTITorchCudaRandTest, RandNullRet0Fails) {
  std::vector<int64_t> sizes = {4};
  int32_t dtype = static_cast<int32_t>(slim_c10::ScalarType::Float);

  AOTITorchError error = aoti_torch_cuda_rand(
      sizes.data(),
      static_cast<int64_t>(sizes.size()),
      &dtype,
      /*layout=*/nullptr,
      /*device=*/nullptr,
      /*device_index_=*/0,
      /*pin_memory=*/nullptr,
      /*ret0=*/nullptr);

  EXPECT_EQ(error, Error::InvalidArgument);
}

// Two invocations should advance the GPU-resident counter and produce
// different sequences (extremely high probability for non-trivial sizes).
TEST_F(AOTITorchCudaRandTest, RandTwoCallsProduceDifferentValues) {
  std::vector<int64_t> sizes = {64};
  int64_t numel = 64;
  int32_t dtype = static_cast<int32_t>(slim_c10::ScalarType::Float);

  Tensor* out_a = nullptr;
  Tensor* out_b = nullptr;

  ASSERT_EQ(
      aoti_torch_cuda_rand(
          sizes.data(),
          static_cast<int64_t>(sizes.size()),
          &dtype,
          nullptr,
          nullptr,
          0,
          nullptr,
          &out_a),
      Error::Ok);
  ASSERT_EQ(
      aoti_torch_cuda_rand(
          sizes.data(),
          static_cast<int64_t>(sizes.size()),
          &dtype,
          nullptr,
          nullptr,
          0,
          nullptr,
          &out_b),
      Error::Ok);

  cudaDeviceSynchronize();
  auto host_a = copy_to_host<float>(out_a, numel);
  auto host_b = copy_to_host<float>(out_b, numel);

  // The two draws must not be bit-for-bit identical.
  bool any_diff = false;
  for (int64_t i = 0; i < numel; ++i) {
    if (host_a[i] != host_b[i]) {
      any_diff = true;
      break;
    }
  }
  EXPECT_TRUE(any_diff)
      << "two consecutive aoti_torch_cuda_rand calls produced identical values";
}

// ----------------------------------------------------------------------------
// aoti_torch_cuda_randint_low_out tests
// ----------------------------------------------------------------------------

// Basic randint into a pre-allocated int64 tensor; values lie in [low, high).
TEST_F(AOTITorchCudaRandTest, RandintBasicRange) {
  std::vector<int64_t> sizes = {32};
  int64_t numel = 32;
  int64_t low = -5;
  int64_t high = 17;

  Tensor* out = create_int64_tensor(sizes);
  ASSERT_NE(out, nullptr);

  AOTITorchError error = aoti_torch_cuda_randint_low_out(
      out, low, high, sizes.data(), static_cast<int64_t>(sizes.size()));
  ASSERT_EQ(error, Error::Ok);

  cudaDeviceSynchronize();
  auto host = copy_to_host<int64_t>(out, numel);
  for (int64_t i = 0; i < numel; ++i) {
    EXPECT_GE(host[i], low);
    EXPECT_LT(host[i], high);
  }
}

// Empty out tensor: numel == 0 should be a no-op success.
TEST_F(AOTITorchCudaRandTest, RandintEmptyTensor) {
  std::vector<int64_t> sizes = {0};
  Tensor* out = create_int64_tensor(sizes);
  ASSERT_NE(out, nullptr);

  AOTITorchError error = aoti_torch_cuda_randint_low_out(
      out, /*low=*/0, /*high=*/10, sizes.data(), 1);
  EXPECT_EQ(error, Error::Ok);
}

// Null `out` tensor must return InvalidArgument.
TEST_F(AOTITorchCudaRandTest, RandintNullOutFails) {
  std::vector<int64_t> sizes = {4};
  AOTITorchError error = aoti_torch_cuda_randint_low_out(
      /*out=*/nullptr, 0, 10, sizes.data(), 1);
  EXPECT_EQ(error, Error::InvalidArgument);
}

// `high <= low` must return InvalidArgument.
TEST_F(AOTITorchCudaRandTest, RandintInvalidRangeFails) {
  std::vector<int64_t> sizes = {4};
  Tensor* out = create_int64_tensor(sizes);
  ASSERT_NE(out, nullptr);

  // high == low
  EXPECT_EQ(
      aoti_torch_cuda_randint_low_out(
          out, /*low=*/3, /*high=*/3, sizes.data(), 1),
      Error::InvalidArgument);
  // high < low
  EXPECT_EQ(
      aoti_torch_cuda_randint_low_out(
          out, /*low=*/5, /*high=*/2, sizes.data(), 1),
      Error::InvalidArgument);
}

// Calling randint twice should produce different sequences via the on-device
// counter advance. With numel=1 (the typical Inductor seed-gen pattern) we
// run a few iterations to make collision extremely unlikely.
TEST_F(AOTITorchCudaRandTest, RandintAdvancesCounter) {
  std::vector<int64_t> sizes = {1};
  Tensor* out = create_int64_tensor(sizes);
  ASSERT_NE(out, nullptr);

  constexpr int kIters = 8;
  std::vector<int64_t> draws;
  draws.reserve(kIters);
  for (int i = 0; i < kIters; ++i) {
    AOTITorchError error = aoti_torch_cuda_randint_low_out(
        out,
        /*low=*/0,
        /*high=*/std::numeric_limits<int32_t>::max(),
        sizes.data(),
        1);
    ASSERT_EQ(error, Error::Ok);
    cudaDeviceSynchronize();
    auto host = copy_to_host<int64_t>(out, 1);
    draws.push_back(host[0]);
  }

  // Not all draws should be equal.
  bool any_diff = false;
  for (int i = 1; i < kIters; ++i) {
    if (draws[i] != draws[0]) {
      any_diff = true;
      break;
    }
  }
  EXPECT_TRUE(any_diff) << "randint counter did not advance across calls";
}
