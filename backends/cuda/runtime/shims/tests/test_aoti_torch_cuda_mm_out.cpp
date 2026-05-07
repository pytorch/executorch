/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <gtest/gtest.h>
#include <cmath>
#include <vector>

#include <executorch/backends/aoti/slim/c10/core/DeviceType.h>
#include <executorch/backends/aoti/slim/c10/core/ScalarType.h>
#include <executorch/backends/cuda/runtime/shims/memory.h>
#include <executorch/backends/cuda/runtime/shims/mm.h>
#include <executorch/runtime/core/error.h>
#include <executorch/runtime/platform/platform.h>

using executorch::backends::cuda::aoti_torch_cuda_mm_out;
using executorch::backends::cuda::aoti_torch_delete_tensor_object;
using executorch::backends::cuda::aoti_torch_empty_strided;
using executorch::backends::cuda::AOTITorchError;
using executorch::runtime::Error;
namespace slim_c10 = executorch::backends::aoti::slim::c10;

using Tensor = executorch::backends::aoti::slim::SlimTensor;

// -- Dtype traits for templated tests ----------------------------------------

template <typename T>
struct DtypeTraits;

template <>
struct DtypeTraits<__nv_bfloat16> {
  static constexpr slim_c10::ScalarType scalar_type =
      slim_c10::ScalarType::BFloat16;
  static __nv_bfloat16 from_float(float v) {
    return __float2bfloat16(v);
  }
  static float to_float(__nv_bfloat16 v) {
    return __bfloat162float(v);
  }
};

template <>
struct DtypeTraits<__half> {
  static constexpr slim_c10::ScalarType scalar_type =
      slim_c10::ScalarType::Half;
  static __half from_float(float v) {
    return __float2half(v);
  }
  static float to_float(__half v) {
    return __half2float(v);
  }
};

template <>
struct DtypeTraits<float> {
  static constexpr slim_c10::ScalarType scalar_type =
      slim_c10::ScalarType::Float;
  static float from_float(float v) {
    return v;
  }
  static float to_float(float v) {
    return v;
  }
};

// -- Test fixture ------------------------------------------------------------

template <typename T>
class AOTITorchMmOutTypedTest : public ::testing::Test {
 protected:
  using Traits = DtypeTraits<T>;

  void SetUp() override {
    et_pal_init();
    int device_count = 0;
    if (cudaGetDeviceCount(&device_count) != cudaSuccess || device_count == 0) {
      GTEST_SKIP() << "CUDA not available";
    }
  }

  Tensor* createTensor(const std::vector<int64_t>& sizes) {
    Tensor* tensor = nullptr;
    AOTITorchError error = aoti_torch_empty_strided(
        sizes.size(),
        sizes.data(),
        nullptr,
        static_cast<int32_t>(Traits::scalar_type),
        static_cast<int32_t>(slim_c10::DeviceType::CUDA),
        0,
        &tensor);
    return (error == Error::Ok) ? tensor : nullptr;
  }

  // Small-integer reference: inputs are exactly representable in all dtypes,
  // so cuBLAS output must match the serial f32 reference exactly.
  void runExactTest(
      int64_t M,
      int64_t K,
      int64_t N,
      const std::vector<float>& h_A,
      const std::vector<float>& h_B) {
    Tensor* self = createTensor({M, K});
    ASSERT_NE(self, nullptr);
    Tensor* mat2 = createTensor({K, N});
    ASSERT_NE(mat2, nullptr);
    Tensor* out = createTensor({M, N});
    ASSERT_NE(out, nullptr);

    std::vector<T> d_A(M * K), d_B(K * N);
    for (int64_t i = 0; i < M * K; i++)
      d_A[i] = Traits::from_float(h_A[i]);
    for (int64_t i = 0; i < K * N; i++)
      d_B[i] = Traits::from_float(h_B[i]);

    cudaMemcpy(
        self->data_ptr(),
        d_A.data(),
        M * K * sizeof(T),
        cudaMemcpyHostToDevice);
    cudaMemcpy(
        mat2->data_ptr(),
        d_B.data(),
        K * N * sizeof(T),
        cudaMemcpyHostToDevice);

    AOTITorchError error = aoti_torch_cuda_mm_out(out, self, mat2);
    EXPECT_EQ(error, Error::Ok);
    cudaDeviceSynchronize();

    std::vector<T> h_out(M * N);
    cudaMemcpy(
        h_out.data(),
        out->data_ptr(),
        M * N * sizeof(T),
        cudaMemcpyDeviceToHost);

    // Serial f32 reference
    for (int64_t i = 0; i < M; i++) {
      for (int64_t j = 0; j < N; j++) {
        float expected = 0.0f;
        for (int64_t p = 0; p < K; p++) {
          expected += h_A[i * K + p] * h_B[p * N + j];
        }
        float actual = Traits::to_float(h_out[i * N + j]);
        EXPECT_EQ(actual, expected) << "Mismatch at [" << i << "," << j << "]";
      }
    }

    EXPECT_EQ(aoti_torch_delete_tensor_object(self), Error::Ok);
    EXPECT_EQ(aoti_torch_delete_tensor_object(mat2), Error::Ok);
    EXPECT_EQ(aoti_torch_delete_tensor_object(out), Error::Ok);
  }
};

using MmOutTestTypes = ::testing::Types<__nv_bfloat16, __half, float>;
TYPED_TEST_SUITE(AOTITorchMmOutTypedTest, MmOutTestTypes);

// -- Typed correctness tests (run for bf16, fp16, fp32) ----------------------
// Use small integers so results are exact in all dtypes.

TYPED_TEST(AOTITorchMmOutTypedTest, SmallSquare) {
  int64_t M = 4, K = 8, N = 6;
  std::vector<float> h_A(M * K), h_B(K * N);
  for (int64_t i = 0; i < M * K; i++)
    h_A[i] = static_cast<float>((i % 5) + 1);
  for (int64_t i = 0; i < K * N; i++)
    h_B[i] = static_cast<float>((i % 3) + 1);
  this->runExactTest(M, K, N, h_A, h_B);
}

TYPED_TEST(AOTITorchMmOutTypedTest, SingleRow) {
  int64_t M = 1, K = 16, N = 8;
  std::vector<float> h_A(M * K), h_B(K * N);
  for (int64_t i = 0; i < M * K; i++)
    h_A[i] = static_cast<float>((i % 4) + 1);
  for (int64_t i = 0; i < K * N; i++)
    h_B[i] = static_cast<float>((i % 3) + 1);
  this->runExactTest(M, K, N, h_A, h_B);
}

TYPED_TEST(AOTITorchMmOutTypedTest, AllOnes) {
  int64_t M = 1, K = 2048, N = 256;
  std::vector<float> h_A(M * K, 1.0f), h_B(K * N, 1.0f);
  this->runExactTest(M, K, N, h_A, h_B);
}

TYPED_TEST(AOTITorchMmOutTypedTest, Identity) {
  int64_t N = 32;
  std::vector<float> h_A(N * N, 0.0f), h_B(N * N, 0.0f);
  for (int64_t i = 0; i < N; i++) {
    h_A[i * N + i] = 1.0f;
    h_B[i * N + i] = static_cast<float>(i + 1);
  }
  this->runExactTest(N, N, N, h_A, h_B);
}

// -- Non-typed tests (contract validation) -----------------------------------

class AOTITorchMmOutTest : public ::testing::Test {
 protected:
  void SetUp() override {
    et_pal_init();
    int device_count = 0;
    if (cudaGetDeviceCount(&device_count) != cudaSuccess || device_count == 0) {
      GTEST_SKIP() << "CUDA not available";
    }
  }

  Tensor* createTensor(
      const std::vector<int64_t>& sizes,
      slim_c10::ScalarType dtype) {
    Tensor* tensor = nullptr;
    AOTITorchError error = aoti_torch_empty_strided(
        sizes.size(),
        sizes.data(),
        nullptr,
        static_cast<int32_t>(dtype),
        static_cast<int32_t>(slim_c10::DeviceType::CUDA),
        0,
        &tensor);
    return (error == Error::Ok) ? tensor : nullptr;
  }
};

TEST_F(AOTITorchMmOutTest, InnerDimensionMismatch) {
  Tensor* self = createTensor({4, 8}, slim_c10::ScalarType::Float);
  Tensor* mat2 = createTensor({6, 6}, slim_c10::ScalarType::Float);
  Tensor* out = createTensor({4, 6}, slim_c10::ScalarType::Float);
  EXPECT_EQ(aoti_torch_cuda_mm_out(out, self, mat2), Error::InvalidArgument);
  aoti_torch_delete_tensor_object(self);
  aoti_torch_delete_tensor_object(mat2);
  aoti_torch_delete_tensor_object(out);
}

TEST_F(AOTITorchMmOutTest, NullOut) {
  Tensor* self = createTensor({4, 8}, slim_c10::ScalarType::Float);
  Tensor* mat2 = createTensor({8, 6}, slim_c10::ScalarType::Float);
  EXPECT_EQ(
      aoti_torch_cuda_mm_out(nullptr, self, mat2), Error::InvalidArgument);
  aoti_torch_delete_tensor_object(self);
  aoti_torch_delete_tensor_object(mat2);
}

TEST_F(AOTITorchMmOutTest, NullSelf) {
  Tensor* mat2 = createTensor({8, 6}, slim_c10::ScalarType::Float);
  Tensor* out = createTensor({4, 6}, slim_c10::ScalarType::Float);
  EXPECT_EQ(aoti_torch_cuda_mm_out(out, nullptr, mat2), Error::InvalidArgument);
  aoti_torch_delete_tensor_object(mat2);
  aoti_torch_delete_tensor_object(out);
}

TEST_F(AOTITorchMmOutTest, NullMat2) {
  Tensor* self = createTensor({4, 8}, slim_c10::ScalarType::Float);
  Tensor* out = createTensor({4, 6}, slim_c10::ScalarType::Float);
  EXPECT_EQ(aoti_torch_cuda_mm_out(out, self, nullptr), Error::InvalidArgument);
  aoti_torch_delete_tensor_object(self);
  aoti_torch_delete_tensor_object(out);
}

TEST_F(AOTITorchMmOutTest, DtypeMismatch) {
  Tensor* self = createTensor({4, 8}, slim_c10::ScalarType::Float);
  Tensor* mat2 = createTensor({8, 6}, slim_c10::ScalarType::BFloat16);
  Tensor* out = createTensor({4, 6}, slim_c10::ScalarType::Float);
  EXPECT_EQ(aoti_torch_cuda_mm_out(out, self, mat2), Error::InvalidArgument);
  aoti_torch_delete_tensor_object(self);
  aoti_torch_delete_tensor_object(mat2);
  aoti_torch_delete_tensor_object(out);
}

TEST_F(AOTITorchMmOutTest, NonContiguousRejected) {
  // Create a [8, 8] tensor and slice rows to get non-contiguous [4, 8]
  int64_t big_sizes[] = {8, 8};
  int64_t big_strides[] = {8, 1};
  Tensor* big = nullptr;
  aoti_torch_empty_strided(
      2,
      big_sizes,
      big_strides,
      static_cast<int32_t>(slim_c10::ScalarType::Float),
      static_cast<int32_t>(slim_c10::DeviceType::CUDA),
      0,
      &big);
  ASSERT_NE(big, nullptr);

  // Create a non-contiguous view by using stride(0)=16 on a [4, 8] shape
  int64_t nc_sizes[] = {4, 8};
  int64_t nc_strides[] = {16, 1}; // stride(0) > size(1), non-contiguous
  Tensor* nc = nullptr;
  aoti_torch_empty_strided(
      2,
      nc_sizes,
      nc_strides,
      static_cast<int32_t>(slim_c10::ScalarType::Float),
      static_cast<int32_t>(slim_c10::DeviceType::CUDA),
      0,
      &nc);
  ASSERT_NE(nc, nullptr);

  Tensor* mat2 = createTensor({8, 6}, slim_c10::ScalarType::Float);
  Tensor* out = createTensor({4, 6}, slim_c10::ScalarType::Float);

  EXPECT_EQ(aoti_torch_cuda_mm_out(out, nc, mat2), Error::InvalidArgument);

  aoti_torch_delete_tensor_object(big);
  aoti_torch_delete_tensor_object(nc);
  aoti_torch_delete_tensor_object(mat2);
  aoti_torch_delete_tensor_object(out);
}
