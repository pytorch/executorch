/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <cuda_runtime.h>
#include <executorch/backends/aoti/common_shims_slim.h>
#include <executorch/backends/cuda/runtime/shims/cuda_guard.h>
#include <executorch/runtime/core/error.h>
#include <executorch/runtime/platform/platform.h>
#include <gtest/gtest.h>

using namespace executorch::backends::aoti;
using namespace executorch::backends::cuda;
using namespace executorch::runtime;

// TODO(gasoonjia): Multiple device tests were not included due to test
// environment limitations. Will be added in the future.
class AOTITorchCUDAGuardTest : public ::testing::Test {
 protected:
  void SetUp() override {
    et_pal_init();

    int device_count = 0;
    cudaError_t err = cudaGetDeviceCount(&device_count);
    if (err != cudaSuccess || device_count == 0) {
      GTEST_SKIP() << "CUDA not available, skipping CUDA tests";
    }

    ASSERT_EQ(cudaGetDevice(&original_device_), cudaSuccess);
  }

  void TearDown() override {
    if (cudaGetDeviceCount(&original_device_) == cudaSuccess) {
      ASSERT_EQ(cudaGetDevice(&original_device_), cudaSuccess);
    }
  }

  int original_device_ = 0;
};

TEST_F(AOTITorchCUDAGuardTest, CreateAndDeleteCUDAGuard) {
  CUDAGuardHandle guard = nullptr;
  AOTITorchError error = aoti_torch_create_cuda_guard(0, &guard);

  EXPECT_EQ(error, Error::Ok);
  ASSERT_NE(guard, nullptr);

  int current_device = -1;
  ASSERT_EQ(cudaGetDevice(&current_device), cudaSuccess);
  EXPECT_EQ(current_device, 0);

  error = aoti_torch_delete_cuda_guard(guard);
  EXPECT_EQ(error, Error::Ok);
}

TEST_F(AOTITorchCUDAGuardTest, CreateCUDAGuardNullReturnPointer) {
  AOTITorchError error = aoti_torch_create_cuda_guard(0, nullptr);
  EXPECT_EQ(error, Error::InvalidArgument);
}

TEST_F(AOTITorchCUDAGuardTest, DeleteCUDAGuardNullHandle) {
  AOTITorchError error = aoti_torch_delete_cuda_guard(nullptr);
  EXPECT_EQ(error, Error::InvalidArgument);
}

TEST_F(AOTITorchCUDAGuardTest, CUDAGuardSetIndexNullHandle) {
  AOTITorchError error = aoti_torch_cuda_guard_set_index(nullptr, 0);
  EXPECT_EQ(error, Error::InvalidArgument);
}

TEST_F(AOTITorchCUDAGuardTest, CUDAGuardSetIndexInvalidDevice) {
  CUDAGuardHandle guard = nullptr;
  AOTITorchError error = aoti_torch_create_cuda_guard(0, &guard);
  EXPECT_EQ(error, Error::Ok);
  ASSERT_NE(guard, nullptr);

  error = aoti_torch_cuda_guard_set_index(guard, 999);
  EXPECT_NE(error, Error::Ok);

  error = aoti_torch_delete_cuda_guard(guard);
  EXPECT_EQ(error, Error::Ok);
}

TEST_F(AOTITorchCUDAGuardTest, CreateAndDeleteCUDAStreamGuard) {
  cudaStream_t stream;
  ASSERT_EQ(cudaStreamCreate(&stream), cudaSuccess);

  CUDAStreamGuardHandle guard = nullptr;
  AOTITorchError error = aoti_torch_create_cuda_stream_guard(stream, 0, &guard);

  EXPECT_EQ(error, Error::Ok);
  ASSERT_NE(guard, nullptr);

  error = aoti_torch_delete_cuda_stream_guard(guard);
  EXPECT_EQ(error, Error::Ok);

  ASSERT_EQ(cudaStreamDestroy(stream), cudaSuccess);
}

TEST_F(AOTITorchCUDAGuardTest, CreateCUDAStreamGuardNullReturnPointer) {
  cudaStream_t stream;
  ASSERT_EQ(cudaStreamCreate(&stream), cudaSuccess);

  AOTITorchError error =
      aoti_torch_create_cuda_stream_guard(stream, 0, nullptr);
  EXPECT_EQ(error, Error::InvalidArgument);

  ASSERT_EQ(cudaStreamDestroy(stream), cudaSuccess);
}

TEST_F(AOTITorchCUDAGuardTest, CreateCUDAStreamGuardNullStream) {
  CUDAStreamGuardHandle guard = nullptr;
  AOTITorchError error =
      aoti_torch_create_cuda_stream_guard(nullptr, 0, &guard);
  EXPECT_EQ(error, Error::InvalidArgument);
}

TEST_F(AOTITorchCUDAGuardTest, DeleteCUDAStreamGuardNullHandle) {
  AOTITorchError error = aoti_torch_delete_cuda_stream_guard(nullptr);
  EXPECT_EQ(error, Error::InvalidArgument);
}

TEST_F(AOTITorchCUDAGuardTest, GetCurrentCUDAStream) {
  void* ret_stream = nullptr;
  AOTITorchError error = aoti_torch_get_current_cuda_stream(0, &ret_stream);

  EXPECT_EQ(error, Error::Ok);
  EXPECT_NE(ret_stream, nullptr);
}

TEST_F(AOTITorchCUDAGuardTest, GetCurrentCUDAStreamNullReturnPointer) {
  AOTITorchError error = aoti_torch_get_current_cuda_stream(0, nullptr);
  EXPECT_EQ(error, Error::InvalidArgument);
}

TEST_F(AOTITorchCUDAGuardTest, StreamGuardWithSameDevice) {
  ASSERT_EQ(cudaSetDevice(0), cudaSuccess);

  cudaStream_t stream1, stream2;
  ASSERT_EQ(cudaStreamCreate(&stream1), cudaSuccess);
  ASSERT_EQ(cudaStreamCreate(&stream2), cudaSuccess);

  CUDAStreamGuardHandle guard1 = nullptr;
  AOTITorchError error =
      aoti_torch_create_cuda_stream_guard(stream1, 0, &guard1);
  EXPECT_EQ(error, Error::Ok);

  void* ret_stream = nullptr;
  error = aoti_torch_get_current_cuda_stream(0, &ret_stream);
  EXPECT_EQ(error, Error::Ok);
  EXPECT_EQ(static_cast<cudaStream_t>(ret_stream), stream1);

  CUDAStreamGuardHandle guard2 = nullptr;
  error = aoti_torch_create_cuda_stream_guard(stream2, 0, &guard2);
  EXPECT_EQ(error, Error::Ok);

  ret_stream = nullptr;
  error = aoti_torch_get_current_cuda_stream(0, &ret_stream);
  EXPECT_EQ(error, Error::Ok);
  EXPECT_EQ(static_cast<cudaStream_t>(ret_stream), stream2);

  error = aoti_torch_delete_cuda_stream_guard(guard2);
  EXPECT_EQ(error, Error::Ok);

  ret_stream = nullptr;
  error = aoti_torch_get_current_cuda_stream(0, &ret_stream);
  EXPECT_EQ(error, Error::Ok);
  EXPECT_EQ(static_cast<cudaStream_t>(ret_stream), stream1);

  error = aoti_torch_delete_cuda_stream_guard(guard1);
  EXPECT_EQ(error, Error::Ok);

  ASSERT_EQ(cudaStreamDestroy(stream1), cudaSuccess);
  ASSERT_EQ(cudaStreamDestroy(stream2), cudaSuccess);
}

TEST_F(AOTITorchCUDAGuardTest, GetCurrentStreamAfterSetStream) {
  cudaStream_t new_stream;
  ASSERT_EQ(cudaStreamCreate(&new_stream), cudaSuccess);

  CUDAStreamGuardHandle guard = nullptr;
  AOTITorchError error =
      aoti_torch_create_cuda_stream_guard(new_stream, 0, &guard);
  EXPECT_EQ(error, Error::Ok);

  void* ret_stream = nullptr;
  error = aoti_torch_get_current_cuda_stream(0, &ret_stream);
  EXPECT_EQ(error, Error::Ok);
  EXPECT_EQ(static_cast<cudaStream_t>(ret_stream), new_stream);

  error = aoti_torch_delete_cuda_stream_guard(guard);
  EXPECT_EQ(error, Error::Ok);

  ASSERT_EQ(cudaStreamDestroy(new_stream), cudaSuccess);
}
