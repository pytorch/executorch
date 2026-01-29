/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <cuda_runtime.h>
#include <gtest/gtest.h>
#include <vector>

#include <executorch/backends/aoti/slim/c10/core/Device.h>
#include <executorch/backends/aoti/slim/c10/core/ScalarType.h>
#include <executorch/backends/cuda/runtime/shims/memory_slim.h>
#include <executorch/runtime/core/error.h>
#include <executorch/runtime/platform/platform.h>

using namespace executorch::backends::cuda;
using executorch::runtime::Error;

namespace slim_c10 = executorch::backends::aoti::slim::c10;

namespace {

bool isCudaAvailable() {
  int device_count = 0;
  cudaError_t err = cudaGetDeviceCount(&device_count);
  return (err == cudaSuccess && device_count > 0);
}

std::vector<int64_t> calculateContiguousStrides(
    const std::vector<int64_t>& sizes) {
  std::vector<int64_t> strides(sizes.size());
  if (sizes.empty()) {
    return strides;
  }
  strides[sizes.size() - 1] = 1;
  for (int64_t i = static_cast<int64_t>(sizes.size()) - 2; i >= 0; i--) {
    strides[i] = strides[i + 1] * sizes[i + 1];
  }
  return strides;
}

} // namespace

class AOTITorchCopySlimTest : public ::testing::Test {
 protected:
  void SetUp() override {
    et_pal_init();
  }

  Tensor* createTestTensor(
      const std::vector<int64_t>& sizes,
      const std::vector<int64_t>& strides = {},
      int32_t dtype = static_cast<int32_t>(slim_c10::ScalarType::Float),
      int32_t device_type = static_cast<int32_t>(slim_c10::DeviceType::CPU),
      int32_t device_index = 0) {
    Tensor* tensor = nullptr;

    std::vector<int64_t> effective_strides = strides;
    if (strides.empty()) {
      effective_strides = calculateContiguousStrides(sizes);
    }

    AOTITorchError error = aoti_torch_empty_strided(
        sizes.size(),
        sizes.data(),
        effective_strides.data(),
        dtype,
        device_type,
        device_index,
        &tensor);

    return (error == Error::Ok) ? tensor : nullptr;
  }
};

// ============================================================================
// Basic Functionality Tests
// ============================================================================

TEST_F(AOTITorchCopySlimTest, BasicCopy_CPU) {
  std::vector<int64_t> sizes = {3, 4};
  Tensor* src = createTestTensor(
      sizes,
      {},
      static_cast<int32_t>(slim_c10::ScalarType::Float),
      static_cast<int32_t>(slim_c10::DeviceType::CPU),
      0);
  ASSERT_NE(src, nullptr);

  float* src_data = static_cast<float*>(src->data_ptr());
  for (int64_t i = 0; i < src->numel(); i++) {
    src_data[i] = static_cast<float>(i + 1);
  }

  Tensor* dst = createTestTensor(
      sizes,
      {},
      static_cast<int32_t>(slim_c10::ScalarType::Float),
      static_cast<int32_t>(slim_c10::DeviceType::CPU),
      0);
  ASSERT_NE(dst, nullptr);

  AOTITorchError error = aoti_torch_copy_(dst, src, 0);
  EXPECT_EQ(error, Error::Ok);

  float* dst_data = static_cast<float*>(dst->data_ptr());
  for (int64_t i = 0; i < dst->numel(); i++) {
    EXPECT_FLOAT_EQ(dst_data[i], static_cast<float>(i + 1));
  }

  EXPECT_EQ(aoti_torch_delete_tensor_object(src), Error::Ok);
  EXPECT_EQ(aoti_torch_delete_tensor_object(dst), Error::Ok);
}

TEST_F(AOTITorchCopySlimTest, NullSelf) {
  std::vector<int64_t> sizes = {2, 3};
  Tensor* src = createTestTensor(
      sizes,
      {},
      static_cast<int32_t>(slim_c10::ScalarType::Float),
      static_cast<int32_t>(slim_c10::DeviceType::CPU),
      0);
  ASSERT_NE(src, nullptr);

  AOTITorchError error = aoti_torch_copy_(nullptr, src, 0);
  EXPECT_EQ(error, Error::InvalidArgument);

  EXPECT_EQ(aoti_torch_delete_tensor_object(src), Error::Ok);
}

TEST_F(AOTITorchCopySlimTest, NullSrc) {
  std::vector<int64_t> sizes = {2, 3};
  Tensor* dst = createTestTensor(
      sizes,
      {},
      static_cast<int32_t>(slim_c10::ScalarType::Float),
      static_cast<int32_t>(slim_c10::DeviceType::CPU),
      0);
  ASSERT_NE(dst, nullptr);

  AOTITorchError error = aoti_torch_copy_(dst, nullptr, 0);
  EXPECT_EQ(error, Error::InvalidArgument);

  EXPECT_EQ(aoti_torch_delete_tensor_object(dst), Error::Ok);
}

// ============================================================================
// Different Dtype Tests
// ============================================================================

TEST_F(AOTITorchCopySlimTest, Int64Copy_CPU) {
  std::vector<int64_t> sizes = {2, 3};
  Tensor* src = createTestTensor(
      sizes,
      {},
      static_cast<int32_t>(slim_c10::ScalarType::Long),
      static_cast<int32_t>(slim_c10::DeviceType::CPU),
      0);
  ASSERT_NE(src, nullptr);

  int64_t* src_data = static_cast<int64_t*>(src->data_ptr());
  for (int64_t i = 0; i < src->numel(); i++) {
    src_data[i] = i * 100;
  }

  Tensor* dst = createTestTensor(
      sizes,
      {},
      static_cast<int32_t>(slim_c10::ScalarType::Long),
      static_cast<int32_t>(slim_c10::DeviceType::CPU),
      0);
  ASSERT_NE(dst, nullptr);

  AOTITorchError error = aoti_torch_copy_(dst, src, 0);
  EXPECT_EQ(error, Error::Ok);

  int64_t* dst_data = static_cast<int64_t*>(dst->data_ptr());
  for (int64_t i = 0; i < dst->numel(); i++) {
    EXPECT_EQ(dst_data[i], i * 100);
  }

  EXPECT_EQ(aoti_torch_delete_tensor_object(src), Error::Ok);
  EXPECT_EQ(aoti_torch_delete_tensor_object(dst), Error::Ok);
}

TEST_F(AOTITorchCopySlimTest, BoolCopy_CPU) {
  std::vector<int64_t> sizes = {4};
  Tensor* src = createTestTensor(
      sizes,
      {},
      static_cast<int32_t>(slim_c10::ScalarType::Bool),
      static_cast<int32_t>(slim_c10::DeviceType::CPU),
      0);
  ASSERT_NE(src, nullptr);

  bool* src_data = static_cast<bool*>(src->data_ptr());
  src_data[0] = true;
  src_data[1] = false;
  src_data[2] = true;
  src_data[3] = false;

  Tensor* dst = createTestTensor(
      sizes,
      {},
      static_cast<int32_t>(slim_c10::ScalarType::Bool),
      static_cast<int32_t>(slim_c10::DeviceType::CPU),
      0);
  ASSERT_NE(dst, nullptr);

  AOTITorchError error = aoti_torch_copy_(dst, src, 0);
  EXPECT_EQ(error, Error::Ok);

  bool* dst_data = static_cast<bool*>(dst->data_ptr());
  EXPECT_EQ(dst_data[0], true);
  EXPECT_EQ(dst_data[1], false);
  EXPECT_EQ(dst_data[2], true);
  EXPECT_EQ(dst_data[3], false);

  EXPECT_EQ(aoti_torch_delete_tensor_object(src), Error::Ok);
  EXPECT_EQ(aoti_torch_delete_tensor_object(dst), Error::Ok);
}

// ============================================================================
// Tensor Shape Tests
// ============================================================================

TEST_F(AOTITorchCopySlimTest, ScalarTensorCopy_CPU) {
  std::vector<int64_t> sizes = {};
  Tensor* src = createTestTensor(
      sizes,
      {},
      static_cast<int32_t>(slim_c10::ScalarType::Float),
      static_cast<int32_t>(slim_c10::DeviceType::CPU),
      0);
  ASSERT_NE(src, nullptr);
  EXPECT_EQ(src->dim(), 0);
  EXPECT_EQ(src->numel(), 1);

  float* src_data = static_cast<float*>(src->data_ptr());
  *src_data = 42.0f;

  Tensor* dst = createTestTensor(
      sizes,
      {},
      static_cast<int32_t>(slim_c10::ScalarType::Float),
      static_cast<int32_t>(slim_c10::DeviceType::CPU),
      0);
  ASSERT_NE(dst, nullptr);

  AOTITorchError error = aoti_torch_copy_(dst, src, 0);
  EXPECT_EQ(error, Error::Ok);

  float* dst_data = static_cast<float*>(dst->data_ptr());
  EXPECT_FLOAT_EQ(*dst_data, 42.0f);

  EXPECT_EQ(aoti_torch_delete_tensor_object(src), Error::Ok);
  EXPECT_EQ(aoti_torch_delete_tensor_object(dst), Error::Ok);
}

TEST_F(AOTITorchCopySlimTest, LargeTensorCopy_CPU) {
  std::vector<int64_t> sizes = {100, 100};
  Tensor* src = createTestTensor(
      sizes,
      {},
      static_cast<int32_t>(slim_c10::ScalarType::Float),
      static_cast<int32_t>(slim_c10::DeviceType::CPU),
      0);
  ASSERT_NE(src, nullptr);

  float* src_data = static_cast<float*>(src->data_ptr());
  for (int64_t i = 0; i < src->numel(); i++) {
    src_data[i] = static_cast<float>(i);
  }

  Tensor* dst = createTestTensor(
      sizes,
      {},
      static_cast<int32_t>(slim_c10::ScalarType::Float),
      static_cast<int32_t>(slim_c10::DeviceType::CPU),
      0);
  ASSERT_NE(dst, nullptr);

  AOTITorchError error = aoti_torch_copy_(dst, src, 0);
  EXPECT_EQ(error, Error::Ok);

  float* dst_data = static_cast<float*>(dst->data_ptr());
  for (int64_t i = 0; i < dst->numel(); i++) {
    EXPECT_FLOAT_EQ(dst_data[i], static_cast<float>(i));
  }

  EXPECT_EQ(aoti_torch_delete_tensor_object(src), Error::Ok);
  EXPECT_EQ(aoti_torch_delete_tensor_object(dst), Error::Ok);
}

// ============================================================================
// CUDA Tests
// ============================================================================

TEST_F(AOTITorchCopySlimTest, CudaToCuda) {
  if (!isCudaAvailable()) {
    GTEST_SKIP() << "CUDA not available";
  }

  std::vector<int64_t> sizes = {3, 4};

  std::vector<float> host_src_data(12);
  for (size_t i = 0; i < host_src_data.size(); i++) {
    host_src_data[i] = static_cast<float>(i + 1);
  }

  Tensor* src = createTestTensor(
      sizes,
      {},
      static_cast<int32_t>(slim_c10::ScalarType::Float),
      static_cast<int32_t>(slim_c10::DeviceType::CUDA),
      0);
  ASSERT_NE(src, nullptr);
  EXPECT_TRUE(src->is_cuda());

  cudaMemcpy(
      src->data_ptr(),
      host_src_data.data(),
      host_src_data.size() * sizeof(float),
      cudaMemcpyHostToDevice);

  Tensor* dst = createTestTensor(
      sizes,
      {},
      static_cast<int32_t>(slim_c10::ScalarType::Float),
      static_cast<int32_t>(slim_c10::DeviceType::CUDA),
      0);
  ASSERT_NE(dst, nullptr);
  EXPECT_TRUE(dst->is_cuda());

  AOTITorchError error = aoti_torch_copy_(dst, src, 0);
  EXPECT_EQ(error, Error::Ok);

  std::vector<float> host_dst_data(12);
  cudaMemcpy(
      host_dst_data.data(),
      dst->data_ptr(),
      host_dst_data.size() * sizeof(float),
      cudaMemcpyDeviceToHost);

  for (size_t i = 0; i < host_dst_data.size(); i++) {
    EXPECT_FLOAT_EQ(host_dst_data[i], static_cast<float>(i + 1));
  }

  EXPECT_EQ(aoti_torch_delete_tensor_object(src), Error::Ok);
  EXPECT_EQ(aoti_torch_delete_tensor_object(dst), Error::Ok);
}

TEST_F(AOTITorchCopySlimTest, CpuToCuda) {
  if (!isCudaAvailable()) {
    GTEST_SKIP() << "CUDA not available";
  }

  std::vector<int64_t> sizes = {2, 3};
  Tensor* src = createTestTensor(
      sizes,
      {},
      static_cast<int32_t>(slim_c10::ScalarType::Float),
      static_cast<int32_t>(slim_c10::DeviceType::CPU),
      0);
  ASSERT_NE(src, nullptr);
  EXPECT_TRUE(src->is_cpu());

  float* src_data = static_cast<float*>(src->data_ptr());
  for (int64_t i = 0; i < src->numel(); i++) {
    src_data[i] = static_cast<float>(i * 10);
  }

  Tensor* dst = createTestTensor(
      sizes,
      {},
      static_cast<int32_t>(slim_c10::ScalarType::Float),
      static_cast<int32_t>(slim_c10::DeviceType::CUDA),
      0);
  ASSERT_NE(dst, nullptr);
  EXPECT_TRUE(dst->is_cuda());

  AOTITorchError error = aoti_torch_copy_(dst, src, 0);
  EXPECT_EQ(error, Error::Ok);

  std::vector<float> host_dst_data(6);
  cudaMemcpy(
      host_dst_data.data(),
      dst->data_ptr(),
      host_dst_data.size() * sizeof(float),
      cudaMemcpyDeviceToHost);

  for (size_t i = 0; i < host_dst_data.size(); i++) {
    EXPECT_FLOAT_EQ(host_dst_data[i], static_cast<float>(i * 10));
  }

  EXPECT_EQ(aoti_torch_delete_tensor_object(src), Error::Ok);
  EXPECT_EQ(aoti_torch_delete_tensor_object(dst), Error::Ok);
}

TEST_F(AOTITorchCopySlimTest, CudaToCpu) {
  if (!isCudaAvailable()) {
    GTEST_SKIP() << "CUDA not available";
  }

  std::vector<int64_t> sizes = {2, 3};

  std::vector<float> host_src_data(6);
  for (size_t i = 0; i < host_src_data.size(); i++) {
    host_src_data[i] = static_cast<float>(i * 5);
  }

  Tensor* src = createTestTensor(
      sizes,
      {},
      static_cast<int32_t>(slim_c10::ScalarType::Float),
      static_cast<int32_t>(slim_c10::DeviceType::CUDA),
      0);
  ASSERT_NE(src, nullptr);

  cudaMemcpy(
      src->data_ptr(),
      host_src_data.data(),
      host_src_data.size() * sizeof(float),
      cudaMemcpyHostToDevice);

  Tensor* dst = createTestTensor(
      sizes,
      {},
      static_cast<int32_t>(slim_c10::ScalarType::Float),
      static_cast<int32_t>(slim_c10::DeviceType::CPU),
      0);
  ASSERT_NE(dst, nullptr);
  EXPECT_TRUE(dst->is_cpu());

  AOTITorchError error = aoti_torch_copy_(dst, src, 0);
  EXPECT_EQ(error, Error::Ok);

  float* dst_data = static_cast<float*>(dst->data_ptr());
  for (int64_t i = 0; i < dst->numel(); i++) {
    EXPECT_FLOAT_EQ(dst_data[i], static_cast<float>(i * 5));
  }

  EXPECT_EQ(aoti_torch_delete_tensor_object(src), Error::Ok);
  EXPECT_EQ(aoti_torch_delete_tensor_object(dst), Error::Ok);
}

// ============================================================================
// Non-blocking Tests
// ============================================================================

TEST_F(AOTITorchCopySlimTest, NonBlockingFlag_CPU) {
  std::vector<int64_t> sizes = {2, 3};
  Tensor* src = createTestTensor(
      sizes,
      {},
      static_cast<int32_t>(slim_c10::ScalarType::Float),
      static_cast<int32_t>(slim_c10::DeviceType::CPU),
      0);
  ASSERT_NE(src, nullptr);

  float* src_data = static_cast<float*>(src->data_ptr());
  for (int64_t i = 0; i < src->numel(); i++) {
    src_data[i] = static_cast<float>(i);
  }

  Tensor* dst = createTestTensor(
      sizes,
      {},
      static_cast<int32_t>(slim_c10::ScalarType::Float),
      static_cast<int32_t>(slim_c10::DeviceType::CPU),
      0);
  ASSERT_NE(dst, nullptr);

  AOTITorchError error = aoti_torch_copy_(dst, src, 1);
  EXPECT_EQ(error, Error::Ok);

  float* dst_data = static_cast<float*>(dst->data_ptr());
  for (int64_t i = 0; i < dst->numel(); i++) {
    EXPECT_FLOAT_EQ(dst_data[i], static_cast<float>(i));
  }

  EXPECT_EQ(aoti_torch_delete_tensor_object(src), Error::Ok);
  EXPECT_EQ(aoti_torch_delete_tensor_object(dst), Error::Ok);
}
