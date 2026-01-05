/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <gtest/gtest.h>
#include <vector>

#include <executorch/backends/aoti/common_shims_slim.h>
#include <executorch/backends/aoti/slim/c10/core/Device.h>
#include <executorch/backends/aoti/slim/c10/core/ScalarType.h>
#include <executorch/backends/aoti/slim/factory/Empty.h>
#include <executorch/runtime/core/error.h>
#include <executorch/runtime/platform/platform.h>

#ifdef CUDA_AVAILABLE
#include <cuda_runtime.h>
#endif

using namespace executorch::backends::aoti;
using executorch::runtime::Error;

namespace slim_c10 = executorch::backends::aoti::slim::c10;
namespace slim = executorch::backends::aoti::slim;

namespace {

#ifdef CUDA_AVAILABLE
bool isCudaAvailable() {
  int device_count = 0;
  cudaError_t err = cudaGetDeviceCount(&device_count);
  return (err == cudaSuccess && device_count > 0);
}
#endif

// Helper to calculate contiguous strides from sizes
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

// Test fixture for common_shims_slim tests
class CommonShimsSlimTest : public ::testing::Test {
 protected:
  void SetUp() override {
    et_pal_init();
  }

  void TearDown() override {
    // Cleanup tracked tensors
    for (Tensor* t : tensors_) {
      delete t;
    }
    tensors_.clear();
  }

  void trackTensor(Tensor* t) {
    if (t != nullptr) {
      tensors_.push_back(t);
    }
  }

  Tensor* createTestTensor(
      const std::vector<int64_t>& sizes,
      slim_c10::DeviceType device_type) {
    std::vector<int64_t> strides = calculateContiguousStrides(sizes);
    slim_c10::Device device(device_type, 0);
    Tensor* tensor = new Tensor(slim::empty_strided(
        slim::makeArrayRef(sizes),
        slim::makeArrayRef(strides),
        slim_c10::ScalarType::Float,
        device));
    trackTensor(tensor);
    return tensor;
  }

 private:
  std::vector<Tensor*> tensors_;
};

// ============================================================================
// Common test body implementations - parameterized by device type
// ============================================================================

void runGetDataPtrTest(slim_c10::DeviceType device_type) {
  std::vector<int64_t> sizes = {2, 3};
  std::vector<int64_t> strides = calculateContiguousStrides(sizes);
  slim_c10::Device device(device_type, 0);

  Tensor* tensor = new Tensor(slim::empty_strided(
      slim::makeArrayRef(sizes),
      slim::makeArrayRef(strides),
      slim_c10::ScalarType::Float,
      device));

  void* data_ptr = nullptr;
  AOTITorchError error = aoti_torch_get_data_ptr(tensor, &data_ptr);

  EXPECT_EQ(error, Error::Ok);
  EXPECT_NE(data_ptr, nullptr);

  // Verify the returned pointer matches tensor's data_ptr
  EXPECT_EQ(data_ptr, tensor->data_ptr());

  delete tensor;
}

void runGetSizesTest(slim_c10::DeviceType device_type) {
  std::vector<int64_t> sizes = {2, 3, 4};
  std::vector<int64_t> strides = calculateContiguousStrides(sizes);
  slim_c10::Device device(device_type, 0);

  Tensor* tensor = new Tensor(slim::empty_strided(
      slim::makeArrayRef(sizes),
      slim::makeArrayRef(strides),
      slim_c10::ScalarType::Float,
      device));

  int64_t* ret_sizes = nullptr;
  AOTITorchError error = aoti_torch_get_sizes(tensor, &ret_sizes);

  EXPECT_EQ(error, Error::Ok);
  EXPECT_NE(ret_sizes, nullptr);

  // Verify sizes match
  EXPECT_EQ(ret_sizes[0], 2);
  EXPECT_EQ(ret_sizes[1], 3);
  EXPECT_EQ(ret_sizes[2], 4);

  delete tensor;
}

void runGetStridesTest(slim_c10::DeviceType device_type) {
  std::vector<int64_t> sizes = {2, 3, 4};
  std::vector<int64_t> strides = calculateContiguousStrides(sizes);
  slim_c10::Device device(device_type, 0);

  Tensor* tensor = new Tensor(slim::empty_strided(
      slim::makeArrayRef(sizes),
      slim::makeArrayRef(strides),
      slim_c10::ScalarType::Float,
      device));

  int64_t* ret_strides = nullptr;
  AOTITorchError error = aoti_torch_get_strides(tensor, &ret_strides);

  EXPECT_EQ(error, Error::Ok);
  EXPECT_NE(ret_strides, nullptr);

  // Verify strides match: [12, 4, 1] for contiguous [2, 3, 4]
  EXPECT_EQ(ret_strides[0], 12);
  EXPECT_EQ(ret_strides[1], 4);
  EXPECT_EQ(ret_strides[2], 1);

  delete tensor;
}

void runGetDtypeTest(slim_c10::DeviceType device_type) {
  std::vector<int64_t> sizes = {2, 3};
  std::vector<int64_t> strides = calculateContiguousStrides(sizes);
  slim_c10::Device device(device_type, 0);

  // Test Float32
  {
    Tensor* tensor = new Tensor(slim::empty_strided(
        slim::makeArrayRef(sizes),
        slim::makeArrayRef(strides),
        slim_c10::ScalarType::Float,
        device));

    int32_t ret_dtype = -1;
    AOTITorchError error = aoti_torch_get_dtype(tensor, &ret_dtype);

    EXPECT_EQ(error, Error::Ok);
    EXPECT_EQ(ret_dtype, static_cast<int32_t>(slim_c10::ScalarType::Float));

    delete tensor;
  }

  // Test Int64
  {
    Tensor* tensor = new Tensor(slim::empty_strided(
        slim::makeArrayRef(sizes),
        slim::makeArrayRef(strides),
        slim_c10::ScalarType::Long,
        device));

    int32_t ret_dtype = -1;
    AOTITorchError error = aoti_torch_get_dtype(tensor, &ret_dtype);

    EXPECT_EQ(error, Error::Ok);
    EXPECT_EQ(ret_dtype, static_cast<int32_t>(slim_c10::ScalarType::Long));

    delete tensor;
  }

  // Test BFloat16
  {
    Tensor* tensor = new Tensor(slim::empty_strided(
        slim::makeArrayRef(sizes),
        slim::makeArrayRef(strides),
        slim_c10::ScalarType::BFloat16,
        device));

    int32_t ret_dtype = -1;
    AOTITorchError error = aoti_torch_get_dtype(tensor, &ret_dtype);

    EXPECT_EQ(error, Error::Ok);
    EXPECT_EQ(ret_dtype, static_cast<int32_t>(slim_c10::ScalarType::BFloat16));

    delete tensor;
  }
}

void runGetDimTest(slim_c10::DeviceType device_type) {
  slim_c10::Device device(device_type, 0);

  // Test 0D tensor (scalar)
  {
    std::vector<int64_t> sizes = {};
    std::vector<int64_t> strides = {};

    Tensor* tensor = new Tensor(slim::empty_strided(
        slim::makeArrayRef(sizes),
        slim::makeArrayRef(strides),
        slim_c10::ScalarType::Float,
        device));

    int64_t ret_dim = -1;
    AOTITorchError error = aoti_torch_get_dim(tensor, &ret_dim);

    EXPECT_EQ(error, Error::Ok);
    EXPECT_EQ(ret_dim, 0);

    delete tensor;
  }

  // Test 1D tensor
  {
    std::vector<int64_t> sizes = {5};
    std::vector<int64_t> strides = calculateContiguousStrides(sizes);

    Tensor* tensor = new Tensor(slim::empty_strided(
        slim::makeArrayRef(sizes),
        slim::makeArrayRef(strides),
        slim_c10::ScalarType::Float,
        device));

    int64_t ret_dim = -1;
    AOTITorchError error = aoti_torch_get_dim(tensor, &ret_dim);

    EXPECT_EQ(error, Error::Ok);
    EXPECT_EQ(ret_dim, 1);

    delete tensor;
  }

  // Test 3D tensor
  {
    std::vector<int64_t> sizes = {2, 3, 4};
    std::vector<int64_t> strides = calculateContiguousStrides(sizes);

    Tensor* tensor = new Tensor(slim::empty_strided(
        slim::makeArrayRef(sizes),
        slim::makeArrayRef(strides),
        slim_c10::ScalarType::Float,
        device));

    int64_t ret_dim = -1;
    AOTITorchError error = aoti_torch_get_dim(tensor, &ret_dim);

    EXPECT_EQ(error, Error::Ok);
    EXPECT_EQ(ret_dim, 3);

    delete tensor;
  }
}

// ============================================================================
// Storage & Device Property Tests
// ============================================================================

void runGetStorageOffsetTest(slim_c10::DeviceType device_type) {
  std::vector<int64_t> sizes = {2, 3};
  std::vector<int64_t> strides = calculateContiguousStrides(sizes);
  slim_c10::Device device(device_type, 0);

  Tensor* tensor = new Tensor(slim::empty_strided(
      slim::makeArrayRef(sizes),
      slim::makeArrayRef(strides),
      slim_c10::ScalarType::Float,
      device));

  int64_t ret_storage_offset = -1;
  AOTITorchError error =
      aoti_torch_get_storage_offset(tensor, &ret_storage_offset);

  EXPECT_EQ(error, Error::Ok);
  // Default storage offset for newly created tensor is 0
  EXPECT_EQ(ret_storage_offset, 0);

  delete tensor;
}

void runGetStorageSizeTest(slim_c10::DeviceType device_type) {
  std::vector<int64_t> sizes = {2, 3};
  std::vector<int64_t> strides = calculateContiguousStrides(sizes);
  slim_c10::Device device(device_type, 0);

  Tensor* tensor = new Tensor(slim::empty_strided(
      slim::makeArrayRef(sizes),
      slim::makeArrayRef(strides),
      slim_c10::ScalarType::Float,
      device));

  int64_t ret_size = -1;
  AOTITorchError error = aoti_torch_get_storage_size(tensor, &ret_size);

  EXPECT_EQ(error, Error::Ok);
  // 2 * 3 * sizeof(float) = 6 * 4 = 24 bytes
  EXPECT_EQ(ret_size, 24);

  delete tensor;
}

void runGetDeviceTypeTest(slim_c10::DeviceType device_type) {
  std::vector<int64_t> sizes = {2, 3};
  std::vector<int64_t> strides = calculateContiguousStrides(sizes);
  slim_c10::Device device(device_type, 0);

  Tensor* tensor = new Tensor(slim::empty_strided(
      slim::makeArrayRef(sizes),
      slim::makeArrayRef(strides),
      slim_c10::ScalarType::Float,
      device));

  int32_t ret_device_type = -1;
  AOTITorchError error = aoti_torch_get_device_type(tensor, &ret_device_type);

  EXPECT_EQ(error, Error::Ok);
  EXPECT_EQ(ret_device_type, static_cast<int32_t>(device_type));

  delete tensor;
}

void runGetDeviceIndexTest(slim_c10::DeviceType device_type) {
  std::vector<int64_t> sizes = {2, 3};
  std::vector<int64_t> strides = calculateContiguousStrides(sizes);
  slim_c10::Device device(device_type, 0);

  Tensor* tensor = new Tensor(slim::empty_strided(
      slim::makeArrayRef(sizes),
      slim::makeArrayRef(strides),
      slim_c10::ScalarType::Float,
      device));

  int32_t ret_device_index = -1;
  AOTITorchError error = aoti_torch_get_device_index(tensor, &ret_device_index);

  EXPECT_EQ(error, Error::Ok);
  EXPECT_EQ(ret_device_index, 0);

  delete tensor;
}

// ============================================================================
// CPU Tests
// ============================================================================

TEST_F(CommonShimsSlimTest, GetDataPtr_CPU) {
  runGetDataPtrTest(slim_c10::DeviceType::CPU);
}

TEST_F(CommonShimsSlimTest, GetSizes_CPU) {
  runGetSizesTest(slim_c10::DeviceType::CPU);
}

TEST_F(CommonShimsSlimTest, GetStrides_CPU) {
  runGetStridesTest(slim_c10::DeviceType::CPU);
}

TEST_F(CommonShimsSlimTest, GetDtype_CPU) {
  runGetDtypeTest(slim_c10::DeviceType::CPU);
}

TEST_F(CommonShimsSlimTest, GetDim_CPU) {
  runGetDimTest(slim_c10::DeviceType::CPU);
}

TEST_F(CommonShimsSlimTest, GetStorageOffset_CPU) {
  runGetStorageOffsetTest(slim_c10::DeviceType::CPU);
}

TEST_F(CommonShimsSlimTest, GetStorageSize_CPU) {
  runGetStorageSizeTest(slim_c10::DeviceType::CPU);
}

TEST_F(CommonShimsSlimTest, GetDeviceType_CPU) {
  runGetDeviceTypeTest(slim_c10::DeviceType::CPU);
}

TEST_F(CommonShimsSlimTest, GetDeviceIndex_CPU) {
  runGetDeviceIndexTest(slim_c10::DeviceType::CPU);
}

// ============================================================================
// CUDA Tests
// ============================================================================

#ifdef CUDA_AVAILABLE
TEST_F(CommonShimsSlimTest, GetDataPtr_CUDA) {
  if (!isCudaAvailable()) {
    GTEST_SKIP() << "CUDA not available";
  }
  runGetDataPtrTest(slim_c10::DeviceType::CUDA);
}

TEST_F(CommonShimsSlimTest, GetSizes_CUDA) {
  if (!isCudaAvailable()) {
    GTEST_SKIP() << "CUDA not available";
  }
  runGetSizesTest(slim_c10::DeviceType::CUDA);
}

TEST_F(CommonShimsSlimTest, GetStrides_CUDA) {
  if (!isCudaAvailable()) {
    GTEST_SKIP() << "CUDA not available";
  }
  runGetStridesTest(slim_c10::DeviceType::CUDA);
}

TEST_F(CommonShimsSlimTest, GetDtype_CUDA) {
  if (!isCudaAvailable()) {
    GTEST_SKIP() << "CUDA not available";
  }
  runGetDtypeTest(slim_c10::DeviceType::CUDA);
}

TEST_F(CommonShimsSlimTest, GetDim_CUDA) {
  if (!isCudaAvailable()) {
    GTEST_SKIP() << "CUDA not available";
  }
  runGetDimTest(slim_c10::DeviceType::CUDA);
}

TEST_F(CommonShimsSlimTest, GetStorageOffset_CUDA) {
  if (!isCudaAvailable()) {
    GTEST_SKIP() << "CUDA not available";
  }
  runGetStorageOffsetTest(slim_c10::DeviceType::CUDA);
}

TEST_F(CommonShimsSlimTest, GetStorageSize_CUDA) {
  if (!isCudaAvailable()) {
    GTEST_SKIP() << "CUDA not available";
  }
  runGetStorageSizeTest(slim_c10::DeviceType::CUDA);
}

TEST_F(CommonShimsSlimTest, GetDeviceType_CUDA) {
  if (!isCudaAvailable()) {
    GTEST_SKIP() << "CUDA not available";
  }
  runGetDeviceTypeTest(slim_c10::DeviceType::CUDA);
}

TEST_F(CommonShimsSlimTest, GetDeviceIndex_CUDA) {
  if (!isCudaAvailable()) {
    GTEST_SKIP() << "CUDA not available";
  }
  runGetDeviceIndexTest(slim_c10::DeviceType::CUDA);
}
#endif

// ============================================================================
// Error Cases
// ============================================================================

TEST_F(CommonShimsSlimTest, NullTensorArgument) {
  void* data_ptr = nullptr;
  int64_t* sizes = nullptr;
  int64_t* strides = nullptr;
  int32_t dtype = -1;
  int64_t dim = -1;

  EXPECT_EQ(
      aoti_torch_get_data_ptr(nullptr, &data_ptr), Error::InvalidArgument);
  EXPECT_EQ(aoti_torch_get_sizes(nullptr, &sizes), Error::InvalidArgument);
  EXPECT_EQ(aoti_torch_get_strides(nullptr, &strides), Error::InvalidArgument);
  EXPECT_EQ(aoti_torch_get_dtype(nullptr, &dtype), Error::InvalidArgument);
  EXPECT_EQ(aoti_torch_get_dim(nullptr, &dim), Error::InvalidArgument);
}

TEST_F(CommonShimsSlimTest, NullReturnPointer) {
  Tensor* tensor = createTestTensor({2, 3}, slim_c10::DeviceType::CPU);

  EXPECT_EQ(aoti_torch_get_data_ptr(tensor, nullptr), Error::InvalidArgument);
  EXPECT_EQ(aoti_torch_get_sizes(tensor, nullptr), Error::InvalidArgument);
  EXPECT_EQ(aoti_torch_get_strides(tensor, nullptr), Error::InvalidArgument);
  EXPECT_EQ(aoti_torch_get_dtype(tensor, nullptr), Error::InvalidArgument);
  EXPECT_EQ(aoti_torch_get_dim(tensor, nullptr), Error::InvalidArgument);
}

// ============================================================================
// Edge Cases
// ============================================================================

TEST_F(CommonShimsSlimTest, ScalarTensor) {
  std::vector<int64_t> sizes = {};
  std::vector<int64_t> strides = {};
  slim_c10::Device device(slim_c10::DeviceType::CPU, 0);

  Tensor* tensor = new Tensor(slim::empty_strided(
      slim::makeArrayRef(sizes),
      slim::makeArrayRef(strides),
      slim_c10::ScalarType::Float,
      device));
  trackTensor(tensor);

  // Get sizes and strides for 0D tensor
  int64_t* ret_sizes = nullptr;
  int64_t* ret_strides = nullptr;
  int64_t ret_dim = -1;

  EXPECT_EQ(aoti_torch_get_sizes(tensor, &ret_sizes), Error::Ok);
  EXPECT_NE(ret_sizes, nullptr);

  EXPECT_EQ(aoti_torch_get_strides(tensor, &ret_strides), Error::Ok);
  EXPECT_NE(ret_strides, nullptr);

  EXPECT_EQ(aoti_torch_get_dim(tensor, &ret_dim), Error::Ok);
  EXPECT_EQ(ret_dim, 0);
}

TEST_F(CommonShimsSlimTest, LargeTensor) {
  std::vector<int64_t> sizes = {100, 200, 300};
  std::vector<int64_t> strides = calculateContiguousStrides(sizes);
  slim_c10::Device device(slim_c10::DeviceType::CPU, 0);

  Tensor* tensor = new Tensor(slim::empty_strided(
      slim::makeArrayRef(sizes),
      slim::makeArrayRef(strides),
      slim_c10::ScalarType::Float,
      device));
  trackTensor(tensor);

  int64_t* ret_sizes = nullptr;
  int64_t* ret_strides = nullptr;

  EXPECT_EQ(aoti_torch_get_sizes(tensor, &ret_sizes), Error::Ok);
  EXPECT_EQ(ret_sizes[0], 100);
  EXPECT_EQ(ret_sizes[1], 200);
  EXPECT_EQ(ret_sizes[2], 300);

  EXPECT_EQ(aoti_torch_get_strides(tensor, &ret_strides), Error::Ok);
  EXPECT_EQ(ret_strides[0], 60000); // 200 * 300
  EXPECT_EQ(ret_strides[1], 300); // 300
  EXPECT_EQ(ret_strides[2], 1);
}

TEST_F(CommonShimsSlimTest, ConsistentPointerReturn) {
  Tensor* tensor = createTestTensor({2, 3, 4}, slim_c10::DeviceType::CPU);

  // Multiple calls should return the same pointer (for SlimTensor)
  int64_t* sizes_ptr1 = nullptr;
  int64_t* sizes_ptr2 = nullptr;

  EXPECT_EQ(aoti_torch_get_sizes(tensor, &sizes_ptr1), Error::Ok);
  EXPECT_EQ(aoti_torch_get_sizes(tensor, &sizes_ptr2), Error::Ok);
  EXPECT_EQ(sizes_ptr1, sizes_ptr2);

  int64_t* strides_ptr1 = nullptr;
  int64_t* strides_ptr2 = nullptr;

  EXPECT_EQ(aoti_torch_get_strides(tensor, &strides_ptr1), Error::Ok);
  EXPECT_EQ(aoti_torch_get_strides(tensor, &strides_ptr2), Error::Ok);
  EXPECT_EQ(strides_ptr1, strides_ptr2);
}

// ============================================================================
// DType Constants Tests
// ============================================================================

TEST_F(CommonShimsSlimTest, DTypeConstants) {
  // Verify dtype constants match expected PyTorch ScalarType values
  EXPECT_EQ(aoti_torch_dtype_float32(), 6); // ScalarType::Float
  EXPECT_EQ(aoti_torch_dtype_bfloat16(), 15); // ScalarType::BFloat16
  EXPECT_EQ(aoti_torch_dtype_int64(), 4); // ScalarType::Long
  EXPECT_EQ(aoti_torch_dtype_int32(), 3); // ScalarType::Int
  EXPECT_EQ(aoti_torch_dtype_int16(), 2); // ScalarType::Short
  EXPECT_EQ(aoti_torch_dtype_int8(), 1); // ScalarType::Char
  EXPECT_EQ(aoti_torch_dtype_bool(), 11); // ScalarType::Bool
}

// ============================================================================
// Device Type Constants Tests
// ============================================================================

TEST_F(CommonShimsSlimTest, DeviceTypeConstants) {
  EXPECT_EQ(aoti_torch_device_type_cpu(), 0); // DeviceType::CPU
  EXPECT_EQ(aoti_torch_device_type_cuda(), 1); // DeviceType::CUDA
}

// ============================================================================
// Grad Mode Tests
// ============================================================================

TEST_F(CommonShimsSlimTest, GradModeIsEnabled) {
  // ExecuTorch doesn't support autograd, so should always return false
  EXPECT_EQ(aoti_torch_grad_mode_is_enabled(), false);
}

TEST_F(CommonShimsSlimTest, GradModeSetEnabled) {
  // Setting to false should succeed
  EXPECT_EQ(aoti_torch_grad_mode_set_enabled(false), Error::Ok);

  // Setting to true should fail (not supported in ExecuTorch)
  EXPECT_EQ(aoti_torch_grad_mode_set_enabled(true), Error::NotSupported);
}
