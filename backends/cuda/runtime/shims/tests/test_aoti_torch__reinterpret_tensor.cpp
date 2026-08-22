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

#include <executorch/backends/aoti/common_shims_slim.h>
#include <executorch/backends/aoti/slim/c10/core/Device.h>
#include <executorch/backends/aoti/slim/c10/core/ScalarType.h>
#include <executorch/backends/cuda/runtime/shims/memory.h>
#include <executorch/runtime/core/error.h>
#include <executorch/runtime/platform/platform.h>

using namespace executorch::backends::cuda;
using namespace executorch::backends::aoti;
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

class AOTITorchReinterpretTensorSlimTest : public ::testing::Test {
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

TEST_F(AOTITorchReinterpretTensorSlimTest, BasicView_CPU) {
  std::vector<int64_t> sizes = {2, 3, 4};
  Tensor* orig_tensor = createTestTensor(
      sizes,
      {},
      static_cast<int32_t>(slim_c10::ScalarType::Float),
      static_cast<int32_t>(slim_c10::DeviceType::CPU),
      0);
  ASSERT_NE(orig_tensor, nullptr);

  std::vector<int64_t> new_sizes = {6, 4};
  std::vector<int64_t> new_strides = {4, 1};
  int64_t storage_offset = 0;

  Tensor* view_tensor = nullptr;
  AOTITorchError error = aoti_torch__reinterpret_tensor(
      orig_tensor,
      new_sizes.size(),
      new_sizes.data(),
      new_strides.data(),
      storage_offset,
      &view_tensor);

  EXPECT_EQ(error, Error::Ok);
  ASSERT_NE(view_tensor, nullptr);

  EXPECT_EQ(view_tensor->dim(), 2);
  EXPECT_EQ(view_tensor->size(0), 6);
  EXPECT_EQ(view_tensor->size(1), 4);
  EXPECT_EQ(view_tensor->stride(0), 4);
  EXPECT_EQ(view_tensor->stride(1), 1);

  EXPECT_EQ(view_tensor->data_ptr(), orig_tensor->data_ptr());

  EXPECT_EQ(aoti_torch_delete_tensor_object(orig_tensor), Error::Ok);
  EXPECT_EQ(aoti_torch_delete_tensor_object(view_tensor), Error::Ok);
}

TEST_F(AOTITorchReinterpretTensorSlimTest, NullSelf) {
  std::vector<int64_t> sizes = {2, 3};
  std::vector<int64_t> strides = {3, 1};

  Tensor* view_tensor = nullptr;
  AOTITorchError error = aoti_torch__reinterpret_tensor(
      nullptr, sizes.size(), sizes.data(), strides.data(), 0, &view_tensor);

  EXPECT_EQ(error, Error::InvalidArgument);
}

TEST_F(AOTITorchReinterpretTensorSlimTest, NullReturnPointer) {
  std::vector<int64_t> sizes = {2, 3};
  Tensor* orig_tensor = createTestTensor(
      sizes,
      {},
      static_cast<int32_t>(slim_c10::ScalarType::Float),
      static_cast<int32_t>(slim_c10::DeviceType::CPU),
      0);
  ASSERT_NE(orig_tensor, nullptr);

  std::vector<int64_t> new_sizes = {6};
  std::vector<int64_t> new_strides = {1};

  AOTITorchError error = aoti_torch__reinterpret_tensor(
      orig_tensor,
      new_sizes.size(),
      new_sizes.data(),
      new_strides.data(),
      0,
      nullptr);

  EXPECT_EQ(error, Error::InvalidArgument);

  EXPECT_EQ(aoti_torch_delete_tensor_object(orig_tensor), Error::Ok);
}

TEST_F(AOTITorchReinterpretTensorSlimTest, NegativeNdim) {
  std::vector<int64_t> sizes = {2, 3};
  Tensor* orig_tensor = createTestTensor(
      sizes,
      {},
      static_cast<int32_t>(slim_c10::ScalarType::Float),
      static_cast<int32_t>(slim_c10::DeviceType::CPU),
      0);
  ASSERT_NE(orig_tensor, nullptr);

  std::vector<int64_t> new_sizes = {6};
  std::vector<int64_t> new_strides = {1};

  Tensor* view_tensor = nullptr;
  AOTITorchError error = aoti_torch__reinterpret_tensor(
      orig_tensor, -1, new_sizes.data(), new_strides.data(), 0, &view_tensor);

  EXPECT_EQ(error, Error::InvalidArgument);

  EXPECT_EQ(aoti_torch_delete_tensor_object(orig_tensor), Error::Ok);
}

// ============================================================================
// Storage Offset Tests
// ============================================================================

TEST_F(AOTITorchReinterpretTensorSlimTest, WithStorageOffset_CPU) {
  std::vector<int64_t> sizes = {4, 4};
  Tensor* orig_tensor = createTestTensor(
      sizes,
      {},
      static_cast<int32_t>(slim_c10::ScalarType::Float),
      static_cast<int32_t>(slim_c10::DeviceType::CPU),
      0);
  ASSERT_NE(orig_tensor, nullptr);

  std::vector<int64_t> new_sizes = {2, 4};
  std::vector<int64_t> new_strides = {4, 1};
  int64_t storage_offset = 4; // Skip first row

  Tensor* view_tensor = nullptr;
  AOTITorchError error = aoti_torch__reinterpret_tensor(
      orig_tensor,
      new_sizes.size(),
      new_sizes.data(),
      new_strides.data(),
      storage_offset,
      &view_tensor);

  EXPECT_EQ(error, Error::Ok);
  ASSERT_NE(view_tensor, nullptr);

  EXPECT_EQ(view_tensor->dim(), 2);
  EXPECT_EQ(view_tensor->size(0), 2);
  EXPECT_EQ(view_tensor->size(1), 4);

  char* orig_ptr = static_cast<char*>(orig_tensor->data_ptr());
  char* view_ptr = static_cast<char*>(view_tensor->data_ptr());
  EXPECT_EQ(view_ptr, orig_ptr + storage_offset * sizeof(float));

  EXPECT_EQ(aoti_torch_delete_tensor_object(orig_tensor), Error::Ok);
  EXPECT_EQ(aoti_torch_delete_tensor_object(view_tensor), Error::Ok);
}

// ============================================================================
// Memory Sharing Tests
// ============================================================================

TEST_F(AOTITorchReinterpretTensorSlimTest, MemorySharing_CPU) {
  std::vector<int64_t> sizes = {6};
  Tensor* orig_tensor = createTestTensor(
      sizes,
      {},
      static_cast<int32_t>(slim_c10::ScalarType::Float),
      static_cast<int32_t>(slim_c10::DeviceType::CPU),
      0);
  ASSERT_NE(orig_tensor, nullptr);

  void* orig_ptr = orig_tensor->data_ptr();

  std::vector<int64_t> new_sizes = {2, 3};
  std::vector<int64_t> new_strides = {3, 1};

  Tensor* view_tensor = nullptr;
  AOTITorchError error = aoti_torch__reinterpret_tensor(
      orig_tensor,
      new_sizes.size(),
      new_sizes.data(),
      new_strides.data(),
      0,
      &view_tensor);

  EXPECT_EQ(error, Error::Ok);
  ASSERT_NE(view_tensor, nullptr);

  EXPECT_EQ(view_tensor->data_ptr(), orig_ptr);

  EXPECT_EQ(aoti_torch_delete_tensor_object(orig_tensor), Error::Ok);

  EXPECT_EQ(view_tensor->data_ptr(), orig_ptr);

  EXPECT_EQ(aoti_torch_delete_tensor_object(view_tensor), Error::Ok);
}

TEST_F(AOTITorchReinterpretTensorSlimTest, MultipleViews_CPU) {
  std::vector<int64_t> sizes = {24};
  Tensor* orig_tensor = createTestTensor(
      sizes,
      {},
      static_cast<int32_t>(slim_c10::ScalarType::Float),
      static_cast<int32_t>(slim_c10::DeviceType::CPU),
      0);
  ASSERT_NE(orig_tensor, nullptr);

  void* orig_ptr = orig_tensor->data_ptr();

  std::vector<int64_t> sizes1 = {2, 12};
  std::vector<int64_t> strides1 = {12, 1};

  std::vector<int64_t> sizes2 = {4, 6};
  std::vector<int64_t> strides2 = {6, 1};

  std::vector<int64_t> sizes3 = {2, 3, 4};
  std::vector<int64_t> strides3 = {12, 4, 1};

  Tensor* view1 = nullptr;
  Tensor* view2 = nullptr;
  Tensor* view3 = nullptr;

  EXPECT_EQ(
      aoti_torch__reinterpret_tensor(
          orig_tensor,
          sizes1.size(),
          sizes1.data(),
          strides1.data(),
          0,
          &view1),
      Error::Ok);
  EXPECT_EQ(
      aoti_torch__reinterpret_tensor(
          orig_tensor,
          sizes2.size(),
          sizes2.data(),
          strides2.data(),
          0,
          &view2),
      Error::Ok);
  EXPECT_EQ(
      aoti_torch__reinterpret_tensor(
          orig_tensor,
          sizes3.size(),
          sizes3.data(),
          strides3.data(),
          0,
          &view3),
      Error::Ok);

  EXPECT_EQ(view1->data_ptr(), orig_ptr);
  EXPECT_EQ(view2->data_ptr(), orig_ptr);
  EXPECT_EQ(view3->data_ptr(), orig_ptr);

  EXPECT_EQ(aoti_torch_delete_tensor_object(orig_tensor), Error::Ok);

  EXPECT_EQ(view1->data_ptr(), orig_ptr);
  EXPECT_EQ(view2->data_ptr(), orig_ptr);
  EXPECT_EQ(view3->data_ptr(), orig_ptr);

  EXPECT_EQ(aoti_torch_delete_tensor_object(view1), Error::Ok);
  EXPECT_EQ(aoti_torch_delete_tensor_object(view2), Error::Ok);
  EXPECT_EQ(aoti_torch_delete_tensor_object(view3), Error::Ok);
}

// ============================================================================
// Dimension Change Tests
// ============================================================================

TEST_F(AOTITorchReinterpretTensorSlimTest, ExpandDimensions_CPU) {
  std::vector<int64_t> sizes = {6};
  Tensor* orig_tensor = createTestTensor(
      sizes,
      {},
      static_cast<int32_t>(slim_c10::ScalarType::Float),
      static_cast<int32_t>(slim_c10::DeviceType::CPU),
      0);
  ASSERT_NE(orig_tensor, nullptr);
  EXPECT_EQ(orig_tensor->dim(), 1);

  std::vector<int64_t> new_sizes = {2, 3};
  std::vector<int64_t> new_strides = {3, 1};

  Tensor* view_tensor = nullptr;
  AOTITorchError error = aoti_torch__reinterpret_tensor(
      orig_tensor,
      new_sizes.size(),
      new_sizes.data(),
      new_strides.data(),
      0,
      &view_tensor);

  EXPECT_EQ(error, Error::Ok);
  ASSERT_NE(view_tensor, nullptr);
  EXPECT_EQ(view_tensor->dim(), 2);

  EXPECT_EQ(aoti_torch_delete_tensor_object(orig_tensor), Error::Ok);
  EXPECT_EQ(aoti_torch_delete_tensor_object(view_tensor), Error::Ok);
}

TEST_F(AOTITorchReinterpretTensorSlimTest, CollapseDimensions_CPU) {
  std::vector<int64_t> sizes = {2, 3, 4};
  Tensor* orig_tensor = createTestTensor(
      sizes,
      {},
      static_cast<int32_t>(slim_c10::ScalarType::Float),
      static_cast<int32_t>(slim_c10::DeviceType::CPU),
      0);
  ASSERT_NE(orig_tensor, nullptr);
  EXPECT_EQ(orig_tensor->dim(), 3);

  std::vector<int64_t> new_sizes = {24};
  std::vector<int64_t> new_strides = {1};

  Tensor* view_tensor = nullptr;
  AOTITorchError error = aoti_torch__reinterpret_tensor(
      orig_tensor,
      new_sizes.size(),
      new_sizes.data(),
      new_strides.data(),
      0,
      &view_tensor);

  EXPECT_EQ(error, Error::Ok);
  ASSERT_NE(view_tensor, nullptr);
  EXPECT_EQ(view_tensor->dim(), 1);
  EXPECT_EQ(view_tensor->numel(), 24);

  EXPECT_EQ(aoti_torch_delete_tensor_object(orig_tensor), Error::Ok);
  EXPECT_EQ(aoti_torch_delete_tensor_object(view_tensor), Error::Ok);
}

TEST_F(AOTITorchReinterpretTensorSlimTest, ScalarTensorView_CPU) {
  std::vector<int64_t> sizes = {1};
  Tensor* orig_tensor = createTestTensor(
      sizes,
      {},
      static_cast<int32_t>(slim_c10::ScalarType::Float),
      static_cast<int32_t>(slim_c10::DeviceType::CPU),
      0);
  ASSERT_NE(orig_tensor, nullptr);

  std::vector<int64_t> new_sizes = {};
  std::vector<int64_t> new_strides = {};

  Tensor* view_tensor = nullptr;
  AOTITorchError error = aoti_torch__reinterpret_tensor(
      orig_tensor, 0, new_sizes.data(), new_strides.data(), 0, &view_tensor);

  EXPECT_EQ(error, Error::Ok);
  ASSERT_NE(view_tensor, nullptr);
  EXPECT_EQ(view_tensor->dim(), 0);
  EXPECT_EQ(view_tensor->numel(), 1);

  EXPECT_EQ(aoti_torch_delete_tensor_object(orig_tensor), Error::Ok);
  EXPECT_EQ(aoti_torch_delete_tensor_object(view_tensor), Error::Ok);
}

// ============================================================================
// Stride Tests
// ============================================================================

TEST_F(AOTITorchReinterpretTensorSlimTest, TransposeViaStrides_CPU) {
  std::vector<int64_t> sizes = {3, 4};
  Tensor* orig_tensor = createTestTensor(
      sizes,
      {},
      static_cast<int32_t>(slim_c10::ScalarType::Float),
      static_cast<int32_t>(slim_c10::DeviceType::CPU),
      0);
  ASSERT_NE(orig_tensor, nullptr);

  std::vector<int64_t> new_sizes = {4, 3};
  std::vector<int64_t> new_strides = {1, 4};

  Tensor* view_tensor = nullptr;
  AOTITorchError error = aoti_torch__reinterpret_tensor(
      orig_tensor,
      new_sizes.size(),
      new_sizes.data(),
      new_strides.data(),
      0,
      &view_tensor);

  EXPECT_EQ(error, Error::Ok);
  ASSERT_NE(view_tensor, nullptr);
  EXPECT_EQ(view_tensor->size(0), 4);
  EXPECT_EQ(view_tensor->size(1), 3);
  EXPECT_EQ(view_tensor->stride(0), 1);
  EXPECT_EQ(view_tensor->stride(1), 4);

  EXPECT_EQ(aoti_torch_delete_tensor_object(orig_tensor), Error::Ok);
  EXPECT_EQ(aoti_torch_delete_tensor_object(view_tensor), Error::Ok);
}

// ============================================================================
// Different Dtype Tests
// ============================================================================

TEST_F(AOTITorchReinterpretTensorSlimTest, Int64Tensor_CPU) {
  std::vector<int64_t> sizes = {6};
  Tensor* orig_tensor = createTestTensor(
      sizes,
      {},
      static_cast<int32_t>(slim_c10::ScalarType::Long),
      static_cast<int32_t>(slim_c10::DeviceType::CPU),
      0);
  ASSERT_NE(orig_tensor, nullptr);

  std::vector<int64_t> new_sizes = {2, 3};
  std::vector<int64_t> new_strides = {3, 1};

  Tensor* view_tensor = nullptr;
  AOTITorchError error = aoti_torch__reinterpret_tensor(
      orig_tensor,
      new_sizes.size(),
      new_sizes.data(),
      new_strides.data(),
      0,
      &view_tensor);

  EXPECT_EQ(error, Error::Ok);
  ASSERT_NE(view_tensor, nullptr);
  EXPECT_EQ(view_tensor->itemsize(), 8);

  EXPECT_EQ(aoti_torch_delete_tensor_object(orig_tensor), Error::Ok);
  EXPECT_EQ(aoti_torch_delete_tensor_object(view_tensor), Error::Ok);
}

TEST_F(AOTITorchReinterpretTensorSlimTest, BFloat16Tensor_CPU) {
  std::vector<int64_t> sizes = {6};
  Tensor* orig_tensor = createTestTensor(
      sizes,
      {},
      static_cast<int32_t>(slim_c10::ScalarType::BFloat16),
      static_cast<int32_t>(slim_c10::DeviceType::CPU),
      0);
  ASSERT_NE(orig_tensor, nullptr);

  std::vector<int64_t> new_sizes = {2, 3};
  std::vector<int64_t> new_strides = {3, 1};

  Tensor* view_tensor = nullptr;
  AOTITorchError error = aoti_torch__reinterpret_tensor(
      orig_tensor,
      new_sizes.size(),
      new_sizes.data(),
      new_strides.data(),
      0,
      &view_tensor);

  EXPECT_EQ(error, Error::Ok);
  ASSERT_NE(view_tensor, nullptr);
  EXPECT_EQ(view_tensor->itemsize(), 2);

  EXPECT_EQ(aoti_torch_delete_tensor_object(orig_tensor), Error::Ok);
  EXPECT_EQ(aoti_torch_delete_tensor_object(view_tensor), Error::Ok);
}

// ============================================================================
// CUDA Tests
// ============================================================================

TEST_F(AOTITorchReinterpretTensorSlimTest, BasicView_CUDA) {
  if (!isCudaAvailable()) {
    GTEST_SKIP() << "CUDA not available";
  }

  std::vector<int64_t> sizes = {2, 3, 4};
  Tensor* orig_tensor = createTestTensor(
      sizes,
      {},
      static_cast<int32_t>(slim_c10::ScalarType::Float),
      static_cast<int32_t>(slim_c10::DeviceType::CUDA),
      0);
  ASSERT_NE(orig_tensor, nullptr);
  EXPECT_TRUE(orig_tensor->is_cuda());

  std::vector<int64_t> new_sizes = {6, 4};
  std::vector<int64_t> new_strides = {4, 1};

  Tensor* view_tensor = nullptr;
  AOTITorchError error = aoti_torch__reinterpret_tensor(
      orig_tensor,
      new_sizes.size(),
      new_sizes.data(),
      new_strides.data(),
      0,
      &view_tensor);

  EXPECT_EQ(error, Error::Ok);
  ASSERT_NE(view_tensor, nullptr);
  EXPECT_TRUE(view_tensor->is_cuda());

  EXPECT_EQ(view_tensor->dim(), 2);
  EXPECT_EQ(view_tensor->size(0), 6);
  EXPECT_EQ(view_tensor->size(1), 4);

  EXPECT_EQ(view_tensor->data_ptr(), orig_tensor->data_ptr());

  EXPECT_EQ(aoti_torch_delete_tensor_object(orig_tensor), Error::Ok);
  EXPECT_EQ(aoti_torch_delete_tensor_object(view_tensor), Error::Ok);
}

TEST_F(AOTITorchReinterpretTensorSlimTest, WithStorageOffset_CUDA) {
  if (!isCudaAvailable()) {
    GTEST_SKIP() << "CUDA not available";
  }

  std::vector<int64_t> sizes = {4, 4};
  Tensor* orig_tensor = createTestTensor(
      sizes,
      {},
      static_cast<int32_t>(slim_c10::ScalarType::Float),
      static_cast<int32_t>(slim_c10::DeviceType::CUDA),
      0);
  ASSERT_NE(orig_tensor, nullptr);

  std::vector<int64_t> new_sizes = {2, 4};
  std::vector<int64_t> new_strides = {4, 1};
  int64_t storage_offset = 8;

  Tensor* view_tensor = nullptr;
  AOTITorchError error = aoti_torch__reinterpret_tensor(
      orig_tensor,
      new_sizes.size(),
      new_sizes.data(),
      new_strides.data(),
      storage_offset,
      &view_tensor);

  EXPECT_EQ(error, Error::Ok);
  ASSERT_NE(view_tensor, nullptr);
  EXPECT_TRUE(view_tensor->is_cuda());

  char* orig_ptr = static_cast<char*>(orig_tensor->data_ptr());
  char* view_ptr = static_cast<char*>(view_tensor->data_ptr());
  EXPECT_EQ(view_ptr, orig_ptr + storage_offset * sizeof(float));

  EXPECT_EQ(aoti_torch_delete_tensor_object(orig_tensor), Error::Ok);
  EXPECT_EQ(aoti_torch_delete_tensor_object(view_tensor), Error::Ok);
}

TEST_F(AOTITorchReinterpretTensorSlimTest, MemorySharing_CUDA) {
  if (!isCudaAvailable()) {
    GTEST_SKIP() << "CUDA not available";
  }

  std::vector<int64_t> sizes = {6};
  Tensor* orig_tensor = createTestTensor(
      sizes,
      {},
      static_cast<int32_t>(slim_c10::ScalarType::Float),
      static_cast<int32_t>(slim_c10::DeviceType::CUDA),
      0);
  ASSERT_NE(orig_tensor, nullptr);

  void* orig_ptr = orig_tensor->data_ptr();

  std::vector<int64_t> new_sizes = {2, 3};
  std::vector<int64_t> new_strides = {3, 1};

  Tensor* view_tensor = nullptr;
  AOTITorchError error = aoti_torch__reinterpret_tensor(
      orig_tensor,
      new_sizes.size(),
      new_sizes.data(),
      new_strides.data(),
      0,
      &view_tensor);

  EXPECT_EQ(error, Error::Ok);
  ASSERT_NE(view_tensor, nullptr);

  EXPECT_EQ(view_tensor->data_ptr(), orig_ptr);

  EXPECT_EQ(aoti_torch_delete_tensor_object(orig_tensor), Error::Ok);
  EXPECT_EQ(view_tensor->data_ptr(), orig_ptr);

  EXPECT_EQ(aoti_torch_delete_tensor_object(view_tensor), Error::Ok);
}

TEST_F(AOTITorchReinterpretTensorSlimTest, ChainedViews_CUDA) {
  if (!isCudaAvailable()) {
    GTEST_SKIP() << "CUDA not available";
  }

  std::vector<int64_t> sizes = {24};
  Tensor* orig_tensor = createTestTensor(
      sizes,
      {},
      static_cast<int32_t>(slim_c10::ScalarType::Float),
      static_cast<int32_t>(slim_c10::DeviceType::CUDA),
      0);
  ASSERT_NE(orig_tensor, nullptr);

  void* orig_ptr = orig_tensor->data_ptr();

  std::vector<int64_t> sizes1 = {4, 6};
  std::vector<int64_t> strides1 = {6, 1};

  Tensor* view1 = nullptr;
  EXPECT_EQ(
      aoti_torch__reinterpret_tensor(
          orig_tensor,
          sizes1.size(),
          sizes1.data(),
          strides1.data(),
          0,
          &view1),
      Error::Ok);

  std::vector<int64_t> sizes2 = {2, 2, 6};
  std::vector<int64_t> strides2 = {12, 6, 1};

  Tensor* view2 = nullptr;
  EXPECT_EQ(
      aoti_torch__reinterpret_tensor(
          view1, sizes2.size(), sizes2.data(), strides2.data(), 0, &view2),
      Error::Ok);

  EXPECT_EQ(view1->data_ptr(), orig_ptr);
  EXPECT_EQ(view2->data_ptr(), orig_ptr);

  EXPECT_EQ(aoti_torch_delete_tensor_object(orig_tensor), Error::Ok);
  EXPECT_EQ(aoti_torch_delete_tensor_object(view1), Error::Ok);
  EXPECT_EQ(aoti_torch_delete_tensor_object(view2), Error::Ok);
}
