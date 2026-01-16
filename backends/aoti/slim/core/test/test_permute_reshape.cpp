/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <gtest/gtest.h>

#include <executorch/backends/aoti/slim/core/slim_tensor.h>
#include <executorch/backends/aoti/slim/core/slim_tensor_view_incl.h>
#include <executorch/backends/aoti/slim/factory/empty.h>

#ifdef CUDA_AVAILABLE
#include <cuda_runtime.h>
#endif

namespace executorch::backends::aoti::slim {

// =============================================================================
// Device trait for parameterized tests
// =============================================================================

struct CPUDevice {
  static c10::Device device() {
    return CPU_DEVICE;
  }
  static constexpr bool is_cuda = false;
};

#ifdef CUDA_AVAILABLE
struct CUDADevice {
  static c10::Device device() {
    return DEFAULT_CUDA_DEVICE;
  }
  static constexpr bool is_cuda = true;
};
#endif

// =============================================================================
// Test fixture for parameterized device tests
// =============================================================================

template <typename DeviceTrait>
class PermuteReshapeDeviceTest : public ::testing::Test {
 protected:
  static c10::Device device() {
    return DeviceTrait::device();
  }

  SlimTensor make_tensor(
      std::initializer_list<int64_t> sizes,
      c10::ScalarType dtype = c10::ScalarType::Float) {
    return empty(sizes, dtype, device());
  }

  // Helper to initialize tensor data from CPU (handles both CPU and CUDA)
  template <typename T>
  void fill_sequential(SlimTensor& tensor, size_t count) {
    if constexpr (DeviceTrait::is_cuda) {
#ifdef CUDA_AVAILABLE
      std::vector<T> cpu_data(count);
      for (size_t i = 0; i < count; ++i) {
        cpu_data[i] = static_cast<T>(i);
      }
      DeviceTraits<c10::DeviceType::CUDA>::memcpy(
          tensor.data_ptr(),
          cpu_data.data(),
          count * sizeof(T),
          DEFAULT_CUDA_DEVICE,
          CPU_DEVICE);
#endif
    } else {
      T* data = static_cast<T*>(tensor.data_ptr());
      for (size_t i = 0; i < count; ++i) {
        data[i] = static_cast<T>(i);
      }
    }
  }

  // Helper to read a value from tensor (handles both CPU and CUDA)
  template <typename T>
  T read_value(void* ptr, size_t offset = 0) {
    if constexpr (DeviceTrait::is_cuda) {
#ifdef CUDA_AVAILABLE
      T value;
      DeviceTraits<c10::DeviceType::CUDA>::memcpy(
          &value,
          static_cast<T*>(ptr) + offset,
          sizeof(T),
          CPU_DEVICE,
          DEFAULT_CUDA_DEVICE);
      return value;
#else
      return T{};
#endif
    } else {
      return *(static_cast<T*>(ptr) + offset);
    }
  }

  // Helper to write a value to tensor (handles both CPU and CUDA)
  template <typename T>
  void write_value(void* ptr, T value, size_t offset = 0) {
    if constexpr (DeviceTrait::is_cuda) {
#ifdef CUDA_AVAILABLE
      DeviceTraits<c10::DeviceType::CUDA>::memcpy(
          static_cast<T*>(ptr) + offset,
          &value,
          sizeof(T),
          DEFAULT_CUDA_DEVICE,
          CPU_DEVICE);
#endif
    } else {
      *(static_cast<T*>(ptr) + offset) = value;
    }
  }
};

// Type list for parameterized tests
using DeviceTypes = ::testing::Types<
    CPUDevice
#ifdef CUDA_AVAILABLE
    ,
    CUDADevice
#endif
    >;

TYPED_TEST_SUITE(PermuteReshapeDeviceTest, DeviceTypes);

// =============================================================================
// permute Basic Tests
// =============================================================================

TYPED_TEST(PermuteReshapeDeviceTest, Basic2DTranspose) {
  SlimTensor tensor = this->make_tensor({3, 4});
  this->template fill_sequential<float>(tensor, 12);

  SlimTensor transposed = tensor.permute({1, 0});

  EXPECT_EQ(transposed.size(0), 4);
  EXPECT_EQ(transposed.size(1), 3);
  EXPECT_EQ(transposed.stride(0), 1);
  EXPECT_EQ(transposed.stride(1), 4);
  EXPECT_FALSE(transposed.is_contiguous());
  EXPECT_EQ(transposed.numel(), 12);

  // Shares storage
  EXPECT_EQ(transposed.storage().get(), tensor.storage().get());
}

TYPED_TEST(PermuteReshapeDeviceTest, 3DPermutation) {
  SlimTensor tensor = this->make_tensor({2, 3, 4});
  this->template fill_sequential<float>(tensor, 24);

  SlimTensor permuted = tensor.permute({2, 0, 1});

  EXPECT_EQ(permuted.size(0), 4);
  EXPECT_EQ(permuted.size(1), 2);
  EXPECT_EQ(permuted.size(2), 3);

  // Original strides: [12, 4, 1]
  // Permuted strides for {2, 0, 1}: [1, 12, 4]
  EXPECT_EQ(permuted.stride(0), 1);
  EXPECT_EQ(permuted.stride(1), 12);
  EXPECT_EQ(permuted.stride(2), 4);
}

TYPED_TEST(PermuteReshapeDeviceTest, NegativeIndices) {
  SlimTensor tensor = this->make_tensor({2, 3, 4});

  // Use negative indices: -1 is the last dimension
  SlimTensor permuted = tensor.permute({-1, -3, -2});

  EXPECT_EQ(permuted.size(0), 4);
  EXPECT_EQ(permuted.size(1), 2);
  EXPECT_EQ(permuted.size(2), 3);
}

TYPED_TEST(PermuteReshapeDeviceTest, IdentityPermutation) {
  SlimTensor tensor = this->make_tensor({2, 3, 4});

  SlimTensor permuted = tensor.permute({0, 1, 2});

  EXPECT_EQ(permuted.sizes(), tensor.sizes());
  EXPECT_EQ(permuted.strides(), tensor.strides());
  EXPECT_TRUE(permuted.is_contiguous());
}

TYPED_TEST(PermuteReshapeDeviceTest, SharedStorageModification) {
  SlimTensor tensor = this->make_tensor({2, 3});
  this->template fill_sequential<float>(tensor, 6);

  SlimTensor transposed = tensor.permute({1, 0});

  // Modify via transposed
  this->template write_value<float>(transposed.data_ptr(), 100.0f, 0);

  EXPECT_FLOAT_EQ(
      this->template read_value<float>(tensor.data_ptr(), 0), 100.0f);
}

// =============================================================================
// reshape Basic Tests
// =============================================================================

TYPED_TEST(PermuteReshapeDeviceTest, ContiguousReshapeToView) {
  SlimTensor tensor = this->make_tensor({2, 3, 4});
  this->template fill_sequential<float>(tensor, 24);

  SlimTensor reshaped = tensor.reshape({6, 4});

  EXPECT_EQ(reshaped.size(0), 6);
  EXPECT_EQ(reshaped.size(1), 4);
  EXPECT_EQ(reshaped.numel(), 24);
  EXPECT_TRUE(reshaped.is_contiguous());

  // Should share storage (view)
  EXPECT_EQ(reshaped.storage().get(), tensor.storage().get());

  // Verify data is accessible
  EXPECT_FLOAT_EQ(
      this->template read_value<float>(reshaped.data_ptr(), 0), 0.0f);
  EXPECT_FLOAT_EQ(
      this->template read_value<float>(reshaped.data_ptr(), 23), 23.0f);
}

TYPED_TEST(PermuteReshapeDeviceTest, Flatten) {
  SlimTensor tensor = this->make_tensor({2, 3, 4});
  this->template fill_sequential<float>(tensor, 24);

  SlimTensor flat = tensor.reshape({24});

  EXPECT_EQ(flat.dim(), 1);
  EXPECT_EQ(flat.size(0), 24);
  EXPECT_TRUE(flat.is_contiguous());

  // Should be a view
  EXPECT_EQ(flat.storage().get(), tensor.storage().get());
}

TYPED_TEST(PermuteReshapeDeviceTest, InferDimension) {
  SlimTensor tensor = this->make_tensor({2, 3, 4});

  // Use -1 to infer dimension: 24 / 6 = 4
  SlimTensor reshaped = tensor.reshape({6, -1});

  EXPECT_EQ(reshaped.size(0), 6);
  EXPECT_EQ(reshaped.size(1), 4);
}

TYPED_TEST(PermuteReshapeDeviceTest, InferFirstDimension) {
  SlimTensor tensor = this->make_tensor({2, 3, 4});

  // Infer first dimension: 24 / 8 = 3
  SlimTensor reshaped = tensor.reshape({-1, 8});

  EXPECT_EQ(reshaped.size(0), 3);
  EXPECT_EQ(reshaped.size(1), 8);
}

TYPED_TEST(PermuteReshapeDeviceTest, NonContiguousTensorCopies) {
  SlimTensor tensor = this->make_tensor({3, 4});
  this->template fill_sequential<float>(tensor, 12);

  // Transpose makes it non-contiguous
  SlimTensor transposed = tensor.permute({1, 0});
  EXPECT_FALSE(transposed.is_contiguous());

  // Reshape of non-contiguous requires copy
  SlimTensor reshaped = transposed.reshape({12});

  EXPECT_EQ(reshaped.dim(), 1);
  EXPECT_EQ(reshaped.size(0), 12);
  EXPECT_TRUE(reshaped.is_contiguous());

  // Should NOT share storage (copy made)
  EXPECT_NE(reshaped.storage().get(), transposed.storage().get());

  // Verify data was copied correctly
  // transposed[0][0] = tensor[0][0] = 0
  // transposed[0][1] = tensor[1][0] = 4
  // transposed[0][2] = tensor[2][0] = 8
  // transposed[1][0] = tensor[0][1] = 1
  EXPECT_FLOAT_EQ(
      this->template read_value<float>(reshaped.data_ptr(), 0), 0.0f);
  EXPECT_FLOAT_EQ(
      this->template read_value<float>(reshaped.data_ptr(), 1), 4.0f);
  EXPECT_FLOAT_EQ(
      this->template read_value<float>(reshaped.data_ptr(), 2), 8.0f);
  EXPECT_FLOAT_EQ(
      this->template read_value<float>(reshaped.data_ptr(), 3), 1.0f);
}

TYPED_TEST(PermuteReshapeDeviceTest, ExpandDimensions) {
  SlimTensor tensor = this->make_tensor({12});
  this->template fill_sequential<float>(tensor, 12);

  SlimTensor reshaped = tensor.reshape({2, 2, 3});

  EXPECT_EQ(reshaped.dim(), 3);
  EXPECT_EQ(reshaped.size(0), 2);
  EXPECT_EQ(reshaped.size(1), 2);
  EXPECT_EQ(reshaped.size(2), 3);

  // Should be a view
  EXPECT_EQ(reshaped.storage().get(), tensor.storage().get());
}

TYPED_TEST(PermuteReshapeDeviceTest, SharedStorageModificationView) {
  SlimTensor tensor = this->make_tensor({2, 6});
  this->template fill_sequential<float>(tensor, 12);

  SlimTensor reshaped = tensor.reshape({3, 4});

  // Modify via reshaped
  this->template write_value<float>(reshaped.data_ptr(), 100.0f, 0);

  // Should be visible in original
  EXPECT_FLOAT_EQ(
      this->template read_value<float>(tensor.data_ptr(), 0), 100.0f);
}

// =============================================================================
// clone_contiguous Tests
// =============================================================================

TYPED_TEST(PermuteReshapeDeviceTest, BasicClone) {
  SlimTensor tensor = this->make_tensor({2, 3});
  this->template fill_sequential<float>(tensor, 6);

  SlimTensor cloned = tensor.clone_contiguous();

  EXPECT_EQ(cloned.sizes(), tensor.sizes());
  EXPECT_TRUE(cloned.is_contiguous());
  EXPECT_NE(cloned.storage().get(), tensor.storage().get());

  // Data should be copied
  for (size_t i = 0; i < 6; ++i) {
    EXPECT_FLOAT_EQ(
        this->template read_value<float>(cloned.data_ptr(), i),
        static_cast<float>(i));
  }

  // Modification should be independent
  this->template write_value<float>(cloned.data_ptr(), 100.0f, 0);
  EXPECT_FLOAT_EQ(this->template read_value<float>(tensor.data_ptr(), 0), 0.0f);
}

TYPED_TEST(PermuteReshapeDeviceTest, NonContiguousToContiguous) {
  SlimTensor tensor = this->make_tensor({3, 4});
  this->template fill_sequential<float>(tensor, 12);

  // Transpose makes it non-contiguous
  SlimTensor transposed = tensor.permute({1, 0});
  EXPECT_FALSE(transposed.is_contiguous());

  SlimTensor cloned = transposed.clone_contiguous();

  EXPECT_EQ(cloned.size(0), 4);
  EXPECT_EQ(cloned.size(1), 3);
  EXPECT_TRUE(cloned.is_contiguous());
  EXPECT_NE(cloned.storage().get(), transposed.storage().get());

  // Verify data was correctly reordered
  // cloned[0][0] = transposed[0][0] = tensor[0][0] = 0
  // cloned[0][1] = transposed[0][1] = tensor[1][0] = 4
  // cloned[0][2] = transposed[0][2] = tensor[2][0] = 8
  // cloned[1][0] = transposed[1][0] = tensor[0][1] = 1
  EXPECT_FLOAT_EQ(this->template read_value<float>(cloned.data_ptr(), 0), 0.0f);
  EXPECT_FLOAT_EQ(this->template read_value<float>(cloned.data_ptr(), 1), 4.0f);
  EXPECT_FLOAT_EQ(this->template read_value<float>(cloned.data_ptr(), 2), 8.0f);
  EXPECT_FLOAT_EQ(this->template read_value<float>(cloned.data_ptr(), 3), 1.0f);
}

// =============================================================================
// Combined Operations Tests
// =============================================================================

TYPED_TEST(PermuteReshapeDeviceTest, PermuteAndReshape) {
  SlimTensor tensor = this->make_tensor({2, 3, 4});
  this->template fill_sequential<float>(tensor, 24);

  // Permute to 3x2x4, then reshape to 6x4
  SlimTensor permuted = tensor.permute({1, 0, 2});
  SlimTensor reshaped = permuted.reshape({6, 4});

  EXPECT_EQ(reshaped.size(0), 6);
  EXPECT_EQ(reshaped.size(1), 4);
  EXPECT_EQ(reshaped.numel(), 24);
}

TYPED_TEST(PermuteReshapeDeviceTest, ReshapeAndPermute) {
  SlimTensor tensor = this->make_tensor({24});
  this->template fill_sequential<float>(tensor, 24);

  // Reshape to 2x3x4, then permute to 4x3x2
  SlimTensor reshaped = tensor.reshape({2, 3, 4});
  SlimTensor permuted = reshaped.permute({2, 1, 0});

  EXPECT_EQ(permuted.size(0), 4);
  EXPECT_EQ(permuted.size(1), 3);
  EXPECT_EQ(permuted.size(2), 2);
}

} // namespace executorch::backends::aoti::slim
