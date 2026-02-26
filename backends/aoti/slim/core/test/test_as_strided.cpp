/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <gtest/gtest.h>

#include <executorch/backends/aoti/slim/core/slim_tensor.h>

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
class AsStridedDeviceTest : public ::testing::Test {
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

TYPED_TEST_SUITE(AsStridedDeviceTest, DeviceTypes);

// =============================================================================
// as_strided Basic Tests
// =============================================================================

TYPED_TEST(AsStridedDeviceTest, BasicView) {
  SlimTensor tensor = this->make_tensor({4, 4});
  this->template fill_sequential<float>(tensor, 16);

  SlimTensor view = tensor.as_strided({2, 2}, {4, 1}, 0);

  EXPECT_EQ(view.size(0), 2);
  EXPECT_EQ(view.size(1), 2);
  EXPECT_EQ(view.stride(0), 4);
  EXPECT_EQ(view.stride(1), 1);
  EXPECT_EQ(view.storage_offset(), 0);
  EXPECT_EQ(view.numel(), 4);

  // View should share storage
  EXPECT_EQ(view.storage().get(), tensor.storage().get());

  // Verify data access through view
  EXPECT_FLOAT_EQ(this->template read_value<float>(view.data_ptr(), 0), 0.0f);
  EXPECT_FLOAT_EQ(this->template read_value<float>(view.data_ptr(), 1), 1.0f);
  EXPECT_FLOAT_EQ(this->template read_value<float>(tensor.data_ptr(), 4), 4.0f);
}

TYPED_TEST(AsStridedDeviceTest, WithStorageOffset) {
  SlimTensor tensor = this->make_tensor({4, 4});
  this->template fill_sequential<float>(tensor, 16);

  SlimTensor view = tensor.as_strided({2, 3}, {4, 1}, 5);

  EXPECT_EQ(view.storage_offset(), 5);
  EXPECT_EQ(view.numel(), 6);

  EXPECT_FLOAT_EQ(this->template read_value<float>(view.data_ptr(), 0), 5.0f);
}

TYPED_TEST(AsStridedDeviceTest, NonContiguousStrides) {
  SlimTensor tensor = this->make_tensor({6});
  this->template fill_sequential<float>(tensor, 6);

  SlimTensor view = tensor.as_strided({3}, {2}, 0);

  EXPECT_EQ(view.size(0), 3);
  EXPECT_EQ(view.stride(0), 2);
  EXPECT_EQ(view.numel(), 3);
  EXPECT_FALSE(view.is_contiguous());

  // Access values through stride (stride=2, so indices 0, 2, 4)
  EXPECT_FLOAT_EQ(
      this->template read_value<float>(view.data_ptr(), 0 * 2), 0.0f);
  EXPECT_FLOAT_EQ(
      this->template read_value<float>(view.data_ptr(), 1 * 2), 2.0f);
  EXPECT_FLOAT_EQ(
      this->template read_value<float>(view.data_ptr(), 2 * 2), 4.0f);
}

TYPED_TEST(AsStridedDeviceTest, TransposeView) {
  SlimTensor tensor = this->make_tensor({3, 4});
  this->template fill_sequential<float>(tensor, 12);

  // Create transposed view (4x3) by swapping sizes and strides
  SlimTensor transposed = tensor.as_strided({4, 3}, {1, 4}, 0);

  EXPECT_EQ(transposed.size(0), 4);
  EXPECT_EQ(transposed.size(1), 3);
  EXPECT_EQ(transposed.stride(0), 1);
  EXPECT_EQ(transposed.stride(1), 4);
  EXPECT_FALSE(transposed.is_contiguous());
}

TYPED_TEST(AsStridedDeviceTest, SharedStorageModification) {
  SlimTensor tensor = this->make_tensor({4});
  this->template fill_sequential<float>(tensor, 4);

  SlimTensor view = tensor.as_strided({2}, {1}, 1);

  // Modify through view
  this->template write_value<float>(view.data_ptr(), 100.0f, 0);
  this->template write_value<float>(view.data_ptr(), 200.0f, 1);

  // Changes should be visible in original tensor
  EXPECT_FLOAT_EQ(
      this->template read_value<float>(tensor.data_ptr(), 1), 100.0f);
  EXPECT_FLOAT_EQ(
      this->template read_value<float>(tensor.data_ptr(), 2), 200.0f);
}

// =============================================================================
// as_strided_ In-Place Tests
// =============================================================================

TYPED_TEST(AsStridedDeviceTest, InPlaceModification) {
  SlimTensor tensor = this->make_tensor({4, 4});
  void* original_data = tensor.data_ptr();

  tensor.as_strided_({2, 8}, {8, 1}, 0);

  EXPECT_EQ(tensor.size(0), 2);
  EXPECT_EQ(tensor.size(1), 8);
  EXPECT_EQ(tensor.stride(0), 8);
  EXPECT_EQ(tensor.stride(1), 1);
  EXPECT_EQ(tensor.numel(), 16);
  EXPECT_TRUE(tensor.is_contiguous());

  EXPECT_EQ(tensor.data_ptr(), original_data);
}

TYPED_TEST(AsStridedDeviceTest, InPlaceWithOffset) {
  SlimTensor tensor = this->make_tensor({16});

  tensor.as_strided_({4}, {1}, 4);

  EXPECT_EQ(tensor.size(0), 4);
  EXPECT_EQ(tensor.storage_offset(), 4);
  EXPECT_EQ(tensor.numel(), 4);
}

// =============================================================================
// as_strided Edge Cases
// =============================================================================

TYPED_TEST(AsStridedDeviceTest, ZeroDimView) {
  SlimTensor tensor = this->make_tensor({4});
  this->template write_value<float>(tensor.data_ptr(), 42.0f, 2);

  SlimTensor scalar_view = tensor.as_strided({}, {}, 2);

  EXPECT_EQ(scalar_view.dim(), 0);
  EXPECT_EQ(scalar_view.numel(), 1);
  EXPECT_EQ(scalar_view.storage_offset(), 2);

  EXPECT_FLOAT_EQ(
      this->template read_value<float>(scalar_view.data_ptr(), 0), 42.0f);
}

TYPED_TEST(AsStridedDeviceTest, SingleElementView) {
  SlimTensor tensor = this->make_tensor({3, 3});
  this->template fill_sequential<float>(tensor, 9);

  SlimTensor view = tensor.as_strided({1, 1}, {3, 1}, 4);

  EXPECT_EQ(view.numel(), 1);

  EXPECT_FLOAT_EQ(this->template read_value<float>(view.data_ptr(), 0), 4.0f);
}

TYPED_TEST(AsStridedDeviceTest, ZeroStridesBroadcast) {
  SlimTensor tensor = this->make_tensor({4});
  this->template write_value<float>(tensor.data_ptr(), 42.0f, 0);

  SlimTensor broadcast = tensor.as_strided({3, 3}, {0, 0}, 0);

  EXPECT_EQ(broadcast.size(0), 3);
  EXPECT_EQ(broadcast.size(1), 3);
  EXPECT_EQ(broadcast.stride(0), 0);
  EXPECT_EQ(broadcast.stride(1), 0);
  EXPECT_EQ(broadcast.numel(), 9);

  EXPECT_FLOAT_EQ(
      this->template read_value<float>(broadcast.data_ptr(), 0), 42.0f);
}

// =============================================================================
// as_strided with Different DTypes
// =============================================================================

TYPED_TEST(AsStridedDeviceTest, Int64View) {
  SlimTensor tensor = this->make_tensor({8}, c10::ScalarType::Long);

  // Fill with values multiplied by 10
  if constexpr (TypeParam::is_cuda) {
#ifdef CUDA_AVAILABLE
    std::vector<int64_t> cpu_data(8);
    for (size_t i = 0; i < 8; ++i) {
      cpu_data[i] = static_cast<int64_t>(i * 10);
    }
    DeviceTraits<c10::DeviceType::CUDA>::memcpy(
        tensor.data_ptr(),
        cpu_data.data(),
        8 * sizeof(int64_t),
        DEFAULT_CUDA_DEVICE,
        CPU_DEVICE);
#endif
  } else {
    int64_t* data = static_cast<int64_t*>(tensor.data_ptr());
    for (size_t i = 0; i < 8; ++i) {
      data[i] = static_cast<int64_t>(i * 10);
    }
  }

  SlimTensor view = tensor.as_strided({2, 3}, {3, 1}, 1);

  EXPECT_EQ(view.dtype(), c10::ScalarType::Long);
  EXPECT_EQ(this->template read_value<int64_t>(view.data_ptr(), 0), 10);
}

TYPED_TEST(AsStridedDeviceTest, Int8View) {
  SlimTensor tensor = this->make_tensor({16}, c10::ScalarType::Char);

  if constexpr (TypeParam::is_cuda) {
#ifdef CUDA_AVAILABLE
    std::vector<int8_t> cpu_data(16);
    for (size_t i = 0; i < 16; ++i) {
      cpu_data[i] = static_cast<int8_t>(i);
    }
    DeviceTraits<c10::DeviceType::CUDA>::memcpy(
        tensor.data_ptr(),
        cpu_data.data(),
        16 * sizeof(int8_t),
        DEFAULT_CUDA_DEVICE,
        CPU_DEVICE);
#endif
  } else {
    int8_t* data = static_cast<int8_t*>(tensor.data_ptr());
    for (size_t i = 0; i < 16; ++i) {
      data[i] = static_cast<int8_t>(i);
    }
  }

  SlimTensor view = tensor.as_strided({4, 2}, {4, 1}, 2);

  EXPECT_EQ(view.dtype(), c10::ScalarType::Char);
  EXPECT_EQ(view.itemsize(), 1);
  EXPECT_EQ(this->template read_value<int8_t>(view.data_ptr(), 0), 2);
}

// =============================================================================
// Multiple Views Share Storage
// =============================================================================

TYPED_TEST(AsStridedDeviceTest, MultipleViews) {
  SlimTensor tensor = this->make_tensor({12});
  this->template fill_sequential<float>(tensor, 12);

  SlimTensor view1 = tensor.as_strided({4}, {1}, 0);
  SlimTensor view2 = tensor.as_strided({4}, {1}, 4);
  SlimTensor view3 = tensor.as_strided({4}, {1}, 8);

  EXPECT_EQ(view1.storage().get(), tensor.storage().get());
  EXPECT_EQ(view2.storage().get(), tensor.storage().get());
  EXPECT_EQ(view3.storage().get(), tensor.storage().get());

  EXPECT_FLOAT_EQ(this->template read_value<float>(view1.data_ptr(), 0), 0.0f);
  EXPECT_FLOAT_EQ(this->template read_value<float>(view2.data_ptr(), 0), 4.0f);
  EXPECT_FLOAT_EQ(this->template read_value<float>(view3.data_ptr(), 0), 8.0f);

  // Modify through one view
  this->template write_value<float>(view2.data_ptr(), 100.0f, 0);

  // Visible in original
  EXPECT_FLOAT_EQ(
      this->template read_value<float>(tensor.data_ptr(), 4), 100.0f);
}

} // namespace executorch::backends::aoti::slim
