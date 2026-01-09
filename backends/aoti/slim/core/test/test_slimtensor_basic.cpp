/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <gtest/gtest.h>

#include <executorch/backends/aoti/slim/core/SlimTensor.h>
#include <executorch/backends/aoti/slim/core/Storage.h>

namespace executorch::backends::aoti::slim {

// =============================================================================
// Device-Parameterized Test Infrastructure
// =============================================================================

// Get list of devices to test based on availability
inline std::vector<c10::Device> get_test_devices() {
  std::vector<c10::Device> devices;
  devices.push_back(CPU_DEVICE);
  return devices;
}

// Device-parameterized test fixture
class SlimTensorBasicDeviceTest : public ::testing::TestWithParam<c10::Device> {
 protected:
  c10::Device device() const {
    return GetParam();
  }

  Storage make_storage(size_t nbytes) const {
    return Storage(new MaybeOwningStorage(device(), nbytes));
  }

  SlimTensor make_2x3_tensor() const {
    std::vector<int64_t> sizes = {2, 3};
    std::vector<int64_t> strides = {3, 1};
    size_t nbytes = 6 * sizeof(float);
    Storage storage = make_storage(nbytes);
    return SlimTensor(
        std::move(storage),
        makeArrayRef(sizes),
        makeArrayRef(strides),
        c10::ScalarType::Float);
  }
};

INSTANTIATE_TEST_SUITE_P(
    DeviceTests,
    SlimTensorBasicDeviceTest,
    ::testing::ValuesIn(get_test_devices()),
    [](const ::testing::TestParamInfo<c10::Device>& info) { return "CPU"; });

// =============================================================================
// Constructor Tests (Device-Parameterized)
// =============================================================================

TEST_P(SlimTensorBasicDeviceTest, ConstructWithStorage) {
  std::vector<int64_t> sizes = {2, 3, 4};
  std::vector<int64_t> strides = {12, 4, 1};
  size_t nbytes = 24 * sizeof(float);
  Storage storage = make_storage(nbytes);

  SlimTensor tensor(
      std::move(storage),
      makeArrayRef(sizes),
      makeArrayRef(strides),
      c10::ScalarType::Float);

  EXPECT_TRUE(tensor.defined());
  EXPECT_EQ(tensor.dim(), 3u);
  EXPECT_EQ(tensor.numel(), 24u);
  EXPECT_TRUE(tensor.is_contiguous());

  if (device().is_cpu()) {
    EXPECT_TRUE(tensor.is_cpu());
  } else {
    EXPECT_TRUE(tensor.is_cuda());
  }
}

TEST_P(SlimTensorBasicDeviceTest, ConstructWithStorageOffset) {
  std::vector<int64_t> sizes = {2, 3};
  std::vector<int64_t> strides = {3, 1};
  size_t nbytes = 100 * sizeof(float);
  Storage storage = make_storage(nbytes);

  SlimTensor tensor(
      std::move(storage),
      makeArrayRef(sizes),
      makeArrayRef(strides),
      c10::ScalarType::Float,
      10);

  EXPECT_EQ(tensor.storage_offset(), 10);
}

// =============================================================================
// Property Accessor Tests (Device-Parameterized)
// =============================================================================

TEST_P(SlimTensorBasicDeviceTest, Sizes) {
  SlimTensor tensor = make_2x3_tensor();

  auto sizes = tensor.sizes();
  EXPECT_EQ(sizes.size(), 2u);
  EXPECT_EQ(sizes[0], 2);
  EXPECT_EQ(sizes[1], 3);
}

TEST_P(SlimTensorBasicDeviceTest, SizeAtDim) {
  SlimTensor tensor = make_2x3_tensor();

  EXPECT_EQ(tensor.size(0), 2);
  EXPECT_EQ(tensor.size(1), 3);
  EXPECT_EQ(tensor.size(-1), 3);
  EXPECT_EQ(tensor.size(-2), 2);
}

TEST_P(SlimTensorBasicDeviceTest, Strides) {
  SlimTensor tensor = make_2x3_tensor();

  auto strides = tensor.strides();
  EXPECT_EQ(strides.size(), 2u);
  EXPECT_EQ(strides[0], 3);
  EXPECT_EQ(strides[1], 1);
}

TEST_P(SlimTensorBasicDeviceTest, StrideAtDim) {
  SlimTensor tensor = make_2x3_tensor();

  EXPECT_EQ(tensor.stride(0), 3);
  EXPECT_EQ(tensor.stride(1), 1);
  EXPECT_EQ(tensor.stride(-1), 1);
  EXPECT_EQ(tensor.stride(-2), 3);
}

TEST_P(SlimTensorBasicDeviceTest, Dtype) {
  SlimTensor tensor = make_2x3_tensor();

  EXPECT_EQ(tensor.dtype(), c10::ScalarType::Float);
  EXPECT_EQ(tensor.itemsize(), sizeof(float));
}

TEST_P(SlimTensorBasicDeviceTest, Device) {
  SlimTensor tensor = make_2x3_tensor();

  if (device().is_cpu()) {
    EXPECT_TRUE(tensor.is_cpu());
    EXPECT_EQ(tensor.device_type(), c10::DeviceType::CPU);
  } else {
    EXPECT_TRUE(tensor.is_cuda());
    EXPECT_EQ(tensor.device_type(), c10::DeviceType::CUDA);
  }
  EXPECT_EQ(tensor.device_index(), device().index());
}

TEST_P(SlimTensorBasicDeviceTest, Numel) {
  SlimTensor tensor = make_2x3_tensor();
  EXPECT_EQ(tensor.numel(), 6u);
}

TEST_P(SlimTensorBasicDeviceTest, Dim) {
  SlimTensor tensor = make_2x3_tensor();
  EXPECT_EQ(tensor.dim(), 2u);
}

TEST_P(SlimTensorBasicDeviceTest, Nbytes) {
  SlimTensor tensor = make_2x3_tensor();
  EXPECT_EQ(tensor.nbytes(), 6 * sizeof(float));
}

TEST_P(SlimTensorBasicDeviceTest, Itemsize) {
  SlimTensor tensor = make_2x3_tensor();
  EXPECT_EQ(tensor.itemsize(), sizeof(float));
}

TEST_P(SlimTensorBasicDeviceTest, DataPtr) {
  SlimTensor tensor = make_2x3_tensor();
  void* data = tensor.data_ptr();
  EXPECT_NE(data, nullptr);
}

TEST_P(SlimTensorBasicDeviceTest, StorageOffset) {
  std::vector<int64_t> sizes = {2, 3};
  std::vector<int64_t> strides = {3, 1};
  size_t nbytes = 100 * sizeof(float);
  Storage storage = make_storage(nbytes);

  SlimTensor tensor(
      std::move(storage),
      makeArrayRef(sizes),
      makeArrayRef(strides),
      c10::ScalarType::Float,
      42);

  EXPECT_EQ(tensor.storage_offset(), 42);
}

// =============================================================================
// Contiguity Tests (Device-Parameterized)
// =============================================================================

TEST_P(SlimTensorBasicDeviceTest, IsContiguousTrue) {
  SlimTensor tensor = make_2x3_tensor();
  EXPECT_TRUE(tensor.is_contiguous());
}

TEST_P(SlimTensorBasicDeviceTest, IsContiguousFalseTransposed) {
  std::vector<int64_t> sizes = {3, 2};
  std::vector<int64_t> strides = {1, 3};
  size_t nbytes = 6 * sizeof(float);
  Storage storage = make_storage(nbytes);

  SlimTensor tensor(
      std::move(storage),
      makeArrayRef(sizes),
      makeArrayRef(strides),
      c10::ScalarType::Float);

  EXPECT_FALSE(tensor.is_contiguous());
}

TEST_P(SlimTensorBasicDeviceTest, IsContiguousEmptyTensor) {
  std::vector<int64_t> sizes = {0, 3};
  std::vector<int64_t> strides = {3, 1};
  size_t nbytes = 0;
  Storage storage = make_storage(nbytes);

  SlimTensor tensor(
      std::move(storage),
      makeArrayRef(sizes),
      makeArrayRef(strides),
      c10::ScalarType::Float);

  EXPECT_TRUE(tensor.is_contiguous());
  EXPECT_TRUE(tensor.is_empty());
}

// =============================================================================
// State Tests (Device-Parameterized)
// =============================================================================

TEST_P(SlimTensorBasicDeviceTest, Defined) {
  SlimTensor tensor = make_2x3_tensor();
  EXPECT_TRUE(tensor.defined());
}

TEST_P(SlimTensorBasicDeviceTest, IsEmpty) {
  std::vector<int64_t> sizes = {0};
  std::vector<int64_t> strides = {1};
  Storage storage = make_storage(0);

  SlimTensor tensor(
      std::move(storage),
      makeArrayRef(sizes),
      makeArrayRef(strides),
      c10::ScalarType::Float);

  EXPECT_TRUE(tensor.is_empty());
  EXPECT_EQ(tensor.numel(), 0u);
}

TEST_P(SlimTensorBasicDeviceTest, Reset) {
  SlimTensor tensor = make_2x3_tensor();
  EXPECT_TRUE(tensor.defined());

  tensor.reset();
  EXPECT_FALSE(tensor.defined());
}

// =============================================================================
// Copy/Move Tests (Device-Parameterized)
// =============================================================================

TEST_P(SlimTensorBasicDeviceTest, MoveConstructor) {
  SlimTensor original = make_2x3_tensor();
  SlimTensor moved = std::move(original);

  EXPECT_TRUE(moved.defined());
  EXPECT_EQ(moved.dim(), 2u);
  EXPECT_EQ(moved.numel(), 6u);
}

// =============================================================================
// Multi-dimensional Tests (Device-Parameterized)
// =============================================================================

TEST_P(SlimTensorBasicDeviceTest, OneDimensional) {
  std::vector<int64_t> sizes = {10};
  std::vector<int64_t> strides = {1};
  Storage storage = make_storage(10 * sizeof(float));

  SlimTensor tensor(
      std::move(storage),
      makeArrayRef(sizes),
      makeArrayRef(strides),
      c10::ScalarType::Float);

  EXPECT_EQ(tensor.dim(), 1u);
  EXPECT_EQ(tensor.size(0), 10);
  EXPECT_EQ(tensor.stride(0), 1);
  EXPECT_TRUE(tensor.is_contiguous());
}

TEST_P(SlimTensorBasicDeviceTest, FourDimensional) {
  std::vector<int64_t> sizes = {2, 3, 4, 5};
  std::vector<int64_t> strides = {60, 20, 5, 1};
  Storage storage = make_storage(120 * sizeof(float));

  SlimTensor tensor(
      std::move(storage),
      makeArrayRef(sizes),
      makeArrayRef(strides),
      c10::ScalarType::Float);

  EXPECT_EQ(tensor.dim(), 4u);
  EXPECT_EQ(tensor.numel(), 120u);
  EXPECT_TRUE(tensor.is_contiguous());
}

// =============================================================================
// Non-Parameterized Tests (CPU-only, device-independent behavior)
// =============================================================================

// Helper function for CPU-only tests
Storage make_cpu_storage(size_t nbytes) {
  return Storage(new MaybeOwningStorage(CPU_DEVICE, nbytes));
}

SlimTensor make_cpu_2x3_tensor() {
  std::vector<int64_t> sizes = {2, 3};
  std::vector<int64_t> strides = {3, 1};
  size_t nbytes = 6 * sizeof(float);
  Storage storage = make_cpu_storage(nbytes);
  return SlimTensor(
      std::move(storage),
      makeArrayRef(sizes),
      makeArrayRef(strides),
      c10::ScalarType::Float);
}

TEST(SlimTensorBasicTest, DefaultConstructor) {
  SlimTensor tensor;

  EXPECT_FALSE(tensor.defined());
  EXPECT_EQ(tensor.numel(), 0u);
  EXPECT_EQ(tensor.dtype(), c10::ScalarType::Float);
  EXPECT_TRUE(tensor.is_contiguous());
}

TEST(SlimTensorBasicTest, NotDefined) {
  SlimTensor tensor;
  EXPECT_FALSE(tensor.defined());
}

TEST(SlimTensorBasicTest, CopyConstructor) {
  SlimTensor original = make_cpu_2x3_tensor();
  const SlimTensor& copy = original;

  EXPECT_TRUE(copy.defined());
  EXPECT_EQ(copy.dim(), 2u);
  EXPECT_EQ(copy.numel(), 6u);
  EXPECT_EQ(copy.dtype(), c10::ScalarType::Float);
}

// CPU-only test for DataPtrWithOffset (requires reading data back)
TEST(SlimTensorBasicTest, DataPtrWithOffset) {
  std::vector<int64_t> sizes = {2, 3};
  std::vector<int64_t> strides = {3, 1};
  size_t nbytes = 100 * sizeof(float);
  Storage storage = make_cpu_storage(nbytes);
  void* base = storage->data();

  SlimTensor tensor(
      std::move(storage),
      makeArrayRef(sizes),
      makeArrayRef(strides),
      c10::ScalarType::Float,
      5);

  void* data = tensor.data_ptr();
  EXPECT_EQ(data, static_cast<char*>(base) + 5 * sizeof(float));
}

} // namespace executorch::backends::aoti::slim
