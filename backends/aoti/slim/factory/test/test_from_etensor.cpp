/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <gtest/gtest.h>

#include <executorch/backends/aoti/slim/core/storage.h>
#include <executorch/backends/aoti/slim/factory/empty.h>
#include <executorch/backends/aoti/slim/factory/from_etensor.h>
#include <executorch/runtime/core/exec_aten/testing_util/tensor_factory.h>
#include <executorch/runtime/platform/runtime.h>

#ifdef CUDA_AVAILABLE
#include <cuda_runtime.h>
#endif

namespace executorch::backends::aoti::slim {

using executorch::runtime::Error;
using executorch::runtime::TensorShapeDynamism;
using executorch::runtime::etensor::ScalarType;
using executorch::runtime::testing::TensorFactory;

// =============================================================================
// Test Device Helpers
// =============================================================================

inline std::vector<c10::Device> getTestDevices() {
  std::vector<c10::Device> devices = {CPU_DEVICE};
#ifdef CUDA_AVAILABLE
  devices.push_back(DEFAULT_CUDA_DEVICE);
#endif
  return devices;
}

inline std::string deviceToString(
    const testing::TestParamInfo<c10::Device>& info) {
  return info.param.is_cpu() ? "CPU" : "CUDA";
}

// =============================================================================
// Helper Functions
// =============================================================================

namespace {

// Helper: Verify SlimTensor data matches expected values
// Handles GPU data by copying to host first
template <typename T>
void verify_slimtensor_data(
    const SlimTensor& tensor,
    const T* expected_data,
    size_t num_elements) {
  size_t nbytes = num_elements * sizeof(T);

  if (tensor.is_cpu()) {
    const T* actual = static_cast<const T*>(tensor.data_ptr());
    for (size_t i = 0; i < num_elements; ++i) {
      EXPECT_EQ(actual[i], expected_data[i])
          << "Mismatch at index " << i << ": expected " << expected_data[i]
          << ", got " << actual[i];
    }
  } else {
#ifdef CUDA_AVAILABLE
    // Copy GPU data to host for verification
    std::vector<T> host_data(num_elements);
    DeviceTraits<c10::DeviceType::CUDA>::memcpy(
        host_data.data(),
        tensor.data_ptr(),
        nbytes,
        CPU_DEVICE,
        tensor.device());
    for (size_t i = 0; i < num_elements; ++i) {
      EXPECT_EQ(host_data[i], expected_data[i])
          << "Mismatch at index " << i << ": expected " << expected_data[i]
          << ", got " << host_data[i];
    }
#else
    FAIL() << "CUDA not available but tensor is on CUDA device";
#endif
  }
}

} // namespace

// =============================================================================
// FromETensor Parameterized Tests (CPU and CUDA)
// =============================================================================

class FromETensorParamTest : public testing::TestWithParam<c10::Device> {
 protected:
  void SetUp() override {
    executorch::runtime::runtime_init();
  }

  c10::Device device() const {
    return GetParam();
  }
};

TEST_P(FromETensorParamTest, BasicConversion) {
  TensorFactory<ScalarType::Float> tf;

  // Create ETensor on CPU with known values
  std::vector<float> data = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
  auto etensor = tf.make({2, 3}, data);

  // Convert to SlimTensor on target device (source is CPU)
  SlimTensor result = from_etensor(etensor, CPU_DEVICE, device());

  // Verify metadata
  EXPECT_EQ(result.dim(), 2u);
  EXPECT_EQ(result.size(0), 2);
  EXPECT_EQ(result.size(1), 3);
  EXPECT_EQ(result.dtype(), c10::ScalarType::Float);
  EXPECT_EQ(result.device().type(), device().type());
  EXPECT_EQ(result.numel(), 6u);
  EXPECT_TRUE(result.is_contiguous());

  // Verify data
  verify_slimtensor_data<float>(result, data.data(), data.size());
}

TEST_P(FromETensorParamTest, PreservesStrides) {
  TensorFactory<ScalarType::Float> tf;

  // Create ETensor with non-default strides (column-major order)
  std::vector<float> data = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
  std::vector<int32_t> strides = {1, 2}; // Column-major for 2x3 tensor
  auto etensor = tf.make({2, 3}, data, strides);

  // Convert to SlimTensor (source is CPU)
  SlimTensor result = from_etensor(etensor, CPU_DEVICE, device());

  // Verify strides are preserved
  EXPECT_EQ(result.stride(0), 1);
  EXPECT_EQ(result.stride(1), 2);
  EXPECT_FALSE(result.is_contiguous());
}

TEST_P(FromETensorParamTest, Float32Dtype) {
  TensorFactory<ScalarType::Float> tf;
  std::vector<float> data = {1.5f, 2.5f, 3.5f, 4.5f};
  auto etensor = tf.make({2, 2}, data);

  SlimTensor result = from_etensor(etensor, CPU_DEVICE, device());

  EXPECT_EQ(result.dtype(), c10::ScalarType::Float);
  EXPECT_EQ(result.itemsize(), sizeof(float));
  verify_slimtensor_data<float>(result, data.data(), data.size());
}

TEST_P(FromETensorParamTest, Int64Dtype) {
  TensorFactory<ScalarType::Long> tf;
  std::vector<int64_t> data = {10, 20, 30, 40, 50, 60};
  auto etensor = tf.make({2, 3}, data);

  SlimTensor result = from_etensor(etensor, CPU_DEVICE, device());

  EXPECT_EQ(result.dtype(), c10::ScalarType::Long);
  EXPECT_EQ(result.itemsize(), sizeof(int64_t));
  verify_slimtensor_data<int64_t>(result, data.data(), data.size());
}

TEST_P(FromETensorParamTest, Int32Dtype) {
  TensorFactory<ScalarType::Int> tf;
  std::vector<int32_t> data = {100, 200, 300, 400};
  auto etensor = tf.make({4}, data);

  SlimTensor result = from_etensor(etensor, CPU_DEVICE, device());

  EXPECT_EQ(result.dtype(), c10::ScalarType::Int);
  EXPECT_EQ(result.itemsize(), sizeof(int32_t));
  verify_slimtensor_data<int32_t>(result, data.data(), data.size());
}

TEST_P(FromETensorParamTest, Int16Dtype) {
  TensorFactory<ScalarType::Short> tf;
  std::vector<int16_t> data = {-1, 0, 1, 2, 3, 4};
  auto etensor = tf.make({2, 3}, data);

  SlimTensor result = from_etensor(etensor, CPU_DEVICE, device());

  EXPECT_EQ(result.dtype(), c10::ScalarType::Short);
  EXPECT_EQ(result.itemsize(), sizeof(int16_t));
  verify_slimtensor_data<int16_t>(result, data.data(), data.size());
}

TEST_P(FromETensorParamTest, Int8Dtype) {
  TensorFactory<ScalarType::Char> tf;
  std::vector<int8_t> data = {-128, -1, 0, 1, 127};
  auto etensor = tf.make({5}, data);

  SlimTensor result = from_etensor(etensor, CPU_DEVICE, device());

  EXPECT_EQ(result.dtype(), c10::ScalarType::Char);
  EXPECT_EQ(result.itemsize(), sizeof(int8_t));
  verify_slimtensor_data<int8_t>(result, data.data(), data.size());
}

TEST_P(FromETensorParamTest, BoolDtype) {
  TensorFactory<ScalarType::Bool> tf;
  // TensorFactory<Bool> uses uint8_t internally
  std::vector<uint8_t> data = {1, 0, 1, 0, 1, 1};
  auto etensor = tf.make({2, 3}, data);

  SlimTensor result = from_etensor(etensor, CPU_DEVICE, device());

  EXPECT_EQ(result.dtype(), c10::ScalarType::Bool);
  EXPECT_EQ(result.numel(), 6u);

  // Verify data using uint8_t representation
  verify_slimtensor_data<uint8_t>(result, data.data(), data.size());
}

TEST_P(FromETensorParamTest, LargeTensor) {
  TensorFactory<ScalarType::Float> tf;

  // Create a larger tensor
  constexpr size_t kSize = 1024;
  std::vector<float> data(kSize);
  for (size_t i = 0; i < kSize; ++i) {
    data[i] = static_cast<float>(i) * 0.5f;
  }
  auto etensor = tf.make({32, 32}, data);

  SlimTensor result = from_etensor(etensor, CPU_DEVICE, device());

  EXPECT_EQ(result.numel(), kSize);
  EXPECT_EQ(result.size(0), 32);
  EXPECT_EQ(result.size(1), 32);

  verify_slimtensor_data<float>(result, data.data(), data.size());
}

TEST_P(FromETensorParamTest, OneDimensional) {
  TensorFactory<ScalarType::Float> tf;
  std::vector<float> data = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f};
  auto etensor = tf.make({5}, data);

  SlimTensor result = from_etensor(etensor, CPU_DEVICE, device());

  EXPECT_EQ(result.dim(), 1u);
  EXPECT_EQ(result.size(0), 5);
  EXPECT_EQ(result.stride(0), 1);
  EXPECT_TRUE(result.is_contiguous());

  verify_slimtensor_data<float>(result, data.data(), data.size());
}

TEST_P(FromETensorParamTest, ThreeDimensional) {
  TensorFactory<ScalarType::Float> tf;
  std::vector<float> data(24);
  for (size_t i = 0; i < 24; ++i) {
    data[i] = static_cast<float>(i);
  }
  auto etensor = tf.make({2, 3, 4}, data);

  SlimTensor result = from_etensor(etensor, CPU_DEVICE, device());

  EXPECT_EQ(result.dim(), 3u);
  EXPECT_EQ(result.size(0), 2);
  EXPECT_EQ(result.size(1), 3);
  EXPECT_EQ(result.size(2), 4);
  EXPECT_TRUE(result.is_contiguous());

  verify_slimtensor_data<float>(result, data.data(), data.size());
}

TEST_P(FromETensorParamTest, PointerOverload) {
  TensorFactory<ScalarType::Float> tf;
  std::vector<float> data = {1.0f, 2.0f, 3.0f, 4.0f};
  auto etensor = tf.make({2, 2}, data);

  // Use pointer overload (source is CPU)
  SlimTensor result = from_etensor(&etensor, CPU_DEVICE, device());

  EXPECT_EQ(result.dim(), 2u);
  EXPECT_EQ(result.numel(), 4u);

  verify_slimtensor_data<float>(result, data.data(), data.size());
}

INSTANTIATE_TEST_SUITE_P(
    DeviceTests,
    FromETensorParamTest,
    testing::ValuesIn(getTestDevices()),
    deviceToString);

// =============================================================================
// CPU-Only Tests (require direct data access without CUDA memcpy)
// =============================================================================

TEST(FromETensorCPUTest, DataIsIndependent) {
  executorch::runtime::runtime_init();
  TensorFactory<ScalarType::Float> tf;

  std::vector<float> data = {1.0f, 2.0f, 3.0f, 4.0f};
  auto etensor = tf.make({2, 2}, data);

  SlimTensor result = from_etensor(etensor, CPU_DEVICE);

  // Modify source data
  float* etensor_data = etensor.mutable_data_ptr<float>();
  etensor_data[0] = 999.0f;

  // SlimTensor should have its own copy
  const float* result_data = static_cast<const float*>(result.data_ptr());
  EXPECT_FLOAT_EQ(result_data[0], 1.0f);
}

TEST(FromETensorCPUTest, ModifySlimTensorDoesNotAffectETensor) {
  executorch::runtime::runtime_init();
  TensorFactory<ScalarType::Float> tf;

  std::vector<float> data = {1.0f, 2.0f, 3.0f, 4.0f};
  auto etensor = tf.make({2, 2}, data);

  SlimTensor result = from_etensor(etensor, CPU_DEVICE);

  // Modify SlimTensor
  float* result_data = static_cast<float*>(result.data_ptr());
  result_data[0] = 999.0f;

  // ETensor should be unchanged
  const float* etensor_data = etensor.const_data_ptr<float>();
  EXPECT_FLOAT_EQ(etensor_data[0], 1.0f);
}

} // namespace executorch::backends::aoti::slim
