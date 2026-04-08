/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/extension/tensor/tensor_ptr.h>

#include <gtest/gtest.h>

#include <cstdlib>
#include <cstring>

#include <executorch/extension/tensor/tensor_ptr_maker.h>
#include <executorch/runtime/core/device_allocator.h>
#include <executorch/runtime/core/test/mock_cuda_allocator.h>
#include <executorch/runtime/platform/runtime.h>
#include <executorch/test/utils/DeathTest.h>

using namespace ::executorch::extension;
using namespace ::executorch::runtime;
using executorch::runtime::etensor::DeviceIndex;
using executorch::runtime::etensor::DeviceType;
using executorch::runtime::testing::MockCudaAllocator;

static MockCudaAllocator g_mock_cuda;

struct RegisterMockAllocator {
  RegisterMockAllocator() {
    register_device_allocator(DeviceType::CUDA, &g_mock_cuda);
  }
};
static RegisterMockAllocator s_register;

class TensorPtrDeviceTest : public ::testing::Test {
 protected:
  static void SetUpTestSuite() {
    runtime_init();
  }

  void SetUp() override {
    g_mock_cuda.allocate_count_ = 0;
    g_mock_cuda.deallocate_count_ = 0;
    g_mock_cuda.h2d_count_ = 0;
    g_mock_cuda.d2h_count_ = 0;
  }
};

TEST_F(TensorPtrDeviceTest, CpuToDeviceTensor) {
  auto cpu_tensor =
      make_tensor_ptr({2, 3}, {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f});
  auto device_tensor = clone_tensor_ptr_to_device(cpu_tensor, DeviceType::CUDA);

  EXPECT_EQ(device_tensor->dim(), 2);
  EXPECT_EQ(device_tensor->size(0), 2);
  EXPECT_EQ(device_tensor->size(1), 3);
  EXPECT_EQ(device_tensor->scalar_type(), executorch::aten::ScalarType::Float);
  EXPECT_NE(device_tensor->const_data_ptr(), nullptr);
  EXPECT_NE(device_tensor->const_data_ptr(), cpu_tensor->const_data_ptr());

#ifndef USE_ATEN_LIB
  EXPECT_EQ(
      device_tensor->unsafeGetTensorImpl()->device_type(), DeviceType::CUDA);
  EXPECT_EQ(device_tensor->unsafeGetTensorImpl()->device_index(), 0);
#endif

  EXPECT_EQ(g_mock_cuda.allocate_count_, 1);
  EXPECT_EQ(g_mock_cuda.h2d_count_, 1);
}

TEST_F(TensorPtrDeviceTest, CpuToDeviceFromRawData) {
  float data[] = {10.0f, 20.0f, 30.0f, 40.0f};
  auto cpu_tensor = make_tensor_ptr({2, 2}, data);
  auto device_tensor =
      make_tensor_ptr(cpu_tensor, {}, {}, {}, DeviceType::CUDA);

  EXPECT_EQ(device_tensor->dim(), 2);
  EXPECT_EQ(device_tensor->size(0), 2);
  EXPECT_EQ(device_tensor->size(1), 2);
  EXPECT_EQ(device_tensor->scalar_type(), executorch::aten::ScalarType::Float);
  EXPECT_NE(device_tensor->const_data_ptr(), nullptr);
  EXPECT_NE(device_tensor->const_data_ptr(), static_cast<void*>(data));

#ifndef USE_ATEN_LIB
  EXPECT_EQ(
      device_tensor->unsafeGetTensorImpl()->device_type(), DeviceType::CUDA);
#endif

  EXPECT_EQ(g_mock_cuda.allocate_count_, 1);
  EXPECT_EQ(g_mock_cuda.h2d_count_, 1);
}

#ifndef USE_ATEN_LIB
// clone_tensor_ptr_to_cpu relies on TensorImpl device metadata which is only
// available in the non-ATen (ExecuTorch portable) path.
TEST_F(TensorPtrDeviceTest, DeviceToCpuTensor) {
  auto cpu_tensor =
      make_tensor_ptr({2, 3}, {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f});
  auto device_tensor = clone_tensor_ptr_to_device(cpu_tensor, DeviceType::CUDA);
  auto result_tensor = clone_tensor_ptr_to_cpu(device_tensor);

  EXPECT_EQ(result_tensor->dim(), 2);
  EXPECT_EQ(result_tensor->size(0), 2);
  EXPECT_EQ(result_tensor->size(1), 3);
  EXPECT_EQ(result_tensor->scalar_type(), executorch::aten::ScalarType::Float);

  auto* result_data = result_tensor->const_data_ptr<float>();
  auto* original_data = cpu_tensor->const_data_ptr<float>();
  for (int i = 0; i < 6; ++i) {
    EXPECT_FLOAT_EQ(result_data[i], original_data[i]);
  }

  EXPECT_EQ(g_mock_cuda.d2h_count_, 1);
}
#endif

#ifndef USE_ATEN_LIB
TEST_F(TensorPtrDeviceTest, RoundtripCpuDeviceCpu) {
  const std::vector<float> original = {1.5f, 2.5f, 3.5f, 4.5f, 5.5f, 6.5f};
  auto cpu_tensor = make_tensor_ptr({2, 3}, original);

  auto device_tensor = clone_tensor_ptr_to_device(cpu_tensor, DeviceType::CUDA);
  auto roundtrip_tensor = clone_tensor_ptr_to_cpu(device_tensor);

  EXPECT_NE(roundtrip_tensor->const_data_ptr(), cpu_tensor->const_data_ptr());
  EXPECT_NE(
      roundtrip_tensor->const_data_ptr(), device_tensor->const_data_ptr());

  auto* result_data = roundtrip_tensor->const_data_ptr<float>();
  for (size_t i = 0; i < original.size(); ++i) {
    EXPECT_FLOAT_EQ(result_data[i], original[i]);
  }

  EXPECT_EQ(roundtrip_tensor->dim(), cpu_tensor->dim());
  EXPECT_EQ(roundtrip_tensor->size(0), cpu_tensor->size(0));
  EXPECT_EQ(roundtrip_tensor->size(1), cpu_tensor->size(1));
  EXPECT_EQ(roundtrip_tensor->scalar_type(), cpu_tensor->scalar_type());
}
#endif

#ifndef USE_ATEN_LIB
TEST_F(TensorPtrDeviceTest, RoundtripInt32) {
  auto cpu_tensor = make_tensor_ptr({4}, std::vector<int32_t>{10, 20, 30, 40});

  auto device_tensor = clone_tensor_ptr_to_device(cpu_tensor, DeviceType::CUDA);
  auto roundtrip = clone_tensor_ptr_to_cpu(device_tensor);

  EXPECT_EQ(roundtrip->scalar_type(), executorch::aten::ScalarType::Int);
  const std::vector<int32_t> expected = {10, 20, 30, 40};
  auto* data = roundtrip->const_data_ptr<int32_t>();
  for (size_t i = 0; i < expected.size(); ++i) {
    EXPECT_EQ(data[i], expected[i]);
  }
}
#endif

#ifndef USE_ATEN_LIB
TEST_F(TensorPtrDeviceTest, DeviceIndexPropagation) {
  auto cpu_tensor = make_tensor_ptr({2}, {1.0f, 2.0f});
  auto device_tensor = clone_tensor_ptr_to_device(
      cpu_tensor, DeviceType::CUDA, /*device_index=*/1);

  EXPECT_EQ(device_tensor->unsafeGetTensorImpl()->device_index(), 1);

  auto roundtrip = clone_tensor_ptr_to_cpu(device_tensor);
  EXPECT_FLOAT_EQ(roundtrip->const_data_ptr<float>()[0], 1.0f);
  EXPECT_FLOAT_EQ(roundtrip->const_data_ptr<float>()[1], 2.0f);
}
#endif

TEST_F(TensorPtrDeviceTest, DeviceMemoryCleanup) {
  {
    auto cpu_tensor = make_tensor_ptr({2}, {1.0f, 2.0f});
    auto device_tensor =
        clone_tensor_ptr_to_device(cpu_tensor, DeviceType::CUDA);
    EXPECT_EQ(g_mock_cuda.allocate_count_, 1);
    EXPECT_EQ(g_mock_cuda.deallocate_count_, 0);
  }
  EXPECT_EQ(g_mock_cuda.deallocate_count_, 1);
}

#ifndef USE_ATEN_LIB
TEST_F(TensorPtrDeviceTest, ScalarTensorRoundtrip) {
  auto cpu_tensor = make_tensor_ptr({}, {42.0f});
  auto device_tensor = clone_tensor_ptr_to_device(cpu_tensor, DeviceType::CUDA);

  EXPECT_EQ(device_tensor->dim(), 0);
  EXPECT_EQ(device_tensor->numel(), 1);

  auto roundtrip = clone_tensor_ptr_to_cpu(device_tensor);
  EXPECT_EQ(roundtrip->dim(), 0);
  EXPECT_EQ(roundtrip->numel(), 1);
  EXPECT_FLOAT_EQ(roundtrip->const_data_ptr<float>()[0], 42.0f);
}
#endif

#ifndef USE_ATEN_LIB
TEST_F(TensorPtrDeviceTest, RawDataRoundtrip) {
  float raw_data[] = {100.0f, 200.0f, 300.0f};
  auto cpu_tensor = make_tensor_ptr({3}, raw_data);
  auto device_tensor =
      make_tensor_ptr(cpu_tensor, {}, {}, {}, DeviceType::CUDA);
  auto roundtrip = clone_tensor_ptr_to_cpu(device_tensor);

  EXPECT_EQ(roundtrip->dim(), 1);
  EXPECT_EQ(roundtrip->size(0), 3);
  auto* data = roundtrip->const_data_ptr<float>();
  EXPECT_FLOAT_EQ(data[0], 100.0f);
  EXPECT_FLOAT_EQ(data[1], 200.0f);
  EXPECT_FLOAT_EQ(data[2], 300.0f);
}
#endif

TEST_F(TensorPtrDeviceTest, ErrorCpuTargetDevice) {
  auto cpu_tensor = make_tensor_ptr({2}, {1.0f, 2.0f});
  ET_EXPECT_DEATH(clone_tensor_ptr_to_device(cpu_tensor, DeviceType::CPU), "");
}

TEST_F(TensorPtrDeviceTest, ErrorNullRawData) {
  auto null_tensor = make_tensor_ptr({2, 2}, nullptr);
  ET_EXPECT_DEATH(
      make_tensor_ptr(null_tensor, {}, {}, {}, DeviceType::CUDA), "");
}

TEST_F(TensorPtrDeviceTest, ErrorNullCpuTensorData) {
  auto null_tensor = make_tensor_ptr({2, 2}, nullptr);
  ET_EXPECT_DEATH(
      clone_tensor_ptr_to_device(null_tensor, DeviceType::CUDA), "");
}

#ifndef USE_ATEN_LIB
TEST_F(TensorPtrDeviceTest, ErrorCpuTensorToCpu) {
  auto cpu_tensor = make_tensor_ptr({2}, {1.0f, 2.0f});
  ET_EXPECT_DEATH(clone_tensor_ptr_to_cpu(cpu_tensor), "");
}
#endif
