/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/extension/tensor/tensor_ptr.h>
#include <executorch/extension/tensor/tensor_ptr_maker.h>

#include <gtest/gtest.h>

#include <array>
#include <cstdlib>
#include <cstring>

#include <executorch/runtime/core/device_allocator.h>
#include <executorch/runtime/core/test/mock_cuda_allocator.h>
#include <executorch/runtime/platform/runtime.h>
#include <executorch/test/utils/DeathTest.h>

using namespace ::executorch::extension;
using namespace ::executorch::runtime;
using executorch::runtime::etensor::Device;
using executorch::runtime::etensor::DeviceType;
using executorch::runtime::testing::MockCudaAllocator;

#ifndef USE_ATEN_LIB
// The device clone helpers rely on the ExecuTorch DeviceAllocator and portable
// tensor metadata APIs, which have no equivalent in USE_ATEN_LIB builds, so the
// entire test fixture is gated to the portable build.

static MockCudaAllocator g_mock_cuda;

struct RegisterMockAllocator {
  RegisterMockAllocator() {
    register_device_allocator(&g_mock_cuda);
  }
};
const RegisterMockAllocator s_register;

class TensorPtrDeviceTest : public ::testing::Test {
 protected:
  static void SetUpTestSuite() {
    runtime_init();
  }

  void SetUp() override {
    g_mock_cuda.reset();
  }
};

TEST_F(TensorPtrDeviceTest, CpuToDeviceTensor) {
  auto cpu_tensor =
      make_tensor_ptr({2, 3}, {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f});
  auto device_tensor = clone_tensor_ptr(cpu_tensor, DeviceType::CUDA);

  EXPECT_EQ(device_tensor->dim(), 2);
  EXPECT_EQ(device_tensor->size(0), 2);
  EXPECT_EQ(device_tensor->size(1), 3);
  EXPECT_EQ(device_tensor->scalar_type(), executorch::aten::ScalarType::Float);
  EXPECT_NE(device_tensor->const_data_ptr(), nullptr);
  EXPECT_NE(device_tensor->const_data_ptr(), cpu_tensor->const_data_ptr());

  EXPECT_EQ(
      device_tensor->unsafeGetTensorImpl()->device_type(), DeviceType::CUDA);
  EXPECT_EQ(device_tensor->unsafeGetTensorImpl()->device_index(), 0);

  EXPECT_EQ(g_mock_cuda.allocate_count_, 1);
  EXPECT_EQ(g_mock_cuda.h2d_count_, 1);
}

TEST_F(TensorPtrDeviceTest, CpuToDeviceFromRawData) {
  constexpr std::array<float, 4> data{10.0f, 20.0f, 30.0f, 40.0f};
  auto cpu_tensor = make_tensor_ptr({2, 2}, const_cast<float*>(data.data()));
  auto device_tensor = clone_tensor_ptr(cpu_tensor, DeviceType::CUDA);

  EXPECT_EQ(device_tensor->dim(), 2);
  EXPECT_EQ(device_tensor->size(0), 2);
  EXPECT_EQ(device_tensor->size(1), 2);
  EXPECT_EQ(device_tensor->scalar_type(), executorch::aten::ScalarType::Float);
  EXPECT_NE(device_tensor->const_data_ptr(), nullptr);
  EXPECT_NE(
      device_tensor->const_data_ptr(), static_cast<const void*>(data.data()));

  EXPECT_EQ(
      device_tensor->unsafeGetTensorImpl()->device_type(), DeviceType::CUDA);

  EXPECT_EQ(g_mock_cuda.allocate_count_, 1);
  EXPECT_EQ(g_mock_cuda.h2d_count_, 1);
}

// Device-to-host clone needs TensorImpl device metadata, available only in the
// non-ATen (ExecuTorch portable) path.
TEST_F(TensorPtrDeviceTest, DeviceToCpuTensor) {
  auto cpu_tensor =
      make_tensor_ptr({2, 3}, {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f});
  auto device_tensor = clone_tensor_ptr(cpu_tensor, DeviceType::CUDA);
  auto result_tensor = clone_tensor_ptr(device_tensor, DeviceType::CPU);

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

TEST_F(TensorPtrDeviceTest, DeviceToCpuPreservesShapeDynamism) {
  auto cpu_tensor = make_tensor_ptr(
      std::vector<executorch::aten::SizesType>{2},
      std::vector<float>{1.0f, 2.0f},
      {},
      {},
      executorch::aten::ScalarType::Float,
      executorch::aten::TensorShapeDynamism::DYNAMIC_BOUND);
  auto device_tensor = clone_tensor_ptr(cpu_tensor, DeviceType::CUDA);
  auto result_tensor = clone_tensor_ptr(device_tensor, DeviceType::CPU);

  EXPECT_EQ(
      result_tensor->shape_dynamism(),
      executorch::aten::TensorShapeDynamism::DYNAMIC_BOUND);
}

TEST_F(TensorPtrDeviceTest, RoundtripCpuDeviceCpu) {
  const std::vector<float> original = {1.5f, 2.5f, 3.5f, 4.5f, 5.5f, 6.5f};
  auto cpu_tensor = make_tensor_ptr({2, 3}, original);

  auto device_tensor = clone_tensor_ptr(cpu_tensor, DeviceType::CUDA);
  auto roundtrip_tensor = clone_tensor_ptr(device_tensor, DeviceType::CPU);

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

TEST_F(TensorPtrDeviceTest, RoundtripInt32) {
  auto cpu_tensor = make_tensor_ptr({4}, std::vector<int32_t>{10, 20, 30, 40});

  auto device_tensor = clone_tensor_ptr(cpu_tensor, DeviceType::CUDA);
  auto roundtrip = clone_tensor_ptr(device_tensor, DeviceType::CPU);

  EXPECT_EQ(roundtrip->scalar_type(), executorch::aten::ScalarType::Int);
  const std::vector<int32_t> expected = {10, 20, 30, 40};
  auto* data = roundtrip->const_data_ptr<int32_t>();
  for (size_t i = 0; i < expected.size(); ++i) {
    EXPECT_EQ(data[i], expected[i]);
  }
}

TEST_F(TensorPtrDeviceTest, DeviceIndexPropagation) {
  auto cpu_tensor = make_tensor_ptr({2}, {1.0f, 2.0f});
  auto device_tensor =
      clone_tensor_ptr(cpu_tensor, Device(DeviceType::CUDA, /*index=*/1));

  EXPECT_EQ(device_tensor->unsafeGetTensorImpl()->device_index(), 1);
  EXPECT_EQ(g_mock_cuda.last_allocate_index_, 1);
  EXPECT_EQ(g_mock_cuda.last_h2d_index_, 1);

  auto roundtrip = clone_tensor_ptr(device_tensor, DeviceType::CPU);
  EXPECT_FLOAT_EQ(roundtrip->const_data_ptr<float>()[0], 1.0f);
  EXPECT_FLOAT_EQ(roundtrip->const_data_ptr<float>()[1], 2.0f);
  EXPECT_EQ(g_mock_cuda.last_d2h_index_, 1);

  device_tensor.reset();
  EXPECT_EQ(g_mock_cuda.last_deallocate_index_, 1);
}

TEST_F(TensorPtrDeviceTest, DeviceMemoryCleanup) {
  {
    auto cpu_tensor = make_tensor_ptr({2}, {1.0f, 2.0f});
    auto device_tensor = clone_tensor_ptr(cpu_tensor, DeviceType::CUDA);
    EXPECT_EQ(g_mock_cuda.allocate_count_, 1);
    EXPECT_EQ(g_mock_cuda.deallocate_count_, 0);
  }
  EXPECT_EQ(g_mock_cuda.deallocate_count_, 1);
  EXPECT_NE(g_mock_cuda.last_allocate_ptr_, nullptr);
  EXPECT_EQ(g_mock_cuda.last_deallocate_ptr_, g_mock_cuda.last_allocate_ptr_);
}

TEST_F(TensorPtrDeviceTest, ScalarTensorRoundtrip) {
  auto cpu_tensor = make_tensor_ptr({}, {42.0f});
  auto device_tensor = clone_tensor_ptr(cpu_tensor, DeviceType::CUDA);

  EXPECT_EQ(device_tensor->dim(), 0);
  EXPECT_EQ(device_tensor->numel(), 1);

  auto roundtrip = clone_tensor_ptr(device_tensor, DeviceType::CPU);
  EXPECT_EQ(roundtrip->dim(), 0);
  EXPECT_EQ(roundtrip->numel(), 1);
  EXPECT_FLOAT_EQ(roundtrip->const_data_ptr<float>()[0], 42.0f);
}

TEST_F(TensorPtrDeviceTest, RawDataRoundtrip) {
  constexpr std::array<float, 3> raw_data{100.0f, 200.0f, 300.0f};
  auto cpu_tensor = make_tensor_ptr({3}, const_cast<float*>(raw_data.data()));
  auto device_tensor = clone_tensor_ptr(cpu_tensor, DeviceType::CUDA);
  auto roundtrip = clone_tensor_ptr(device_tensor, DeviceType::CPU);

  EXPECT_EQ(roundtrip->dim(), 1);
  EXPECT_EQ(roundtrip->size(0), 3);
  auto* data = roundtrip->const_data_ptr<float>();
  EXPECT_FLOAT_EQ(data[0], 100.0f);
  EXPECT_FLOAT_EQ(data[1], 200.0f);
  EXPECT_FLOAT_EQ(data[2], 300.0f);
}

TEST_F(TensorPtrDeviceTest, RoundtripKeepsChannelsLastLayout) {
  const std::vector<executorch::aten::DimOrderType> dim_order = {0, 2, 3, 1};
  const std::vector<executorch::aten::StridesType> strides = {12, 1, 6, 3};
  auto cpu_tensor = make_tensor_ptr(
      {1, 3, 2, 2}, std::vector<float>(12, 1.0f), dim_order, strides);

  auto device_tensor = clone_tensor_ptr(cpu_tensor, DeviceType::CUDA);
  auto roundtrip = clone_tensor_ptr(device_tensor, DeviceType::CPU);

  for (size_t i = 0; i < dim_order.size(); ++i) {
    EXPECT_EQ(device_tensor->dim_order()[i], dim_order[i]);
    EXPECT_EQ(device_tensor->strides()[i], strides[i]);
    EXPECT_EQ(roundtrip->dim_order()[i], dim_order[i]);
    EXPECT_EQ(roundtrip->strides()[i], strides[i]);
  }
}

TEST_F(TensorPtrDeviceTest, CpuToCpuCopiesWithoutTheAllocator) {
  auto cpu_tensor = make_tensor_ptr({2}, {1.0f, 2.0f});
  auto copy = clone_tensor_ptr(cpu_tensor, DeviceType::CPU);

  EXPECT_TRUE(copy->device().is_cpu());
  EXPECT_NE(copy->const_data_ptr(), cpu_tensor->const_data_ptr());
  EXPECT_FLOAT_EQ(copy->const_data_ptr<float>()[0], 1.0f);
  EXPECT_FLOAT_EQ(copy->const_data_ptr<float>()[1], 2.0f);
  EXPECT_EQ(g_mock_cuda.allocate_count_, 0);
  EXPECT_EQ(g_mock_cuda.h2d_count_, 0);
  EXPECT_EQ(g_mock_cuda.d2h_count_, 0);
}

TEST_F(TensorPtrDeviceTest, CpuCloneReportsOneDeviceWhicheverBranchItTakes) {
  auto with_data = make_tensor_ptr({2}, {1.0f, 2.0f});
  auto without_data = make_tensor_ptr({2}, nullptr);
  const auto target = Device(DeviceType::CPU, /*index=*/3);

  auto copied = clone_tensor_ptr(with_data, target);
  auto copied_null = clone_tensor_ptr(without_data, target);
  auto copied_no_target = clone_tensor_ptr(with_data);

  EXPECT_EQ(copied->device(), Device(DeviceType::CPU, /*index=*/0));
  EXPECT_EQ(copied_null->device(), copied->device());
  EXPECT_EQ(copied_no_target->device(), copied->device());
  EXPECT_NE(copied->const_data_ptr(), with_data->const_data_ptr());
  EXPECT_FLOAT_EQ(copied->const_data_ptr<float>()[1], 2.0f);
  EXPECT_EQ(copied_null->const_data_ptr(), nullptr);
  EXPECT_EQ(g_mock_cuda.allocate_count_, 0);
}

TEST_F(TensorPtrDeviceTest, ErrorNullCpuTensorData) {
  auto null_tensor = make_tensor_ptr({2, 2}, nullptr);
  ET_EXPECT_DEATH(
      clone_tensor_ptr(null_tensor, DeviceType::CUDA),
      "Source tensor has no data");
}

TEST_F(TensorPtrDeviceTest, ErrorDeviceToDevice) {
  auto cpu_tensor = make_tensor_ptr({2}, {1.0f, 2.0f});
  auto device_tensor = clone_tensor_ptr(cpu_tensor, DeviceType::CUDA);
  ET_EXPECT_DEATH(
      clone_tensor_ptr(device_tensor, Device(DeviceType::CUDA, /*index=*/1)),
      "can only be copied to CPU");
}

TEST_F(TensorPtrDeviceTest, ErrorNoTargetOnADeviceTensor) {
  auto cpu_tensor = make_tensor_ptr({2}, {1.0f, 2.0f});
  auto device_tensor = clone_tensor_ptr(cpu_tensor, DeviceType::CUDA);
  ET_EXPECT_DEATH(clone_tensor_ptr(device_tensor), "can only be copied to CPU");
}

TEST_F(TensorPtrDeviceTest, ErrorConvertingADeviceTensor) {
  auto cpu_tensor = make_tensor_ptr({2}, {1.0f, 2.0f});
  auto device_tensor = clone_tensor_ptr(cpu_tensor, DeviceType::CUDA);
  ET_EXPECT_DEATH(
      convert_tensor_ptr(device_tensor, executorch::aten::ScalarType::Double),
      "only supports CPU tensors");
}

TEST_F(TensorPtrDeviceTest, ErrorAllocatingDeviceMemory) {
  auto cpu_tensor = make_tensor_ptr({2}, {1.0f, 2.0f});
  g_mock_cuda.fail_allocations_ = true;
  ET_EXPECT_DEATH(
      clone_tensor_ptr(cpu_tensor, DeviceType::CUDA),
      "Failed to allocate device memory");
}

#ifdef __GNUC__
// Disable -Wdeprecated-declarations, as some builds use 'Werror'. This test
// exists to keep the deprecated spelling working until it is removed.
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
#endif

TEST_F(TensorPtrDeviceTest, DeprecatedCloneToStillCopiesToTheDevice) {
  auto cpu_tensor = make_tensor_ptr({2}, {1.0f, 2.0f});
  auto device_tensor =
      clone_tensor_ptr_to(cpu_tensor, Device(DeviceType::CUDA, /*index=*/1));

  ASSERT_NE(device_tensor, nullptr);
  EXPECT_EQ(
      device_tensor->unsafeGetTensorImpl()->device_type(), DeviceType::CUDA);
  EXPECT_EQ(device_tensor->unsafeGetTensorImpl()->device_index(), 1);
  EXPECT_EQ(g_mock_cuda.allocate_count_, 1);
  EXPECT_EQ(g_mock_cuda.h2d_count_, 1);

  auto roundtrip = clone_tensor_ptr(device_tensor, DeviceType::CPU);
  EXPECT_FLOAT_EQ(roundtrip->const_data_ptr<float>()[1], 2.0f);
}

TEST_F(TensorPtrDeviceTest, DeprecatedCloneToNowCopiesCpuToCpu) {
  auto cpu_tensor = make_tensor_ptr({2}, {1.0f, 2.0f});
  auto copy = clone_tensor_ptr_to(cpu_tensor, DeviceType::CPU);

  ASSERT_NE(copy, nullptr);
  EXPECT_TRUE(copy->device().is_cpu());
  EXPECT_NE(copy->const_data_ptr(), cpu_tensor->const_data_ptr());
  EXPECT_FLOAT_EQ(copy->const_data_ptr<float>()[1], 2.0f);
  EXPECT_EQ(g_mock_cuda.allocate_count_, 0);
}

#ifdef __GNUC__
#pragma GCC diagnostic pop
#endif

TEST_F(TensorPtrDeviceTest, MakeTensorPtrVectorToDevice) {
  auto cpu_tensor =
      make_tensor_ptr({2, 2}, std::vector<float>{1.0f, 2.0f, 3.0f, 4.0f});
  auto device_tensor = clone_tensor_ptr(cpu_tensor, DeviceType::CUDA);

  EXPECT_EQ(device_tensor->dim(), 2);
  EXPECT_EQ(device_tensor->size(0), 2);
  EXPECT_EQ(device_tensor->size(1), 2);
  EXPECT_EQ(device_tensor->scalar_type(), executorch::aten::ScalarType::Float);
  EXPECT_EQ(
      device_tensor->unsafeGetTensorImpl()->device_type(), DeviceType::CUDA);
  EXPECT_EQ(g_mock_cuda.allocate_count_, 1);
  EXPECT_EQ(g_mock_cuda.h2d_count_, 1);

  auto roundtrip = clone_tensor_ptr(device_tensor, DeviceType::CPU);
  auto* data = roundtrip->const_data_ptr<float>();
  EXPECT_FLOAT_EQ(data[0], 1.0f);
  EXPECT_FLOAT_EQ(data[1], 2.0f);
  EXPECT_FLOAT_EQ(data[2], 3.0f);
  EXPECT_FLOAT_EQ(data[3], 4.0f);
}

TEST_F(TensorPtrDeviceTest, MakeTensorPtrRawPointerToDevice) {
  constexpr std::array<float, 3> raw{5.0f, 6.0f, 7.0f};
  auto cpu_tensor = make_tensor_ptr({3}, const_cast<float*>(raw.data()));
  auto device_tensor = clone_tensor_ptr(cpu_tensor, DeviceType::CUDA);

  EXPECT_EQ(device_tensor->dim(), 1);
  EXPECT_EQ(device_tensor->size(0), 3);
  EXPECT_EQ(
      device_tensor->unsafeGetTensorImpl()->device_type(), DeviceType::CUDA);
  EXPECT_NE(
      device_tensor->const_data_ptr(), static_cast<const void*>(raw.data()));
  EXPECT_EQ(g_mock_cuda.allocate_count_, 1);
  EXPECT_EQ(g_mock_cuda.h2d_count_, 1);

  auto roundtrip = clone_tensor_ptr(device_tensor, DeviceType::CPU);
  auto* data = roundtrip->const_data_ptr<float>();
  EXPECT_FLOAT_EQ(data[0], 5.0f);
  EXPECT_FLOAT_EQ(data[1], 6.0f);
  EXPECT_FLOAT_EQ(data[2], 7.0f);
}

TEST_F(TensorPtrDeviceTest, CloneToCpuVerifiesCpuDeviceMetadata) {
  auto cpu_tensor = make_tensor_ptr({3}, {1.0f, 2.0f, 3.0f});
  auto device_tensor = clone_tensor_ptr(cpu_tensor, DeviceType::CUDA);
  auto result = clone_tensor_ptr(device_tensor, DeviceType::CPU);

  EXPECT_EQ(result->unsafeGetTensorImpl()->device_type(), DeviceType::CPU);
  EXPECT_EQ(result->unsafeGetTensorImpl()->device_index(), 0);
}

TEST_F(TensorPtrDeviceTest, MultipleClonesFromSameSource) {
  auto cpu_tensor = make_tensor_ptr({3}, {1.0f, 2.0f, 3.0f});
  auto device1 = clone_tensor_ptr(cpu_tensor, DeviceType::CUDA);
  auto device2 = clone_tensor_ptr(cpu_tensor, DeviceType::CUDA);

  EXPECT_NE(device1->const_data_ptr(), device2->const_data_ptr());
  EXPECT_EQ(g_mock_cuda.allocate_count_, 2);
  EXPECT_EQ(g_mock_cuda.h2d_count_, 2);
}

TEST_F(TensorPtrDeviceTest, HighDimensionalTensorRoundtrip) {
  std::vector<float> data(24);
  for (size_t i = 0; i < 24; ++i) {
    data[i] = static_cast<float>(i);
  }
  auto cpu_tensor = make_tensor_ptr({2, 3, 4}, data);
  auto device_tensor = clone_tensor_ptr(cpu_tensor, DeviceType::CUDA);

  EXPECT_EQ(device_tensor->dim(), 3);
  EXPECT_EQ(device_tensor->size(0), 2);
  EXPECT_EQ(device_tensor->size(1), 3);
  EXPECT_EQ(device_tensor->size(2), 4);

  auto roundtrip = clone_tensor_ptr(device_tensor, DeviceType::CPU);
  auto* result = roundtrip->const_data_ptr<float>();
  for (size_t i = 0; i < 24; ++i) {
    EXPECT_FLOAT_EQ(result[i], static_cast<float>(i));
  }
}

TEST_F(TensorPtrDeviceTest, RoundtripDouble) {
  auto cpu_tensor = make_tensor_ptr({3}, std::vector<double>{1.1, 2.2, 3.3});
  auto device_tensor = clone_tensor_ptr(cpu_tensor, DeviceType::CUDA);
  auto roundtrip = clone_tensor_ptr(device_tensor, DeviceType::CPU);

  EXPECT_EQ(roundtrip->scalar_type(), executorch::aten::ScalarType::Double);
  auto* data = roundtrip->const_data_ptr<double>();
  EXPECT_DOUBLE_EQ(data[0], 1.1);
  EXPECT_DOUBLE_EQ(data[1], 2.2);
  EXPECT_DOUBLE_EQ(data[2], 3.3);
}

TEST_F(TensorPtrDeviceTest, RoundtripInt64) {
  auto cpu_tensor = make_tensor_ptr({3}, std::vector<int64_t>{100, 200, 300});
  auto device_tensor = clone_tensor_ptr(cpu_tensor, DeviceType::CUDA);
  auto roundtrip = clone_tensor_ptr(device_tensor, DeviceType::CPU);

  EXPECT_EQ(roundtrip->scalar_type(), executorch::aten::ScalarType::Long);
  auto* data = roundtrip->const_data_ptr<int64_t>();
  EXPECT_EQ(data[0], 100);
  EXPECT_EQ(data[1], 200);
  EXPECT_EQ(data[2], 300);
}

TEST_F(TensorPtrDeviceTest, LargeTensorRoundtrip) {
  const size_t n = 10000;
  std::vector<float> data(n);
  for (size_t i = 0; i < n; ++i) {
    data[i] = static_cast<float>(i) * 0.1f;
  }
  auto cpu_tensor = make_tensor_ptr({static_cast<int32_t>(n)}, data);
  auto device_tensor = clone_tensor_ptr(cpu_tensor, DeviceType::CUDA);
  auto roundtrip = clone_tensor_ptr(device_tensor, DeviceType::CPU);

  auto* result = roundtrip->const_data_ptr<float>();
  for (size_t i = 0; i < n; ++i) {
    EXPECT_FLOAT_EQ(result[i], data[i]);
  }
}

// The `device` argument sits immediately after `type` in the `make_tensor_ptr`
// and `from_blob` overloads that take a raw pointer, so tagging a buffer that
// already lives on an accelerator never needs the trailing arguments spelled
// out. `for_blob` is the exception: it carries no `device` parameter and takes
// the device through its `.device()` builder method instead. Tagging is
// metadata only: it must not allocate or copy.

TEST_F(TensorPtrDeviceTest, MakeTensorPtrTagsDeviceAfterType) {
  std::array<float, 4> raw{1.0f, 2.0f, 3.0f, 4.0f};
  auto tensor = make_tensor_ptr(
      {2, 2},
      raw.data(),
      executorch::aten::ScalarType::Float,
      DeviceType::CUDA);

  EXPECT_EQ(tensor->unsafeGetTensorImpl()->device_type(), DeviceType::CUDA);
  EXPECT_EQ(tensor->unsafeGetTensorImpl()->device_index(), 0);
  EXPECT_EQ(tensor->const_data_ptr(), raw.data());
  EXPECT_EQ(g_mock_cuda.allocate_count_, 0);
  EXPECT_EQ(g_mock_cuda.h2d_count_, 0);
}

TEST_F(TensorPtrDeviceTest, MakeTensorPtrDefaultsToCpu) {
  std::array<float, 4> raw{1.0f, 2.0f, 3.0f, 4.0f};
  auto bare = make_tensor_ptr({2, 2}, raw.data());
  auto typed =
      make_tensor_ptr({2, 2}, raw.data(), executorch::aten::ScalarType::Float);

  EXPECT_EQ(bare->unsafeGetTensorImpl()->device_type(), DeviceType::CPU);
  EXPECT_EQ(typed->unsafeGetTensorImpl()->device_type(), DeviceType::CPU);
}

TEST_F(TensorPtrDeviceTest, MakeTensorPtrKeepsDynamismAndDeleterAfterDevice) {
  auto* raw = new float[4]{1.0f, 2.0f, 3.0f, 4.0f};
  bool deleted = false;
  {
    auto tensor = make_tensor_ptr(
        {2, 2},
        raw,
        executorch::aten::ScalarType::Float,
        Device(DeviceType::CUDA, 1),
        executorch::aten::TensorShapeDynamism::STATIC,
        [&deleted](void* p) {
          deleted = true;
          delete[] static_cast<float*>(p);
        });

    EXPECT_EQ(tensor->unsafeGetTensorImpl()->device_type(), DeviceType::CUDA);
    EXPECT_EQ(tensor->unsafeGetTensorImpl()->device_index(), 1);
    EXPECT_EQ(
        tensor->shape_dynamism(),
        executorch::aten::TensorShapeDynamism::STATIC);
    EXPECT_FALSE(deleted);
  }
  EXPECT_TRUE(deleted);
}

TEST_F(TensorPtrDeviceTest, MakeTensorPtrPrimaryTagsDeviceAfterType) {
  std::array<float, 4> raw{1.0f, 2.0f, 3.0f, 4.0f};
  auto tensor = make_tensor_ptr(
      {2, 2},
      raw.data(),
      {0, 1},
      {2, 1},
      executorch::aten::ScalarType::Float,
      DeviceType::CUDA);

  EXPECT_EQ(tensor->unsafeGetTensorImpl()->device_type(), DeviceType::CUDA);
  EXPECT_EQ(tensor->const_data_ptr(), raw.data());
  EXPECT_EQ(g_mock_cuda.allocate_count_, 0);
}

TEST_F(TensorPtrDeviceTest, FromBlobTagsDeviceAfterType) {
  std::array<float, 4> raw{1.0f, 2.0f, 3.0f, 4.0f};
  auto tensor = from_blob(
      raw.data(),
      {2, 2},
      executorch::aten::ScalarType::Float,
      DeviceType::CUDA);

  EXPECT_EQ(tensor->unsafeGetTensorImpl()->device_type(), DeviceType::CUDA);
  EXPECT_EQ(tensor->const_data_ptr(), raw.data());
  EXPECT_EQ(g_mock_cuda.allocate_count_, 0);
}

TEST_F(TensorPtrDeviceTest, FromBlobDefaultsToCpu) {
  std::array<float, 4> raw{1.0f, 2.0f, 3.0f, 4.0f};
  auto tensor = from_blob(raw.data(), {2, 2});

  EXPECT_EQ(tensor->unsafeGetTensorImpl()->device_type(), DeviceType::CPU);
}

TEST_F(TensorPtrDeviceTest, FromBlobWithStridesTagsDevice) {
  std::array<float, 4> raw{1.0f, 2.0f, 3.0f, 4.0f};
  auto tensor = from_blob(
      raw.data(),
      {2, 2},
      {2, 1},
      executorch::aten::ScalarType::Float,
      DeviceType::CUDA);

  EXPECT_EQ(tensor->unsafeGetTensorImpl()->device_type(), DeviceType::CUDA);
  EXPECT_EQ(tensor->strides()[0], 2);
  EXPECT_EQ(tensor->strides()[1], 1);
}

TEST_F(TensorPtrDeviceTest, FromBlobRunsDeleterAfterDevice) {
  auto* raw = new float[4]{1.0f, 2.0f, 3.0f, 4.0f};
  bool deleted = false;
  {
    auto tensor = from_blob(
        raw,
        {2, 2},
        executorch::aten::ScalarType::Float,
        DeviceType::CUDA,
        [&deleted](void* p) {
          deleted = true;
          delete[] static_cast<float*>(p);
        });

    EXPECT_EQ(tensor->unsafeGetTensorImpl()->device_type(), DeviceType::CUDA);
    EXPECT_FALSE(deleted);
  }
  EXPECT_TRUE(deleted);
}

TEST_F(TensorPtrDeviceTest, FromBlobWithStridesRunsDeleterAfterDevice) {
  auto* raw = new float[4]{1.0f, 2.0f, 3.0f, 4.0f};
  bool deleted = false;
  {
    auto tensor = from_blob(
        raw,
        {2, 2},
        {2, 1},
        executorch::aten::ScalarType::Float,
        DeviceType::CUDA,
        [&deleted](void* p) {
          deleted = true;
          delete[] static_cast<float*>(p);
        },
        executorch::aten::TensorShapeDynamism::STATIC);

    EXPECT_EQ(tensor->unsafeGetTensorImpl()->device_type(), DeviceType::CUDA);
    EXPECT_EQ(
        tensor->shape_dynamism(),
        executorch::aten::TensorShapeDynamism::STATIC);
  }
  EXPECT_TRUE(deleted);
}

TEST_F(TensorPtrDeviceTest, ForBlobBuilderMatchesFromBlob) {
  std::array<float, 4> raw{1.0f, 2.0f, 3.0f, 4.0f};
  auto built =
      for_blob(raw.data(), {2, 2}).device(DeviceType::CUDA).make_tensor_ptr();
  auto direct = from_blob(
      raw.data(),
      {2, 2},
      executorch::aten::ScalarType::Float,
      DeviceType::CUDA);

  // Assert the device absolutely, not just that the two agree. Comparing them
  // to each other alone would still pass if both paths dropped the device.
  EXPECT_EQ(built->unsafeGetTensorImpl()->device_type(), DeviceType::CUDA);
  EXPECT_EQ(direct->unsafeGetTensorImpl()->device_type(), DeviceType::CUDA);
  EXPECT_EQ(
      built->unsafeGetTensorImpl()->device_type(),
      direct->unsafeGetTensorImpl()->device_type());
  EXPECT_EQ(built->const_data_ptr(), direct->const_data_ptr());
}

// A tensor built as a view of another tensor has to inherit where that data
// lives. Aliasing device memory and reporting it as CPU would hand the delegate
// a pointer it refuses, or worse, one a host memcpy would try to touch.
TEST_F(TensorPtrDeviceTest, ViewOfDeviceTensorInheritsDeviceAndIndex) {
  std::array<float, 4> raw{1.0f, 2.0f, 3.0f, 4.0f};
  auto source = make_tensor_ptr(
      {2, 2},
      raw.data(),
      executorch::aten::ScalarType::Float,
      Device(DeviceType::CUDA, 1));

  auto view = make_tensor_ptr(*source);
  auto reshaped = make_tensor_ptr(*source, {4});

  EXPECT_EQ(view->unsafeGetTensorImpl()->device_type(), DeviceType::CUDA);
  EXPECT_EQ(view->unsafeGetTensorImpl()->device_index(), 1);
  EXPECT_EQ(reshaped->unsafeGetTensorImpl()->device_type(), DeviceType::CUDA);
  EXPECT_EQ(reshaped->unsafeGetTensorImpl()->device_index(), 1);
  EXPECT_EQ(view->const_data_ptr(), raw.data());
  EXPECT_EQ(reshaped->const_data_ptr(), raw.data());
  EXPECT_EQ(g_mock_cuda.allocate_count_, 0);
  EXPECT_EQ(g_mock_cuda.h2d_count_, 0);
}

// Every `from_blob` overload has to carry the device index through, not just
// the device type. A delegate picks its stream from the index.
TEST_F(TensorPtrDeviceTest, FromBlobPreservesNonZeroDeviceIndex) {
  std::array<float, 4> raw{1.0f, 2.0f, 3.0f, 4.0f};
  const auto device = Device(DeviceType::CUDA, 3);

  auto plain = from_blob(
      raw.data(), {2, 2}, executorch::aten::ScalarType::Float, device);
  auto strided = from_blob(
      raw.data(), {2, 2}, {2, 1}, executorch::aten::ScalarType::Float, device);

  EXPECT_EQ(plain->unsafeGetTensorImpl()->device_index(), 3);
  EXPECT_EQ(strided->unsafeGetTensorImpl()->device_index(), 3);

  bool plain_deleted = false;
  bool strided_deleted = false;
  {
    auto with_deleter = from_blob(
        raw.data(),
        {2, 2},
        executorch::aten::ScalarType::Float,
        device,
        [&plain_deleted](void*) { plain_deleted = true; });
    auto strided_with_deleter = from_blob(
        raw.data(),
        {2, 2},
        {2, 1},
        executorch::aten::ScalarType::Float,
        device,
        [&strided_deleted](void*) { strided_deleted = true; });

    EXPECT_EQ(with_deleter->unsafeGetTensorImpl()->device_index(), 3);
    EXPECT_EQ(strided_with_deleter->unsafeGetTensorImpl()->device_index(), 3);
    EXPECT_FALSE(plain_deleted);
    EXPECT_FALSE(strided_deleted);
  }
  EXPECT_TRUE(plain_deleted);
  EXPECT_TRUE(strided_deleted);
}

// `FromBlobDefaultsToCpu` above covers the bare overload. The strided overload
// has its own defaulted `device` parameter, and nothing else exercises it
// without naming a device explicitly.
TEST_F(TensorPtrDeviceTest, FromBlobWithStridesDefaultsToCpu) {
  std::array<float, 4> raw{1.0f, 2.0f, 3.0f, 4.0f};
  auto strided = from_blob(
      raw.data(), {2, 2}, {2, 1}, executorch::aten::ScalarType::Float);

  EXPECT_EQ(strided->unsafeGetTensorImpl()->device_type(), DeviceType::CPU);
}

#endif // USE_ATEN_LIB
