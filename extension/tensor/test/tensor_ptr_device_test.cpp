/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/extension/tensor/tensor_ptr.h>

#include <gtest/gtest.h>

#include <array>
#include <cstdlib>
#include <cstring>

#include <executorch/runtime/core/device_allocator.h>
#include <executorch/runtime/platform/runtime.h>
#include <executorch/test/utils/DeathTest.h>

using namespace ::executorch::extension;
using namespace ::executorch::runtime;
using executorch::runtime::etensor::Device;
using executorch::runtime::etensor::DeviceIndex;
using executorch::runtime::etensor::DeviceType;

#ifndef USE_ATEN_LIB
// The device clone helpers rely on the ExecuTorch DeviceAllocator and portable
// tensor metadata APIs, which have no equivalent in USE_ATEN_LIB builds, so the
// entire test fixture is gated to the portable build.

namespace {

// A fake device allocator that uses host memory (malloc/free/memcpy) to
// simulate device memory operations, enabling end-to-end data roundtrip
// verification without requiring actual device hardware.
class FakeDeviceAllocator : public DeviceAllocator {
 public:
  explicit FakeDeviceAllocator(DeviceType type) : type_(type) {}

  Result<void*> allocate(
      size_t nbytes,
      DeviceIndex /*index*/,
      size_t /*alignment*/ = kDefaultAlignment) override {
    void* ptr = std::malloc(nbytes);
    if (!ptr) {
      return Error::MemoryAllocationFailed;
    }
    allocate_count_++;
    return ptr;
  }

  void deallocate(void* ptr, DeviceIndex /*index*/) override {
    std::free(ptr);
    deallocate_count_++;
  }

  Error copy_host_to_device(
      void* dst,
      const void* src,
      size_t nbytes,
      DeviceIndex /*index*/) override {
    std::memcpy(dst, src, nbytes);
    h2d_count_++;
    return Error::Ok;
  }

  Error copy_device_to_host(
      void* dst,
      const void* src,
      size_t nbytes,
      DeviceIndex /*index*/) override {
    std::memcpy(dst, src, nbytes);
    d2h_count_++;
    return Error::Ok;
  }

  DeviceType device_type() const override {
    return type_;
  }

  void reset_counters() {
    allocate_count_ = 0;
    deallocate_count_ = 0;
    h2d_count_ = 0;
    d2h_count_ = 0;
  }

  int allocate_count_ = 0;
  int deallocate_count_ = 0;
  int h2d_count_ = 0;
  int d2h_count_ = 0;

 private:
  DeviceType type_;
};

// Function-static singleton avoids non-const global allocator state.
FakeDeviceAllocator& fake_cuda_allocator() {
  static FakeDeviceAllocator allocator(DeviceType::CUDA);
  return allocator;
}

// One-shot registration; the constructor runs at static init time and the
// instance itself is immutable afterwards.
struct RegisterFakeAllocator {
  RegisterFakeAllocator() {
    register_device_allocator(&fake_cuda_allocator());
  }
};
const RegisterFakeAllocator s_register;

} // namespace

class TensorPtrDeviceTest : public ::testing::Test {
 protected:
  static void SetUpTestSuite() {
    runtime_init();
  }

  void SetUp() override {
    fake_cuda_allocator().reset_counters();
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

  EXPECT_EQ(
      device_tensor->unsafeGetTensorImpl()->device_type(), DeviceType::CUDA);
  EXPECT_EQ(device_tensor->unsafeGetTensorImpl()->device_index(), 0);

  EXPECT_EQ(fake_cuda_allocator().allocate_count_, 1);
  EXPECT_EQ(fake_cuda_allocator().h2d_count_, 1);
}

TEST_F(TensorPtrDeviceTest, CpuToDeviceFromRawData) {
  constexpr std::array<float, 4> data{10.0f, 20.0f, 30.0f, 40.0f};
  auto cpu_tensor = make_tensor_ptr({2, 2}, const_cast<float*>(data.data()));
  auto device_tensor = clone_tensor_ptr_to_device(cpu_tensor, DeviceType::CUDA);

  EXPECT_EQ(device_tensor->dim(), 2);
  EXPECT_EQ(device_tensor->size(0), 2);
  EXPECT_EQ(device_tensor->size(1), 2);
  EXPECT_EQ(device_tensor->scalar_type(), executorch::aten::ScalarType::Float);
  EXPECT_NE(device_tensor->const_data_ptr(), nullptr);
  EXPECT_NE(
      device_tensor->const_data_ptr(), static_cast<const void*>(data.data()));

  EXPECT_EQ(
      device_tensor->unsafeGetTensorImpl()->device_type(), DeviceType::CUDA);

  EXPECT_EQ(fake_cuda_allocator().allocate_count_, 1);
  EXPECT_EQ(fake_cuda_allocator().h2d_count_, 1);
}

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

  EXPECT_EQ(fake_cuda_allocator().d2h_count_, 1);
}

TEST_F(TensorPtrDeviceTest, DeviceToCpuPreservesShapeDynamism) {
  auto cpu_tensor = make_tensor_ptr(
      std::vector<executorch::aten::SizesType>{2},
      std::vector<float>{1.0f, 2.0f},
      {},
      {},
      executorch::aten::ScalarType::Float,
      executorch::aten::TensorShapeDynamism::STATIC);
  auto device_tensor = clone_tensor_ptr_to_device(cpu_tensor, DeviceType::CUDA);
  auto result_tensor = clone_tensor_ptr_to_cpu(device_tensor);

  EXPECT_EQ(
      result_tensor->shape_dynamism(),
      executorch::aten::TensorShapeDynamism::STATIC);
}

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

TEST_F(TensorPtrDeviceTest, DeviceIndexPropagation) {
  auto cpu_tensor = make_tensor_ptr({2}, {1.0f, 2.0f});
  auto device_tensor = clone_tensor_ptr_to_device(
      cpu_tensor, Device(DeviceType::CUDA, /*index=*/1));

  EXPECT_EQ(device_tensor->unsafeGetTensorImpl()->device_index(), 1);

  auto roundtrip = clone_tensor_ptr_to_cpu(device_tensor);
  EXPECT_FLOAT_EQ(roundtrip->const_data_ptr<float>()[0], 1.0f);
  EXPECT_FLOAT_EQ(roundtrip->const_data_ptr<float>()[1], 2.0f);
}

TEST_F(TensorPtrDeviceTest, DeviceMemoryCleanup) {
  {
    auto cpu_tensor = make_tensor_ptr({2}, {1.0f, 2.0f});
    auto device_tensor =
        clone_tensor_ptr_to_device(cpu_tensor, DeviceType::CUDA);
    EXPECT_EQ(fake_cuda_allocator().allocate_count_, 1);
    EXPECT_EQ(fake_cuda_allocator().deallocate_count_, 0);
  }
  EXPECT_EQ(fake_cuda_allocator().deallocate_count_, 1);
}

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

TEST_F(TensorPtrDeviceTest, RawDataRoundtrip) {
  constexpr std::array<float, 3> raw_data{100.0f, 200.0f, 300.0f};
  auto cpu_tensor = make_tensor_ptr({3}, const_cast<float*>(raw_data.data()));
  auto device_tensor = clone_tensor_ptr_to_device(cpu_tensor, DeviceType::CUDA);
  auto roundtrip = clone_tensor_ptr_to_cpu(device_tensor);

  EXPECT_EQ(roundtrip->dim(), 1);
  EXPECT_EQ(roundtrip->size(0), 3);
  auto* data = roundtrip->const_data_ptr<float>();
  EXPECT_FLOAT_EQ(data[0], 100.0f);
  EXPECT_FLOAT_EQ(data[1], 200.0f);
  EXPECT_FLOAT_EQ(data[2], 300.0f);
}

TEST_F(TensorPtrDeviceTest, ErrorCpuTargetDevice) {
  auto cpu_tensor = make_tensor_ptr({2}, {1.0f, 2.0f});
  ET_EXPECT_DEATH(clone_tensor_ptr_to_device(cpu_tensor, DeviceType::CPU), "");
}

TEST_F(TensorPtrDeviceTest, ErrorNullCpuTensorData) {
  auto null_tensor = make_tensor_ptr({2, 2}, nullptr);
  ET_EXPECT_DEATH(
      clone_tensor_ptr_to_device(null_tensor, DeviceType::CUDA), "");
}

TEST_F(TensorPtrDeviceTest, ErrorCpuTensorToCpu) {
  auto cpu_tensor = make_tensor_ptr({2}, {1.0f, 2.0f});
  ET_EXPECT_DEATH(clone_tensor_ptr_to_cpu(cpu_tensor), "");
}

TEST_F(TensorPtrDeviceTest, MakeTensorPtrVectorToDevice) {
  auto cpu_tensor =
      make_tensor_ptr({2, 2}, std::vector<float>{1.0f, 2.0f, 3.0f, 4.0f});
  auto device_tensor = clone_tensor_ptr_to_device(cpu_tensor, DeviceType::CUDA);

  EXPECT_EQ(device_tensor->dim(), 2);
  EXPECT_EQ(device_tensor->size(0), 2);
  EXPECT_EQ(device_tensor->size(1), 2);
  EXPECT_EQ(device_tensor->scalar_type(), executorch::aten::ScalarType::Float);
  EXPECT_EQ(
      device_tensor->unsafeGetTensorImpl()->device_type(), DeviceType::CUDA);
  EXPECT_EQ(fake_cuda_allocator().allocate_count_, 1);
  EXPECT_EQ(fake_cuda_allocator().h2d_count_, 1);

  auto roundtrip = clone_tensor_ptr_to_cpu(device_tensor);
  auto* data = roundtrip->const_data_ptr<float>();
  EXPECT_FLOAT_EQ(data[0], 1.0f);
  EXPECT_FLOAT_EQ(data[1], 2.0f);
  EXPECT_FLOAT_EQ(data[2], 3.0f);
  EXPECT_FLOAT_EQ(data[3], 4.0f);
}

TEST_F(TensorPtrDeviceTest, MakeTensorPtrRawPointerToDevice) {
  constexpr std::array<float, 3> raw{5.0f, 6.0f, 7.0f};
  auto cpu_tensor = make_tensor_ptr({3}, const_cast<float*>(raw.data()));
  auto device_tensor = clone_tensor_ptr_to_device(cpu_tensor, DeviceType::CUDA);

  EXPECT_EQ(device_tensor->dim(), 1);
  EXPECT_EQ(device_tensor->size(0), 3);
  EXPECT_EQ(
      device_tensor->unsafeGetTensorImpl()->device_type(), DeviceType::CUDA);
  EXPECT_NE(
      device_tensor->const_data_ptr(), static_cast<const void*>(raw.data()));
  EXPECT_EQ(fake_cuda_allocator().allocate_count_, 1);
  EXPECT_EQ(fake_cuda_allocator().h2d_count_, 1);

  auto roundtrip = clone_tensor_ptr_to_cpu(device_tensor);
  auto* data = roundtrip->const_data_ptr<float>();
  EXPECT_FLOAT_EQ(data[0], 5.0f);
  EXPECT_FLOAT_EQ(data[1], 6.0f);
  EXPECT_FLOAT_EQ(data[2], 7.0f);
}

TEST_F(TensorPtrDeviceTest, CloneToCpuVerifiesCpuDeviceMetadata) {
  auto cpu_tensor = make_tensor_ptr({3}, {1.0f, 2.0f, 3.0f});
  auto device_tensor = clone_tensor_ptr_to_device(cpu_tensor, DeviceType::CUDA);
  auto result = clone_tensor_ptr_to_cpu(device_tensor);

  EXPECT_EQ(result->unsafeGetTensorImpl()->device_type(), DeviceType::CPU);
  EXPECT_EQ(result->unsafeGetTensorImpl()->device_index(), 0);
}

TEST_F(TensorPtrDeviceTest, MultipleClonesFromSameSource) {
  auto cpu_tensor = make_tensor_ptr({3}, {1.0f, 2.0f, 3.0f});
  auto device1 = clone_tensor_ptr_to_device(cpu_tensor, DeviceType::CUDA);
  auto device2 = clone_tensor_ptr_to_device(cpu_tensor, DeviceType::CUDA);

  EXPECT_NE(device1->const_data_ptr(), device2->const_data_ptr());
  EXPECT_EQ(fake_cuda_allocator().allocate_count_, 2);
  EXPECT_EQ(fake_cuda_allocator().h2d_count_, 2);
}

TEST_F(TensorPtrDeviceTest, HighDimensionalTensorRoundtrip) {
  std::vector<float> data(24);
  for (size_t i = 0; i < 24; ++i) {
    data[i] = static_cast<float>(i);
  }
  auto cpu_tensor = make_tensor_ptr({2, 3, 4}, data);
  auto device_tensor = clone_tensor_ptr_to_device(cpu_tensor, DeviceType::CUDA);

  EXPECT_EQ(device_tensor->dim(), 3);
  EXPECT_EQ(device_tensor->size(0), 2);
  EXPECT_EQ(device_tensor->size(1), 3);
  EXPECT_EQ(device_tensor->size(2), 4);

  auto roundtrip = clone_tensor_ptr_to_cpu(device_tensor);
  auto* result = roundtrip->const_data_ptr<float>();
  for (size_t i = 0; i < 24; ++i) {
    EXPECT_FLOAT_EQ(result[i], static_cast<float>(i));
  }
}

TEST_F(TensorPtrDeviceTest, RoundtripDouble) {
  auto cpu_tensor = make_tensor_ptr({3}, std::vector<double>{1.1, 2.2, 3.3});
  auto device_tensor = clone_tensor_ptr_to_device(cpu_tensor, DeviceType::CUDA);
  auto roundtrip = clone_tensor_ptr_to_cpu(device_tensor);

  EXPECT_EQ(roundtrip->scalar_type(), executorch::aten::ScalarType::Double);
  auto* data = roundtrip->const_data_ptr<double>();
  EXPECT_DOUBLE_EQ(data[0], 1.1);
  EXPECT_DOUBLE_EQ(data[1], 2.2);
  EXPECT_DOUBLE_EQ(data[2], 3.3);
}

TEST_F(TensorPtrDeviceTest, RoundtripInt64) {
  auto cpu_tensor = make_tensor_ptr({3}, std::vector<int64_t>{100, 200, 300});
  auto device_tensor = clone_tensor_ptr_to_device(cpu_tensor, DeviceType::CUDA);
  auto roundtrip = clone_tensor_ptr_to_cpu(device_tensor);

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
  auto device_tensor = clone_tensor_ptr_to_device(cpu_tensor, DeviceType::CUDA);
  auto roundtrip = clone_tensor_ptr_to_cpu(device_tensor);

  auto* result = roundtrip->const_data_ptr<float>();
  for (size_t i = 0; i < n; ++i) {
    EXPECT_FLOAT_EQ(result[i], data[i]);
  }
}

#endif // USE_ATEN_LIB
