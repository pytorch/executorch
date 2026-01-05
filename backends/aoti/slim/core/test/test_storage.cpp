/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <gtest/gtest.h>

#include <executorch/backends/aoti/slim/core/Storage.h>

#ifdef CUDA_AVAILABLE
#include <cuda_runtime.h>
#endif

namespace executorch::backends::aoti::slim {

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
// DeviceTraits<CPU> Tests
// =============================================================================

TEST(DeviceTraitsCPUTest, AllocateAndFree) {
  constexpr size_t kSize = 1024;
  void* ptr = DeviceTraits<c10::DeviceType::CPU>::allocate(kSize);
  ASSERT_NE(ptr, nullptr);

  DeviceTraits<c10::DeviceType::CPU>::free(ptr);
}

TEST(DeviceTraitsCPUTest, AllocateZeroBytes) {
  void* ptr = DeviceTraits<c10::DeviceType::CPU>::allocate(0);
  DeviceTraits<c10::DeviceType::CPU>::free(ptr);
}

TEST(DeviceTraitsCPUTest, MemcpyCPUToCPU) {
  constexpr size_t kSize = 256;
  float* src = static_cast<float*>(
      DeviceTraits<c10::DeviceType::CPU>::allocate(kSize * sizeof(float)));
  float* dst = static_cast<float*>(
      DeviceTraits<c10::DeviceType::CPU>::allocate(kSize * sizeof(float)));

  for (size_t i = 0; i < kSize; ++i) {
    src[i] = static_cast<float>(i) * 1.5f;
  }

  DeviceTraits<c10::DeviceType::CPU>::memcpy(
      dst, src, kSize * sizeof(float), CPU_DEVICE, CPU_DEVICE);

  for (size_t i = 0; i < kSize; ++i) {
    EXPECT_FLOAT_EQ(dst[i], static_cast<float>(i) * 1.5f);
  }

  DeviceTraits<c10::DeviceType::CPU>::free(src);
  DeviceTraits<c10::DeviceType::CPU>::free(dst);
}

// =============================================================================
// MaybeOwningStorage Tests - Non-Owning Mode
// =============================================================================

TEST(MaybeOwningStorageNonOwningTest, ConstructNonOwning) {
  constexpr size_t kNumFloats = 64;
  constexpr size_t kNbytes = kNumFloats * sizeof(float);

  // Allocate external memory
  float* external_data = static_cast<float*>(
      DeviceTraits<c10::DeviceType::CPU>::allocate(kNbytes));

  // Initialize external data
  for (size_t i = 0; i < kNumFloats; ++i) {
    external_data[i] = static_cast<float>(i) * 2.5f;
  }

  {
    // Create non-owning storage
    MaybeOwningStorage storage(CPU_DEVICE, external_data, kNbytes);

    EXPECT_EQ(storage.data(), external_data);
    EXPECT_EQ(storage.nbytes(), kNbytes);
    EXPECT_TRUE(storage.device().is_cpu());
    EXPECT_FALSE(storage.is_owning());
    EXPECT_FALSE(storage.is_resizable());

    // Verify data is accessible through storage
    float* data = static_cast<float*>(storage.data());
    for (size_t i = 0; i < kNumFloats; ++i) {
      EXPECT_FLOAT_EQ(data[i], static_cast<float>(i) * 2.5f);
    }
  }
  // After storage goes out of scope, external_data should still be valid
  // because the storage did not own it

  // Verify external data is still accessible after storage is destroyed
  for (size_t i = 0; i < kNumFloats; ++i) {
    EXPECT_FLOAT_EQ(external_data[i], static_cast<float>(i) * 2.5f);
  }

  // Clean up external data manually
  DeviceTraits<c10::DeviceType::CPU>::free(external_data);
}

TEST(MaybeOwningStorageNonOwningTest, ModifyThroughStorage) {
  constexpr size_t kNumFloats = 32;
  constexpr size_t kNbytes = kNumFloats * sizeof(float);

  // Allocate and initialize external memory
  float* external_data = static_cast<float*>(
      DeviceTraits<c10::DeviceType::CPU>::allocate(kNbytes));
  for (size_t i = 0; i < kNumFloats; ++i) {
    external_data[i] = 0.0f;
  }

  {
    MaybeOwningStorage storage(CPU_DEVICE, external_data, kNbytes);

    // Modify data through storage
    float* data = static_cast<float*>(storage.data());
    for (size_t i = 0; i < kNumFloats; ++i) {
      data[i] = static_cast<float>(i) * 10.0f;
    }
  }

  // Verify external data was modified after storage is destroyed
  for (size_t i = 0; i < kNumFloats; ++i) {
    EXPECT_FLOAT_EQ(external_data[i], static_cast<float>(i) * 10.0f);
  }

  DeviceTraits<c10::DeviceType::CPU>::free(external_data);
}

TEST(MaybeOwningStorageNonOwningTest, MoveConstruct) {
  constexpr size_t kNbytes = 256;
  float* external_data = static_cast<float*>(
      DeviceTraits<c10::DeviceType::CPU>::allocate(kNbytes));

  MaybeOwningStorage original(CPU_DEVICE, external_data, kNbytes);

  MaybeOwningStorage moved(std::move(original));

  EXPECT_EQ(moved.data(), external_data);
  EXPECT_EQ(moved.nbytes(), kNbytes);
  EXPECT_FALSE(moved.is_owning());

  EXPECT_EQ(original.data(), nullptr);
  EXPECT_EQ(original.nbytes(), 0);
  EXPECT_FALSE(original.is_owning());

  DeviceTraits<c10::DeviceType::CPU>::free(external_data);
}

TEST(MaybeOwningStorageNonOwningTest, MoveAssign) {
  constexpr size_t kNbytes1 = 256;
  constexpr size_t kNbytes2 = 512;

  // Create two external buffers
  float* external_data1 = static_cast<float*>(
      DeviceTraits<c10::DeviceType::CPU>::allocate(kNbytes1));
  float* external_data2 = static_cast<float*>(
      DeviceTraits<c10::DeviceType::CPU>::allocate(kNbytes2));

  MaybeOwningStorage storage1(CPU_DEVICE, external_data1, kNbytes1);
  MaybeOwningStorage storage2(CPU_DEVICE, external_data2, kNbytes2);

  storage1 = std::move(storage2);

  EXPECT_EQ(storage1.data(), external_data2);
  EXPECT_EQ(storage1.nbytes(), kNbytes2);
  EXPECT_FALSE(storage1.is_owning());

  EXPECT_EQ(storage2.data(), nullptr);
  EXPECT_EQ(storage2.nbytes(), 0);
  EXPECT_FALSE(storage2.is_owning());

  // Clean up both external buffers
  DeviceTraits<c10::DeviceType::CPU>::free(external_data1);
  DeviceTraits<c10::DeviceType::CPU>::free(external_data2);
}

TEST(MaybeOwningStorageNonOwningTest, ZeroBytes) {
  // Non-owning with nullptr and zero bytes
  MaybeOwningStorage storage(CPU_DEVICE, nullptr, 0);

  EXPECT_EQ(storage.data(), nullptr);
  EXPECT_EQ(storage.nbytes(), 0);
  EXPECT_FALSE(storage.is_owning());
}

// =============================================================================
// MaybeOwningStorage Parameterized Tests (CPU and CUDA)
// =============================================================================

class MaybeOwningStorageParamTest : public testing::TestWithParam<c10::Device> {
 protected:
  c10::Device device() const {
    return GetParam();
  }
};

TEST_P(MaybeOwningStorageParamTest, ConstructOwning) {
  constexpr size_t kNbytes = 512;
  MaybeOwningStorage storage(device(), kNbytes);

  EXPECT_NE(storage.data(), nullptr);
  EXPECT_EQ(storage.nbytes(), kNbytes);
  EXPECT_EQ(storage.device().type(), device().type());
  EXPECT_TRUE(storage.is_owning());
  EXPECT_TRUE(storage.is_resizable());
}

TEST_P(MaybeOwningStorageParamTest, ConstructOwningZeroBytes) {
  MaybeOwningStorage storage(device(), 0);

  EXPECT_EQ(storage.data(), nullptr);
  EXPECT_EQ(storage.nbytes(), 0);
  EXPECT_EQ(storage.device().type(), device().type());
  EXPECT_TRUE(storage.is_owning());
}

TEST_P(MaybeOwningStorageParamTest, MoveConstruct) {
  constexpr size_t kNbytes = 256;
  MaybeOwningStorage original(device(), kNbytes);
  void* original_data = original.data();

  MaybeOwningStorage moved(std::move(original));

  EXPECT_EQ(moved.data(), original_data);
  EXPECT_EQ(moved.nbytes(), kNbytes);
  EXPECT_TRUE(moved.is_owning());
  EXPECT_EQ(moved.device().type(), device().type());

  EXPECT_EQ(original.data(), nullptr);
  EXPECT_EQ(original.nbytes(), 0);
  EXPECT_FALSE(original.is_owning());
}

TEST_P(MaybeOwningStorageParamTest, MoveAssign) {
  constexpr size_t kNbytes1 = 256;
  constexpr size_t kNbytes2 = 512;
  MaybeOwningStorage storage1(device(), kNbytes1);
  MaybeOwningStorage storage2(device(), kNbytes2);
  void* storage2_data = storage2.data();

  storage1 = std::move(storage2);

  EXPECT_EQ(storage1.data(), storage2_data);
  EXPECT_EQ(storage1.nbytes(), kNbytes2);
  EXPECT_TRUE(storage1.is_owning());

  EXPECT_EQ(storage2.data(), nullptr);
  EXPECT_EQ(storage2.nbytes(), 0);
  EXPECT_FALSE(storage2.is_owning());
}

INSTANTIATE_TEST_SUITE_P(
    DeviceTests,
    MaybeOwningStorageParamTest,
    testing::ValuesIn(getTestDevices()),
    deviceToString);

// =============================================================================
// MaybeOwningStorage CPU-Only Tests (require direct data access)
// =============================================================================

TEST(MaybeOwningStorageCPUTest, DataPersistence) {
  constexpr size_t kNumFloats = 64;
  constexpr size_t kNbytes = kNumFloats * sizeof(float);
  MaybeOwningStorage storage(CPU_DEVICE, kNbytes);

  float* data = static_cast<float*>(storage.data());
  for (size_t i = 0; i < kNumFloats; ++i) {
    data[i] = static_cast<float>(i) * 2.0f;
  }

  float* read_data = static_cast<float*>(storage.data());
  for (size_t i = 0; i < kNumFloats; ++i) {
    EXPECT_FLOAT_EQ(read_data[i], static_cast<float>(i) * 2.0f);
  }
}

TEST(MaybeOwningStorageCPUTest, Clone) {
  constexpr size_t kNumFloats = 32;
  constexpr size_t kNbytes = kNumFloats * sizeof(float);
  MaybeOwningStorage original(CPU_DEVICE, kNbytes);

  float* data = static_cast<float*>(original.data());
  for (size_t i = 0; i < kNumFloats; ++i) {
    data[i] = static_cast<float>(i) * 3.0f;
  }

  MaybeOwningStorage cloned = original.clone(CPU_DEVICE);

  EXPECT_NE(cloned.data(), original.data());
  EXPECT_EQ(cloned.nbytes(), original.nbytes());
  EXPECT_TRUE(cloned.is_owning());

  float* cloned_data = static_cast<float*>(cloned.data());
  for (size_t i = 0; i < kNumFloats; ++i) {
    EXPECT_FLOAT_EQ(cloned_data[i], static_cast<float>(i) * 3.0f);
  }

  data[0] = 999.0f;
  EXPECT_FLOAT_EQ(cloned_data[0], 0.0f);
}

TEST(MaybeOwningStorageCPUTest, CopyFunction) {
  constexpr size_t kNumFloats = 16;
  constexpr size_t kNbytes = kNumFloats * sizeof(float);
  MaybeOwningStorage src_storage(CPU_DEVICE, kNbytes);
  MaybeOwningStorage dst_storage(CPU_DEVICE, kNbytes);

  float* src_data = static_cast<float*>(src_storage.data());
  for (size_t i = 0; i < kNumFloats; ++i) {
    src_data[i] = static_cast<float>(i) + 0.5f;
  }

  dst_storage.copy_(
      dst_storage.data(), src_storage.data(), kNbytes, CPU_DEVICE);

  float* dst_data = static_cast<float*>(dst_storage.data());
  for (size_t i = 0; i < kNumFloats; ++i) {
    EXPECT_FLOAT_EQ(dst_data[i], static_cast<float>(i) + 0.5f);
  }
}

// =============================================================================
// Storage (SharedPtr<MaybeOwningStorage>) Parameterized Tests
// =============================================================================

class StorageSharedPtrParamTest : public testing::TestWithParam<c10::Device> {
 protected:
  c10::Device device() const {
    return GetParam();
  }
};

TEST_P(StorageSharedPtrParamTest, BasicUsage) {
  constexpr size_t kNbytes = 128;
  Storage storage(new MaybeOwningStorage(device(), kNbytes));

  EXPECT_NE(storage.get(), nullptr);
  EXPECT_NE(storage->data(), nullptr);
  EXPECT_EQ(storage->nbytes(), kNbytes);
  EXPECT_EQ(storage->device().type(), device().type());
  EXPECT_EQ(storage.use_count(), 1);
}

TEST_P(StorageSharedPtrParamTest, SharedOwnership) {
  constexpr size_t kNbytes = 128;
  Storage storage1(new MaybeOwningStorage(device(), kNbytes));
  void* data_ptr = storage1->data();

  Storage storage2 = storage1;

  EXPECT_EQ(storage1.use_count(), 2);
  EXPECT_EQ(storage2.use_count(), 2);
  EXPECT_EQ(storage1->data(), storage2->data());
  EXPECT_EQ(storage2->data(), data_ptr);
}

TEST_P(StorageSharedPtrParamTest, ReferenceCountDecrement) {
  constexpr size_t kNbytes = 64;
  Storage storage1(new MaybeOwningStorage(device(), kNbytes));
  EXPECT_EQ(storage1.use_count(), 1);

  {
    Storage storage2 = storage1;
    EXPECT_EQ(storage1.use_count(), 2);
  }

  EXPECT_EQ(storage1.use_count(), 1);
}

TEST_P(StorageSharedPtrParamTest, MoveSemantics) {
  constexpr size_t kNbytes = 64;
  Storage storage1(new MaybeOwningStorage(device(), kNbytes));
  void* data_ptr = storage1->data();

  Storage storage2 = std::move(storage1);

  EXPECT_EQ(storage1.get(), nullptr);
  EXPECT_EQ(storage2->data(), data_ptr);
  EXPECT_EQ(storage2.use_count(), 1);
}

TEST_P(StorageSharedPtrParamTest, MakeShared) {
  constexpr size_t kNbytes = 256;
  Storage storage = make_shared<MaybeOwningStorage>(device(), kNbytes);

  EXPECT_NE(storage.get(), nullptr);
  EXPECT_NE(storage->data(), nullptr);
  EXPECT_EQ(storage->nbytes(), kNbytes);
  EXPECT_EQ(storage.use_count(), 1);
}

INSTANTIATE_TEST_SUITE_P(
    DeviceTests,
    StorageSharedPtrParamTest,
    testing::ValuesIn(getTestDevices()),
    deviceToString);

// =============================================================================
// Storage CPU-Only Tests (require direct data access)
// =============================================================================

TEST(StorageSharedPtrCPUTest, SharedOwnershipModification) {
  constexpr size_t kNumFloats = 8;
  constexpr size_t kNbytes = kNumFloats * sizeof(float);
  Storage storage1(new MaybeOwningStorage(CPU_DEVICE, kNbytes));

  float* data = static_cast<float*>(storage1->data());
  for (size_t i = 0; i < kNumFloats; ++i) {
    data[i] = 0.0f;
  }

  Storage storage2 = storage1;

  float* data2 = static_cast<float*>(storage2->data());
  for (size_t i = 0; i < kNumFloats; ++i) {
    data2[i] = static_cast<float>(i) * 10.0f;
  }

  float* data1 = static_cast<float*>(storage1->data());
  for (size_t i = 0; i < kNumFloats; ++i) {
    EXPECT_FLOAT_EQ(data1[i], static_cast<float>(i) * 10.0f);
  }
}

#ifdef CUDA_AVAILABLE

// =============================================================================
// DeviceTraits<CUDA> Tests
// =============================================================================

TEST(DeviceTraitsCUDATest, AllocateAndFree) {
  constexpr size_t kSize = 1024;
  void* ptr =
      DeviceTraits<c10::DeviceType::CUDA>::allocate(kSize, DEFAULT_CUDA_DEVICE);
  ASSERT_NE(ptr, nullptr);

  DeviceTraits<c10::DeviceType::CUDA>::free(ptr);
}

TEST(DeviceTraitsCUDATest, AllocateZeroBytes) {
  void* ptr =
      DeviceTraits<c10::DeviceType::CUDA>::allocate(0, DEFAULT_CUDA_DEVICE);
  DeviceTraits<c10::DeviceType::CUDA>::free(ptr);
}

TEST(DeviceTraitsCUDATest, MemcpyCPUToCUDA) {
  constexpr size_t kSize = 256;
  float* cpu_src = static_cast<float*>(
      DeviceTraits<c10::DeviceType::CPU>::allocate(kSize * sizeof(float)));
  float* cuda_dst =
      static_cast<float*>(DeviceTraits<c10::DeviceType::CUDA>::allocate(
          kSize * sizeof(float), DEFAULT_CUDA_DEVICE));
  float* cpu_verify = static_cast<float*>(
      DeviceTraits<c10::DeviceType::CPU>::allocate(kSize * sizeof(float)));

  for (size_t i = 0; i < kSize; ++i) {
    cpu_src[i] = static_cast<float>(i) * 2.5f;
  }

  // Copy CPU -> CUDA
  DeviceTraits<c10::DeviceType::CUDA>::memcpy(
      cuda_dst,
      cpu_src,
      kSize * sizeof(float),
      DEFAULT_CUDA_DEVICE,
      CPU_DEVICE);

  // Copy CUDA -> CPU to verify
  DeviceTraits<c10::DeviceType::CUDA>::memcpy(
      cpu_verify,
      cuda_dst,
      kSize * sizeof(float),
      CPU_DEVICE,
      DEFAULT_CUDA_DEVICE);

  for (size_t i = 0; i < kSize; ++i) {
    EXPECT_FLOAT_EQ(cpu_verify[i], static_cast<float>(i) * 2.5f);
  }

  DeviceTraits<c10::DeviceType::CPU>::free(cpu_src);
  DeviceTraits<c10::DeviceType::CUDA>::free(cuda_dst);
  DeviceTraits<c10::DeviceType::CPU>::free(cpu_verify);
}

TEST(DeviceTraitsCUDATest, MemcpyCUDAToCPU) {
  constexpr size_t kSize = 128;
  float* cpu_src = static_cast<float*>(
      DeviceTraits<c10::DeviceType::CPU>::allocate(kSize * sizeof(float)));
  float* cuda_mem =
      static_cast<float*>(DeviceTraits<c10::DeviceType::CUDA>::allocate(
          kSize * sizeof(float), DEFAULT_CUDA_DEVICE));
  float* cpu_dst = static_cast<float*>(
      DeviceTraits<c10::DeviceType::CPU>::allocate(kSize * sizeof(float)));

  for (size_t i = 0; i < kSize; ++i) {
    cpu_src[i] = static_cast<float>(i) + 100.0f;
  }

  // Copy CPU -> CUDA
  DeviceTraits<c10::DeviceType::CUDA>::memcpy(
      cuda_mem,
      cpu_src,
      kSize * sizeof(float),
      DEFAULT_CUDA_DEVICE,
      CPU_DEVICE);

  // Copy CUDA -> CPU
  DeviceTraits<c10::DeviceType::CUDA>::memcpy(
      cpu_dst,
      cuda_mem,
      kSize * sizeof(float),
      CPU_DEVICE,
      DEFAULT_CUDA_DEVICE);

  for (size_t i = 0; i < kSize; ++i) {
    EXPECT_FLOAT_EQ(cpu_dst[i], static_cast<float>(i) + 100.0f);
  }

  DeviceTraits<c10::DeviceType::CPU>::free(cpu_src);
  DeviceTraits<c10::DeviceType::CUDA>::free(cuda_mem);
  DeviceTraits<c10::DeviceType::CPU>::free(cpu_dst);
}

TEST(DeviceTraitsCUDATest, MemcpyCUDAToCUDA) {
  constexpr size_t kSize = 64;
  float* cpu_src = static_cast<float*>(
      DeviceTraits<c10::DeviceType::CPU>::allocate(kSize * sizeof(float)));
  float* cuda_src =
      static_cast<float*>(DeviceTraits<c10::DeviceType::CUDA>::allocate(
          kSize * sizeof(float), DEFAULT_CUDA_DEVICE));
  float* cuda_dst =
      static_cast<float*>(DeviceTraits<c10::DeviceType::CUDA>::allocate(
          kSize * sizeof(float), DEFAULT_CUDA_DEVICE));
  float* cpu_verify = static_cast<float*>(
      DeviceTraits<c10::DeviceType::CPU>::allocate(kSize * sizeof(float)));

  for (size_t i = 0; i < kSize; ++i) {
    cpu_src[i] = static_cast<float>(i) * 3.0f;
  }

  // Copy CPU -> CUDA src
  DeviceTraits<c10::DeviceType::CUDA>::memcpy(
      cuda_src,
      cpu_src,
      kSize * sizeof(float),
      DEFAULT_CUDA_DEVICE,
      CPU_DEVICE);

  // Copy CUDA src -> CUDA dst
  DeviceTraits<c10::DeviceType::CUDA>::memcpy(
      cuda_dst,
      cuda_src,
      kSize * sizeof(float),
      DEFAULT_CUDA_DEVICE,
      DEFAULT_CUDA_DEVICE);

  // Copy CUDA dst -> CPU to verify
  DeviceTraits<c10::DeviceType::CUDA>::memcpy(
      cpu_verify,
      cuda_dst,
      kSize * sizeof(float),
      CPU_DEVICE,
      DEFAULT_CUDA_DEVICE);

  for (size_t i = 0; i < kSize; ++i) {
    EXPECT_FLOAT_EQ(cpu_verify[i], static_cast<float>(i) * 3.0f);
  }

  DeviceTraits<c10::DeviceType::CPU>::free(cpu_src);
  DeviceTraits<c10::DeviceType::CUDA>::free(cuda_src);
  DeviceTraits<c10::DeviceType::CUDA>::free(cuda_dst);
  DeviceTraits<c10::DeviceType::CPU>::free(cpu_verify);
}

// =============================================================================
// MaybeOwningStorage CUDA Tests
// =============================================================================

TEST(MaybeOwningStorageCUDATest, ConstructOwning) {
  constexpr size_t kNbytes = 512;
  MaybeOwningStorage storage(DEFAULT_CUDA_DEVICE, kNbytes);

  EXPECT_NE(storage.data(), nullptr);
  EXPECT_EQ(storage.nbytes(), kNbytes);
  EXPECT_TRUE(storage.device().is_cuda());
  EXPECT_FALSE(storage.device().is_cpu());
  EXPECT_TRUE(storage.is_owning());
  EXPECT_TRUE(storage.is_resizable());
  EXPECT_EQ(storage.device().index(), 0);
}

TEST(MaybeOwningStorageCUDATest, ConstructOwningZeroBytes) {
  MaybeOwningStorage storage(DEFAULT_CUDA_DEVICE, 0);

  EXPECT_EQ(storage.data(), nullptr);
  EXPECT_EQ(storage.nbytes(), 0);
  EXPECT_TRUE(storage.device().is_cuda());
  EXPECT_TRUE(storage.is_owning());
}

TEST(MaybeOwningStorageCUDATest, MoveConstruct) {
  constexpr size_t kNbytes = 256;
  MaybeOwningStorage original(DEFAULT_CUDA_DEVICE, kNbytes);
  void* original_data = original.data();

  MaybeOwningStorage moved(std::move(original));

  EXPECT_EQ(moved.data(), original_data);
  EXPECT_EQ(moved.nbytes(), kNbytes);
  EXPECT_TRUE(moved.is_owning());
  EXPECT_TRUE(moved.device().is_cuda());

  EXPECT_EQ(original.data(), nullptr);
  EXPECT_EQ(original.nbytes(), 0);
  EXPECT_FALSE(original.is_owning());
}

TEST(MaybeOwningStorageCUDATest, MoveAssign) {
  constexpr size_t kNbytes1 = 256;
  constexpr size_t kNbytes2 = 512;
  MaybeOwningStorage storage1(DEFAULT_CUDA_DEVICE, kNbytes1);
  MaybeOwningStorage storage2(DEFAULT_CUDA_DEVICE, kNbytes2);
  void* storage2_data = storage2.data();

  storage1 = std::move(storage2);

  EXPECT_EQ(storage1.data(), storage2_data);
  EXPECT_EQ(storage1.nbytes(), kNbytes2);
  EXPECT_TRUE(storage1.is_owning());

  EXPECT_EQ(storage2.data(), nullptr);
  EXPECT_EQ(storage2.nbytes(), 0);
  EXPECT_FALSE(storage2.is_owning());
}

// =============================================================================
// MaybeOwningStorage CUDA Tests - Non-Owning Mode
// =============================================================================

TEST(MaybeOwningStorageCUDANonOwningTest, ConstructNonOwning) {
  constexpr size_t kNumFloats = 64;
  constexpr size_t kNbytes = kNumFloats * sizeof(float);

  // Allocate external CUDA memory
  float* external_data =
      static_cast<float*>(DeviceTraits<c10::DeviceType::CUDA>::allocate(
          kNbytes, DEFAULT_CUDA_DEVICE));

  // Initialize external data via CPU buffer
  float* cpu_buffer = static_cast<float*>(
      DeviceTraits<c10::DeviceType::CPU>::allocate(kNbytes));
  for (size_t i = 0; i < kNumFloats; ++i) {
    cpu_buffer[i] = static_cast<float>(i) * 2.5f;
  }
  DeviceTraits<c10::DeviceType::CUDA>::memcpy(
      external_data, cpu_buffer, kNbytes, DEFAULT_CUDA_DEVICE, CPU_DEVICE);

  {
    // Create non-owning storage
    MaybeOwningStorage storage(DEFAULT_CUDA_DEVICE, external_data, kNbytes);

    EXPECT_EQ(storage.data(), external_data);
    EXPECT_EQ(storage.nbytes(), kNbytes);
    EXPECT_TRUE(storage.device().is_cuda());
    EXPECT_FALSE(storage.is_owning());
    EXPECT_FALSE(storage.is_resizable());

    // Verify data is accessible through storage by copying back to CPU
    float* verify_buffer = static_cast<float*>(
        DeviceTraits<c10::DeviceType::CPU>::allocate(kNbytes));
    DeviceTraits<c10::DeviceType::CUDA>::memcpy(
        verify_buffer,
        storage.data(),
        kNbytes,
        CPU_DEVICE,
        DEFAULT_CUDA_DEVICE);
    for (size_t i = 0; i < kNumFloats; ++i) {
      EXPECT_FLOAT_EQ(verify_buffer[i], static_cast<float>(i) * 2.5f);
    }
    DeviceTraits<c10::DeviceType::CPU>::free(verify_buffer);
  }
  // After storage goes out of scope, external_data should still be valid
  // because the storage did not own it

  // Verify external data is still accessible after storage is destroyed
  float* verify_buffer2 = static_cast<float*>(
      DeviceTraits<c10::DeviceType::CPU>::allocate(kNbytes));
  DeviceTraits<c10::DeviceType::CUDA>::memcpy(
      verify_buffer2, external_data, kNbytes, CPU_DEVICE, DEFAULT_CUDA_DEVICE);
  for (size_t i = 0; i < kNumFloats; ++i) {
    EXPECT_FLOAT_EQ(verify_buffer2[i], static_cast<float>(i) * 2.5f);
  }

  // Clean up
  DeviceTraits<c10::DeviceType::CPU>::free(verify_buffer2);
  DeviceTraits<c10::DeviceType::CPU>::free(cpu_buffer);
  DeviceTraits<c10::DeviceType::CUDA>::free(external_data);
}

TEST(MaybeOwningStorageCUDANonOwningTest, ModifyThroughStorage) {
  constexpr size_t kNumFloats = 32;
  constexpr size_t kNbytes = kNumFloats * sizeof(float);

  // Allocate external CUDA memory
  float* external_data =
      static_cast<float*>(DeviceTraits<c10::DeviceType::CUDA>::allocate(
          kNbytes, DEFAULT_CUDA_DEVICE));

  // Initialize to zeros
  float* cpu_buffer = static_cast<float*>(
      DeviceTraits<c10::DeviceType::CPU>::allocate(kNbytes));
  for (size_t i = 0; i < kNumFloats; ++i) {
    cpu_buffer[i] = 0.0f;
  }
  DeviceTraits<c10::DeviceType::CUDA>::memcpy(
      external_data, cpu_buffer, kNbytes, DEFAULT_CUDA_DEVICE, CPU_DEVICE);

  {
    MaybeOwningStorage storage(DEFAULT_CUDA_DEVICE, external_data, kNbytes);

    // Modify data through storage by copying new data
    for (size_t i = 0; i < kNumFloats; ++i) {
      cpu_buffer[i] = static_cast<float>(i) * 10.0f;
    }
    DeviceTraits<c10::DeviceType::CUDA>::memcpy(
        storage.data(), cpu_buffer, kNbytes, DEFAULT_CUDA_DEVICE, CPU_DEVICE);
  }

  // Verify external data was modified after storage is destroyed
  float* verify_buffer = static_cast<float*>(
      DeviceTraits<c10::DeviceType::CPU>::allocate(kNbytes));
  DeviceTraits<c10::DeviceType::CUDA>::memcpy(
      verify_buffer, external_data, kNbytes, CPU_DEVICE, DEFAULT_CUDA_DEVICE);
  for (size_t i = 0; i < kNumFloats; ++i) {
    EXPECT_FLOAT_EQ(verify_buffer[i], static_cast<float>(i) * 10.0f);
  }

  // Clean up
  DeviceTraits<c10::DeviceType::CPU>::free(verify_buffer);
  DeviceTraits<c10::DeviceType::CPU>::free(cpu_buffer);
  DeviceTraits<c10::DeviceType::CUDA>::free(external_data);
}

TEST(MaybeOwningStorageCUDANonOwningTest, MoveConstruct) {
  constexpr size_t kNbytes = 256;
  float* external_data =
      static_cast<float*>(DeviceTraits<c10::DeviceType::CUDA>::allocate(
          kNbytes, DEFAULT_CUDA_DEVICE));

  MaybeOwningStorage original(DEFAULT_CUDA_DEVICE, external_data, kNbytes);

  MaybeOwningStorage moved(std::move(original));

  EXPECT_EQ(moved.data(), external_data);
  EXPECT_EQ(moved.nbytes(), kNbytes);
  EXPECT_FALSE(moved.is_owning());
  EXPECT_TRUE(moved.device().is_cuda());

  EXPECT_EQ(original.data(), nullptr);
  EXPECT_EQ(original.nbytes(), 0);
  EXPECT_FALSE(original.is_owning());

  DeviceTraits<c10::DeviceType::CUDA>::free(external_data);
}

TEST(MaybeOwningStorageCUDANonOwningTest, MoveAssign) {
  constexpr size_t kNbytes1 = 256;
  constexpr size_t kNbytes2 = 512;

  // Create two external CUDA buffers
  float* external_data1 =
      static_cast<float*>(DeviceTraits<c10::DeviceType::CUDA>::allocate(
          kNbytes1, DEFAULT_CUDA_DEVICE));
  float* external_data2 =
      static_cast<float*>(DeviceTraits<c10::DeviceType::CUDA>::allocate(
          kNbytes2, DEFAULT_CUDA_DEVICE));

  MaybeOwningStorage storage1(DEFAULT_CUDA_DEVICE, external_data1, kNbytes1);
  MaybeOwningStorage storage2(DEFAULT_CUDA_DEVICE, external_data2, kNbytes2);

  storage1 = std::move(storage2);

  EXPECT_EQ(storage1.data(), external_data2);
  EXPECT_EQ(storage1.nbytes(), kNbytes2);
  EXPECT_FALSE(storage1.is_owning());
  EXPECT_TRUE(storage1.device().is_cuda());

  EXPECT_EQ(storage2.data(), nullptr);
  EXPECT_EQ(storage2.nbytes(), 0);
  EXPECT_FALSE(storage2.is_owning());

  // Clean up both external buffers
  DeviceTraits<c10::DeviceType::CUDA>::free(external_data1);
  DeviceTraits<c10::DeviceType::CUDA>::free(external_data2);
}

TEST(MaybeOwningStorageCUDANonOwningTest, ZeroBytes) {
  // Non-owning with nullptr and zero bytes
  MaybeOwningStorage storage(DEFAULT_CUDA_DEVICE, nullptr, 0);

  EXPECT_EQ(storage.data(), nullptr);
  EXPECT_EQ(storage.nbytes(), 0);
  EXPECT_FALSE(storage.is_owning());
  EXPECT_TRUE(storage.device().is_cuda());
}

// =============================================================================
// Storage (SharedPtr<MaybeOwningStorage>) CUDA Tests
// =============================================================================

TEST(StorageSharedPtrCUDATest, BasicUsage) {
  constexpr size_t kNbytes = 128;
  Storage storage(new MaybeOwningStorage(DEFAULT_CUDA_DEVICE, kNbytes));

  EXPECT_NE(storage.get(), nullptr);
  EXPECT_NE(storage->data(), nullptr);
  EXPECT_EQ(storage->nbytes(), kNbytes);
  EXPECT_TRUE(storage->device().is_cuda());
  EXPECT_EQ(storage.use_count(), 1);
}

TEST(StorageSharedPtrCUDATest, SharedOwnership) {
  constexpr size_t kNbytes = 128;
  Storage storage1(new MaybeOwningStorage(DEFAULT_CUDA_DEVICE, kNbytes));
  void* data_ptr = storage1->data();

  Storage storage2 = storage1;

  EXPECT_EQ(storage1.use_count(), 2);
  EXPECT_EQ(storage2.use_count(), 2);
  EXPECT_EQ(storage1->data(), storage2->data());
  EXPECT_EQ(storage2->data(), data_ptr);
}

TEST(StorageSharedPtrCUDATest, MoveSemantics) {
  constexpr size_t kNbytes = 64;
  Storage storage1(new MaybeOwningStorage(DEFAULT_CUDA_DEVICE, kNbytes));
  void* data_ptr = storage1->data();

  Storage storage2 = std::move(storage1);

  EXPECT_EQ(storage1.get(), nullptr);
  EXPECT_EQ(storage2->data(), data_ptr);
  EXPECT_EQ(storage2.use_count(), 1);
}

#endif // CUDA_AVAILABLE

} // namespace executorch::backends::aoti::slim
