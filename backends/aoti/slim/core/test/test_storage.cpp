/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <gtest/gtest.h>

#include <executorch/backends/aoti/slim/core/Storage.h>

namespace executorch::backends::aoti::slim {

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
// MaybeOwningStorage Tests - Owning Mode
// =============================================================================

TEST(MaybeOwningStorageTest, ConstructOwning) {
  constexpr size_t kNbytes = 512;
  MaybeOwningStorage storage(CPU_DEVICE, kNbytes);

  EXPECT_NE(storage.data(), nullptr);
  EXPECT_EQ(storage.nbytes(), kNbytes);
  EXPECT_TRUE(storage.device().is_cpu());
  EXPECT_TRUE(storage.is_owning());
  EXPECT_TRUE(storage.is_resizable());
}

TEST(MaybeOwningStorageTest, ConstructOwningZeroBytes) {
  MaybeOwningStorage storage(CPU_DEVICE, 0);

  EXPECT_EQ(storage.data(), nullptr);
  EXPECT_EQ(storage.nbytes(), 0);
  EXPECT_TRUE(storage.device().is_cpu());
  EXPECT_TRUE(storage.is_owning());
}

TEST(MaybeOwningStorageTest, DataPersistence) {
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

TEST(MaybeOwningStorageTest, MoveConstruct) {
  constexpr size_t kNbytes = 256;
  MaybeOwningStorage original(CPU_DEVICE, kNbytes);
  void* original_data = original.data();

  MaybeOwningStorage moved(std::move(original));

  EXPECT_EQ(moved.data(), original_data);
  EXPECT_EQ(moved.nbytes(), kNbytes);
  EXPECT_TRUE(moved.is_owning());

  EXPECT_EQ(original.data(), nullptr);
  EXPECT_EQ(original.nbytes(), 0);
  EXPECT_FALSE(original.is_owning());
}

TEST(MaybeOwningStorageTest, MoveAssign) {
  constexpr size_t kNbytes1 = 256;
  constexpr size_t kNbytes2 = 512;
  MaybeOwningStorage storage1(CPU_DEVICE, kNbytes1);
  MaybeOwningStorage storage2(CPU_DEVICE, kNbytes2);
  void* storage2_data = storage2.data();

  storage1 = std::move(storage2);

  EXPECT_EQ(storage1.data(), storage2_data);
  EXPECT_EQ(storage1.nbytes(), kNbytes2);
  EXPECT_TRUE(storage1.is_owning());

  EXPECT_EQ(storage2.data(), nullptr);
  EXPECT_EQ(storage2.nbytes(), 0);
  EXPECT_FALSE(storage2.is_owning());
}

TEST(MaybeOwningStorageTest, Clone) {
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

TEST(MaybeOwningStorageTest, CopyFunction) {
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
// Storage (SharedPtr<MaybeOwningStorage>) Tests
// =============================================================================

TEST(StorageSharedPtrTest, BasicUsage) {
  constexpr size_t kNbytes = 128;
  Storage storage(new MaybeOwningStorage(CPU_DEVICE, kNbytes));

  EXPECT_NE(storage.get(), nullptr);
  EXPECT_NE(storage->data(), nullptr);
  EXPECT_EQ(storage->nbytes(), kNbytes);
  EXPECT_TRUE(storage->device().is_cpu());
  EXPECT_EQ(storage.use_count(), 1);
}

TEST(StorageSharedPtrTest, SharedOwnership) {
  constexpr size_t kNbytes = 128;
  Storage storage1(new MaybeOwningStorage(CPU_DEVICE, kNbytes));
  void* data_ptr = storage1->data();

  Storage storage2 = storage1; // Copy, not reference - increments ref count

  EXPECT_EQ(storage1.use_count(), 2);
  EXPECT_EQ(storage2.use_count(), 2);
  EXPECT_EQ(storage1->data(), storage2->data());
  EXPECT_EQ(storage2->data(), data_ptr);
}

TEST(StorageSharedPtrTest, SharedOwnershipModification) {
  constexpr size_t kNumFloats = 8;
  constexpr size_t kNbytes = kNumFloats * sizeof(float);
  Storage storage1(new MaybeOwningStorage(CPU_DEVICE, kNbytes));

  float* data = static_cast<float*>(storage1->data());
  for (size_t i = 0; i < kNumFloats; ++i) {
    data[i] = 0.0f;
  }

  const Storage& storage2 = storage1;

  float* data2 = static_cast<float*>(storage2->data());
  for (size_t i = 0; i < kNumFloats; ++i) {
    data2[i] = static_cast<float>(i) * 10.0f;
  }

  float* data1 = static_cast<float*>(storage1->data());
  for (size_t i = 0; i < kNumFloats; ++i) {
    EXPECT_FLOAT_EQ(data1[i], static_cast<float>(i) * 10.0f);
  }
}

TEST(StorageSharedPtrTest, ReferenceCountDecrement) {
  constexpr size_t kNbytes = 64;
  Storage storage1(new MaybeOwningStorage(CPU_DEVICE, kNbytes));
  EXPECT_EQ(storage1.use_count(), 1);

  {
    Storage storage2 = storage1; // Copy increments ref count
    EXPECT_EQ(storage1.use_count(), 2);
  } // storage2 destroyed, ref count decrements

  EXPECT_EQ(storage1.use_count(), 1);
}

TEST(StorageSharedPtrTest, MoveSemantics) {
  constexpr size_t kNbytes = 64;
  Storage storage1(new MaybeOwningStorage(CPU_DEVICE, kNbytes));
  void* data_ptr = storage1->data();

  Storage storage2 = std::move(storage1);

  EXPECT_EQ(storage1.get(), nullptr);
  EXPECT_EQ(storage2->data(), data_ptr);
  EXPECT_EQ(storage2.use_count(), 1);
}

TEST(StorageSharedPtrTest, MakeShared) {
  constexpr size_t kNbytes = 256;
  Storage storage = make_shared<MaybeOwningStorage>(CPU_DEVICE, kNbytes);

  EXPECT_NE(storage.get(), nullptr);
  EXPECT_NE(storage->data(), nullptr);
  EXPECT_EQ(storage->nbytes(), kNbytes);
  EXPECT_EQ(storage.use_count(), 1);
}

} // namespace executorch::backends::aoti::slim
