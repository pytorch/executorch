/*
 * Copyright (c) 2025 MediaTek Inc.
 *
 * Licensed under the BSD License (the "License"); you may not use this file
 * except in compliance with the License. See the license file in the root
 * directory of this source tree for more details.
 */

#pragma once

#include "NeuronExecutor.h"
#include "NeuronLog.h"
#include "api/NeuronAdapter.h"

#include <android/hardware_buffer.h>

#include <executorch/runtime/core/memory_allocator.h>

#include <dlfcn.h>
#include <iostream>
#include <map>
#include <memory>
#include <mutex>
#include <new>

#define GET_NEURON_ALLOCATOR \
  ::executorch::backends::neuron::BufferAllocator::GetInstance()

// Define the types for the function pointers
typedef void* (*CreateFunc)();
typedef void* (*AllocateFunc)(void*, size_t);
typedef bool (*RemoveFunc)(void*, void*);
typedef const void* (*FindFunc)(void*, void*);
typedef void (*ClearFunc)(void*);

static CreateFunc create_func = nullptr;
static AllocateFunc allocate_func = nullptr;
static RemoveFunc remove_func = nullptr;
static FindFunc find_func = nullptr;
static ClearFunc clear_func = nullptr;
static void* handle = nullptr;
static void* allocatorHandle = nullptr;

namespace executorch {
namespace backends {
namespace neuron {

struct BufferDeleter {
  void operator()(AHardwareBuffer* buffer) {
    if (buffer != nullptr) {
      AHardwareBuffer_unlock(buffer, nullptr);
      AHardwareBuffer_release(buffer);
    }
  }
};

class MemoryUnit {
 public:
  static std::unique_ptr<MemoryUnit> Create(size_t size) {
    auto obj = std::unique_ptr<MemoryUnit>(new (std::nothrow) MemoryUnit(size));
    return (obj && (obj->Allocate() == NEURON_NO_ERROR)) ? std::move(obj)
                                                         : nullptr;
  }

  ~MemoryUnit() {
    mNeuronMemory.reset();
    mAhwb.reset();
  }

  size_t GetSize() const {
    return mSize;
  }

  void* GetAddress() const {
    return mAddress;
  }

  NeuronMemory* GetNeuronMemory() const {
    return mNeuronMemory.get();
  }

 private:
  explicit MemoryUnit(size_t size) : mSize(size) {}

  int Allocate() {
    AHardwareBuffer_Desc iDesc{
        .width = static_cast<uint32_t>(mSize),
        .height = 1,
        .layers = 1,
        .format = AHARDWAREBUFFER_FORMAT_BLOB,
        .usage = mAhwbType,
        .stride = static_cast<uint32_t>(mSize),
    };
    AHardwareBuffer* Abuffer = nullptr;
    AHardwareBuffer_allocate(&iDesc, &Abuffer);
    CHECK_VALID_PTR(Abuffer);
    mAhwb = std::unique_ptr<AHardwareBuffer, BufferDeleter>(Abuffer);

    NeuronMemory* memory = nullptr;
    NeuronMemory_createFromAHardwareBuffer(Abuffer, &memory);
    CHECK_VALID_PTR(memory);
    mNeuronMemory = std::
        unique_ptr<NeuronMemory, executorch::backends::neuron::NeuronDeleter>(
            memory);

    AHardwareBuffer_lock(Abuffer, mAhwbType, -1, nullptr, &mAddress);
    CHECK_VALID_PTR(mAddress);
    return NEURON_NO_ERROR;
  }

 private:
  std::unique_ptr<NeuronMemory, executorch::backends::neuron::NeuronDeleter>
      mNeuronMemory;

  std::unique_ptr<AHardwareBuffer, BufferDeleter> mAhwb;

  uint64_t mAhwbType = AHARDWAREBUFFER_USAGE_CPU_READ_OFTEN |
      AHARDWAREBUFFER_USAGE_CPU_WRITE_OFTEN;

  void* mAddress = nullptr;

  size_t mSize = 0;
};

class BufferAllocator : public executorch::runtime::MemoryAllocator {
 public:
  static BufferAllocator& GetInstance();

  void* Allocate(size_t size);

  void* allocate(size_t size, size_t alignment = kDefaultAlignment) override {
    return Allocate(size);
  }

  bool RemoveBuffer(void* address);

  const MemoryUnit* Find(void* address);

  void Clear();

 private:
  BufferAllocator() : executorch::runtime::MemoryAllocator(0, nullptr) {}

  BufferAllocator(const BufferAllocator&) = delete;

  BufferAllocator& operator=(const BufferAllocator&) = delete;

  ~BufferAllocator() override {
    Clear();
  }

 private:
  std::mutex mMutex;
};

} // namespace neuron
} // namespace backends
} // namespace executorch
