/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/backends/qualcomm/runtime/SharedBuffer.h>
#include <executorch/runtime/platform/compiler.h>
#include <gtest/gtest.h>
#include <atomic>
#include <cstdint>
#include <cstdlib>
#include <limits>
#include <memory>
#include <new>
#include <thread>
#include <vector>

#if ET_HAS_EXCEPTIONS && !defined(_GLIBCXX_RESTRICT_EXCEPTIONS)
namespace {
thread_local int allocationsUntilFailure = -1;
} // namespace

void* operator new(std::size_t size) {
  if (allocationsUntilFailure >= 0 && allocationsUntilFailure-- == 0) {
    throw std::bad_alloc();
  }
  if (void* pointer = std::malloc(size == 0 ? 1 : size)) {
    return pointer;
  }
  throw std::bad_alloc();
}

void operator delete(void* pointer) noexcept {
  std::free(pointer);
}

void operator delete(void* pointer, std::size_t) noexcept {
  std::free(pointer);
}

void* operator new(std::size_t size, const std::nothrow_t&) noexcept {
  try {
    return ::operator new(size);
  } catch (const std::bad_alloc&) {
    return nullptr;
  }
}

void operator delete(void* pointer, const std::nothrow_t&) noexcept {
  std::free(pointer);
}
#endif

namespace executorch::backends::qnn {

class SharedBufferTest : public ::testing::Test {
 protected:
  void SetUp() override {
    allocationCalls_ = 0;
    buffer_.reset(new SharedBuffer());
    buffer_->rpc_mem_alloc_ = [](int, uint32_t, int bytes) -> void* {
      ++allocationCalls_;
      return bytes > 0 && bytes <= 1024 * 1024
          ? std::malloc(static_cast<size_t>(bytes))
          : nullptr;
    };
    buffer_->rpc_mem_free_ = [](void* pointer) { std::free(pointer); };
    buffer_->rpc_mem_to_fd_ = [](void*) { return 17; };
    buffer_->SetInitialize(true);
  }

  void TearDown() override {
    for (const auto& allocation : buffer_->restore_map_) {
      std::free(allocation.second);
    }
    buffer_->SetInitialize(false);
  }

  void useOffsetRpcAllocator() {
    buffer_->rpc_mem_alloc_ = [](int, uint32_t, int bytes) -> void* {
      rpcAllocation_ =
          static_cast<uint8_t*>(std::malloc(static_cast<size_t>(bytes) + 1)) +
          1;
      return rpcAllocation_;
    };
    buffer_->rpc_mem_free_ = [](void* pointer) {
      rpcFreed_ = pointer;
      std::free(static_cast<uint8_t*>(pointer) - 1);
    };
  }

  bool empty() const {
    return buffer_->restore_map_.empty() &&
        buffer_->allocated_size_map_.empty() &&
        buffer_->tensor_addr_to_custom_mem_.empty() &&
        buffer_->custom_mem_to_tensor_addr_.empty();
  }

#if ET_HAS_EXCEPTIONS && !defined(_GLIBCXX_RESTRICT_EXCEPTIONS)
  void failBookkeepingAllocation(int successfulAllocations) {
    buffer_->restore_map_.reserve(1);
    buffer_->allocated_size_map_.reserve(1);
    buffer_->rpc_mem_alloc_ = [](int, uint32_t, int bytes) -> void* {
      rpcAllocation_ = std::malloc(static_cast<size_t>(bytes));
      return rpcAllocation_;
    };
    buffer_->rpc_mem_free_ = [](void* pointer) {
      rpcFreed_ = pointer;
      std::free(pointer);
    };
    rpcFreed_ = nullptr;
    allocationsUntilFailure = successfulAllocations;
  }
#endif

  std::unique_ptr<SharedBuffer> buffer_;
  inline static std::atomic<size_t> allocationCalls_{0};
  inline static void* rpcAllocation_{nullptr};
  inline static void* rpcFreed_{nullptr};
};

TEST_F(SharedBufferTest, AlignsOffsetRpcAllocationAndFreesTheOriginal) {
  constexpr size_t kAlignment = 64;
  constexpr size_t kBytes = 128;
  useOffsetRpcAllocator();

  auto* aligned = static_cast<uint8_t*>(buffer_->AllocMem(kBytes, kAlignment));
  ASSERT_NE(aligned, nullptr);
  auto* original = static_cast<uint8_t*>(rpcAllocation_);
  EXPECT_EQ(reinterpret_cast<uintptr_t>(aligned) % kAlignment, 0);
  EXPECT_GT(aligned, original);
  EXPECT_LE(aligned + kBytes, original + buffer_->GetAllocatedSize(aligned));
  buffer_->FreeMem(aligned);

  EXPECT_EQ(rpcFreed_, original);
  EXPECT_TRUE(empty());
}

TEST_F(SharedBufferTest, InvalidSizeAndAlignmentDoNotCallRpcAllocator) {
  EXPECT_EQ(buffer_->AllocMem(0, alignof(void*)), nullptr);
  EXPECT_EQ(buffer_->AllocMem(64, 0), nullptr);
  EXPECT_EQ(buffer_->AllocMem(64, 3), nullptr);
  EXPECT_EQ(
      buffer_->AllocMem(std::numeric_limits<int32_t>::max(), alignof(void*)),
      nullptr);
  EXPECT_EQ(
      buffer_->AllocMem(std::numeric_limits<size_t>::max(), alignof(void*)),
      nullptr);
  EXPECT_EQ(allocationCalls_.load(), 0);
  EXPECT_TRUE(empty());
}

TEST_F(SharedBufferTest, ConcurrentAllocateRegisterLookupAndFree) {
  constexpr size_t kThreads = 4;
  constexpr size_t kAllocations = 256;
  constexpr size_t kBytes = 64;
  std::vector<void*> buffers(kThreads * kAllocations, nullptr);
  std::vector<std::thread> workers;
  workers.reserve(kThreads);
  std::atomic<bool> start{false};
  for (size_t worker = 0; worker < kThreads; ++worker) {
    workers.emplace_back([&, worker] {
      while (!start.load(std::memory_order_acquire)) {
        std::this_thread::yield();
      }
      for (size_t index = worker; index < buffers.size(); index += kThreads) {
        void* pointer = buffer_->AllocMem(kBytes, alignof(void*));
        ASSERT_NE(pointer, nullptr);
        buffers[index] = pointer;
        void* alias = static_cast<uint8_t*>(pointer) + sizeof(uint64_t);
        buffer_->AddCusomMemTensorAddr(alias, pointer);
        EXPECT_TRUE(buffer_->IsAllocated(pointer));
        EXPECT_EQ(buffer_->GetCustomMemBase(alias), pointer);
        EXPECT_GE(buffer_->GetAllocatedSize(pointer), kBytes);
        EXPECT_EQ(buffer_->MemToFd(pointer), 17);
      }
    });
  }
  start.store(true, std::memory_order_release);
  for (auto& worker : workers) {
    worker.join();
  }
  EXPECT_EQ(allocationCalls_.load(), buffers.size());

  workers.clear();
  for (size_t worker = 0; worker < kThreads; ++worker) {
    workers.emplace_back([&, worker] {
      for (size_t index = worker; index < buffers.size(); index += kThreads) {
        void* pointer = buffers[index];
        ASSERT_NE(pointer, nullptr);
        void* alias = static_cast<uint8_t*>(pointer) + sizeof(uint64_t);
        buffer_->FreeMem(pointer);
        EXPECT_FALSE(buffer_->IsAllocated(pointer));
        EXPECT_EQ(buffer_->GetAllocatedSize(pointer), 0);
        EXPECT_EQ(buffer_->GetCustomMemBase(alias), nullptr);
      }
    });
  }
  for (auto& worker : workers) {
    worker.join();
  }
  EXPECT_TRUE(empty());
}

#if ET_HAS_EXCEPTIONS && !defined(_GLIBCXX_RESTRICT_EXCEPTIONS)
TEST_F(SharedBufferTest, BookkeepingAllocationFailureRollsBackAndReturnsNull) {
  // Reserved buckets leave one node allocation per map insertion.
  for (int successfulAllocations : {0, 1}) {
    failBookkeepingAllocation(successfulAllocations);
    void* result = buffer_->AllocMem(64, alignof(void*));
    allocationsUntilFailure = -1;
    EXPECT_EQ(result, nullptr);
    EXPECT_EQ(rpcFreed_, rpcAllocation_);
    EXPECT_TRUE(empty());

    void* retry = buffer_->AllocMem(64, alignof(void*));
    ASSERT_NE(retry, nullptr);
    buffer_->FreeMem(retry);
    EXPECT_TRUE(empty());
  }
}
#endif

} // namespace executorch::backends::qnn
