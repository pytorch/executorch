/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/extension/cuda/cuda_allocator.h>

#include <cstdio>
#include <cstdlib>
#include <new>

using namespace executorch::extension::cuda;

namespace {
alignas(4096) char storage[8192];
void* rounded = nullptr;
executorch::runtime::DeviceAllocator* allocator = nullptr;
int frees = 0;
int trims = 0;
bool forbid_host_allocation = false;
const auto fake_pool = reinterpret_cast<cudaMemPool_t>(storage + 256);

[[noreturn]] void fail(const char* message) {
  std::fprintf(stderr, "%s\n", message);
  std::_Exit(1);
}

// Delegates free memory and trim pools from their own static destructors, so
// the allocator and both of its tables must outlive every exit handler.
void free_at_exit() {
  frees = 0;
  // Through the base class, as the device allocator registry calls it.
  allocator->deallocate(rounded, -1);
  if (frees != 1) {
    fail("late deallocation did not free the original block");
  }
  CudaAllocator::release_cached_memory(-1);
  if (trims != 1) {
    fail("late trim did not reach the pool");
  }
}
} // namespace

// A per-device trim is called after the KV pool commits its replacement. It
// must not allocate a snapshot that could throw before pitch reconciliation.
void* operator new(std::size_t size) {
  if (forbid_host_allocation) {
    fail("per-device trim allocated host memory after storage commit");
  }
  if (void* memory = std::malloc(size == 0 ? 1 : size)) {
    return memory;
  }
  fail("host allocation failed");
}

void operator delete(void* memory) noexcept {
  std::free(memory);
}

void operator delete(void* memory, std::size_t) noexcept {
  std::free(memory);
}

// Stand-ins keep the test independent of GPU packing and driver teardown order.
extern "C" {
cudaError_t cudaGetDevice(int* device) {
  *device = 0;
  return cudaSuccess;
}

cudaError_t cudaMalloc(void** ptr, size_t) {
  *ptr = storage + 256;
  return cudaSuccess;
}

cudaError_t cudaFree(void* ptr) {
  if (ptr != storage + 256) {
    fail("late deallocation lost the original block");
  }
  ++frees;
  return cudaSuccess;
}

cudaError_t cudaMemPoolCreate(cudaMemPool_t* pool, const cudaMemPoolProps*) {
  *pool = fake_pool;
  return cudaSuccess;
}

cudaError_t cudaMemPoolSetAttribute(cudaMemPool_t, cudaMemPoolAttr, void*) {
  return cudaSuccess;
}

cudaError_t
cudaMallocFromPoolAsync(void** ptr, size_t, cudaMemPool_t, cudaStream_t) {
  *ptr = storage + 4096;
  return cudaSuccess;
}

cudaError_t cudaFreeAsync(void*, cudaStream_t) {
  return cudaSuccess;
}

cudaError_t cudaMemPoolTrimTo(cudaMemPool_t pool, size_t) {
  if (pool != fake_pool) {
    fail("late trim lost the pool");
  }
  ++trims;
  return cudaSuccess;
}

cudaError_t cudaDeviceGraphMemTrim(int) {
  return cudaSuccess;
}

// The pooled path, whatever the machine's device reports.
cudaError_t cudaDeviceGetAttribute(int* value, cudaDeviceAttr, int) {
  *value = 1;
  return cudaSuccess;
}
}

int main() {
  // Registered before the allocator's state exists, so this handler runs after
  // the destructors of anything built later.
  if (std::atexit(free_at_exit) != 0) {
    return 1;
  }
  allocator = &CudaAllocator::instance();
  auto result = allocator->allocate(1024, -1, 4096);
  if (!result.ok() || result.get() != storage + 4096) {
    fail("allocate did not round the block up");
  }
  rounded = result.get();

  auto pooled = CudaAllocator::allocate_async(1024, -1, nullptr);
  if (!pooled.ok()) {
    fail("allocate_async did not create the pool");
  }
  CudaAllocator::deallocate_async(pooled.get(), -1, nullptr);
  forbid_host_allocation = true;
  CudaAllocator::release_cached_memory(0);
  CudaAllocator::release_cached_memory(1);
  forbid_host_allocation = false;
  if (trims != 1) {
    fail("per-device trim did not reach exactly one existing pool");
  }
  trims = 0;
  return 0;
}
