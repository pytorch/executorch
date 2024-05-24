/*
 * Copyright (c) Qualcomm Innovation Center, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
#pragma once
#include <executorch/runtime/core/error.h>
#include <atomic>
#include <cstdint>
#include <memory>
#include <mutex>
#include <unordered_map>

using RpcMemAllocFn_t = void* (*)(int, uint32_t, int);
using RpcMemFreeFn_t = void (*)(void*);
using RpcMemToFdFn_t = int (*)(void*);

namespace torch {
namespace executor {
namespace qnn {
class SharedBuffer final {
 public:
  SharedBuffer(const SharedBuffer&) = delete;
  SharedBuffer& operator=(const SharedBuffer&) = delete;
  SharedBuffer(SharedBuffer&&) = delete;
  SharedBuffer& operator=(SharedBuffer&&) = delete;
  ~SharedBuffer();

  static SharedBuffer& GetSharedBufferManager();
  void* AllocMem(size_t bytes, size_t alignment);
  // map a buffer allocated via RPCMem to a file descriptor so it can be
  // registered with a backend via QnnMem_register()
  int32_t MemToFd(void* buf);

  void FreeMem(void* buf);

  bool IsAllocated(void* buf);

  bool GetInitialize() {
    return initialize_;
  }
  void SetInitialize(bool initialize) {
    initialize_ = initialize;
  }

 private:
  SharedBuffer() = default;

  // dlopen RPCMem library and dlysm required functions
  Error Load();

  Error UnLoad();

  // Pointer to the dlopen'd libcdsprpc.so shared library which contains
  // rpcmem_alloc, rpcmem_free, rpcmem_to_fd APIs
  [[maybe_unused]] void* lib_cdsp_rpc_;
  // Function pointer to rpcmem_alloc
  RpcMemAllocFn_t rpc_mem_alloc_;
  // Function pointer to rpcmem_free
  RpcMemFreeFn_t rpc_mem_free_;
  // Function pointer to rpcmem_to_fd
  RpcMemToFdFn_t rpc_mem_to_fd_;
  std::unordered_map<void*, void*> restore_map_;
  std::atomic_bool initialize_{false};
  static std::mutex init_mutex_;
};

} // namespace qnn
} // namespace executor
} // namespace torch
