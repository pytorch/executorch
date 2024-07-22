/*
 * Copyright (c) Qualcomm Innovation Center, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
#pragma once
#include <QnnTypes.h>
#include <executorch/backends/qualcomm/runtime/QnnExecuTorch.h>
#include <executorch/runtime/core/error.h>
#include <atomic>
#include <cstdint>
#include <memory>
#include <mutex>
#include <unordered_map>
#include <unordered_set>

using RpcMemAllocFn_t = void* (*)(int, uint32_t, int);
using RpcMemFreeFn_t = void (*)(void*);
using RpcMemToFdFn_t = int (*)(void*);

// TODO Finad a better file to place CustomMemTensorInfo
bool operator==(const CustomMemTensorInfo& lhs, const CustomMemTensorInfo& rhs);
template <>
struct std::hash<CustomMemTensorInfo> {
  std::size_t operator()(const CustomMemTensorInfo& info) const noexcept;
};

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

  // memory handle is registered during execution
  void AddCusomMemTensorAddr(void* tensor_addr, void* custom_mem);

  // memory handle can be registered before execution
  void AddCusomMemTensorInfo(const CustomMemTensorInfo& info);

  size_t GetAllocatedSize(void* buf);

  void* GetCustomMemBase(void* buf);

  void* GetUnAlignedAddr(void* buf);

  const std::unordered_set<CustomMemTensorInfo>& GetCustomMemTensorInfoSet() {
    return custom_mem_tensor_info_set_;
  };

 private:
  SharedBuffer() = default;

  // dlopen RPCMem library and dlysm required functions
  Error Load();

  Error UnLoad();

  // Pointer to the dlopen'd libcdsprpc.so shared library which contains
  // rpcmem_alloc, rpcmem_free, rpcmem_to_fd APIs
  void* lib_cdsp_rpc_;
  // Function pointer to rpcmem_alloc
  RpcMemAllocFn_t rpc_mem_alloc_;
  // Function pointer to rpcmem_free
  RpcMemFreeFn_t rpc_mem_free_;
  // Function pointer to rpcmem_to_fd
  RpcMemToFdFn_t rpc_mem_to_fd_;
  std::unordered_map<void*, void*> restore_map_;
  std::unordered_map<void*, size_t> allocated_size_map_;
  // Maps for the custom memory
  std::unordered_map<void*, void*> tensor_addr_to_custom_mem_;
  std::unordered_set<CustomMemTensorInfo> custom_mem_tensor_info_set_;
  std::atomic_bool initialize_{false};
  static std::mutex init_mutex_;
};

} // namespace qnn
} // namespace executor
} // namespace torch
