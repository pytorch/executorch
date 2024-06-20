/*
 * Copyright (c) Qualcomm Innovation Center, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
#ifdef __ANDROID__
#include <dlfcn.h>
#endif
#include <executorch/backends/qualcomm/runtime/Logging.h>
#include <executorch/backends/qualcomm/runtime/SharedBuffer.h>

// Refer to the QNN HTP Shared Buffer Tutorial
// in Qualcomm® AI Engine Direct document
constexpr uint8_t RPCMEM_HEAP_ID_SYSTEM = 25;
constexpr uint8_t RPCMEM_DEFAULT_FLAGS = 1;

namespace torch {
namespace executor {
namespace qnn {

namespace {

intptr_t alignTo(size_t alignment, intptr_t offset) {
  return offset % alignment == 0 ? offset
                                 : offset +
          (static_cast<intptr_t>(alignment) -
           offset % static_cast<intptr_t>(alignment));
}

} // namespace

std::mutex SharedBuffer::init_mutex_;

SharedBuffer& SharedBuffer::GetSharedBufferManager() {
  std::lock_guard<std::mutex> lk(init_mutex_);
  static SharedBuffer shared_buffer_manager;
  if (!shared_buffer_manager.GetInitialize()) {
    Error status = shared_buffer_manager.Load();
    if (status == Error::Ok) {
      shared_buffer_manager.SetInitialize(true);
    }
  }
  return shared_buffer_manager;
}

SharedBuffer::~SharedBuffer() {
  if (initialize_) {
    SharedBuffer::GetSharedBufferManager().UnLoad();
  }
};

void* SharedBuffer::AllocMem(size_t bytes, size_t alignment) {
  if (!initialize_) {
    QNN_EXECUTORCH_LOG_ERROR("Shared memory not initialized.");
    return nullptr;
  }
  // do alignment:
  auto allocate_bytes = static_cast<int32_t>(bytes + alignment);
  void* buf = rpc_mem_alloc_(
      RPCMEM_HEAP_ID_SYSTEM, RPCMEM_DEFAULT_FLAGS, allocate_bytes);
  if (buf == nullptr) {
    QNN_EXECUTORCH_LOG_WARN("Failed to allocate the tensor by RPC memory.");
    return nullptr;
  }
  auto aligned_buf = reinterpret_cast<void*>(
      alignTo(alignment, reinterpret_cast<intptr_t>(buf)));
  bool status =
      restore_map_.insert(std::pair<void*, void*>(aligned_buf, buf)).second;
  if (!status) {
    QNN_EXECUTORCH_LOG_ERROR("Failed to allocate the tensor by RPC memory.");
    rpc_mem_free_(buf);
  }
  return aligned_buf;
}

int32_t SharedBuffer::MemToFd(void* buf) {
  int32_t memFd = -1;
  if (!initialize_) {
    QNN_EXECUTORCH_LOG_ERROR("Shared memory not initialized.");
  } else {
    memFd = rpc_mem_to_fd_(buf);
  }
  return memFd;
}

void SharedBuffer::FreeMem(void* buf) {
  if (!initialize_) {
    QNN_EXECUTORCH_LOG_ERROR("Shared memory not initialized.");
  } else if (restore_map_.count(buf) == 0) {
    QNN_EXECUTORCH_LOG_WARN("Don't free an unallocated tensor.");
  } else {
    rpc_mem_free_(restore_map_[buf]);
    restore_map_.erase(buf);
  }
}

bool SharedBuffer::IsAllocated(void* buf) {
  return restore_map_.count(buf) != 0U;
}

Error SharedBuffer::Load() {
#ifndef __ANDROID__
  QNN_EXECUTORCH_LOG_WARN("Shared buffer is not supported on this platform.");
  return Error::Ok;
#else
  // On Android, 32-bit and 64-bit libcdsprpc.so can be found at /vendor/lib/
  // and /vendor/lib64/ respectively.
  lib_cdsp_rpc_ = dlopen("libcdsprpc.so", RTLD_NOW | RTLD_LOCAL);
  if (lib_cdsp_rpc_ == nullptr) {
    QNN_EXECUTORCH_LOG_ERROR(
        "Unable to load shared buffer. dlerror(): %s", dlerror());
    return Error::Internal;
  }
  rpc_mem_alloc_ = reinterpret_cast<RpcMemAllocFn_t>( // NOLINT
      dlsym(lib_cdsp_rpc_, "rpcmem_alloc"));
  rpc_mem_free_ = reinterpret_cast<RpcMemFreeFn_t>( // NOLINT
      dlsym(lib_cdsp_rpc_, "rpcmem_free"));
  rpc_mem_to_fd_ = reinterpret_cast<RpcMemToFdFn_t>( // NOLINT
      dlsym(lib_cdsp_rpc_, "rpcmem_to_fd"));
  if (nullptr == rpc_mem_alloc_ || nullptr == rpc_mem_free_ ||
      nullptr == rpc_mem_to_fd_) {
    QNN_EXECUTORCH_LOG_ERROR(
        "Unable to access symbols in shared buffer. dlerror(): %s", dlerror());
    dlclose(lib_cdsp_rpc_);
    return Error::Internal;
  }
  return Error::Ok;
#endif
}

Error SharedBuffer::UnLoad() {
#ifndef __ANDROID__
  return Error::Ok;
#else
  if (dlclose(lib_cdsp_rpc_) != 0) {
    QNN_EXECUTORCH_LOG_ERROR(
        "Unable to close shared buffer. dlerror(): %s", dlerror());
    return Error::Internal;
  };
  return Error::Ok;
#endif
}
} // namespace qnn
} // namespace executor
} // namespace torch
