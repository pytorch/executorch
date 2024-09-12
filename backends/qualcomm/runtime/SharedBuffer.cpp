/*
 * Copyright (c) Qualcomm Innovation Center, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
#include <dlfcn.h>
#include <executorch/backends/qualcomm/runtime/Logging.h>
#include <executorch/backends/qualcomm/runtime/SharedBuffer.h>

// Refer to the QNN HTP Shared Buffer Tutorial
// in Qualcomm® AI Engine Direct document
constexpr uint8_t RPCMEM_HEAP_ID_SYSTEM = 25;
constexpr uint8_t RPCMEM_DEFAULT_FLAGS = 1;

std::size_t std::hash<CustomMemTensorInfo>::operator()(
    const CustomMemTensorInfo& info) const noexcept {
  size_t hash_val = 0;
  hash_val ^= std::hash<void*>()(info.tensor_addr);
  hash_val ^= std::hash<void*>()(info.custom_mem);
  hash_val ^= std::hash<size_t>()(info.pos);
  hash_val ^= std::hash<size_t>()(info.tensor_bytes);
  for (int i = 0; i < info.rank; ++i) {
    hash_val ^= info.shape[i];
  }
  hash_val ^= std::hash<uint32_t>()(info.rank);
  hash_val ^= std::hash<exec_aten::ScalarType>()(info.dtype);
  return hash_val;
}

bool operator==(
    const CustomMemTensorInfo& lhs,
    const CustomMemTensorInfo& rhs) {
  bool is_same =
      (lhs.tensor_addr == rhs.tensor_addr && lhs.custom_mem == rhs.custom_mem &&
       lhs.pos == rhs.pos && lhs.tensor_bytes == rhs.tensor_bytes &&
       lhs.rank == rhs.rank && lhs.dtype == rhs.dtype);
  for (int i = 0; i < lhs.rank; ++i) {
    is_same &= lhs.shape[i] == rhs.shape[i];
  }
  return is_same;
}

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

void* SharedBuffer::GetCustomMemBase(void* buf) {
  auto it = tensor_addr_to_custom_mem_.find(buf);
  if (it == tensor_addr_to_custom_mem_.end()) {
    return nullptr;
  }
  return it->second;
}

void* SharedBuffer::GetUnAlignedAddr(void* buf) {
  auto it = restore_map_.find(buf);
  if (it == restore_map_.end()) {
    return nullptr;
  }
  return it->second;
}

size_t SharedBuffer::GetAllocatedSize(void* buf) {
  auto it = allocated_size_map_.find(buf);
  if (it == allocated_size_map_.end()) {
    return 0;
  }
  return it->second;
}

SharedBuffer& SharedBuffer::GetSharedBufferManager() {
  std::lock_guard<std::mutex> lk(init_mutex_);
  static SharedBuffer shared_buffer_manager;
  if (!shared_buffer_manager.GetInitialize()) {
#if defined(__aarch64__)
    Error status = shared_buffer_manager.Load();
#else
    // For x86_64 platform
    Error status = Error::Ok;
#endif
    if (status == Error::Ok) {
      shared_buffer_manager.SetInitialize(true);
    }
  }
  return shared_buffer_manager;
}

SharedBuffer::~SharedBuffer() {
#if defined(__aarch64__)
  if (initialize_) {
    SharedBuffer::GetSharedBufferManager().UnLoad();
  }
#endif
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
  allocated_size_map_.insert({buf, allocate_bytes});
  auto aligned_buf = reinterpret_cast<void*>(
      alignTo(alignment, reinterpret_cast<intptr_t>(buf)));
  bool status = restore_map_.insert({aligned_buf, buf}).second;
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
}

void SharedBuffer::AddCusomMemTensorAddr(void* tensor_addr, void* custom_mem) {
  tensor_addr_to_custom_mem_.insert({tensor_addr, custom_mem});
};

void SharedBuffer::AddCusomMemTensorInfo(const CustomMemTensorInfo& info) {
  custom_mem_tensor_info_set_.insert(info);
  tensor_addr_to_custom_mem_.insert({info.tensor_addr, info.custom_mem});
}

Error SharedBuffer::UnLoad() {
  if (dlclose(lib_cdsp_rpc_) != 0) {
    QNN_EXECUTORCH_LOG_ERROR(
        "Unable to close shared buffer. dlerror(): %s", dlerror());
    return Error::Internal;
  };
  return Error::Ok;
}
} // namespace qnn
} // namespace executor
} // namespace torch
