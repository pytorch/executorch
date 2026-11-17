/*
 * Copyright (c) Qualcomm Innovation Center, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
#include <executorch/backends/qualcomm/runtime/Logging.h>
#include <executorch/backends/qualcomm/runtime/SharedBuffer.h>
#include <executorch/runtime/platform/compiler.h>

#include <pal/DynamicLoading.h>

#include <limits>
#include <new>

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
  for (size_t i = 0; i < info.rank; ++i) {
    hash_val ^= std::hash<uint32_t>()(info.shape[i]);
  }
  hash_val ^= std::hash<uint32_t>()(info.rank);
  hash_val ^= std::hash<executorch::aten::ScalarType>()(info.dtype);
  return hash_val;
}

bool operator==(
    const CustomMemTensorInfo& lhs,
    const CustomMemTensorInfo& rhs) {
  bool is_same =
      (lhs.tensor_addr == rhs.tensor_addr && lhs.custom_mem == rhs.custom_mem &&
       lhs.pos == rhs.pos && lhs.tensor_bytes == rhs.tensor_bytes &&
       lhs.rank == rhs.rank && lhs.dtype == rhs.dtype);
  for (size_t i = 0; i < lhs.rank; ++i) {
    is_same &= lhs.shape[i] == rhs.shape[i];
  }
  return is_same;
}

namespace executorch {
namespace backends {
namespace qnn {

using executorch::runtime::Error;

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
  std::lock_guard<std::mutex> lock(memory_mutex_);
  auto it = tensor_addr_to_custom_mem_.find(buf);
  if (it == tensor_addr_to_custom_mem_.end()) {
    return nullptr;
  }
  return it->second;
}

SharedBuffer::AllocationInfo SharedBuffer::GetAllocationInfo(void* buf) {
  std::lock_guard<std::mutex> lock(memory_mutex_);
  auto it = allocated_size_map_.find(buf);
  if (it == allocated_size_map_.end()) {
    return {};
  }
  return {it->second, rpc_mem_to_fd_(buf)};
}

SharedBuffer& SharedBuffer::GetSharedBufferManager() {
  std::lock_guard<std::mutex> lk(init_mutex_);
  static SharedBuffer shared_buffer_manager;
  if (!shared_buffer_manager.GetInitialize()) {
    // libcdsprpc is only available on aarch64 targets: Linux/Android and
    // Windows-on-ARM64. There is no equivalent on Windows-x86.
#if defined(__aarch64__) || defined(_M_ARM64)
    Error status = shared_buffer_manager.Load();
#else
    Error status = Error::NotSupported;
#endif
    if (status == Error::Ok) {
      shared_buffer_manager.SetInitialize(true);
    }
  }
  return shared_buffer_manager;
}

SharedBuffer::~SharedBuffer() {
#if defined(__aarch64__) || defined(_M_ARM64)
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
  const size_t max_bytes = std::numeric_limits<int32_t>::max();
  if (alignment == 0 || (alignment & (alignment - 1)) != 0 ||
      alignment > max_bytes || bytes > max_bytes - alignment) {
    QNN_EXECUTORCH_LOG_ERROR("Invalid shared memory size or alignment.");
    return nullptr;
  }
  std::lock_guard<std::mutex> lock(memory_mutex_);
  // do alignment:
  auto allocate_bytes = static_cast<int32_t>(bytes + alignment);
  void* buf = rpc_mem_alloc_(
      RPCMEM_HEAP_ID_SYSTEM, RPCMEM_DEFAULT_FLAGS, allocate_bytes);
  if (buf == nullptr) {
    QNN_EXECUTORCH_LOG_WARN("Failed to allocate the tensor by RPC memory.");
    return nullptr;
  }
  // NOLINTNEXTLINE(performance-no-int-to-ptr)
  auto aligned_buf = reinterpret_cast<void*>(
      alignTo(alignment, reinterpret_cast<intptr_t>(buf)));
  if (restore_map_.count(aligned_buf) != 0) {
    // The RPC allocator returned memory that is still live, so freeing buf
    // would release the existing allocation.
    QNN_EXECUTORCH_LOG_ERROR(
        "RPC memory %p is already allocated.", aligned_buf);
    return nullptr;
  }
#if ET_HAS_EXCEPTIONS
  try {
#endif
    restore_map_.insert({aligned_buf, buf});
    allocated_size_map_.insert({aligned_buf, allocate_bytes});
#if ET_HAS_EXCEPTIONS
  } catch (const std::bad_alloc&) {
    restore_map_.erase(aligned_buf);
    rpc_mem_free_(buf);
    QNN_EXECUTORCH_LOG_ERROR("Failed to allocate shared memory bookkeeping.");
    return nullptr;
  }
#endif
  return aligned_buf;
}

int32_t SharedBuffer::MemToFd(void* buf) {
  std::lock_guard<std::mutex> lock(memory_mutex_);
  int32_t memFd = -1;
  if (!initialize_) {
    QNN_EXECUTORCH_LOG_ERROR("Shared memory not initialized.");
  } else {
    memFd = rpc_mem_to_fd_(buf);
  }
  return memFd;
}

void SharedBuffer::FreeMem(void* buf) {
  std::lock_guard<std::mutex> lock(memory_mutex_);
  if (!initialize_) {
    QNN_EXECUTORCH_LOG_ERROR("Shared memory not initialized.");
  } else if (restore_map_.count(buf) == 0) {
    QNN_EXECUTORCH_LOG_WARN("Don't free an unallocated tensor.");
  } else {
    rpc_mem_free_(restore_map_[buf]);
    restore_map_.erase(buf);
    allocated_size_map_.erase(buf);
    // Unbind the custom memory from tensor address.
    auto mit = custom_mem_to_tensor_addr_.find(buf);
    if (mit != custom_mem_to_tensor_addr_.end()) {
      for (auto it = mit->second.begin(); it != mit->second.end(); ++it) {
        tensor_addr_to_custom_mem_.erase(*it);
      }
      custom_mem_to_tensor_addr_.erase(buf);
    }
  }
}

bool SharedBuffer::IsAllocated(void* buf) {
  std::lock_guard<std::mutex> lock(memory_mutex_);
  return restore_map_.count(buf) != 0U;
}

Error SharedBuffer::Load() {
  // On Android, 32-bit and 64-bit libcdsprpc.so can be found at /vendor/lib/
  // and /vendor/lib64/ respectively.
#if defined(_WIN32)
  const char* lib_name = "libcdsprpc.dll";
#else
  const char* lib_name = "libcdsprpc.so";
#endif
  lib_cdsp_rpc_ = pal::dynamic_loading::DlOpen(
      lib_name, pal::dynamic_loading::DL_NOW | pal::dynamic_loading::DL_LOCAL);
  if (lib_cdsp_rpc_ == nullptr) {
    QNN_EXECUTORCH_LOG_ERROR(
        "Unable to load shared buffer. dlerror(): %s",
        pal::dynamic_loading::DlError());
    return Error::Internal;
  }
  rpc_mem_alloc_ = reinterpret_cast<RpcMemAllocFn_t>( // NOLINT
      pal::dynamic_loading::DlSym(lib_cdsp_rpc_, "rpcmem_alloc"));
  rpc_mem_free_ = reinterpret_cast<RpcMemFreeFn_t>( // NOLINT
      pal::dynamic_loading::DlSym(lib_cdsp_rpc_, "rpcmem_free"));
  rpc_mem_to_fd_ = reinterpret_cast<RpcMemToFdFn_t>( // NOLINT
      pal::dynamic_loading::DlSym(lib_cdsp_rpc_, "rpcmem_to_fd"));
  if (nullptr == rpc_mem_alloc_ || nullptr == rpc_mem_free_ ||
      nullptr == rpc_mem_to_fd_) {
    QNN_EXECUTORCH_LOG_ERROR(
        "Unable to access symbols in shared buffer. dlerror(): %s",
        pal::dynamic_loading::DlError());
    pal::dynamic_loading::DlClose(lib_cdsp_rpc_);
    return Error::Internal;
  }
  QNN_EXECUTORCH_LOG_INFO("Shared buffer initialized via %s", lib_name);
  return Error::Ok;
}

void SharedBuffer::AddCusomMemTensorAddr(void* tensor_addr, void* custom_mem) {
  std::lock_guard<std::mutex> lock(memory_mutex_);
  bool status =
      tensor_addr_to_custom_mem_.insert({tensor_addr, custom_mem}).second;
  if (!status) {
    QNN_EXECUTORCH_LOG_WARN(
        "Tensor address %p already associated with custom memory %p",
        tensor_addr,
        custom_mem);
    return;
  }
  custom_mem_to_tensor_addr_[custom_mem].insert(tensor_addr);
};

Error SharedBuffer::UnLoad() {
  if (pal::dynamic_loading::DlClose(lib_cdsp_rpc_) != 0) {
    QNN_EXECUTORCH_LOG_ERROR(
        "Unable to close shared buffer. dlerror(): %s",
        pal::dynamic_loading::DlError());
    return Error::Internal;
  };
  return Error::Ok;
}
} // namespace qnn
} // namespace backends
} // namespace executorch
