/*
 *  Copyright (c) 2025 Samsung Electronics Co. LTD
 *  All rights reserved
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 *
 */
#include <executorch/backends/samsung/runtime/enn_shared_memory_manager.h>

#include <executorch/backends/samsung/runtime/enn_api_implementation.h>
#include <executorch/backends/samsung/runtime/logging.h>

#include <inttypes.h>
#include <mutex>

namespace executorch {
namespace backends {
namespace enn {
namespace shared_memory_manager {

using torch::executor::enn::EnnApi;

static std::mutex instance_mutex_;

SharedMemoryManager* SharedMemoryManager::getInstance() {
  // Touch the EnnApi singleton first so that it outlives this instance: the
  // destructor below releases buffers through the ENN API.
  EnnApi::getEnnApiInstance();
  static SharedMemoryManager instance;
  return &instance;
}

SharedMemoryManager::~SharedMemoryManager() {
  std::lock_guard<std::mutex> lgd(instance_mutex_);
  auto enn_api_inst = EnnApi::getEnnApiInstance();
  for (auto& buffer : buffers_) {
    if (enn_api_inst->EnnReleaseBuffer(buffer)) {
      ET_LOG(Error, "Failed to destroy buffer: %p", buffer->va);
    }
  }
  buffers_.clear();
}

void* SharedMemoryManager::alloc(const size_t size) {
  std::lock_guard<std::mutex> lgd(instance_mutex_);
  auto enn_api_inst = EnnApi::getEnnApiInstance();
  EnnBufferPtr bufferPtr;
  auto ret = enn_api_inst->EnnCreateBuffer(size, 0, &bufferPtr);
  if (ret) {
    ET_LOG(Error, "Buffer Creation Error");
    return nullptr;
  }
  buffers_.emplace_back(bufferPtr);
  return bufferPtr->va;
}

bool SharedMemoryManager::query(
    EnnBufferPtr* out,
    const void* ptr,
    const size_t size) {
  std::lock_guard<std::mutex> lgd(instance_mutex_);
  auto enn_api_inst = EnnApi::getEnnApiInstance();
  for (const auto& buffer : buffers_) {
    if (buffer->va <= ptr &&
        ptr < static_cast<char*>(buffer->va) + buffer->size) {
      int32_t fd;
      auto ret = enn_api_inst->EnnGetFileDescriptorFromEnnBuffer(buffer, &fd);
      if (ret) {
        ET_LOG(
            Info,
            "va: %p, size: %zu is in LUT, but failed to get FileDescriptor",
            ptr,
            size);
        return false;
      }
      *out = buffer;
      return true;
    }
  }
  ET_LOG(Info, "va: %p, size: %zu is not in LUT", ptr, size);
  *out = nullptr;
  return false;
}

void SharedMemoryManager::free(void* ptr) {
  std::lock_guard<std::mutex> lgd(instance_mutex_);
  auto enn_api_inst = EnnApi::getEnnApiInstance();
  for (auto it = buffers_.begin(); it != buffers_.end(); ++it) {
    if ((*it)->va == ptr) {
      ET_LOG(
          Info,
          "va(%p), size(%" PRIu32 "), offset(%" PRIu32 ") is erased from LUT",
          ptr,
          (*it)->size,
          (*it)->offset);
      if (enn_api_inst->EnnReleaseBuffer(*it)) {
        ET_LOG(Error, "Failed to destroy buffer: %p", ptr);
      }
      buffers_.erase(it);
      return;
    }
  }
}

} // namespace shared_memory_manager
} // namespace enn
} // namespace backends
} // namespace executorch
