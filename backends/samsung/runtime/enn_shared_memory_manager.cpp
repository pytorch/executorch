/*
 *  Copyright (c) 2025 Samsung Electronics Co. LTD
 *  All rights reserved
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 *
 */
#include <executorch/backends/samsung/runtime/enn_api_implementation.h>
#include <executorch/backends/samsung/runtime/enn_shared_memory_manager.h>
#include <executorch/backends/samsung/runtime/enn_type.h>
#include <executorch/backends/samsung/runtime/logging.h>
#include <executorch/runtime/core/error.h>

#include <mutex>
#include <vector>

using namespace torch::executor::enn;
using namespace torch::executor;

namespace executorch {
namespace backends {
namespace enn {
namespace shared_memory_manager {

static std::mutex instance_mutex_;

SharedMemoryManager* SharedMemoryManager::getInstance() {
  static SharedMemoryManager instance;
  return &instance;
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
  EnnBufferPtrList.emplace_back(bufferPtr);
  return bufferPtr->va;
}

bool SharedMemoryManager::query(
    EnnBufferPtr* out,
    const void* ptr,
    const size_t size) {
  std::lock_guard<std::mutex> lgd(instance_mutex_);
  auto enn_api_inst = EnnApi::getEnnApiInstance();
  for (const auto& buffer : EnnBufferPtrList) {
    if (buffer->va <= ptr &&
        ptr < static_cast<char*>(buffer->va) + buffer->size) {
      int fd;
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
  free(ptr, {});
}
void SharedMemoryManager::free(void* ptr, std::align_val_t alignment) {
  std::lock_guard<std::mutex> lgd(instance_mutex_);
  auto enn_api_inst = EnnApi::getEnnApiInstance();
  for (auto it = EnnBufferPtrList.begin(); it != EnnBufferPtrList.end(); ++it) {
    if ((*it)->va == ptr) {
      ET_LOG(
          Info,
          "va(%p), size(%d), offset(%d) is erased from LUT",
          ptr,
          (*it)->size,
          (*it)->offset);
      auto ret = enn_api_inst->EnnReleaseBuffer(*it);
      if (ret) {
        ET_LOG(Error, "Failed to destroy buffer: %p", ptr);
      }
      EnnBufferPtrList.erase(it);
      ET_LOG(Info, "Buffer Erased(%p)", ptr);
      return;
    }
  }
}

} // namespace shared_memory_manager
} // namespace enn
} // namespace backends
} // namespace executorch
