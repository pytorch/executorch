/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

//===----------------------------------------------------------------------===//
// Compiled with -fobjc-arc (see backends/metal/CMakeLists.txt). The
// Entry struct's id<MTLBuffer> mtl field is __strong; map insert retains,
// erase releases.
//
// Subregion entries take a +1 strong ref on the parent's mtl (extending
// parent's lifetime while a subregion exists). This is strictly safer
// than borrowing (no dangling ref if parent removed first) at the cost
// of a refcount bump.
//===----------------------------------------------------------------------===//

#import "BufferRegistry.h"

#include <executorch/runtime/platform/assert.h>

#include <algorithm>
#include <cstring>

namespace executorch {
namespace backends {
namespace metal_v2 {

BufferRegistry::~BufferRegistry() {
  clear();
}

void BufferRegistry::insert(
    void* ptr,
    id<MTLBuffer> mtl,
    Origin origin,
    size_t size,
    ResidencyClass residency_class) {
  if (!ptr || !mtl) {
    return;
  }
  ET_DCHECK_MSG(
      map_.find(ptr) == map_.end(),
      "BufferRegistry::insert: duplicate ptr=%p; call remove() first to avoid "
      "silently dropping the prior entry's strong ref.",
      ptr);
  map_[ptr] = Entry{mtl, origin, size, /*offset=*/0, residency_class};
}

void BufferRegistry::insertSubregion(
    void* child_ptr,
    id<MTLBuffer> parent_mtl,
    size_t offset,
    size_t size) {
  if (!child_ptr || !parent_mtl) {
    return;
  }
  ET_DCHECK_MSG(
      map_.find(child_ptr) == map_.end(),
      "BufferRegistry::insertSubregion: duplicate child_ptr=%p; call "
      "unregisterSubregion() first.",
      child_ptr);
  // Retains parent_mtl: extends parent's lifetime to cover the subregion.
  // On remove() the ref drops symmetrically. Caller is not responsible
  // for ordering parent vs subregion removals.
  map_[child_ptr] = Entry{parent_mtl, Origin::Subregion, size, offset};
}

const BufferRegistry::Entry* BufferRegistry::find(void* ptr) const {
  auto it = map_.find(ptr);
  return it == map_.end() ? nullptr : &it->second;
}

id<MTLBuffer> BufferRegistry::findBuffer(void* ptr) const {
  auto it = map_.find(ptr);
  return it == map_.end() ? nil : it->second.mtl;
}

std::optional<BufferRegistry::Entry> BufferRegistry::remove(void* ptr) {
  auto it = map_.find(ptr);
  if (it == map_.end()) {
    return std::nullopt;
  }
  Entry entry = it->second;
  map_.erase(it);
  // The returned Entry carries +1 on its mtl via value semantics; ARC
  // releases on scope exit at the caller.
  return entry;
}

void BufferRegistry::refreshIfCopied(void* ptr, size_t size) {
  auto it = map_.find(ptr);
  if (it == map_.end() || it->second.origin != Origin::ExternalCopied) {
    return;
  }
  if (size == 0) {
    return;
  }
  // Hard-fail when the caller asks to refresh more bytes than the entry
  // was created with: silently truncating would let the kernel that
  // thinks it has `size` bytes read stale / garbage data past entry.size.
  // (Smaller `size` is fine: the caller asked to refresh a prefix of the
  // registered region.)
  ET_CHECK_MSG(
      size <= it->second.size,
      "BufferRegistry::refreshIfCopied: caller asked to refresh %zu bytes "
      "but ExternalCopied entry for ptr %p was registered with only %zu "
      "bytes. Re-register with the larger size or use a different region.",
      size, ptr, it->second.size);
  // For ExternalCopied entries we used MTLResourceStorageModeShared, so
  // [mtl contents] is host-accessible.
  std::memcpy([it->second.mtl contents], ptr, size);
}

void BufferRegistry::forEach(
    const std::function<void(void* ptr, const Entry&)>& fn) const {
  for (auto& [ptr, entry] : map_) {
    fn(ptr, entry);
  }
}

void BufferRegistry::clear() {
  map_.clear();
}

} // namespace metal_v2
} // namespace backends
} // namespace executorch
