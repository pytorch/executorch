/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

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
    size_t size) {
  if (!ptr || !mtl) {
    return;
  }
  // Take an additional retain — the entry holds it for the duration the
  // ptr is registered.
  [mtl retain];
  map_[ptr] = Entry{mtl, origin, size, /*offset=*/0};
}

void BufferRegistry::insertSubregion(
    void* child_ptr,
    id<MTLBuffer> parent_mtl,
    size_t offset,
    size_t size) {
  if (!child_ptr || !parent_mtl) {
    return;
  }
  // R8.3: Subregion entries BORROW the parent's MTLBuffer ref — no extra
  // retain. Lifetime is bounded by the parent's entry; if the parent is
  // remove()d before the subregion is, the subregion's mtl becomes
  // dangling. Caller (AOTI shim / MetalBuffer) must ensure correct order.
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
  Entry entry = it->second;  // copy out
  map_.erase(it);
  // OWNERSHIP TRANSFER: for Pool/External* entries the registry's +1 retain
  // is handed to the caller via the returned Entry. Caller MUST release
  // entry.mtl exactly once when done (or hand it to bufferPool_->release
  // for Pool entries — pool.release retains internally, so caller's
  // [release] balances the original acquire+insert chain).
  // R8.3: Subregion entries borrow their mtl from the parent's entry —
  // there is no +1 to transfer. Caller MUST NOT release entry.mtl for
  // Subregion entries.
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
  // previously this silently truncated to entry.size when the
  // caller asked to refresh more bytes than the entry was created with —
  // the kernel that thinks it has `size` bytes would then read stale /
  // garbage data past entry.size. Hard-fail instead so the bug is
  // immediate and visible. (Smaller `size` is fine: the caller asked
  // to refresh a prefix of the registered region.)
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
  for (auto& [ptr, entry] : map_) {
    // R8.3: Subregion entries borrow their mtl from the parent entry; do
    // not release them here. Pool/External* entries own their +1 retain.
    if (entry.mtl && entry.origin != Origin::Subregion) {
      [entry.mtl release];
    }
  }
  map_.clear();
}

} // namespace metal_v2
} // namespace backends
} // namespace executorch
