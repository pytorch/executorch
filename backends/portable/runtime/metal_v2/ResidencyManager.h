/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

// ResidencyManager — wraps an MTLResidencySet (macOS 15+ / iOS 18+).
// Purpose
// -------
// The residency set tells the driver "these MTLBuffers are GPU-resident
// for the foreseeable future, please page them in / pin them so command
// buffers don't take faults". MetalStream registers every buffer it
// owns or wraps (Pool, ExternalAliased, ExternalCopied — but not
// Subregion since those borrow the parent's allocation).
// Lifetime
// --------
// - Constructor calls -newResidencySetWithDescriptor: (gated on
//   @available + ET_METAL4_AVAILABLE). On older OSes the manager is
//   constructed but isEnabled() returns false and all calls are no-ops.
// - Destructor releases the set.
// API surface (intentionally tiny)
// --------------------------------
// - add(buffer)    — schedule the buffer for residency on next commit().
// - remove(buffer) — paired inverse; the audit B7 fix.
// - commit()       — flush pending changes + requestResidency on the
//                    GPU. Called from MetalStream::flush() and
//                    mtl4CommitAndSignal().
// - isEnabled()    — false if @available gate failed or set creation
//                    didn't succeed.
// - nativeSet()    — read-only accessor; needed only by MetalStream's
//                    ctor to wire the set into MetalMTL4Backend's queue
//                    via addQueueResidency:.
// NOT thread-safe.

#import <Metal/Metal.h>

#if !defined(ET_METAL4_AVAILABLE)
#if defined(__MAC_OS_X_VERSION_MAX_ALLOWED) && __MAC_OS_X_VERSION_MAX_ALLOWED >= 150000
#define ET_METAL4_AVAILABLE 1
#else
#define ET_METAL4_AVAILABLE 0
#endif
#endif

namespace executorch {
namespace backends {
namespace metal_v2 {

class ResidencyManager {
 public:
  // device is borrowed (caller owns lifetime).
  explicit ResidencyManager(id<MTLDevice> device);
  ~ResidencyManager();

  ResidencyManager(const ResidencyManager&) = delete;
  ResidencyManager& operator=(const ResidencyManager&) = delete;

  void add(id<MTLBuffer> buffer);
  void remove(id<MTLBuffer> buffer);
  void commit();

  bool isEnabled() const { return enabled_; }

  // Read-only accessor for MetalStream's ctor wiring of
  // mtl4Backend_->addQueueResidency(...). Returns nil when disabled.
#if ET_METAL4_AVAILABLE
  id<MTLResidencySet> nativeSet() const API_AVAILABLE(macos(15.0), ios(18.0)) {
    return set_;
  }
#endif

 private:
  // The set is gated on @available macOS 15+ / iOS 18+. When the gate
  // doesn't pass we leave set_ as nil and enabled_ as false; all calls
  // become no-ops. Stored as id (untyped) so the field declaration
  // doesn't need API_AVAILABLE.
  id set_ = nil;
  bool enabled_ = false;
};

} // namespace metal_v2
} // namespace backends
} // namespace executorch
