/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#import "MetalDeviceInfo.h"

#include <mutex>
#include <unordered_map>

namespace executorch {
namespace backends {
namespace metal_v2 {

namespace {

// Cached MTLDevice. We retain +1 for process lifetime — production never
// swaps the device mid-process. Tests can call resetForTesting() to drop
// the cache.
std::once_flag s_device_once;
id<MTLDevice> s_device = nil;

void ensureDeviceInitialized() {
  std::call_once(s_device_once, []() {
    @autoreleasepool {
      s_device = MTLCreateSystemDefaultDevice();
      if (s_device) [s_device retain];
    }
  });
}

}  // namespace

id<MTLDevice> MetalDeviceInfo::device() {
  ensureDeviceInitialized();
  return s_device;
}

bool MetalDeviceInfo::isAvailable() {
  // Probe without committing to caching the device — getDefault used to
  // hard-fail in its ctor, this lets callers gate before construction.
  // We DO cache through the same singleton because MTLCreateSystemDefaultDevice
  // is itself cached at the OS level, and we'd rather one stable identity
  // than two separate releases.
  ensureDeviceInitialized();
  return s_device != nil;
}

bool MetalDeviceInfo::isNaxAvailable() {
  static const bool cached = []() {
    id<MTLDevice> dev = device();
    return dev && [dev supportsFamily:MTLGPUFamilyApple9];
  }();
  return cached;
}

bool MetalDeviceInfo::supportsFamily(MTLGPUFamily family) {
  // Tiny per-family cache. Most call sites only ever ask about one family,
  // but kept generic so callers can probe Apple7/8/9 etc. without
  // adding methods.
  static std::mutex s_mu;
  static std::unordered_map<int, bool> s_cache;

  const int key = static_cast<int>(family);
  {
    std::lock_guard<std::mutex> lk(s_mu);
    auto it = s_cache.find(key);
    if (it != s_cache.end()) return it->second;
  }
  id<MTLDevice> dev = device();
  bool supported = dev && [dev supportsFamily:family];
  std::lock_guard<std::mutex> lk(s_mu);
  s_cache.emplace(key, supported);
  return supported;
}

DeviceTier MetalDeviceInfo::tier() {
  static DeviceTier cached = []() {
    id<MTLDevice> dev = device();
    if (!dev) return DeviceTier::MacBase;
    return getDeviceTierFromName([[dev name] UTF8String]);
  }();
  return cached;
}

void MetalDeviceInfo::resetForTesting() {
  if (s_device) {
    [s_device release];
    s_device = nil;
  }
  // Re-arm the once_flag by trick: rely on s_device==nil to trigger
  // re-probe via ensureDeviceInitialized. But std::once_flag can't be
  // reset; for testing we just leave it triggered and let device() return
  // nil. Tests that need a re-probe should design around this.
  // (Production never uses this.)
}

} // namespace metal_v2
} // namespace backends
} // namespace executorch
