/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

//===----------------------------------------------------------------------===//
// Compiled with -fobjc-arc (see backends/metal/CMakeLists.txt). The
// process-singleton id<MTLDevice> s_device is __strong by default; ARC
// retains on assignment from MTLCreateSystemDefaultDevice() and releases
// when assigned nil. No manual [retain]/[release] needed.
//===----------------------------------------------------------------------===//

#import "MetalDeviceInfo.h"

#include <mutex>
#include <unordered_map>

namespace executorch {
namespace backends {
namespace metal_v2 {

namespace {

// Cached MTLDevice. Strong reference held for process lifetime — production
// never swaps the device mid-process. Tests can call resetForTesting() to
// drop the cache and force the next device() call to re-probe.
std::mutex s_device_mu;
bool s_device_initialized = false;
id<MTLDevice> s_device = nil;

void ensureDeviceInitialized() {
  std::lock_guard<std::mutex> lk(s_device_mu);
  if (s_device_initialized) return;
  @autoreleasepool {
    s_device = MTLCreateSystemDefaultDevice();
  }
  s_device_initialized = true;
}

}  // namespace

id<MTLDevice> MetalDeviceInfo::device() {
  ensureDeviceInitialized();
  return s_device;
}

bool MetalDeviceInfo::isAvailable() {
  // Probe without forcing the caller to construct first — callers can
  // gate before relying on device(). Caches via the same singleton because
  // MTLCreateSystemDefaultDevice itself caches at the OS level.
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
  std::lock_guard<std::mutex> lk(s_device_mu);
  s_device = nil;
  s_device_initialized = false;  // next device() call re-probes
}

} // namespace metal_v2
} // namespace backends
} // namespace executorch
