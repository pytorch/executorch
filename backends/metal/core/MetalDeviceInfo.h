/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

//===----------------------------------------------------------------------===//
// MetalDeviceInfo — pure device-info queries (read-only, no encoder /
// command queue / buffer pool state).
// Replaces the pattern where ops needed a MetalStream just to read the
// active MTLDevice's properties (e.g., `isNaxAvailable`, `currentDeviceTier`,
// `supportsFamily(MTLGPUFamilyApple9)`). Constructing a full MetalStream is
// ~5ms (queue + backend + buffer pool + residency manager + MPS bridge);
// none of that is needed to answer "is this Apple9?".
// All methods cache on first call. The MTLDevice doesn't change at runtime
// so cached values are always valid.
// Thread-safe by construction: each cached value uses a function-local
// static (Meyers singleton). C++11 magic statics give exactly-once
// initialization with concurrent first-callers.
//===----------------------------------------------------------------------===//

#include <executorch/backends/metal/core/MetalTypes.h>

#import <Metal/Metal.h>

#include <cstring>

namespace executorch {
namespace backends {
namespace metal_v2 {

// Coarse classification of the GPU's perf bucket. Used by ops that pick
// different tile sizes / thresholds per device class. Mirrors v1's
// MatMulConfig::forDevice but lifted to a generic helper.
enum class DeviceTier {
  Phone,    // iPhone, iPad
  MacBase,  // M-series base / Pro
  MacUltra, // M-series Max / Ultra
};

// Pure helper — caller passes [[device name] UTF8String] to keep this
// function free of Metal imports.
inline DeviceTier getDeviceTierFromName(const char* deviceName) {
  if (deviceName == nullptr) return DeviceTier::MacBase;
  // Order matters: check Ultra/Max before generic substring matches.
  if (std::strstr(deviceName, "Ultra") || std::strstr(deviceName, "Max")) {
    return DeviceTier::MacUltra;
  }
  if (std::strstr(deviceName, "iPhone") || std::strstr(deviceName, "iPad") ||
      std::strstr(deviceName, "Apple A")) {
    return DeviceTier::Phone;
  }
  return DeviceTier::MacBase;
}

class MetalDeviceInfo {
 public:
  // Returns the system default MTLDevice (cached). Returns nil only on
  // headless / no-GPU environments.
  static id<MTLDevice> device();

  // Cheap probe: can we build a MetalStream on this machine? Calls
  // MTLCreateSystemDefaultDevice() once and releases — no side effects.
  static bool isAvailable();

  // Apple9+ (M3 / A17 family) — the cooperative-tensor matmul tier ("NAX").
  // Cached on first call.
  static bool isNaxAvailable();

  // Generic family check, cached per-family.
  static bool supportsFamily(MTLGPUFamily family);

  // Device class (Phone / MacBase / MacUltra) inferred from the device
  // name. Cached on first call.
  static DeviceTier tier();

  // For tests / advanced callers: drop the cached singleton MTLDevice so
  // a subsequent device() call re-probes. Does NOT invalidate the boolean
  // / tier caches (those are bound to the FIRST device that was probed;
  // production never swaps devices mid-process). Production code never
  // calls this.
  static void resetForTesting();

 private:
  MetalDeviceInfo() = delete;
};

} // namespace metal_v2
} // namespace backends
} // namespace executorch
