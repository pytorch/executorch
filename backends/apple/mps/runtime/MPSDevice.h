//
//  Copyright (c) 2023 Apple Inc. All rights reserved.
//  Provided subject to the LICENSE file in the top level directory.
//

#pragma once

#include <Foundation/Foundation.h>
#include <Metal/Metal.h>
#include <MetalPerformanceShaders/MetalPerformanceShaders.h>

#define MB(x) (x * 1048576UL)

namespace torch {
namespace executor {
namespace mps {
namespace delegate {

// Helper enum to check if a MPSGraph op is supported in a given macOS version
enum class MacOSVersion : uint32_t {
  MACOS_VER_13_0_PLUS = 0,
  MACOS_VER_13_1_PLUS,
  MACOS_VER_13_2_PLUS,
  MACOS_VER_13_3_PLUS,
  MACOS_VER_14_0_PLUS,
};

class MPSDevice {
 public:
  /**
   * MPSDevice should not be cloneable.
   */
  MPSDevice(MPSDevice& other) = delete;
  /**
   * MPSDevice should not be assignable.
   */
  void operator=(const MPSDevice&) = delete;
  /**
   * Gets single instance of the Device.
   */
  static MPSDevice* getInstance();
  /**
   * Returns the single device.
   */
  id<MTLDevice> device() {
    return _mtl_device;
  }

  /**
   * Returns whether running on Ventura or newer
   */
  bool isMacOS13Plus(MacOSVersion version) const;

  ~MPSDevice();

 private:
  static MPSDevice* _device;
  id<MTLDevice> _mtl_device;
  MPSDevice();
};

bool isMacOS13OrNewer(MacOSVersion version = MacOSVersion::MACOS_VER_13_0_PLUS);

} // namespace delegate
} // namespace mps
} // namespace executor
} // namespace torch
