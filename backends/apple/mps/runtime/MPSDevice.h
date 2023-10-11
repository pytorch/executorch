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

  ~MPSDevice();

 private:
  static MPSDevice* _device;
  id<MTLDevice> _mtl_device;
  MPSDevice();
};

} // namespace delegate
} // namespace mps
} // namespace executor
} // namespace torch
