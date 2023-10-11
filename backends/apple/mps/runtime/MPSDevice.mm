//
//  Copyright (c) 2023 Apple Inc. All rights reserved.
//  Provided subject to the LICENSE file in the top level directory.
//

#include "MPSDevice.h"
#include <executorch/runtime/platform/assert.h>
#include <memory>
#include <mutex>

namespace torch {
namespace executor {
namespace mps {
namespace delegate {

static std::unique_ptr<MPSDevice> mps_device;
static std::once_flag mpsdev_init;

MPSDevice::~MPSDevice() {
  [_mtl_device release];
  _mtl_device = nil;
}

MPSDevice::MPSDevice(): _mtl_device(nil) {
  @autoreleasepool {
#if TARGET_OS_IPHONE
    _mtl_device = MTLCreateSystemDefaultDevice();
#else
    NSArray* devices = MTLCopyAllDevices();
    for (unsigned long i = 0 ; i < [devices count] ; i++) {
      id<MTLDevice>  device = devices[i];
      if(![device isLowPower]) { // exclude Intel GPUs
        _mtl_device = [device retain];
        break;
      }
    }
#endif
  }
  // MPS TODO: Replace with `ET_CHECK_OR_RETURN_ERROR` and propagate back the error.
  ET_CHECK(_mtl_device != nil);
}

MPSDevice* MPSDevice::getInstance() {
  std::call_once(mpsdev_init, [] {
      mps_device = std::unique_ptr<MPSDevice>(new MPSDevice());
  });
  return mps_device.get();
}

} // namespace delegate
} // namespace mps
} // namespace executor
} // namespace torch
