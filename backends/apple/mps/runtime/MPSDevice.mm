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

static inline MTLLanguageVersion getMetalLanguageVersion(const id<MTLDevice>& device, bool macOS13Plus) {
  // MPS Advanced Indexing needs at least Metal 2.0 (support for Argument Buffers and function constants)
  // host_name attribute needs at least Metal 2.2 and ulong needs Metal 2.3 (supported on MacOS 11+)
  MTLLanguageVersion languageVersion = MTLLanguageVersion2_3;
#if defined(__MAC_13_0)
  if (macOS13Plus) {
    languageVersion = MTLLanguageVersion3_0;
  }
#endif

  ET_CHECK_MSG([device supportsFamily:MTLGPUFamilyMac2], "Missing Metal support for MTLGPUFamilyMac2");
  return languageVersion;
}

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

bool MPSDevice::isMacOS13Plus(MacOSVersion version) const {
  id mpsCD = NSClassFromString(@"MPSGraph");
  static auto compileOptions = [[[MTLCompileOptions alloc] init] autorelease];
  static bool _macos_13_0_plus = [mpsCD instancesRespondToSelector:@selector(cumulativeSumWithTensor:
                                                                                                axis:name:)] == YES;
  static bool _macos_13_1_plus =
      [mpsCD instancesRespondToSelector:@selector
             (sampleGridWithSourceTensor:
                        coordinateTensor:layout:normalizeCoordinates:relativeCoordinates:alignCorners:paddingMode
                                        :samplingMode:constantValue:name:)] == YES;
  static bool _macos_13_2_plus =
      [mpsCD instancesRespondToSelector:@selector(convolution3DWithSourceTensor:weightsTensor:descriptor:name:)] == YES;
  static bool _macos_13_3_plus = [compileOptions respondsToSelector:@selector(maxTotalThreadsPerThreadgroup)] == YES;

  static bool _macos_14_0_plus = [mpsCD instancesRespondToSelector:@selector(conjugateWithTensor:name:)] == YES;
  static bool _macos_15_0_plus = [mpsCD instancesRespondToSelector:@selector(scaledDotProductAttentionWithQueryTensor:keyTensor:valueTensor:maskTensor:scale:name:)] == YES;
  switch (version) {
    case MacOSVersion::MACOS_VER_13_0_PLUS:
      return _macos_13_0_plus;
    case MacOSVersion::MACOS_VER_13_1_PLUS:
      return _macos_13_1_plus;
    case MacOSVersion::MACOS_VER_13_2_PLUS:
      return _macos_13_2_plus;
    case MacOSVersion::MACOS_VER_13_3_PLUS:
      return _macos_13_3_plus;
    case MacOSVersion::MACOS_VER_14_0_PLUS:
      return _macos_14_0_plus;
    case MacOSVersion::MACOS_VER_15_0_PLUS:
      return _macos_15_0_plus;
    default:
      return false;
  }
}

const char* getLibraryCString(LibraryType libraryType) {
  switch (libraryType) {
    case LibraryType::INDEXING_KERNELS:
      return "TODO";
    default:
      ET_CHECK_MSG(false, "Unhandled library type!");
  }
}

Error
MPSDevice::compileLibrary(LibraryType libraryType) {
  Error err = Error::Ok;
  NSError* error = nil;
  MTLCompileOptions* options = [MTLCompileOptions new];
  [options setLanguageVersion:getMetalLanguageVersion(_mtl_device, isMacOS13Plus(MacOSVersion::MACOS_VER_13_0_PLUS))];
  [options setFastMathEnabled:YES];
  id<MTLLibrary> lib =
      [_mtl_device newLibraryWithSource:[NSString stringWithCString:getLibraryCString(libraryType)
                                                           encoding:NSASCIIStringEncoding]
                                options:options
                                  error:&error];

  ET_CHECK_OR_RETURN_ERROR(
    lib != nil,
    Internal,
    "Failed to create indexing library, error: %s", [[error description] UTF8String]
  );

  _m_library_cache[libraryType] = lib;
  return err;
}

Error
MPSDevice::compilePSO(LibraryType libraryType, const char* kernelName) {
  Error err = Error::Ok;
  if (_m_library_cache.find(libraryType) == _m_library_cache.end()) {
    ET_LOG(Debug, "Compiling library type: %d", libraryType);
    err = compileLibrary(libraryType);
    ET_CHECK_OR_RETURN_ERROR(
      err == Error::Ok,
      Internal,
      "An error occured occured while compiling library %d", libraryType
    );
  }
  if (_m_pso_cache.find(kernelName) == _m_pso_cache.end()) {
    ET_LOG(Debug, "Compiling kernel: %s", kernelName);
    // err = compilePSO(libraryType, kernelName);
  }
  return err;
}

bool is_macos_13_or_newer(MacOSVersion version) {
  return MPSDevice::getInstance()->isMacOS13Plus(version);
}

} // namespace delegate
} // namespace mps
} // namespace executor
} // namespace torch
