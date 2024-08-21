//
//  Copyright (c) 2023 Apple Inc. All rights reserved.
//  Provided subject to the LICENSE file in the top level directory.
//

#pragma once

// Obj-C headers
#include <Foundation/Foundation.h>
#include <Metal/Metal.h>

// Runtime headers
#include <executorch/runtime/backend/interface.h>

// MPS headers
#include <MetalPerformanceShaders/MetalPerformanceShaders.h>

#include <unordered_map>
#include <vector>

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
  MACOS_VER_15_0_PLUS,
};

enum class LibraryType : uint32_t {
  INDEXING_KERNELS = 0,
  MAX = INDEXING_KERNELS,
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

  /**
   * Compile a PSO for a given library type.
   * Once compiled, the library and PSOs are cached.
   */
  Error compilePSO(LibraryType libraryType, const char* kernelName);
  Error compileLibrary(LibraryType);

 private:
  static MPSDevice* _device;
  id<MTLDevice> _mtl_device;
  std::unordered_map<LibraryType, id<MTLLibrary>> _m_library_cache;
  std::unordered_map<std::string, id<MTLComputePipelineState>> _m_pso_cache;
  MPSDevice();
};

bool is_macos_13_or_newer(
    MacOSVersion version = MacOSVersion::MACOS_VER_13_0_PLUS);

} // namespace delegate
} // namespace mps
} // namespace executor
} // namespace torch
