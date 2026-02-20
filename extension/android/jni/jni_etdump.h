/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#ifdef EXECUTORCH_ANDROID_PROFILING
#include <executorch/devtools/etdump/etdump_flatcc.h>

namespace executorch {
namespace extension {
namespace jni {

/**
 * ETDumpManager manages the global profiling enabled state.
 * This is a singleton that controls whether newly loaded modules
 * should capture profiling data.
 */
class ETDumpManager {
 public:
  ETDumpManager() : profiling_enabled_(false) {}

  /**
   * Enable profiling for subsequently loaded modules.
   */
  void enableProfiling() {
    profiling_enabled_ = true;
  }

  /**
   * Disable profiling.
   */
  void disableProfiling() {
    profiling_enabled_ = false;
  }

  /**
   * Check if profiling is currently enabled.
   * @return true if profiling is enabled
   */
  bool isProfilingEnabled() const {
    return profiling_enabled_;
  }

 private:
  bool profiling_enabled_;
};

/**
 * Get the global ETDump manager instance.
 * @return Pointer to the global ETDumpManager, or nullptr if not initialized
 */
extern "C" ETDumpManager* getGlobalETDumpManager();

} // namespace jni
} // namespace extension
} // namespace executorch

#endif // EXECUTORCH_ANDROID_PROFILING
