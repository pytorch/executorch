/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

package org.pytorch.executorch;

import com.facebook.soloader.nativeloader.NativeLoader;
import com.facebook.soloader.nativeloader.SystemDelegate;

class Runtime {

  static {
    if (!NativeLoader.isInitialized()) {
      NativeLoader.init(new SystemDelegate());
    }
    // Loads libexecutorch.so from jniLibs
    NativeLoader.loadLibrary("executorch");
  }

  private static final Runtime sInstance = new Runtime();

  private Runtime() {}

  /**
   * Get the runtime instance.
   */
  static Runtime getRuntime() {
    return sInstance;
  }
}
