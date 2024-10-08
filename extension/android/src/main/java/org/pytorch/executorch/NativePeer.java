/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

package org.pytorch.executorch;

import com.facebook.jni.HybridData;
import com.facebook.jni.annotations.DoNotStrip;
import com.facebook.soloader.nativeloader.NativeLoader;
import java.util.Map;
import org.pytorch.executorch.annotations.Experimental;

/**
 * Interface for the native peer object for entry points to the Module
 *
 * <p>Warning: These APIs are experimental and subject to change without notice
 */
@Experimental
class NativePeer {
  static {
    // Loads libexecutorch.so from jniLibs
    NativeLoader.loadLibrary("executorch");
  }

  private final HybridData mHybridData;

  @DoNotStrip
  private static native HybridData initHybrid(
      String moduleAbsolutePath, Map<String, String> extraFiles, int loadMode);

  NativePeer(String moduleAbsolutePath, Map<String, String> extraFiles, int loadMode) {
    mHybridData = initHybrid(moduleAbsolutePath, extraFiles, loadMode);
  }

  /** Clean up the native resources associated with this instance */
  public void resetNative() {
    mHybridData.resetNative();
  }

  /** Run a "forward" call with the given inputs */
  @DoNotStrip
  public native EValue[] forward(EValue... inputs);

  /** Run an arbitrary method on the module */
  @DoNotStrip
  public native EValue[] execute(String methodName, EValue... inputs);

  /**
   * Load a method on this module.
   *
   * @return the Error code if there was an error loading the method
   */
  @DoNotStrip
  public native int loadMethod(String methodName);
}
